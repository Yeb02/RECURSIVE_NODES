#include "ComplexNode_P.h"



ComplexNode_P::ComplexNode_P(ComplexNode_G* _type) : 
	type(_type),
	toComplex(&_type->toComplex),
	toMemory(&_type->toMemory),
	toModulation(&_type->toModulation),
	toOutput(&_type->toOutput)
{
	// create COMPLEX children recursively 
	complexChildren.reserve(type->complexChildren.size());
	for (int i = 0; i < type->complexChildren.size(); i++) {
		complexChildren.emplace_back(type->complexChildren[i]);
	}

	// create MEMORY children  
	memoryChildren.reserve(type->memoryChildren.size());
	for (int i = 0; i < type->memoryChildren.size(); i++) {
		memoryChildren.emplace_back(type->memoryChildren[i]);
	}

	// TotalM is not initialized (i.e. zeroed) here because a call to preTrialReset() 
	// must be made before any forward pass. 
};

#if defined GUIDED_MUTATIONS
void ComplexNode_P::accumulateW(float factor) {
	
	type->toComplex.accumulateW(factor, toComplex.wLifetime.get());
	type->toMemory.accumulateW(factor, toMemory.wLifetime.get());
	type->toModulation.accumulateW(factor, toModulation.wLifetime.get());
	type->toOutput.accumulateW(factor, toOutput.wLifetime.get());

	for (int i = 0; i < type->memoryChildren.size(); i++) {
		memoryChildren[i].accumulateW(factor);
	}

	for (int i = 0; i < complexChildren.size(); i++) {
		complexChildren[i].accumulateW(factor);
	}
}
#endif

#ifndef CONTINUOUS_LEARNING
void ComplexNode_P::updateWatTrialEnd(float invNInferencesOverTrial) {

	for (int i = 0; i < complexChildren.size(); i++) {
		complexChildren[i].updateWatTrialEnd(invNInferencesOverTrial);
	}

	toComplex.updateWatTrialEnd(invNInferencesOverTrial);
	toMemory.updateWatTrialEnd(invNInferencesOverTrial);
	toModulation.updateWatTrialEnd(invNInferencesOverTrial);
	toOutput.updateWatTrialEnd(invNInferencesOverTrial);
	
	for (int i = 0; i < memoryChildren.size(); i++) {
		memoryChildren[i].pLink.updateWatTrialEnd(invNInferencesOverTrial);
	}
}
#endif

void ComplexNode_P::setArrayPointers(float** ppsa, float** cpsa, float** psa, float** aa) {

	// TODO ? if the program runs out of heap memory, one could make it so that a node does not store its own 
	// output. But prevents in place matmul, and complexifies things.

	previousPostSynAct = *ppsa;
	currentPostSynAct = *cpsa;
	preSynAct = *psa;
#ifdef SATURATION_PENALIZING
	averageActivation = *aa;
#endif

	*ppsa += type->inputSize + MODULATION_VECTOR_SIZE;
	*cpsa += type->inputSize + MODULATION_VECTOR_SIZE;
	*psa += type->outputSize + MODULATION_VECTOR_SIZE;
#ifdef SATURATION_PENALIZING
	*aa += type->outputSize + type->inputSize + MODULATION_VECTOR_SIZE;
#endif

	for (int i = 0; i < complexChildren.size(); i++) {
		*ppsa += complexChildren[i].type->outputSize;
		*cpsa += complexChildren[i].type->outputSize;
		*psa += complexChildren[i].type->inputSize;
	}
	for (int i = 0; i < memoryChildren.size(); i++) {
		memoryChildren[i].setArrayPointers(cpsa, psa, currentPostSynAct + type->inputSize);
		*ppsa += memoryChildren[i].type->outputSize;
		*cpsa += memoryChildren[i].type->outputSize;
		*psa += memoryChildren[i].type->inputSize;
	}

	for (int i = 0; i < complexChildren.size(); i++) {
		complexChildren[i].setArrayPointers(ppsa, cpsa, psa, aa);
	}
}


void ComplexNode_P::preTrialReset() {

	for (int i = 0; i < complexChildren.size(); i++) {
		complexChildren[i].preTrialReset();
	}

	for (int i = 0; i < memoryChildren.size(); i++) {
		memoryChildren[i].preTrialReset();
	}

	toComplex.zero();
	toMemory.zero();
	toModulation.zero();
	toOutput.zero();
}


#ifdef SATURATION_PENALIZING
void ComplexNode_P::setglobalSaturationAccumulator(float* globalSaturationAccumulator) {
	this->globalSaturationAccumulator = globalSaturationAccumulator;
	for (int i = 0; i < complexChildren.size(); i++) {
		complexChildren[i].setglobalSaturationAccumulator(globalSaturationAccumulator);
	}
}
#endif


void ComplexNode_P::forward() {
	// TODO is it global modulation or local modulation that should
	// be used as a modulation node output, after its own hebbian update ?

	// TODO there are no reasons not to propagate several times through certain types, for instance:
	// MODULATION -> MEMORY -> COMPLEX -> MODULATION -> MEMORY -> OUTPUT ...
	// And it can even be node specific. To be evolved ?

	// TODO hebbian update before or after non linearity ?
	// After is problematic in this implementation, because the original (currentPostSynAct) input
	// from the node of the same type has changed... 
	// We could use previousPostSynAct for this type, which has it stored, but it requires extra code.
	// (thats previousPostSynAct sole use at the moment) Also be wary of global modulation addition.

#ifdef SATURATION_PENALIZING
	constexpr float saturationExponent = 10.0f;  // one could try lower values... But they must be 2*integer.
#endif

	// STEP 0: initialize all pre-synaptic activations with the associated weights

	{
		int id = 0;
		for (int i = 0; i < type->outputSize; i++) {
			preSynAct[id] = type->outputBias[id];
			id++;
		}
		for (int i = 0; i < MODULATION_VECTOR_SIZE; i++) {
			preSynAct[id] = type->modulationBias[i];
			id++;
		}
		for (int i = 0; i < type->complexBiasSize; i++) {
			preSynAct[id] = type->complexBias[i];
			id++;
		}
		for (int i = 0; i < type->memoryBiasSize; i++) {
			preSynAct[id] = type->memoryBias[i];
			id++;
		}
	}
	


	// STEP 1 to 4: for each of the 4 types of nodes (output, memory, complex, modulation), do:
	
	// - propagate currentPostSynAct in preSynAct of all children of the type
	// - apply all children of the type's forward
	// - apply hebbian update to the involved connexion matrices
	
	// This could be done simultaneously for all types, but doing it this way drastically speeds up information transmission
	// through the network. The following order is used :
	// MODULATION -> COMPLEX -> MEMORY -> OUTPUT

	// These 3 lambdas, hopefully inline, avoid repetition, as they are used for each child type.

	auto propagate = [this](InternalConnexion_P& icp, float* destinationArray)
	{
		int nl = icp.type->nLines;
		int nc = icp.type->nColumns;
		int matID = 0;

		float* H = icp.H.get();
		float* wLifetime = icp.wLifetime.get();
		float* alpha = icp.type->alpha.get();
		float* w = icp.type->w.get();

			
		for (int i = 0; i < nl; i++) {
			for (int j = 0; j < nc; j++) {
				// += (H * alpha + w) * prevAct
				destinationArray[i] += (H[matID] * alpha[matID] + w[matID] + wLifetime[matID]) * currentPostSynAct[j];
				matID++;
			}
		}
	};

	auto hebbianUpdate = [this](InternalConnexion_P& icp, float* destinationArray) {
		int nl = icp.type->nLines;
		int nc = icp.type->nColumns;
		int matID = 0;

		float* A = icp.type->A.get();
		float* B = icp.type->B.get();
		float* C = icp.type->C.get();
		float* D = icp.type->D.get();
		float* eta = icp.type->eta.get();
		float* H = icp.H.get();
		float* E = icp.E.get();

#ifdef CONTINUOUS_LEARNING
		float* wLifetime = icp.wLifetime.get();
		float* gamma = icp.type->gamma.get();
		float* alpha = icp.type->alpha.get();
#else
		float* avgH = icp.avgH.get();
#endif


		for (int i = 0; i < nl; i++) {
			for (int j = 0; j < nc; j++) {
#ifdef CONTINUOUS_LEARNING
				wLifetime[matID] = (1 - gamma[matID]) * wLifetime[matID] + gamma[matID] * alpha[matID] * H[matID] * totalM[1];
#endif
				E[matID] = (1.0f - eta[matID]) * E[matID] + eta[matID] *
					(A[matID] * destinationArray[i] * currentPostSynAct[j] + B[matID] * destinationArray[i] + C[matID] * currentPostSynAct[j] + D[matID]);

				H[matID] += E[matID] * totalM[0];
				H[matID] = std::max(-1.0f, std::min(H[matID], 1.0f));
#ifndef CONTINUOUS_LEARNING
				avgH[matID] += H[matID];
#endif
				matID++;

			}
		}
	};

	// Dont forget to update the copy of this function in network.step when this one changes.
	auto applyNonLinearities = [](float* src, float* dst, ACTIVATION* fcts, int size) 
	{
		for (int i = 0; i < size; i++) {
			switch (fcts[i]) {
			case TANH:
				dst[i] = tanhf(src[i]);
				break;
			case GAUSSIAN:
				dst[i] = 2.0f * expf(-src[i] * src[i]) - 1.0f; // technically the bias is not correctly put in. Does it matter ?
				break;
			case RELU:
				dst[i] = std::max(src[i], 0.0f);
				break;
			case LOG2:
				dst[i] = log2f(abs(src[i]));
				break;
			case EXP2:
				dst[i] = exp2f(src[i]);
				break;
			case SINE:
				dst[i] = sinf(src[i]);
				break;
			case CENTERED_TANH:
				constexpr float z = 1.0f / .375261f; // to map to [-1, 1]
				dst[i] = tanhf(src[i]) * expf(-powf(src[i], 2.0f)) * z;
				break;
			}
		}
	};

	// STEP 1: MODULATION 
	{
		propagate(toModulation, preSynAct + type->outputSize);
		hebbianUpdate(toModulation, preSynAct + type->outputSize);
		applyNonLinearities(
			preSynAct + type->outputSize,
			currentPostSynAct + type->inputSize, 
			type->modulationActivations, 
			MODULATION_VECTOR_SIZE
		);

		for (int i = 0; i < MODULATION_VECTOR_SIZE; i++) {
			totalM[i] += currentPostSynAct[i + type->inputSize];
		}

		

#ifdef SATURATION_PENALIZING
		for (int j = type->inputSize; j < MODULATION_VECTOR_SIZE + type->inputSize; j++) {
			*globalSaturationAccumulator += powf(currentPostSynAct[j], saturationExponent);
			averageActivation[j + type->outputSize] += currentPostSynAct[j];
		}
#endif

	}

	// STEP 2: COMPLEX
	{
		float* ptrToInputs = preSynAct + type->outputSize + MODULATION_VECTOR_SIZE;
		propagate(toComplex, ptrToInputs);
		hebbianUpdate(toComplex, ptrToInputs);
		

		// transmit modulation and input, apply forward, then retrieve the child's output.
		int id = 0;
		float* childOut = currentPostSynAct + type->inputSize + MODULATION_VECTOR_SIZE;
		for (int i = 0; i < complexChildren.size(); i++) {

			for (int j = 0; j < MODULATION_VECTOR_SIZE; j++) {
				complexChildren[i].totalM[j] = this->totalM[j];
			}

			applyNonLinearities(
				ptrToInputs + id,
				complexChildren[i].currentPostSynAct,
				&type->complexActivations[id],
				complexChildren[i].type->inputSize
			);

			complexChildren[i].forward();

			applyNonLinearities(
				complexChildren[i].preSynAct,
				childOut,
				type->complexChildren[i]->outputActivations.data(),
				complexChildren[i].type->outputSize
			);

#ifdef SATURATION_PENALIZING
			for (int j = 0; j < complexChildren[i].type->inputSize; j++) {
				* globalSaturationAccumulator += powf(complexChildren[i].currentPostSynAct[j], saturationExponent);
				complexChildren[i].averageActivation[j] += complexChildren[i].currentPostSynAct[j];
			}
			for (int j = 0; j < complexChildren[i].type->outputSize; j++) {
				*globalSaturationAccumulator += powf(complexChildren[i].preSynAct[j], saturationExponent);
				complexChildren[i].averageActivation[j + complexChildren[i].type->inputSize] += childOut[j];
			}
#endif

			id += complexChildren[i].type->inputSize;
			childOut += complexChildren[i].type->outputSize;
		}

	}

	// STEP 3: MEMORY
	{
		float* ptrToInputs = preSynAct + type->memoryPreSynOffset;
		propagate(toMemory, ptrToInputs);
		hebbianUpdate(toMemory, ptrToInputs);


		// modulation is not transmitted, as it is shared with this node.
		int id = 0;
		for (int i = 0; i < memoryChildren.size(); i++) {

			// in place, as the memory child's input is shared with this node.
			applyNonLinearities(
				ptrToInputs + id,
				ptrToInputs + id, // = memoryChildren[i].input
				&type->memoryActivations[id],
				memoryChildren[i].type->inputSize
			);

			memoryChildren[i].forward();

			// no retrieval, as the memory child's output is shared with this node.

			id += memoryChildren[i].type->inputSize;
		}

	}

	// STEP 4: OUTPUT
	{
		propagate(toOutput, preSynAct);
		hebbianUpdate(toOutput, preSynAct);
		
		// no call to applyNonLinearities, because it is the parent that calls
		// it to avoid one unnecessary copy. (the parent being Network for the top Node)
		// If saturation penalization is enabled, it is also handled by the parent.
	}
	
}
