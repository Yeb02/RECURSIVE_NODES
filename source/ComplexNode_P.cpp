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

void ComplexNode_P::setArrayPointers(float** post_syn_acts, float** pre_syn_acts, float** aa, float** acc_pre_syn_acts) {

	// TODO ? if the program runs out of heap memory, one could make it so that a node does not store its own 
	// output. But prevents in place matmul, and complexifies things.

	postSynActs = *post_syn_acts;
	preSynActs = *pre_syn_acts;


	*post_syn_acts += type->inputSize + MODULATION_VECTOR_SIZE;
	*pre_syn_acts += type->outputSize + MODULATION_VECTOR_SIZE;


#ifdef SATURATION_PENALIZING
	averageActivation = *aa;
	*aa += type->outputSize + type->inputSize + MODULATION_VECTOR_SIZE;
#endif

#ifdef STDP
	accumulatedPreSynActs = *acc_pre_syn_acts;
	*acc_pre_syn_acts += type->outputSize + MODULATION_VECTOR_SIZE;
#endif

	for (int i = 0; i < complexChildren.size(); i++) {
		*post_syn_acts += complexChildren[i].type->outputSize;
		*pre_syn_acts += complexChildren[i].type->inputSize;
#ifdef STDP
		*acc_pre_syn_acts += complexChildren[i].type->inputSize;
#endif
	}
	for (int i = 0; i < memoryChildren.size(); i++) {
		memoryChildren[i].setArrayPointers(*post_syn_acts, *pre_syn_acts, totalM, aa, acc_pre_syn_acts);
		*post_syn_acts += memoryChildren[i].type->outputSize;
		*pre_syn_acts += memoryChildren[i].type->inputSize;
	}

	for (int i = 0; i < complexChildren.size(); i++) {
		complexChildren[i].setArrayPointers(post_syn_acts, pre_syn_acts, aa, acc_pre_syn_acts);
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


#ifdef RANDOM_W
	toComplex.randomInitW();
	toMemory.randomInitW();
	toModulation.randomInitW();
	toOutput.randomInitW();
#endif
	
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
	

	// TODO there are no reasons not to propagate several times through MODULATION and MEMORY, for instance:
	// MODULATION -> MEMORY -> COMPLEX -> MODULATION -> MEMORY -> OUTPUT ...
	// And it can even be node specific. To be evolved ?

#ifdef SATURATION_PENALIZING
	constexpr float saturationExponent = .5f; 
#endif

	// STEP 0: initialize all pre-synaptic activations with the associated weights

	{
		int id = 0;
		for (int i = 0; i < type->outputSize; i++) {
			preSynActs[id] = type->outputBias[id];
			id++;
		}
		for (int i = 0; i < MODULATION_VECTOR_SIZE; i++) {
			preSynActs[id] = type->modulationBias[i];
			id++;
		}
		for (int i = 0; i < type->complexBiasSize; i++) {
			preSynActs[id] = type->complexBias[i];
			id++;
		}
		for (int i = 0; i < type->memoryBiasSize; i++) {
			preSynActs[id] = type->memoryBias[i];
			id++;
		}
	}
	


	// STEP 1 to 4: for each of the 4 types of nodes (output, memory, complex, modulation), do:
	
	// - propagate postSynActs in preSynActs of all children of the type, and apply their non linearities.
	// - apply hebbian update to the involved connexion matrices
	// - apply all children of the type's forward
	
	
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

#ifdef RANDOM_W
		float* w = icp.w.get();
#else
		float* w = icp.type->w.get();
#endif
		

			
		for (int i = 0; i < nl; i++) {
			for (int j = 0; j < nc; j++) {
				// += (H * alpha + w + wL) * prevAct
				destinationArray[i] += (H[matID] * alpha[matID] + w[matID] + wLifetime[matID]) * postSynActs[j];
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


#ifdef OJA
		float* delta = icp.type->delta.get();
#ifdef RANDOM_W
		float* w = icp.w.get();
#else
		float* w = icp.type->delta.get();
#endif
#ifndef CONTINUOUS_LEARNING
		float* wLifetime = icp.wLifetime.get();
		float* alpha = icp.type->alpha.get();
#endif
#endif


		for (int i = 0; i < nl; i++) {
			for (int j = 0; j < nc; j++) {
#ifdef CONTINUOUS_LEARNING
				wLifetime[matID] = (1 - gamma[matID]) * wLifetime[matID] + gamma[matID] * H[matID] * alpha[matID] * totalM[1];
#endif
				E[matID] = (1.0f - eta[matID]) * E[matID] + eta[matID] *
					(A[matID] * destinationArray[i] * postSynActs[j] + B[matID] * destinationArray[i] + C[matID] * postSynActs[j] + D[matID]);

#ifdef OJA
				E[matID] -= eta[matID] * destinationArray[i] * destinationArray[i] * delta[matID] * (w[matID] + alpha[matID]*H[matID] + wLifetime[matID]);
#endif

				H[matID] += E[matID] * totalM[0];
				H[matID] = std::clamp(H[matID], -1.0f, 1.0f);
#ifndef CONTINUOUS_LEARNING
				avgH[matID] += H[matID];
#endif
				matID++;

			}
		}
	};

	auto applyNonLinearities = [](float* src, float* dst, ACTIVATION* fcts, int size
#ifdef STDP
		, float* acc_src, float decay
#endif
		) 
	{
#ifdef STDP
		for (int i = 0; i < size; i++) {
			acc_src[i] = acc_src[i] * (1.0f - decay) + decay * src[i];
		}

		src = acc_src;
#endif

		for (int i = 0; i < size; i++) {
			switch (fcts[i]) {
			case TANH:
				dst[i] = tanhf(src[i]);
				break;
			case GAUSSIAN:
				dst[i] = 2.0f * expf(-std::clamp(powf(src[i], 2.0f), -10.0f, 10.0f)) - 1.0f; // technically the bias is not correctly put in. Does it matter ?
				break;
			case RELU:
				dst[i] = std::max(src[i], 0.0f);
				break;
			case LOG2:
				dst[i] = std::max(log2f(abs(src[i])), -100.0f);
				break;
			case EXP2:
				dst[i] = exp2f(std::clamp(src[i], -30.0f, 30.0f));
				break;
			case SINE:
				dst[i] = sinf(src[i]);
				break;
			case CENTERED_TANH:
				constexpr float z = 1.0f / .375261f; // to map to [-1, 1]
				dst[i] = tanhf(src[i]) * expf(-std::clamp(powf(src[i], 2.0f), -10.0f, 10.0f)) * z;
				break;
			}
		}
	};



	// STEP 1: MODULATION  A
	{
		propagate(toModulation, preSynActs + type->outputSize);
		applyNonLinearities(
			preSynActs + type->outputSize,
			postSynActs + type->inputSize, 
			type->modulationActivations, 
			MODULATION_VECTOR_SIZE
#ifdef STDP
			, accumulatedPreSynActs + type->outputSize, type->STDP_decays[2]
#endif
		);
		hebbianUpdate(toModulation, postSynActs + type->inputSize);

		for (int i = 0; i < MODULATION_VECTOR_SIZE; i++) {
			totalM[i] += postSynActs[i + type->inputSize];
		}

#ifdef SATURATION_PENALIZING 
		for (int j = type->outputSize; j < MODULATION_VECTOR_SIZE + type->outputSize; j++) {
			*globalSaturationAccumulator += powf(abs(preSynActs[j]), saturationExponent);
			averageActivation[j] += preSynActs[j];
		}
#endif
	}



	// STEP 2: MEMORY A
	if (memoryChildren.size() != 0) {
		// Nothing is transmitted between this and the memory children, as their pointers
		// point towards the same data arrays.

		propagate(toMemory, memoryChildren[0].input);


		int activationFunctionId = 0;
		for (int i = 0; i < memoryChildren.size(); i++) {

			// in place, as the memory child's input is shared with this node.
			applyNonLinearities(
				memoryChildren[i].input,
				memoryChildren[i].input,
				&type->memoryActivations[activationFunctionId],
				memoryChildren[i].type->inputSize
#ifdef STDP
				, memoryChildren[i].accumulatedInput, memoryChildren[i].type->STDP_decay
#endif
			);

#ifdef SATURATION_PENALIZING
			for (int j = 0; j < memoryChildren[i].type->inputSize; j++) {
				float v = memoryChildren[i].input[j];
				*globalSaturationAccumulator += powf(v, saturationExponent);
				memoryChildren[i].saturationArray[j] += v;
			}
#endif

			activationFunctionId += memoryChildren[i].type->inputSize;
		}


		hebbianUpdate(toMemory, memoryChildren[0].input);


		for (int i = 0; i < memoryChildren.size(); i++) {
			memoryChildren[i].forward();
		}

	}



	// STEP 2: COMPLEX
	if (complexChildren.size() != 0) {
		float* ptrToInputs = preSynActs + type->outputSize + MODULATION_VECTOR_SIZE;
#ifdef STDP
		float* ptrToAccInputs = accumulatedPreSynActs + type->outputSize + MODULATION_VECTOR_SIZE;
#endif
		propagate(toComplex, ptrToInputs);
		
		

		// Apply non-linearities
		int id = 0;
		for (int i = 0; i < complexChildren.size(); i++) {


			applyNonLinearities(
				ptrToInputs + id,
				complexChildren[i].postSynActs,
				&type->complexActivations[id],
				complexChildren[i].type->inputSize
#ifdef STDP
				, ptrToAccInputs + id, complexChildren[i].type->STDP_decays[0]
#endif
			);

#ifdef SATURATION_PENALIZING 
			// child post-syn input
			for (int j = 0; j < complexChildren[i].type->inputSize; j++) {
				float v = *(ptrToInputs + id + j);
				*globalSaturationAccumulator += powf(abs(v), saturationExponent);
				complexChildren[i].averageActivation[j] += v;
			}
#endif

			id += complexChildren[i].type->inputSize;
		}

		// has to happen after non linearities but before forward, 
		// for children's output not to have changed yet.
		hebbianUpdate(toComplex, ptrToInputs);


		// transmit modulation and apply forward, then retrieve the child's output.
		float* childOut = postSynActs + type->inputSize + MODULATION_VECTOR_SIZE;
		for (int i = 0; i < complexChildren.size(); i++) {

			for (int j = 0; j < MODULATION_VECTOR_SIZE; j++) {
				complexChildren[i].totalM[j] = this->totalM[j];
			}

			complexChildren[i].forward();

			std::copy(complexChildren[i].preSynActs, complexChildren[i].preSynActs + complexChildren[i].type->outputSize, childOut);
			childOut += complexChildren[i].type->outputSize;
		}

	}


	// STEP 4: MODULATION B
	{
		propagate(toModulation, preSynActs + type->outputSize);
		applyNonLinearities(
			preSynActs + type->outputSize,
			postSynActs + type->inputSize,
			type->modulationActivations,
			MODULATION_VECTOR_SIZE
#ifdef STDP
			, accumulatedPreSynActs + type->outputSize, type->STDP_decays[2]
#endif
		);
		hebbianUpdate(toModulation, postSynActs + type->inputSize);

		for (int i = 0; i < MODULATION_VECTOR_SIZE; i++) {
			totalM[i] += postSynActs[i + type->inputSize];
		}

#ifdef SATURATION_PENALIZING 
		for (int j = type->outputSize; j < MODULATION_VECTOR_SIZE + type->outputSize; j++) {
			*globalSaturationAccumulator += powf(abs(preSynActs[j]), saturationExponent);
			averageActivation[j] += preSynActs[j];
		}
#endif
	}


	// STEP 5: MEMORY B
	if (memoryChildren.size() != 0) {
		// Nothing is transmitted between this and the memory children, as their pointers
		// point towards the same data arrays.

		propagate(toMemory, memoryChildren[0].input);


		int activationFunctionId = 0;
		for (int i = 0; i < memoryChildren.size(); i++) {

			// in place, as the memory child's input is shared with this node.
			applyNonLinearities(
				memoryChildren[i].input,
				memoryChildren[i].input,
				&type->memoryActivations[activationFunctionId],
				memoryChildren[i].type->inputSize
#ifdef STDP
				, memoryChildren[i].accumulatedInput, memoryChildren[i].type->STDP_decay
#endif
			);

#ifdef SATURATION_PENALIZING
			for (int j = 0; j < memoryChildren[i].type->inputSize; j++) {
				float v = memoryChildren[i].input[j];
				*globalSaturationAccumulator += powf(v, saturationExponent);
				memoryChildren[i].saturationArray[j] += v;
			}
#endif

			activationFunctionId += memoryChildren[i].type->inputSize;
		}


		hebbianUpdate(toMemory, memoryChildren[0].input);


		for (int i = 0; i < memoryChildren.size(); i++) {
			memoryChildren[i].forward();
		}

	}


	// STEP 6: OUTPUT
	{
		propagate(toOutput, preSynActs);
		
		
		applyNonLinearities(
			preSynActs,
			preSynActs,
			type->outputActivations.data(),
			type->outputSize
#ifdef STDP
			, accumulatedPreSynActs, type->STDP_decays[1]
#endif
		);

		hebbianUpdate(toOutput, preSynActs);
	}
	
}
