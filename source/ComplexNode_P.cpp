#include "ComplexNode_P.h"



ComplexNode_P::ComplexNode_P(ComplexNode_G* type) : type(type)
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

	// create connexions ( in the appropriate order: TODO )
	internalConnexions.reserve(type->internalConnexions.size());
	for (int i = 0; i < type->internalConnexions.size(); i++) {
		internalConnexions.emplace_back(
			type->internalConnexions[i].nLines *
			type->internalConnexions[i].nColumns
		);
	}

	// LocalM, TotalM and nInferencesP are not instantiated here because a call to preTrialReset() 
	// must be made before any forward pass. (Or call to postTrialUpdate)
};

#if defined GUIDED_MUTATIONS
void ComplexNode_P::accumulateW(float factor) {
	type->nAccumulations++;
	for (int i = 0; i < internalConnexions.size(); i++) {
		int s = type->internalConnexions[i].nLines * type->internalConnexions[i].nColumns;
		for (int j = 0; j < s; j++) {
			type->internalConnexions[i].accumulator[j] += factor * internalConnexions[i].wLifetime[j];
			internalConnexions[i].wLifetime[j] = 0.0f;
		}
	}

	for (int i = 0; i < type->memoryChildren.size(); i++) {
		memoryChildren[i].accumulateW(factor);
	}

	for (int i = 0; i < complexChildren.size(); i++) {
		complexChildren[i].accumulateW(factor);
	}
}
#endif

#ifndef CONTINUOUS_LEARNING
void ComplexNode_P::updateWatTrialEnd(float invnInferencesP) {
	if (nInferencesP == 0) return; // should not have been called in the first place.

	for (int i = 0; i < complexChildren.size(); i++) {
		complexChildren[i].updateWatTrialEnd(invnInferencesP);
	}

	for (int i = 0; i < internalConnexions.size(); i++) {
		int s = type->internalConnexions[i].nLines * type->internalConnexions[i].nColumns;
		internalConnexions[i].updateWatTrialEnd(s, invnInferencesP, type->internalConnexions[i].alpha.get());
	}
}
#endif

void ComplexNode_P::setArrayPointers(float** ppsa, float** cpsa, float** psa, float** aa) {

	previousPostSynAct = *ppsa;
	currentPostSynAct = *cpsa;
	preSynAct = *psa;
#ifdef SATURATION_PENALIZING
	averageActivation = *aa;
#endif

	int s = type->inputSize + type->outputSize + (int) type->simpleChildren.size();
	*ppsa += s;
	*cpsa += s;
	*psa += s;
#ifdef SATURATION_PENALIZING
	*aa += ...;
#endif


	for (int i = 0; i < complexChildren.size(); i++) {
		complexChildren[i].setArrayPointers(ppsa, cpsa, psa, aa);
	}
	for (int i = 0; i < memoryChildren.size(); i++) {
		memoryChildren[i].setArrayPointers(ppsa, cpsa, psa, aa);
	}
}


void ComplexNode_P::preTrialReset() {

	for (int i = 0; i < complexChildren.size(); i++) {
		complexChildren[i].preTrialReset();
	}

	for (int i = 0; i < memoryChildren.size(); i++) {
		memoryChildren[i].preTrialReset();
	}

	for (int i = 0; i < internalConnexions.size(); i++) {
		int s = type->internalConnexions[i].nLines * type->internalConnexions[i].nColumns;
		internalConnexions[i].zero(s); // zero E, H, and AVG_H if need be. 
	}

	for (int i = 0; i < MODULATION_VECTOR_SIZE; i++) {
		totalM[i] = 0.0f;
		localM[i] = 0.0f;
	}

	nInferencesP = 0;
}


#ifdef SATURATION_PENALIZING
void ComplexNode_P::setSaturationPenalizationPtr(float* saturationPenalizationPtr) {
	this->saturationPenalizationPtr = saturationPenalizationPtr;
	for (int i = 0; i < children.size(); i++) {
		if (children[i].type->nodeType == ComplexNode_G::COMPLEX) {
			children[i].setSaturationPenalizationPtr(saturationPenalizationPtr);
		}
	}
}
#endif


void ComplexNode_P::forward() {

#ifdef SATURATION_PENALIZING
	constexpr float modulationMultiplier = 0.0f; // must be set to the same value in Genotype::getNnonLinearities. TODO cleaner.
	constexpr float saturationExponent = 20.0f;  // one could try lower values... But they must be 2*integer.
#endif

	nInferencesP++;


	// step 1: save this node's previous local m, for hebbian rules later on.

	float previousLocalM[MODULATION_VECTOR_SIZE];
	float localMpreSyn[MODULATION_VECTOR_SIZE];
	for (int i = 0; i < MODULATION_VECTOR_SIZE; i++) {
		previousLocalM[i] = localM[i];
	}


	// step 2: initialize all pre-synaptic activations with the associated weight

	{
		for (int i = 0; i < MODULATION_VECTOR_SIZE; i++) {
			localMpreSyn[i] = type->modulationBias[i];
		}

		int id = 0;
		for (int i = 0; i < type->outputSize; i++) {
			preSynAct[i + type->inputSize] = type->internalBias[id];
			id++;
		}
		for (int i = 0; i < type->simpleChildren.size(); i++) {
			preSynAct[i + type->inputSize] = type->internalBias[id];
			id++;
		}
		for (int i = 0; i < complexChildren.size(); i++) {
			for (int j = 0; j < complexChildren[i].type->inputSize; j++) {
				complexChildren[i].preSynAct[j] = type->internalBias[id];
				id++;
			}
		}
		for (int i = 0; i < memoryChildren.size(); i++) {
			for (int j = 0; j < memoryChildren[i].type->inputSize; j++) {
				memoryChildren[i].preSynAct[j] = type->internalBias[id];
				id++;
			}
		}
	}
	


	// STEP 3 to 7: NODE_TYPE by NODE_TYPE, follow those 3 substeps: 
	// - propagate currentPostSynAct in preSynAct of all NODE_TYPE children of "this"
	// - apply all NODE_TYPE children's forward
	// - apply hebbian update to the involved connexion
	
	// This could be done simultaneously for all NODE_TYPEs, but doing it this way drastically speeds up information transmition
	// through the network. The following order is respected (INPUT_NODE is handled by the parent, or by Network):
	// MODULATION -> COMPLEX -> SIMPLE -> MEMORY -> OUTPUT


	// These 2 lambdas, hopefully inline, avoid repetition, as they are used for each NODE_TYPE:

	// For each node type, to be called at the beginning of the process 
	auto propagate = [this](NODE_TYPE targetType, float* destinationArray)
	{
		int nc, nl, destinationID, originID, matID;
		for (int id = 0; id < internalConnexions.size(); id++) {

			if (type->internalConnexions[id].destinationType != targetType) {
				continue;
			}

			destinationID = type->internalConnexions[id].destinationID;
			originID = type->internalConnexions[id].originID;
			nl = type->internalConnexions[id].nLines;
			nc = type->internalConnexions[id].nColumns;
			matID = 0;

			float* H = internalConnexions[id].H.get();
			float* wLifetime = internalConnexions[id].wLifetime.get();
			float* alpha = type->internalConnexions[id].alpha.get();
			float* w = type->internalConnexions[id].w.get();

			float* originArray=nullptr;
			switch (type->internalConnexions[id].originType) {
			case INPUT_NODE:
				originArray = currentPostSynAct;
				break;
			case MODULATION:
				originArray = localM;
				break;
			case COMPLEX:
				originArray = complexChildren[originID].currentPostSynAct + type->complexChildren[originID]->inputSize;
				break;
			case MEMORY:
				originArray = memoryChildren[originID].currentPostSynAct + type->memoryChildren[originID]->inputSize;
				break;
			case SIMPLE:
				originArray = currentPostSynAct + type->inputSize + type->outputSize + originID;
				break;
			}

			if (targetType == COMPLEX) {
				destinationArray = complexChildren[destinationID].preSynAct;
			}
			else if (targetType == MEMORY) {
				destinationArray = memoryChildren[destinationID].preSynAct;
			}
			
			for (int i = 0; i < nl; i++) {
				for (int j = 0; j < nc; j++) {
					// += (H * alpha + w) * prevAct
					destinationArray[i] += (H[matID] * alpha[matID] + w[matID] + wLifetime[matID]) * originArray[j];
					matID++;
				}
			}
		}
		return;
	};

	// For each node type, to be called after applying the activation function on the presynatic inputs, but before forward.
	//  Since for simple neurons and modulation, these are merged,  this lambda is called after forward().
	auto hebbianUpdate = [this, &previousLocalM](NODE_TYPE targetType, float* iArray) {
		int nc, nl, destinationID, originID, matID;
		for (int id = 0; id < internalConnexions.size(); id++) {
			if (type->internalConnexions[id].destinationType != targetType) {
				continue;
			}
			originID = type->internalConnexions[id].originID;
			destinationID = type->internalConnexions[id].destinationID;

			float* A = type->internalConnexions[id].A.get();
			float* B = type->internalConnexions[id].B.get();
			float* C = type->internalConnexions[id].C.get();
			float* D = type->internalConnexions[id].D.get();
			float* eta = type->internalConnexions[id].eta.get();
			float* H = internalConnexions[id].H.get();
			float* E = internalConnexions[id].E.get();

#ifdef CONTINUOUS_LEARNING
			float* wLifetime = internalConnexions[id].wLifetime.get();
			float* gamma = type->internalConnexions[id].gamma.get();
			float* alpha = type->internalConnexions[id].alpha.get();
#else
			float* avgH = internalConnexions[id].avgH.get();
#endif

			// the output of the origin node
			float* jArray=nullptr;

			switch (type->internalConnexions[id].originType) {
			case INPUT_NODE:
				jArray = currentPostSynAct;
				break;
			case MODULATION:
				if (targetType == MODULATION) {
					jArray = previousLocalM;
				}
				else {
					jArray = localM;
				}
				break;
			case COMPLEX:
				jArray = complexChildren[originID].currentPostSynAct + complexChildren[originID].type->inputSize;
				break;
			case MEMORY:
				jArray = memoryChildren[originID].currentPostSynAct + memoryChildren[originID].type->inputSize;
				break;
			case SIMPLE:
				if (targetType == SIMPLE) {
					jArray = previousPostSynAct + type->inputSize + type->outputSize + originID;
				}
				else {
					jArray = currentPostSynAct + type->inputSize + type->outputSize + originID;
				}
				break;
			}


			// the post-synaptic input of the destination node
			// float* iArray;
			if (targetType == COMPLEX) {
				iArray = complexChildren[destinationID].currentPostSynAct;
			}
			else if (targetType == MEMORY) {
				iArray = memoryChildren[destinationID].currentPostSynAct;
			}


			nl = type->internalConnexions[id].nLines;
			nc = type->internalConnexions[id].nColumns;
			matID = 0;  // = i*nc+j
			for (int i = 0; i < nl; i++) {
				for (int j = 0; j < nc; j++) {
#ifdef CONTINUOUS_LEARNING
					wLifetime[matID] = (1 - gamma[matID]) * wLifetime[matID] + gamma[matID] * alpha[matID] * H[matID] * totalM[1];
#endif
					E[matID] = (1.0f - eta[matID]) * E[matID] + eta[matID] *
						(A[matID] * iArray[i] * jArray[j] + B[matID] * iArray[i] + C[matID] * jArray[j] + D[matID]);

					H[matID] += E[matID] * totalM[0];
					H[matID] = std::max(-1.0f, std::min(H[matID], 1.0f));
#ifndef CONTINUOUS_LEARNING
					avgH[matID] += H[matID];
#endif
					matID++;

				}
			}
		}

		return;
	};

	// STEP 3: MODULATION
	{
		propagate(MODULATION, localMpreSyn);
		
		// Apply forward.
		for (int i = 0; i < MODULATION_VECTOR_SIZE; i++) {
			localM[i] = tanhf(localMpreSyn[i]);
			totalM[i] += localM[i];
#ifdef SATURATION_PENALIZING
			* saturationPenalizationPtr += modulationMultiplier * powf(localM[i], saturationExponent);
			averageActivation[type->INPUT_NODESize + type->outputSize + i] += localM[i];
#endif
		}

		hebbianUpdate(MODULATION, localM);
	}

	// STEP 4: COMPLEX
	{
		propagate(COMPLEX, NULL);

		// transmit modulation and INPUT_NODE, then apply forward:
		for (int i = 0; i < complexChildren.size(); i++) {
			for (int j = 0; j < MODULATION_VECTOR_SIZE; j++) {
				complexChildren[i].totalM[j] = this->totalM[j];
			}

			for (int j = 0; j < complexChildren[i].type->inputSize; j++) {
				complexChildren[i].currentPostSynAct[j] = tanhf(complexChildren[i].preSynAct[j]);
#ifdef SATURATION_PENALIZING
				* saturationPenalizationPtr += powf(complexChildren[i].currentPostSynAct[j], saturationExponent);
#endif
			}
		}

		hebbianUpdate(COMPLEX, NULL);

		for (int i = 0; i < complexChildren.size(); i++) {
			complexChildren[i].forward();

#ifdef SATURATION_PENALIZING
			float* array = complexChildren[i].currentPostSynAct + complexChildren[i].type->INPUT_NODESize;
			for (int j = 0; j < complexChildren[i].type->outputSize; j++) {
				* saturationPenalizationPtr += powf(array[j], saturationExponent);
			}
#endif
		}

	}

	// STEP 5: SIMPLE
	{
		propagate(SIMPLE, preSynAct + type->inputSize + type->outputSize);

		for (int i = 0; i < type->simpleChildren.size(); i++) {
			int id = i + type->outputSize;

			previousPostSynAct[id] = currentPostSynAct[id];

			switch (type->simpleChildren[i]->activation) {
			case TANH:
				currentPostSynAct[id] = tanhf(preSynAct[id]);
				break;
			case GAUSSIAN:
				currentPostSynAct[id] = 2.0f * expf(-preSynAct[id] * preSynAct[id]) - 1.0f; // technically the bias is not correctly put in. Does it matter ?
				break;
			case SINE:
				currentPostSynAct[id] = sinf(preSynAct[id]);
				break;
			}

#ifdef SATURATION_PENALIZING
				* saturationPenalizationPtr += powf(currentPostSynAct[id], saturationExponent);
				averageActivation[...] += currentPostSynAct[id];
#endif
		}

		hebbianUpdate(SIMPLE, preSynAct + type->inputSize + type->outputSize);
	}

	// STEP 6: MEMORY
	{
		propagate(MEMORY, NULL);
			
		for (int i = 0; i < memoryChildren.size(); i++) {
			for (int j = 0; j < MODULATION_VECTOR_SIZE; j++) {
				memoryChildren[i].localM[j] = this->totalM[j];
			}

			for (int j = 0; j < memoryChildren[i].type->inputSize; j++) {
				memoryChildren[i].currentPostSynAct[j] = tanhf(memoryChildren[i].preSynAct[j]);
#ifdef SATURATION_PENALIZING
				* saturationPenalizationPtr += powf(memoryChildren[i].currentPostSynAct[j], saturationExponent);
#endif
			}
		}

		hebbianUpdate(MEMORY, NULL);

		for (int i = 0; i < memoryChildren.size(); i++) {
			memoryChildren[i].forward();

#ifdef SATURATION_PENALIZING
			float* array = memoryChildren[i].currentPostSynAct + memoryChildren[i].type->INPUT_NODESize;
			for (int j = 0; j < memoryChildren[i].type->outputSize; j++) {
				*saturationPenalizationPtr += powf(array[j], saturationExponent);
			}
#endif
		}
	}

	// STEP 7: OUTPUT
	{
		propagate(OUTPUT, preSynAct + type->inputSize);

		for (int i = type->inputSize; i < type->outputSize + type->inputSize; i++) {
			previousPostSynAct[i] = currentPostSynAct[i];
			currentPostSynAct[i] = tanhf(preSynAct[i]);
#ifdef SATURATION_PENALIZING
			* saturationPenalizationPtr += powf(currentPostSynAct[i], saturationExponent);
#endif
		}

		hebbianUpdate(OUTPUT, currentPostSynAct + type->inputSize);
	}
	
}
