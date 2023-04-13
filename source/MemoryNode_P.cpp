#include "MemoryNode_P.h"

MemoryNode_P::MemoryNode_P(MemoryNode_G* _type) :
	type(_type), pLink(_type->inputSize*_type->outputSize)
{
	
	candidateMemory = std::make_unique<float[]>(type->outputSize);

	// Useless initializations:
	{
		memory = std::make_unique<float[]>(0);
		transformedMemory = std::make_unique<float[]>(0);
		nMemorizedVectors = 0;
		currentPostSynAct = nullptr;
		preSynAct = nullptr;
		previousPostSynAct = nullptr;
		for (int i = 0; i < MODULATION_VECTOR_SIZE; i++) {
			localM[i] = 0.0f;
		}
	}
}

void MemoryNode_P::memorize() {

}

void MemoryNode_P::forward() {

	// vars defined for readability:
	float ksi1 = (localM[2] + 1.0f) * .5f;
	float ksi2 = localM[3];
	float ksi3 = localM[4];
	float* preSynOutput = preSynAct + type->inputSize;
	float* postSynOutput = currentPostSynAct + type->inputSize;


	int nl = type->kernelDimension;
	int nc = type->inputSize;
	int matID = 0;

	float* H = pLink.H.get();
	float* wLifetime = pLink.wLifetime.get();
	float* alpha = type->link.alpha.get();
	float* w = type->link.w.get();

	for (int i = 0; i < nl; i++) {
		for (int j = 0; j < nc; j++) {
			// += (H * alpha + w) * prevAct
			*(preSynOutput + i) += (H[matID] * alpha[matID] + w[matID] + wLifetime[matID]) * currentPostSynAct[j];
			matID++;
		}
	}
	// apply tanh ??
	float maxSigma = -1.0f;
	if (nMemorizedVectors>0){
		// Compute cosinuses, in sigmas[].
		for (int i = 0; i < nMemorizedVectors; i++) {
			sigmas[i] = 0.0f;
			for (int j = 0; j < type->inputSize; j++) {
				sigmas[i] += currentPostSynAct[j] * transformedMemory[i * type->inputSize + j];
			}
		}

		// applies softmax
		float softmaxNormalizationFactor = 0.0f;
		maxSigma = -1000.0f;
		for (int i = 0; i < nMemorizedVectors; i++) {
			sigmas[i] *= type->beta;
			if (maxSigma > sigmas[i]) [[unlikely]] { maxSigma = sigmas[i]; }
			sigmas[i] = expf(sigmas[i]);
			softmaxNormalizationFactor += sigmas[i];
		}
		softmaxNormalizationFactor = 1.0f / softmaxNormalizationFactor;
		for (int i = 0; i < nMemorizedVectors; i++) {
			sigmas[i] *= softmaxNormalizationFactor;
		}

		// accumulates the memory vector weighted by the sigmas.
		for (int j = 0; j < type->outputSize; j++) {
			postSynOutput[j] = 0.0f;
		}

		for (int i = 0; i < nMemorizedVectors; i++) {
			for (int j = 0; j < type->outputSize; j++) {
				postSynOutput[j] += sigmas[i] * transformedMemory[i * type->outputSize + j];
			}
		}

		for (int i = 0; i < type->outputSize; i++) {
			postSynOutput[i] = ksi1 * preSynOutput[i] + (1.0f-ksi1) * postSynOutput[i];
		}
	}


	// accumulate tin*tQ*K in preSynOutput
	for (int i = 0; i < type->outputSize; i++) {
		preSynOutput[i] = 0.0f;
		for (int j = 0; j < type->inputSize; j++) {
			preSynOutput[i] += postSynOutput[j] * type->tQxK[i + j * type->outputSize];
		}
	}


	//Update candidate memory
	float F = ksi1 * ksi2 * powf(1.0f - maxSigma, 1.0f); // std::max(std::min(powf(1.0f - maxSigma, 1.0f), 1.0f)) , -1.0f) ?
	float f = type->beta * type->K;
	float candidateL2Norm = 0.0f;
	for (int i = 0; i < type->outputSize; i++) {
		candidateMemory[i] += (preSynOutput[i] * f + postSynOutput[i]) * F;
		candidateL2Norm += candidateMemory[i] * candidateMemory[i];
	}
	
	// Add candidate memory to memory if a certain treshold is passed. TODO make it more efficient in copy and reallocations.
	if (sqrtf(candidateL2Norm) * ksi3 > 1) {

		int oldMemorySize = nMemorizedVectors * type->outputSize;
		float* newMemory = new float[oldMemorySize + type->outputSize];
		std::copy(memory.get(), memory.get() + oldMemorySize, newMemory);
		memory.reset(newMemory);
		for (int i = 0; i < type->outputSize; i++) {
			memory[i + oldMemorySize] = candidateMemory[i];
		}


		int oldTransormedMemorySize = nMemorizedVectors * type->inputSize;
		float* newTransformedMemory = new float[oldMemorySize + type->inputSize];
		std::copy(transformedMemory.get(), transformedMemory.get() + oldTransormedMemorySize, newTransformedMemory);
		transformedMemory.reset(newTransformedMemory);
		for (int i = 0; i < type->inputSize; i++) {
			transformedMemory[i + oldTransormedMemorySize] = 0.0f;
			for (int j = 0; j < type->outputSize; j++) {
				transformedMemory[i + oldTransormedMemorySize] += type->tQxK[i* type->outputSize + j] * candidateMemory[j];
			}
		}


		for (int i = 0; i < type->outputSize; i++) {
			candidateMemory[i] = 0.0f;
		}
		nMemorizedVectors++;
	}

}

void MemoryNode_P::setArrayPointers(float* ppsa, float* cpsa, float* psa, float* aa) {
	previousPostSynAct = ppsa;
	currentPostSynAct = cpsa;
	preSynAct = psa;
#ifdef SATURATION_PENALIZING
	averageActivation = aa;
#endif

	int s = type->inputSize + type->outputSize;
	ppsa += s;
	cpsa += s;
	psa += s;
#ifdef SATURATION_PENALIZING
	aa += ...;
#endif
}

void MemoryNode_P::preTrialReset() {
	nMemorizedVectors = 0;
	memory = std::make_unique<float[]>(NULL);
	for (int j = 0; j < type->outputSize; j++) {
		candidateMemory[j] = 0.0f;
	}

	int s = type->link.nLines * type->link.nColumns;;
	pLink.zero(s); // zero E, H, and AVG_H if need be.


	for (int j = 0; j < MODULATION_VECTOR_SIZE; j++) {
		localM[j] = 0.0f;
	}

	sigmas = new float[0];
}

#ifdef GUIDED_MUTATIONS
void MemoryNode_P::accumulateW(float factor) {
	type->nAccumulations++;

	int s = type->link.nLines * type->link.nColumns;
	for (int j = 0; j < s; j++) {
		type->link.accumulator[j] += factor * pLink.wLifetime[j];
		pLink.wLifetime[j] = 0.0f;
	}
}
#endif