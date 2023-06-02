#include "MemoryNode_P.h"

MemoryNode_P::MemoryNode_P(MemoryNode_G* _type) :
	type(_type), pLink(&_type->link)
{
	
	candidateMemory = std::make_unique<float[]>(type->outputSize+type->kernelDimension);
	QX = std::make_unique<float[]>(type->kernelDimension);
	outputBuffer = std::make_unique<float[]>(type->outputSize);

	// Useless initializations:
	{
		memory.resize(0);
		invNorms.resize(0);
		unnormalizedCosines.resize(0);
		nMemorizedVectors = 0;
		input = nullptr;
		output = nullptr;
		modulation = nullptr;
	}
}


void MemoryNode_P::forward() {
	
	// vars defined for readability:
	
	float ksi1 = (tanhf(modulation[2]) + 1.0f) * .5f; // from R to [0, 1]
	float ksi2 = modulation[3];
	float ksi3 = (tanhf(modulation[4]) + 1.0f) * .5f; // from R to [0, 1]

	//float ksi1 = .5f; 
	//float ksi2 = .2f;
	//float ksi3 = 1.0f;

	int memoryVectorSize = type->kernelDimension + type->outputSize;

	// link operations: propagation and hebbian update.
	{
		// propagate

		int nl = type->outputSize;
		int nc = type->inputSize;
		int matID = 0;

		std::fill(outputBuffer.get(), outputBuffer.get() + nl, 0.0f);

		float* H = pLink.H.get();
		float* wLifetime = pLink.wLifetime.get();
		float* alpha = type->link.alpha.get();

#ifdef RANDOM_W
		float* w = pLink.w.get();
#else
		float* w = type->link.w.get();
#endif
		

		for (int i = 0; i < nl; i++) {
			for (int j = 0; j < nc; j++) {
				// += (H * alpha + w + wL) * prevAct
				outputBuffer[i] += (H[matID] * alpha[matID] + w[matID] + wLifetime[matID]) * input[j];
				matID++;
			}
		}
		for (int i = 0; i < type->outputSize; i++) {
			output[i] = tanhf(outputBuffer[i]);
		}


		// hebbian update. TODO could happen after memory.
		float* A = type->link.A.get();
		float* B = type->link.B.get();
		float* C = type->link.C.get();
		float* D = type->link.D.get();
		float* eta = type->link.eta.get();
		float* E = pLink.E.get();

#ifdef OJA
		float* delta = type->link.delta.get();
#endif

#ifdef CONTINUOUS_LEARNING
		float* gamma = type->link.gamma.get();
#else
		float* avgH = pLink.avgH.get();
#endif


		matID = 0;  // = i*nc+j
		for (int i = 0; i < nl; i++) {
			for (int j = 0; j < nc; j++) {
#ifdef CONTINUOUS_LEARNING
				wLifetime[matID] = (1 - gamma[matID]) * wLifetime[matID] + gamma[matID] * E[matID] * modulation[1];
#endif
				E[matID] = (1.0f - eta[matID]) * E[matID] + eta[matID] *
					(A[matID] * outputBuffer[i] * input[j] + 
					 B[matID] * outputBuffer[i] +
					 C[matID] * input[j] + 
				     D[matID]);
#ifdef OJA
				E[matID] -= eta[matID] * outputBuffer[i] * outputBuffer[i] * delta[matID] * (w[matID] + alpha[matID] * H[matID] + wLifetime[matID]);
#endif
				H[matID] += E[matID] * modulation[0];
				H[matID] = std::max(-1.0f, std::min(H[matID], 1.0f));
#ifndef CONTINUOUS_LEARNING
				avgH[matID] += H[matID];
#endif
				matID++;

			}
		}
	}

	float maxCos = -1.0f;
	if (nMemorizedVectors>0){
		// Compute QX and its norm
		int QID = 0;
		float invQXnorm = 0.0f;
		for (int i = 0; i < type->kernelDimension; i++) {
			QX[i] = 0.0f;
			for (int j = 0; j < type->inputSize; j++) {
				QX[i] += type->Q[QID] * input[j];
				QID++;
			}
			invQXnorm += QX[i] * QX[i];
		}
		invQXnorm = powf(invQXnorm, -.5f);


		// Compute unnormalized cosinuses
		for (int i = 0; i < nMemorizedVectors; i++) {
			unnormalizedCosines[i] = 0.0f;
			int id = i * memoryVectorSize;
			for (int j = 0; j < type->kernelDimension; j++) {
				unnormalizedCosines[i] += QX[j] * memory[id + j];
			}
		}

		// compute softmax(beta* unnormalizedCosines), and extract the max of the cosines.
		float softmaxNormalizationFactor = 0.0f;
		for (int i = 0; i < nMemorizedVectors; i++) {
			
			float cosine = unnormalizedCosines[i] * invQXnorm * invNorms[i];
			if (maxCos < cosine) [[unlikely]] { maxCos = cosine; }

			// from now on unnormalizedCosines will contain coefficients for the weighted sum of the
			// memorized responses, and not unnormalized cosines.

			unnormalizedCosines[i] *= type->beta;
			unnormalizedCosines[i] = expf(unnormalizedCosines[i]);
			softmaxNormalizationFactor += unnormalizedCosines[i];
		}
		softmaxNormalizationFactor = 1.0f / softmaxNormalizationFactor;
		for (int i = 0; i < nMemorizedVectors; i++) {
			unnormalizedCosines[i] *= softmaxNormalizationFactor;
		}

		std::fill(outputBuffer.get(), outputBuffer.get() + type->outputSize, 0.0f);

		for (int i = 0; i < nMemorizedVectors; i++) {
			int id = i * memoryVectorSize + type->kernelDimension;
			for (int j = 0; j < type->outputSize; j++) {
				outputBuffer[j] += unnormalizedCosines[i] * memory[id + j];
			}
		}

		for (int i = 0; i < type->outputSize; i++) {
			output[i] = ksi1 * output[i] + (1.0f-ksi1) * outputBuffer[i]; // preSynOutput's content can be discarded now.
		}

	}

	// TODO which vector's norm should be used ? Using key as of now
	//Compute candidate memory norm to check if it is to be added to memory
	float candidateL2Norm = 0.0f;
	for (int i = 0; i < type->kernelDimension; i++) {
		candidateL2Norm += candidateMemory[i] * candidateMemory[i];
	}
	
	// If the treshold is reached, add candidate memory to memory, and its inverse norm to
	// invNorms. TODO make it more efficient in copy and reallocations.
	if (sqrtf(candidateL2Norm) * ksi3 > 1) {

		int oldMemorySize = nMemorizedVectors * memoryVectorSize;
		memory.insert(memory.end(), candidateMemory.get(), candidateMemory.get()+ memoryVectorSize);

		invNorms.push_back(candidateL2Norm);

		unnormalizedCosines.resize(nMemorizedVectors+1);

		std::fill(candidateMemory.get(), candidateMemory.get() + memoryVectorSize, 0.0f);

		nMemorizedVectors++;
	}

	//Update candidate memory
	float F = ksi1 * ksi2 * type->decay * powf(1.0f - maxCos, 1.0f);
	for (int i = 0; i < type->kernelDimension; i++) {
		candidateMemory[i] = (1.0f - type->decay) * candidateMemory[i] + F * QX[i];
	}
	for (int i = type->kernelDimension; i < memoryVectorSize; i++) {
		candidateMemory[i] = (1.0f - type->decay) * candidateMemory[i] + F * output[i - type->kernelDimension];
	}
}

void MemoryNode_P::setArrayPointers(float* post_syn_acts, float* pre_syn_acts, float* globalModulation, float** aa, float** acc_pre_syn_acts) {
	output = post_syn_acts;
	input = pre_syn_acts;
	modulation = globalModulation;


#ifdef SATURATION_PENALIZING
	saturationArray = *aa;
	*aa += type->inputSize + type->outputSize;
#endif
	

#ifdef STDP
	accumulatedInput = *acc_pre_syn_acts;
	*acc_pre_syn_acts += type->inputSize;
#endif
}

void MemoryNode_P::preTrialReset() {
	
	// TODO keep memory between trial ?
	{
		nMemorizedVectors = 0;
		memory.resize(0);
		invNorms.resize(0);
		unnormalizedCosines.resize(0);
	}

	std::fill(candidateMemory.get(), candidateMemory.get() + type->outputSize + type->kernelDimension, 0.0f);

	int s = type->link.nLines * type->link.nColumns;
	pLink.zero(); // zero E, H, and AVG_H if need be.

#ifdef RANDOM_W
	pLink.randomInitW(); 
#endif
	
}

#ifdef GUIDED_MUTATIONS
void MemoryNode_P::accumulateW(float factor) {

	type->link.accumulateW(factor, pLink.wLifetime.get());
}
#endif