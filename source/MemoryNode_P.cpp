#include "MemoryNode_P.h"

MemoryNode_P::MemoryNode_P(MemoryNode_G* _type) :
	type(_type)
#ifdef QKV_MEMORY
	,pLink(&_type->link)
#endif
{

	// Useless initializations:
	{
		input = nullptr;
		output = nullptr;
		modulation = nullptr;
#ifdef SATURATION_PENALIZING
		saturationArray = nullptr;
#endif

#ifdef STDP
		accumulatedInput = nullptr;
#endif
	}

#ifdef SRWM
	W = std::make_unique<float[]>(type->nLinesW0 * type->inputSize);
	vBuffer = std::make_unique<float[]>(type->nLinesW0);
	matMulResult = std::make_unique<float[]>(type->nLinesW0);

	k = matMulResult.get() + type->outputSize;
	q = k + type->inputSize;
	beta = q + type->inputSize;
#endif

#ifdef QKV_MEMORY
	candidateMemory = std::make_unique<float[]>(type->outputSize + type->kernelDimension);
	QX = std::make_unique<float[]>(type->kernelDimension);
	outputBuffer = std::make_unique<float[]>(type->outputSize);

	// Useless initializations:
	{
		memory.resize(0);
		invNorms.resize(0);
		unnormalizedCosines.resize(0);
		nMemorizedVectors = 0;
	}
#endif


#ifdef DNN_MEMORY
	int activationS = 0;
	for (int i = 0; i < type->nLayers+1; i++) {
		activationS += type->sizes[i];
	}
	activations = std::make_unique<float[]>(activationS);
	delta = std::make_unique<float[]>(activationS - type->inputSize);

	candX = std::make_unique<float[]>(type->inputSize);
	candY = std::make_unique<float[]>(type->outputSize);

	Ws.reserve(type->nLayers);
	Bs.reserve(type->nLayers);

	for (int i = 0; i < type->nLayers; i++)
	{
		int sW = type->sizes[i] * type->sizes[i + 1];
		Ws.emplace_back(new float[sW]);
		for (int j = 0; j < sW; j++) {
			Ws[i][j] = type->W0s[i][j];
		}

		int sB = type->sizes[i + 1];
		Bs.emplace_back(new float[sB]);
		for (int j = 0; j < sB; j++) {
			Bs[i][j] = type->B0s[i][j];
		}
	}
#endif

}

// forward for each preprocessor directive:
#ifdef QKV_MEMORY

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
				wLifetime[matID] = (1 - gamma[matID]) * wLifetime[matID] + gamma[matID] * H[matID] * alpha[matID] * modulation[1];
#endif
				E[matID] = (1.0f - eta[matID]) * E[matID] + eta[matID] *
					(A[matID] * output[i] * input[j] +
						B[matID] * output[i] +
						C[matID] * input[j] +
						D[matID]);
#ifdef OJA
				E[matID] -= eta[matID] * output[i] * output[i] * delta[matID] * (w[matID] + alpha[matID] * H[matID] + wLifetime[matID]);
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
	if (nMemorizedVectors > 0) {
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
			output[i] = ksi1 * output[i] + (1.0f - ksi1) * outputBuffer[i]; // "input" 's content can be discarded now.
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
		memory.insert(memory.end(), candidateMemory.get(), candidateMemory.get() + memoryVectorSize);

		invNorms.push_back(candidateL2Norm);

		unnormalizedCosines.resize(nMemorizedVectors + 1);

		std::fill(candidateMemory.get(), candidateMemory.get() + memoryVectorSize, 0.0f);

		nMemorizedVectors++;
	}

	//Update candidate memory
	float F = ksi1 * ksi2 * type->QKV_decay * powf(1.0f - maxCos, 1.0f);
	for (int i = 0; i < type->kernelDimension; i++) {
		candidateMemory[i] = (1.0f - type->QKV_decay) * candidateMemory[i] + F * QX[i];
	}
	for (int i = type->kernelDimension; i < memoryVectorSize; i++) {
		candidateMemory[i] = (1.0f - type->QKV_decay) * candidateMemory[i] + F * output[i - type->kernelDimension];
	}
}

#endif

#ifdef SRWM
void MemoryNode_P::forward()
{
	int inS = type->inputSize;

	// W * x
	std::fill(matMulResult.get(), matMulResult.get() + type->nLinesW0, 0.0f);
	for (int i = 0; i < type->nLinesW0; i++) {
		for (int j = 0; j < inS; j++) {
			matMulResult[i] += W[i * inS + j] * input[j];
		}
	}

	// Write pseudoSigmoid(y) to output
	for (int i = 0; i < type->outputSize; i++) {
		output[i] = .5f + .5f * tanhf(matMulResult[i]);
	}

	// Softmax k
	float normalizationFactor = 0.0f;
	for (int i = 0; i < inS; i++) {
		k[i] = expf(k[i]);
		normalizationFactor += k[i];
	}

	normalizationFactor = 1.0f / normalizationFactor;
	for (int i = 0; i < inS; i++) {
		k[i] *= normalizationFactor;
	}

	// Softmax q
	normalizationFactor = 0.0f;
	for (int i = 0; i < inS; i++) {
		q[i] = expf(q[i]);
		normalizationFactor += q[i];
	}

	normalizationFactor = 1.0f / normalizationFactor;
	for (int i = 0; i < inS; i++) {
		q[i] *= normalizationFactor;
	}

	// pseudo sigmoid(beta) * neuromodulation.
	for (int i = 0; i < 4; i++) {
		beta[i] = (tanhf(beta[i]) * .5f + .5f) * (tanhf(modulation[2 + i]) * .5f + .5f);
	}

	// ready v - vBar
	std::fill(vBuffer.get(), vBuffer.get() + type->nLinesW0, 0.0f);

	// W * softmax q ( = v)
	for (int i = 0; i < type->nLinesW0; i++) {
		for (int j = 0; j < inS; j++) {
			vBuffer[i] += W[i * inS + j] * q[j];
		}
	}

	// v -= v bar (= W * softmax k)
	for (int i = 0; i < type->nLinesW0; i++) {
		for (int j = 0; j < inS; j++) {
			vBuffer[i] -= W[i * inS + j] * k[j];
		}
	}


	int f, l;

	// update Wy
	f = 0;
	l = type->outputSize;
	for (int i = f; i < l; i++) {
		for (int j = 0; j < inS; j++) {
			float f = beta[0] * k[j];										// *type->SRWM_DECAY[0] TODO stability ?
			W[i * inS + j] = W[i * inS + j] * (1.0f - type->SRWM_decay[0]) + beta[0] * vBuffer[i] * k[j];
		}
	}

	// update Wk
	f = l;
	l += inS;
	for (int i = f; i < l; i++) {
		for (int j = 0; j < inS; j++) {
			float f = beta[1] * k[j];
			W[i * inS + j] = W[i * inS + j] * (1.0f - type->SRWM_decay[1]) + beta[1] * vBuffer[i] * k[j];
		}
	}

	// update Wq
	f = l;
	l += inS;
	for (int i = f; i < l; i++) {
		for (int j = 0; j < inS; j++) {
			float f = beta[2] * k[j];
			W[i * inS + j] = W[i * inS + j] * (1.0f - type->SRWM_decay[2]) + beta[2] * vBuffer[i] * k[j];
		}
	}

	// update Wbeta
	f = l;
	l += 4;
	for (int i = f; i < l; i++) {
		for (int j = 0; j < inS; j++) {
			float f = beta[3] * k[j];
			W[i * inS + j] = W[i * inS + j] * (1.0f - type->SRWM_decay[3]) + beta[3] * vBuffer[i] * k[j];
		}
	}

}
#endif

#ifdef DNN_MEMORY
void MemoryNode_P::forward()
{
	
	// TODO GD or cand update first ?

	// forward pass. Out = M0 * NORMAL(0,1)*.25f + (1-M0) * DNN(in)
	{
		float M0 = (tanhf(modulation[2]) + 1.0f) * .5f; // from R to [0, 1]

		float* prevOutput = &activations[0];
		std::copy(input, input + type->inputSize, prevOutput); // Not really CPU efficient, but better for later GPU-isation.
		float* currInput = &activations[type->inputSize];
		for (int i = 0; i < type->nLayers; i++) {
			for (int j = 0; j < type->sizes[i + 1]; j++) {
				currInput[j] = Bs[i][j];
			}

			int matID = 0;
			for (int j = 0; j < type->sizes[i + 1]; j++) {
				for (int k = 0; k < type->sizes[i]; k++) {
					currInput[j] += Ws[i][matID] * prevOutput[k];
					matID++;
				}
			}

			for (int j = 0; j < type->sizes[i + 1]; j++) {
				currInput[j] = tanhf(currInput[j]); // Could be in the previous loop, but GPUisation.
			}

			prevOutput = currInput;
			currInput = currInput + type->sizes[i + 1];
		}


		for (int i = 0; i < type->outputSize; i++) 
		{
			output[i] = prevOutput[i] * (1.0f - M0) + NORMAL_01 * .25f * M0;
		}
	}


	// One step of Gradient Descent with (candX, candY).
	//if (modulation[3] > 0.0f) { // to be tested.
	{
		float lr = modulation[3] * type->learningRate;  // = M1 * type->lr

		// forward
		float* prevActs = &activations[0];
		std::copy(candX.get(), candX.get() + type->inputSize, prevActs); // Not really CPU efficient, but better for later GPU-isation.
		float* currActs = &activations[type->inputSize];
		for (int i = 0; i < type->nLayers; i++) {
			for (int j = 0; j < type->sizes[i + 1]; j++) {
				currActs[j] = Bs[i][j];
			}

			int matID = 0;
			for (int j = 0; j < type->sizes[i + 1]; j++) {
				for (int k = 0; k < type->sizes[i]; k++) {
					currActs[j] += Ws[i][matID] * prevActs[k];
					matID++;
				}
			}

			for (int j = 0; j < type->sizes[i + 1]; j++) {
				currActs[j] = tanhf(currActs[j]); // Could be in the previous loop, but GPUisation.
			}

			prevActs = currActs;
			currActs = currActs + type->sizes[i + 1];
		}

		
		
		// backward. We will use tanh'(x) = 1 - tanh(x)², most stable and re-uses forward's calculations.
		float* prevDelta;
		float* currDelta = &delta[0];
		currActs = prevActs;
		prevActs = currActs - type->sizes[type->nLayers - 1];
		for (int i = 0; i < type->outputSize; i++) // euclidean distance loss
		{
			currDelta[i] =  (currActs[i] - candY[i]) * (1.0f - currActs[i] * currActs[i]) * lr; // Who cares about the 2. I inject the lr here for efficiency.
		}

		// w
		int matID = 0;
		for (int j = 0; j < type->sizes[type->nLayers]; j++) {
			for (int k = 0; k < type->sizes[type->nLayers-1]; k++) {
				Ws[type->nLayers-1][matID] += currDelta[j] * prevActs[k];
				matID++;
			}
		}

		// b
		for (int j = 0; j < type->sizes[type->nLayers]; j++)
		{
			Bs[type->nLayers - 1][j] += currDelta[j]; 
		}

		
		for (int i = type->nLayers-1; i >= 1; i--) {
			prevDelta = currDelta;
			currDelta = currDelta + type->sizes[i+1];
			currActs = prevActs; // Yes opposed to delta. I swear this makes sense.
			prevActs = currActs - type->sizes[i - 1];

			for (int j = 0; j < type->sizes[i]; j++) {
				currDelta[j] = 0.0f;
			}

			// Update deltas. This way of seeing the matmul avoids non-trivial indices induced by the transposition.
			matID = 0;
			for (int j = 0; j < type->sizes[i+1]; j++) {
				for (int k = 0; k < type->sizes[i]; k++) {
					currDelta[k] += Ws[i][matID+k] * prevDelta[j];
				}
				matID += type->sizes[i];
			}
			for (int j = 0; j < type->sizes[i]; j++) {
				currDelta[j] *= (1.0f - currActs[j] * currActs[j]);
			}

			// w
			matID = 0;
			for (int j = 0; j < type->sizes[i]; j++) {
				for (int k = 0; k < type->sizes[i - 1]; k++) {
					Ws[i - 1][matID] += currDelta[j] * prevActs[k];
					matID++;
				}
			}

			// b
			for (int j = 0; j < type->sizes[i]; j++)
			{
				Bs[i - 1][j] += currDelta[j];
			}

		}
	
	}


	// Update candX and candY.
	{
		float M2 = (tanhf(modulation[4]) + 1.0f) * .5f; // from R to [0, 1]

		for (int i = 0; i < type->inputSize; i++) {
			candX[i] = (1.0f - M2) * candX[i] + M2 * input[i];
		}
		for (int i = 0; i < type->outputSize; i++) {
			candY[i] = (1.0f - M2) * candY[i] + M2 * output[i];
		}
	}
}
#endif

void MemoryNode_P::setArrayPointers(float* post_syn_acts, float* pre_syn_acts, float* globalModulation, float** aa, float** acc_pre_syn_acts) {
	output = post_syn_acts;
	input = pre_syn_acts;
	modulation = globalModulation;


#ifdef SATURATION_PENALIZING
	saturationArray = *aa;
	*aa += type->inputSize;
#endif
	

#ifdef STDP
	accumulatedInput = *acc_pre_syn_acts;
	*acc_pre_syn_acts += type->inputSize;
#endif
}

void MemoryNode_P::preTrialReset() {	

#ifdef QKV_MEMORY
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

#endif

#ifdef SRWM
	int s = type->nLinesW0 * type->inputSize;
	std::copy(type->W0.get(), type->W0.get() + s, W.get());
#endif


#ifdef DNN_MEMORY
	std::fill(candX.get(), candX.get() + type->inputSize, 0.0f);
	std::fill(candY.get(), candY.get() + type->outputSize, 0.0f);
#endif
}

#ifdef GUIDED_MUTATIONS
void MemoryNode_P::accumulateW(float factor) {
#ifdef QKV_MEMORY
	type->link.accumulateW(factor, pLink.wLifetime.get());
#endif


#ifdef DNN_MEMORY
	int id = 0;
	for (int i = 0; i < type->nLayers; i++)
	{
		int sW = type->sizes[i] * type->sizes[i + 1];
		for (int j = 0; j < sW; j++) {
			type->accumulator[id + j] += factor * Ws[i][j];
		}
		id += sW;

		int sB = type->sizes[i + 1];
		for (int j = 0; j < sB; j++) {
			type->accumulator[id + j] += factor * Bs[i][j];
		}
		id += sB;
	}
#endif

}
#endif


#ifndef CONTINUOUS_LEARNING
void MemoryNode_P::updateWatTrialEnd(float invNInferencesOverTrial)
{
#ifdef QKV_MEMORY
	pLink.updateWatTrialEnd(invNInferencesOverTrial);
#endif

	// DNN_MEMORY and SRWM have nothing to do here.
}
#endif