#pragma once

#include <memory>

#include "MemoryNode_G.h"
#include "InternalConnexion_P.h"





struct MemoryNode_P {
	MemoryNode_G* type;


#ifdef QKV_MEMORY
	int nMemorizedVectors;

	InternalConnexion_P pLink;

	// a set of vectors, each of size kernelDimension + outputSize. Size nMemorized vectors.
	// Starts empty, vectors are added during inferences when a treshold is reached.
	// For each vector, the first kernelDimension values are the stored key, and the  
	// last outputSize values are the stored response.
	std::vector<float> memory;

	// Size nMemorized vectors. Contains the inverses of the norms of the memorized keys, 
	// for cosinus normalization.
	std::vector<float> invNorms;

	// stores the unnormalized cosines. Size nMemorized vectors.
	std::vector<float> unnormalizedCosines;

	// stores QX. Size kernelDimension.
	std::unique_ptr<float[]> QX;

	// a vector of size kernelDimension + outputSize, eligible to be appended to memory.
	std::unique_ptr<float[]> candidateMemory;

	// Size outputSize, used to hold temporary value during inference.
	std::unique_ptr<float[]> outputBuffer;
#endif

#ifdef SRWM
	std::unique_ptr<float[]> W;

	std::unique_ptr<float[]> vBuffer;
	std::unique_ptr<float[]> matMulResult;

	// To avoid setting them at runtime.
	float* k;
	float* q;
	float* beta;
#endif

#ifdef DNN_MEMORY
	std::vector<std::unique_ptr<float[]>> Ws;

	std::vector<std::unique_ptr<float[]>> Bs;

	std::unique_ptr<float[]> candX;
	std::unique_ptr<float[]> candY;

	// Layer by layer activations of the network. 
	std::unique_ptr<float[]> activations;

	// d(cost)/d(preSynAct). Could be only 2 * max (type->sizes) in size, but
	// it is a negligible optimization as of yet.
	std::unique_ptr<float[]> delta;

#endif
	

	// These 3 pointers point towards memory that is shared between this node and its complex parent.
	float* modulation;
	float* input;
	float* output;
#ifdef SATURATION_PENALIZING
	float* saturationArray;
#endif
#ifdef STDP
	// used by its complex parent.
	float* accumulatedInput;
#endif

#ifdef GUIDED_MUTATIONS
	void accumulateW(float factor);
#endif

#ifndef CONTINUOUS_LEARNING
	void updateWatTrialEnd(float invNInferencesOverTrial);
#endif

	MemoryNode_P(MemoryNode_G* type);


	// Should never be called
	MemoryNode_P() {
		__debugbreak();
	}

	// Should never be called
	MemoryNode_P(MemoryNode_P&& n) noexcept {
		__debugbreak();
	}

	// Should never be called
	MemoryNode_P(MemoryNode_P& n)  {
		__debugbreak();
	}

	void forward();

	void setArrayPointers(float* post_syn_acts, float* pre_syn_acts, float* globalModulation, float** aa, float** acc_pre_syn_acts);

	void preTrialReset();
};