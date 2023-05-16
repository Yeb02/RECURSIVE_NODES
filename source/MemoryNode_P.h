#pragma once

#include <memory>

#include "MemoryNode_G.h"
#include "InternalConnexion_P.h"

struct MemoryNode_P {
	MemoryNode_G* type;

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

	// These 3 pointers point towards memory that is shared between this node and its complex parent.
	float* modulation;
	float* input;
	float* output;

	// Size outputSize, used to hold temporary value during inference.
	std::unique_ptr<float[]> outputBuffer;

#ifdef GUIDED_MUTATIONS
	void accumulateW(float factor);
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

	void setArrayPointers(float** cpsa, float** psa, float* globalModulation);

	void preTrialReset();
};