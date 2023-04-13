#pragma once

#include <memory>

#include "MemoryNode_G.h"
#include "PhenotypeConnexion.h"

struct MemoryNode_P {
	MemoryNode_G* type;

	int nMemorizedVectors;

	PhenotypeConnexion pLink;

	// a set of vectors, each of size outputSize. Starts empty, vectors are added during lifetime / trial.
	std::unique_ptr<float[]> memory;

	// a set of vector, each of size kernelDimension. transformedMemory[i] = t(queryM)*keyM*memory[i]. 
	// Exists only for computational efficiency.
	std::unique_ptr<float[]> transformedMemory;

	// a vector of size outputSize, eligible to be appended to memory.
	std::unique_ptr<float[]> candidateMemory;

	// stores the unnormalized cosinuses. Size nMemorized vectors.
	float* sigmas;

	float localM[MODULATION_VECTOR_SIZE];

	// Runtime memory layout: inputSize-outputSize
	float* previousPostSynAct;
	float* currentPostSynAct;
	float* preSynAct;
#ifdef SATURATION_PENALIZING
	float* averageActivation;
#endif

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

	// Add candidateMemory to memory, and transform it to add to transformedMemory. 
	// Zero candidateMemory afterwards.
	void memorize();

	void forward();

	void setArrayPointers(float* ppsa, float* cpsa, float* psa, float* aa);

	void preTrialReset();
};