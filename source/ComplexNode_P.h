#pragma once

#include <vector>
#include <memory>
#include <cmath>

#include "Random.h"
#include "ComplexNode_G.h"
#include "MemoryNode_P.h"
#include "InternalConnexion_P.h"

struct ComplexNode_P {
	ComplexNode_G* type;

	float totalM[MODULATION_VECTOR_SIZE]; // parent's + local M.    

#ifdef SATURATION_PENALIZING
	// A parent updates it for its children (in and out), not for itself.
	float* globalSaturationAccumulator;		 
#endif
	 

	std::vector<ComplexNode_P> complexChildren;
	std::vector<MemoryNode_P>  memoryChildren;

	InternalConnexion_P toComplex, toMemory, toModulation, toOutput;
	

	// These arrays are not managed by Complex node, but by Network:
	 

	// Arranged in this order: input -> modulation.out -> complexChildren.out -> memoryChildren.out
	float* previousPostSynAct; 

	// Used as the multiplied vector in matrix operations.
	// Arranged in this order: input -> modulation.out -> complexChildren.out -> memoryChildren.out
	float* currentPostSynAct;

	// Used as the result vector in matrix operations.
	// Arranged in this order: output -> modulation.int -> complexChildren.in -> memoryChildren.in
	float* preSynAct;
#ifdef SATURATION_PENALIZING
	float* averageActivation;
#endif


	ComplexNode_P(ComplexNode_G* type);
	ComplexNode_P() {
		__debugbreak();
	}
	~ComplexNode_P() {};

	void preTrialReset();

	void forward();

#ifndef CONTINUOUS_LEARNING
	void updateWatTrialEnd(float invnInferencesP);
#endif

#ifdef GUIDED_MUTATIONS 
	// Accumulates wLifetime of the phenotype connexion in the accumulate[] array of their corresponding
	// genotype connexion template, then OPTIONNALY zeroes wLifetime. Weighted by "factor", a decreasing 
	// function of the number of inferences. If CONTINUOUS_LEARNING is not defined
	// 
	void accumulateW(float factor);
#endif

	// The last parameters is optional and only used when SATURATION_PENALIZING is defined.
	void setArrayPointers(float** ppsa, float** cpsa, float** psa, float** aa = nullptr);

#ifdef SATURATION_PENALIZING
	void setglobalSaturationAccumulator(float* globalSaturationAccumulator);
#endif
};
