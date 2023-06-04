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
	 


	// Used as the multiplied vector in matrix operations. Layout:
	// input -> modulation.out -> complexChildren.out -> memoryChildren.out
	float* postSynActs;

	// Used as the result vector in matrix operations. Layout:
	// output -> modulation.in -> complexChildren.in -> memoryChildren.in
	float* preSynActs;

#ifdef STDP
	// Same layout as PreSynActs, i.e.
	// output -> modulation.in -> complexChildren.in -> memoryChildren.in
	float* accumulatedPreSynActs;
#endif

#ifdef SATURATION_PENALIZING
	// Layout:
	// Modulation -> complexChildren->inputSize
	// inputSize for memory nodes
	float* averageActivation;
#endif


	ComplexNode_P(ComplexNode_G* type);

	// Should never be called.
	ComplexNode_P() 
	{
		__debugbreak();
#ifdef STDP
		accumulatedPreSynActs = nullptr;
#endif
#ifdef SATURATION_PENALIZING
		averageActivation = nullptr;
		globalSaturationAccumulator = nullptr;
#endif
		preSynActs = nullptr;
		postSynActs = nullptr;
		std::fill(totalM, totalM + MODULATION_VECTOR_SIZE, 0.0f);
		type = nullptr;
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

	// The last 2 parameters are optional :
	// - aa only used when SATURATION_PENALIZING is defined
	// - acc_pre_syn_acts only used when STDP is defined
	void setArrayPointers(float** pre_syn_acts, float** post_syn_acts, float** aa, float** acc_pre_syn_acts);

#ifdef SATURATION_PENALIZING
	void setglobalSaturationAccumulator(float* globalSaturationAccumulator);
#endif
};
