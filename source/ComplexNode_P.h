#pragma once

#include <vector>
#include <memory>
#include <cmath>

#include "Random.h"
#include "ComplexNode_G.h"
#include "MemoryNode_P.h"
#include "PhenotypeConnexion.h"

struct ComplexNode_P {
	ComplexNode_G* type;

	int nInferencesP; // store how many inferences this node has performed since last preTrialReset.
	float localM[MODULATION_VECTOR_SIZE]; // computed in this node. Wasted space for simple neurons. TODO
	float totalM[MODULATION_VECTOR_SIZE]; // parent's + local M.    Wasted space for simple neurons. TODO

#ifdef SATURATION_PENALIZING
	// A parent updates it for its children (in and out), not for itself.
	float* saturationPenalizationPtr;		 //Wasted space for simple neurons. TODO
#endif
	 

	std::vector<ComplexNode_P> complexChildren;
	std::vector<MemoryNode_P>  memoryChildren;

	// Vector of structs containing pointers to the dynamic connexion matrices linking children
	std::vector<PhenotypeConnexion> internalConnexions;
	
	// Not managed by Complex node, but by network. Never call free or delete on these: 
	float* previousPostSynAct; 
	float* currentPostSynAct;
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
	// Accumulates wLifetime of the phenotype connexion in the accumulate[] array of their 
	// corresponding genotype connexion template, then zeros wLifetime. Weighted by "factor".
	void accumulateW(float factor);
#endif

	// The last parameters is optional and only used when SATURATION_PENALIZING is defined.
	void setArrayPointers(float** ppsa, float** cpsa, float** psa, float** aa = nullptr);

#ifdef SATURATION_PENALIZING
	void setSaturationPenalizationPtr(float* saturationPenalizationPtr);
#endif
};
