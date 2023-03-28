#pragma once

#include <vector>
#include <memory>
#include <cmath>

#include "Random.h"
#include "Genotype.h"


struct PhenotypeConnexion {   // responsible of its pointers

	std::unique_ptr<float[]> H;
	std::unique_ptr<float[]> E;

	// Initialized to 0 when and only when the connexion is created. If CONTINUOUS_LEARNING, updated at each inference,
	// otherwise updated at the end of each trial.
	std::unique_ptr<float[]> wLifetime;

#ifndef CONTINUOUS_LEARNING
	// Arithmetic avg of H over trial duration.
	std::unique_ptr<float[]> avgH;
#endif



	// Should not be called !
	PhenotypeConnexion(const PhenotypeConnexion&) {};

	PhenotypeConnexion(int s);

	void zero(int s);

	// only called at construction.
	void zeroWlifetime(int s); 

#ifndef CONTINUOUS_LEARNING
	// factor = 1/nInferencesP., wLifetime += alpha*avgH*factor
	void updateWatTrialEnd(int s, float factor, float* alpha);
#endif

	~PhenotypeConnexion() {};
};

struct PhenotypeNode {
	GenotypeNode* type;

	int nInferencesP; // store how many inferences this node has performed since last preTrialReset.
	float localM[2]; // computed in this node. Wasted space for simple neurons. TODO
	float totalM[2]; // parent's + local M.    Wasted space for simple neurons. TODO

#ifdef SATURATION_PENALIZING
	// A parent updates it for its children (in and out), not for itself.
	float* saturationPenalizationPtr;		 //Wasted space for simple neurons. TODO
#endif
	 

	// Pointers to its children.
	std::vector<PhenotypeNode> children;

	// Vector of structs containing pointers to the dynamic connexion matrices linking children
	std::vector<PhenotypeConnexion> childrenConnexions;
	
	// Not managed by Phenotype node, but by network. Never call free or delete on these. They
	// are guaranteed to be valid when accessed from a phenotype node.
	float* previousOutput; 
	float* currentOutput;
	float* previousInput;
	float* currentInput;
#ifdef SATURATION_PENALIZING
	float* averageActivation;
#endif

	PhenotypeNode(GenotypeNode* type);
	~PhenotypeNode() {};

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
	void setArrayPointers(float* po, float* co, float* pi, float* ci, float* aa = nullptr);

#ifdef SATURATION_PENALIZING
	void setSaturationPenalizationPtr(float* saturationPenalizationPtr);
#endif
};
