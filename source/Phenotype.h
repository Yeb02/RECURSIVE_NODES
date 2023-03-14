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
	// factor = 1/nInferences., wLifetime += alpha*avgH*factor
	void updateWatTrialEnd(int s, float factor, float* alpha);
#endif

	~PhenotypeConnexion() {};
};

struct PhenotypeNode {
	GenotypeNode* type;

	int nInferences; // store how many inferences this node has performed since last interTrialReset.
	float neuromodulatorySignal; //initialized at 1 at the beginning of a trial
	float M[2];
	// Pointers to its children. Responsible for their lifetime !
	std::vector<PhenotypeNode> children;

	// Vector of structs containing pointers to the dynamic connexion matrices linking children
	std::vector<PhenotypeConnexion> childrenConnexions;

	// For plasticity based updates, must be reset to all 0s at the start of each trial
	std::vector<float> previousOutput, currentOutput, previousInput;

	PhenotypeNode(GenotypeNode* type);
	~PhenotypeNode() {};

	void interTrialReset();
	void forward(const float* input);
};
