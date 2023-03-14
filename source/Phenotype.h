#pragma once

#include <vector>
#include <memory>
#include <cmath>

#include "Random.h"
#include "Genotype.h"


struct PhenotypeConnexion {   // responsible of its pointers

	std::unique_ptr<float[]> H;

	std::unique_ptr<float[]> E;

	// Should not be called !
	PhenotypeConnexion(const PhenotypeConnexion&) {};

	PhenotypeConnexion(int s);

	void zero(int s);

	~PhenotypeConnexion() {};
};

struct PhenotypeNode {
	GenotypeNode* type;

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

	void zero();
	void forward(const float* input);
};
