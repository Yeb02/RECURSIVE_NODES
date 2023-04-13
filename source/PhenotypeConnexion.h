#pragma once

#include <memory>

#include "config.h"


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
	// Should not be called !
	PhenotypeConnexion() {};

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