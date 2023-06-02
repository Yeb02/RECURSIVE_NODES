#pragma once

#include <memory>
#include <InternalConnexion_G.h>
#include "config.h"


struct InternalConnexion_P {   // responsible of its pointers

	InternalConnexion_G* type;

	std::unique_ptr<float[]> H;
	std::unique_ptr<float[]> E;

	// Initialized to 0 when and only when the connexion is created. If CONTINUOUS_LEARNING, updated at each inference,
	// otherwise updated at the end of each trial.
	std::unique_ptr<float[]> wLifetime;

#ifndef CONTINUOUS_LEARNING
	// Arithmetic avg of H over trial duration.
	std::unique_ptr<float[]> avgH;
#endif

#ifdef RANDOM_W
	// Reset to random values at the beginning of each trial
	std::unique_ptr<float[]> w;

	void randomInitW() 
	{
		float normalizator = powf((float)type->nColumns, -.5f);
		int s = type->nLines * type->nColumns;

		for (int i = 0; i < s; i++) {
			w[i] = NORMAL_01 * normalizator;
		}
	}
#endif


	// Should not be called !
	InternalConnexion_P(const InternalConnexion_P&) {};
	
	// Should not be called !
	InternalConnexion_P() {};

	InternalConnexion_P(InternalConnexion_G* type);

	void zero();

	// only called at construction.
	void zeroWlifetime();

#ifndef CONTINUOUS_LEARNING
	// factor = 1/nInferencesP., wLifetime += alpha*avgH*factor
	void updateWatTrialEnd(float factor);
#endif

	~InternalConnexion_P() {};
};