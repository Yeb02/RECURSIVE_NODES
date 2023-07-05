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

#ifdef RANDOM_WB
	// Reset to random values at the beginning of each trial
	std::unique_ptr<float[]> w;
	std::unique_ptr<float[]> biases;

	void randomInitWB();
#endif


#ifdef DROPOUT
	void dropout();
#endif

	// Should not be called !
	// And strangely, is never called but removing its declaration causes an error.
	InternalConnexion_P(const InternalConnexion_P&) { __debugbreak();  type = nullptr; };
	
	// Should not be called !
	InternalConnexion_P() { __debugbreak();  type = nullptr; };

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