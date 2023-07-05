#include "InternalConnexion_P.h"

InternalConnexion_P::InternalConnexion_P(InternalConnexion_G* type) : type(type)
{
	int s = type->nLines * type->nColumns;

	H = std::make_unique<float[]>(s);
	E = std::make_unique<float[]>(s);
	wLifetime = std::make_unique<float[]>(s);

#ifndef CONTINUOUS_LEARNING
	avgH = std::make_unique<float[]>(s);
#endif

#ifdef RANDOM_WB
	w = std::make_unique<float[]>(s);
	biases = std::make_unique<float[]>(type->nLines);
#endif

	zeroWlifetime(); // necessary because ComplexNode_P::preTrialReset() does not have to do it.

	// zero(s); not necessary, because ComplexNode_P::preTrialReset() should be called before any computation. 
}

#ifndef CONTINUOUS_LEARNING
void InternalConnexion_P::updateWatTrialEnd(float factor) {
	int s = type->nLines * type->nColumns;
	float* alpha = type->alpha.get();

	for (int i = 0; i < s; i++) {
		wLifetime[i] += alpha[i] * avgH[i] * factor;
	}
}
#endif

#ifdef DROPOUT
void InternalConnexion_P::dropout() {
	int s = type->nLines * type->nColumns;
	float normalizator = .3f * powf((float)type->nColumns, -.5f);

	SET_BINOMIAL(s, .002f);
	int _nMutations = BINOMIAL;
	for (int i = 0; i < _nMutations; i++) {
		int id = INT_0X(s);

#ifdef CONTINUOUS_LEARNING
		wLifetime[id] = 0.0f;
#endif
		H[id] = 0.0f;
		E[id] = 0.0f;

#ifdef RANDOM_WB
		w[id] = NORMAL_01 * normalizator;
#endif 

	}
}
#endif

void InternalConnexion_P::zero() {
	int s = type->nLines * type->nColumns;

	for (int i = 0; i < s; i++) {
		H[i] = 0.0f;
		E[i] = 0.0f;
#ifndef CONTINUOUS_LEARNING
		avgH[i] = 0.0f;
#endif
	}
}


void InternalConnexion_P::zeroWlifetime()
{
	int s = type->nLines * type->nColumns;

	for (int i = 0; i < s; i++) {
		wLifetime[i] = 0.0f;
	}
}

#ifdef RANDOM_WB
void InternalConnexion_P::randomInitWB()
{
	float normalizator = .3f * powf((float)type->nColumns, -.5f);
	int s = type->nLines * type->nColumns;

	for (int i = 0; i < s; i++) {
		//w[i] = .2f * (UNIFORM_01 - .5f);
		w[i] = NORMAL_01 * normalizator;
	}

	for (int i = 0; i < type->nLines; i++) {
		//biases[i] = NORMAL_01 * .1f; //Seems to hurt results (that was kinda expected.)
		biases[i] = 0.0f;
	}
}
#endif

