#include "PhenotypeConnexion.h"

PhenotypeConnexion::PhenotypeConnexion(int s)
{
	H = std::make_unique<float[]>(s);
	E = std::make_unique<float[]>(s);
	wLifetime = std::make_unique<float[]>(s);

#ifndef CONTINUOUS_LEARNING
	avgH = std::make_unique<float[]>(s);
#endif

	zeroWlifetime(s); // necessary because ComplexNode_P::preTrialReset() does not have to do it.

	// zero(s); not necessary, because ComplexNode_P::preTrialReset() should be called before any computation. 
}

#ifndef CONTINUOUS_LEARNING
void PhenotypeConnexion::updateWatTrialEnd(int s, float factor, float* alpha) {
	for (int i = 0; i < s; i++) {
		// When the following line is commented, there is no learning at all between trials. 
		// Useful for benchmarking.
		wLifetime[i] += alpha[i] * avgH[i] * factor;
	}
}
#endif

void PhenotypeConnexion::zero(int s) {
	for (int i = 0; i < s; i++) {
		H[i] = 0.0f;
		E[i] = 0.0f;
#ifndef CONTINUOUS_LEARNING
		avgH[i] = 0.0f;
#endif
	}
}

void PhenotypeConnexion::zeroWlifetime(int s) {
	for (int i = 0; i < s; i++) {
		wLifetime[i] = 0.0f;
	}
}