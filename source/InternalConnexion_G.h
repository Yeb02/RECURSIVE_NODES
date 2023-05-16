#pragma once

#include <memory>

#include "Random.h"

#include "config.h"



struct InternalConnexion_G {

	const enum INITIALIZATION { ZERO, RANDOM };


	int nLines, nColumns;

	std::unique_ptr<float[]> A;
	std::unique_ptr<float[]> B;
	std::unique_ptr<float[]> C;
	std::unique_ptr<float[]> D;
	std::unique_ptr<float[]> eta;
	std::unique_ptr<float[]> alpha;
	std::unique_ptr<float[]> w;
#ifdef CONTINUOUS_LEARNING
	std::unique_ptr<float[]> gamma;
#endif

#ifdef GUIDED_MUTATIONS
	std::unique_ptr<float[]> accumulator;

	void zeroAccumulator() {
		for (int i = 0; i < nLines * nColumns; i++) {
			accumulator[i] = 0.0f;
		}
	}
#endif

	InternalConnexion_G() { nLines = -1; nColumns = -1; };

	InternalConnexion_G(int nLines, int nColumns, INITIALIZATION init);

	InternalConnexion_G(const InternalConnexion_G& gc);

	InternalConnexion_G operator=(const InternalConnexion_G& gc);

	~InternalConnexion_G() {};

	void mutateFloats(float p);

#if defined GUIDED_MUTATIONS
	void accumulateW(float factor, float* wLifetime);
#endif

	void insertLineRange(int id, int s);
	
	void insertColumnRange(int id, int s);
	
	void removeLineRange(int id, int s);
	
	void removeColumnRange(int id, int s);
};
