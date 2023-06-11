#pragma once

#include <memory>
#include <fstream>

#define WRITE_4B(i, os) os.write(reinterpret_cast<const char*>(&i), 4);
#define READ_4B(i, is) is.read(reinterpret_cast<char*>(&i), 4);

#include "Random.h"

#include "config.h"



struct InternalConnexion_G {

	const enum INITIALIZATION { ZERO, RANDOM };

	// Depends on which preprocessor directives are active. Parameters that have a storage version and a
	// runtime version are not counted twice.
	static int nEvolvedArrays;

	int nLines, nColumns;

	std::unique_ptr<float[]> A;
	std::unique_ptr<float[]> B;
	std::unique_ptr<float[]> C;
	std::unique_ptr<float[]> D;
	std::unique_ptr<float[]> eta;	// in [0, 1]
	std::unique_ptr<float[]> storage_eta;   // in R
	std::unique_ptr<float[]> alpha;

	std::unique_ptr<ACTIVATION[]> activationFunctions;
	std::unique_ptr<float[]> biases;

#ifndef RANDOM_W
	std::unique_ptr<float[]> w;
#endif

#ifdef OJA
	std::unique_ptr<float[]> delta; // in [0, 1]
	std::unique_ptr<float[]> storage_delta; // in R
#endif

#ifdef CONTINUOUS_LEARNING
	std::unique_ptr<float[]> gamma; // in [0, 1]
	std::unique_ptr<float[]> storage_gamma; // in R
#endif

#ifdef GUIDED_MUTATIONS
	std::unique_ptr<float[]> accumulator;

	void zeroAccumulator() {
		for (int i = 0; i < nLines * nColumns; i++) {
			accumulator[i] = 0.0f;
		}
	}
#endif


#ifdef STDP
	std::unique_ptr<float[]> STDP_mu;
	std::unique_ptr<float[]> STDP_storage_mu;

	std::unique_ptr<float[]> STDP_lambda;
	std::unique_ptr<float[]> STDP_storage_lambda;
#endif


	InternalConnexion_G() { nLines = -1; nColumns = -1; };

	InternalConnexion_G(int nLines, int nColumns, INITIALIZATION init);

	InternalConnexion_G(const InternalConnexion_G& gc);

	InternalConnexion_G operator=(const InternalConnexion_G& gc);

	~InternalConnexion_G() {};

	InternalConnexion_G(std::ifstream& is);
	void save(std::ofstream& os);

	int getNParameters() {
		return nEvolvedArrays * nLines * nColumns;
	}

	// Maps stored_X to X for all parameters X that are used at runtime in the range [0,1]
	// but stored in the range R. Typically exponential average decays. 
	// To be called at phenotype creation.
	void transform01Parameters();

	void mutateFloats(float p);

#if defined GUIDED_MUTATIONS
	void accumulateW(float factor, float* wLifetime);
#endif

	void insertLineRange(int id, int s);
	
	void insertColumnRange(int id, int s);
	
	void removeLineRange(int id, int s);
	
	void removeColumnRange(int id, int s);
};
