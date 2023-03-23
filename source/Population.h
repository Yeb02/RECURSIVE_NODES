#pragma once

#include <memory>
#include <cmath>
#include <string>
#include <thread>

#include "Network.h"
#include "Trial.h"

/*
The optimizer is a non canon version of the genetic algorithm, this is how the loop goes:

- All specimen are mutated, which is handled by their own class. Topology and real-valued 
parameters are subject to mutations, whose probabilities are fixed in Network.mutateFloats()
(Mutations first in the loop makes more sense.)

- Each individual is evaluated once on all trials of the set.
For each trial, initialization is random, but the same values are kept within the step. It drastically
diminishes noise and speeds up convergence. The scores are saved into a matrix, one axis corresponding 
to the trials, the other to the specimens.

- A simple artificial niching algorithm is applied, based on ressource sharing principles, and implicit novelty search. 
In the matrix, the vectors of scores per trial are linearly transformed to have min = 0 and max = 1.  
Score per specimen is then a p-norm of its score vector. p < 1 is allowed since all values are >=0 (in [0, 1]). 
Increasing p beyond 1 fosters specialisation, p=inf being the extreme with only the best trial taken into consideration
when computing fitnesses. p=1 encourages digging where other's do not. Decreasing p furthermore weakens the "outwards push".
Which is better is up to the problem's setting and its hyperparameters. I recommend disabling it completely, 
(nichingNorm = 0.0), when nTrials is small( < 8), AND the difference between trials is not especially meaningful,
like running the same trial with slightly different random init.

TODO experiment.

- Regularization is a function of both sheer number of parameters (in the phenotype !!) and their amplitude. I have observed
a strange phenomenon where increasing regularization strength INCREASES networks sizes... That was unexpected. (And I'm pretty
sure it is not a bug.)

- Fitness is computed as a linear combination of trial score and regularization score.

- A roulettte wheel selection with parametrized pressure is applied to pick a specimen, which is simply 
copied to be in the next generation. The selected specimens is not eliminated from the potential parents pool.
The process is repeated as many times as their are specimens, we then proceed to the next iteration of the loop.
*/


// Contains the parameters for population evolution. Detail on each parameter in the Population class
// declaration. (Population.h)
struct PopulationEvolutionParameters {
	float regularizationFactor;
	float nichingNorm;
	float selectionPressure;
	bool useSameTrialInit;
	bool normalizedScoreGradients;

	//defaults:
	PopulationEvolutionParameters() {
		selectionPressure = 0.0f;
		regularizationFactor = 0.1f;
		nichingNorm = 0.0f;
		useSameTrialInit = false;
		normalizedScoreGradients = false;
	}
};


// A group of a fixed number of individuals, optimized with a genetic algorithm.
class Population {

public:	
	// Not exposed to the DLL interface:
	~Population();
	void startThreads(int N_THREADS);
	void stopThreads();
	// Only the last nTrialsEvaluated are used for fitness calculations. Previous ones are used for lifelong learning.
	void step(std::vector<std::unique_ptr<Trial>>& trials, int nTrialsEvaluated);

	// Exposed to the DLL interface:
	Population(int IN_SIZE, int OUT_SIZE, int N_SPECIMENS);
	void computeFitnesses(std::vector<float> avgScorePerSpecimen);
	void createOffsprings();
	void setEvolutionParameters(PopulationEvolutionParameters params) {
		this->regularizationFactor = params.regularizationFactor;
		this->selectionPressure = params.selectionPressure;
		this->nichingNorm = params.nichingNorm;
		this->useSameTrialInit = params.useSameTrialInit;
		this->normalizedScoreGradients = params.normalizedScoreGradients;
	}

	// DLL util only:
	void mutatePopulation() { for (int i = 0; i < N_SPECIMENS; i++) networks[i]->mutate(); };

	// TODO (exposed):
	std::string save() { 
		return std::string("");
	}; 
	void destroyThenLoad(std::string fileName) {};
	void defragmentate() {
		std::string fileName = save();
		// delete population
		destroyThenLoad(fileName);
		// delete file
	}  

	// I dont like getters and setters, but it seems to me like a justified use here. Meant to be called from the DLL part.
	int get_N_SPECIMENS() { return N_SPECIMENS; };
	Network* getSpecimenPointer(int i) { return networks[i]; };

	// Indice in the networks list of the fittest specimen at this step.
	int fittestSpecimen;

	// Current number of generations since initialization.
	int evolutionStep;

private:

	void threadLoop(const int i0, const int subArraySize);
	void evaluate(const int i0, const int subArraySize, Trial* trial, float* scores);
	int N_SPECIMENS, N_THREADS;
	std::vector<Network*> networks;
	

	// The relative importance of the normalized regularization factor to the score. 0 means no regularization,
	// 1 means regularization is as important as score. Recommended value varies with the task, .03f is a safe baseline.
	float regularizationFactor;

	// Enables artificial niching, when > 0. Value of pNorm's p, but can be < 1 because applied to positive values.
	// Requires (mathematically) all trials at a time step have similar distributions of scores,
	// and equal weight in the global fitness.
	float nichingNorm;

	// Should be < 1. When selecting "parents" for the next generation, rejects all specimens
	// below selectionPressure in the gaussian of fitnesses (mean 0 var ~1). Values too high may cause a
	// warning and undesired behaviour. Can be constant or changed at each steps, with for instance:
	// UNIFORM_01 -.5f     OR   sinf((float)iteration / 3.0f)   OR ...
	float selectionPressure;

	// If set to true, each trial of the vector passed to the step function will be reset to the same initial values for
	// each specimen. In most cases, this completely eliminates the stochasticity of fitness function.
	bool useSameTrialInit;

	// Experimental, default=false. Only used with CONTINUOUS_LEARNING && GUIDED_MUTATIONS. If true, for each specimen,
	// postTrialUpdate adds the learned weights wLifetime to the accumulators, wLifetime being weighted by either:
	//   - score in the [NORMALIZED score per specimen on this trial]  array.
	//   - relative score to that of its parent ON THE SAME TRIAL WITH SIMILAR INIT at last step.
	// AND THEN ZEROS wLifetime !
	bool normalizedScoreGradients; 

	// The vector of fitness per specimen, >0. Fitness 0 = probability 0 of generating offspring.
	std::vector<float> fitnesses;


	// Used only if N_THREADS > 1 (i.e. multithreading enabled):

	std::vector<std::thread> threads;
	std::vector<Trial*> globalTrials;
	float* pScores;
	int threadIteration;
	bool mustTerminate;
};