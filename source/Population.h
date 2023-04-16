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


// Contains the parameters for population evolution. They can be changed at each step.
struct PopulationEvolutionParameters {

	// The importance of the (normalized) regularization factor relative to the score. 0 means no regularization,
	// 1 means regularization is as important as score. Recommended value depends on the task.
	float regularizationFactor;


	// Enables artificial niching, when > 0. 0.0f to disable. Value of pNorm's p. When on, encourages high diversity
	// of solution, so one may want to increase the selection pressure. The higher the value, the more specialists,
	// those that perform very well on one trial and poorly on the other, are encouraged. At the opposite,
	// low values favor generalists.
	// When enabled, all trials at a time step should have similar distributions of scores, at a linear transformation,
	// and have equal "value" in the global fitness. 0.8 is a good baseline, dont go under 0.5 or over 10.
	float nichingNorm;


	// Should be < 1.0 , safe value is 0.0 . Works like this: 
	// The final fitness array has mean 0 and variance ~1. Each specimen is assigned a probability to generate offsprings.
	// Probability is proportionnal to max(0, fitness - selectionPressure).	
	// Example of values at each evolution step:  UNIFORM_01 -.5f    OR   .5f*sinf((float)evolutionStep / 10.0f)   OR ...
	float selectionPressure;


	// If set to true, each trial of the vector passed to the step function will be reset to the same initial values for
	// each specimen. This means that all specimens are evaluated on the exact same tasks.
	bool useSameTrialInit;


	// Experimental, default=false. Only used with CONTINUOUS_LEARNING && GUIDED_MUTATIONS, and when enabled in 
	// Network::postTrialUpdate().  If true, for each specimen, postTrialUpdate adds to an accumulator for each weight:
	// [wLifetime * normalizedScoreOnThisTrial] * constant 
	// Then zeros wLifetime, so there is no memory between trials ! TODO wise ?
	bool normalizedScoreGradients;


	// Experimental, default=0.0. Only used when SATURATION_PENALIZING is defined, as it slows down forward() (a bit).
	// The higher, the stronger the penalty for saturated activations. It may be important to use it when GUIDED_MUTATIONS
	// is defined, as with it networks are prone to oversaturation.
	float saturationFactor;

	// Must be a multiple of N_THREADS. If 0, ignored. The number of specimens after the next call to step(). 
	int targetNSpecimens;


	// EvaluateFitness() is supposed to receive a vector of fitnesses, 1 value per specimen, more or less normally 
	// distributed (the function handles centering and reducing). However, many trials have no such measure
	// of the fitness : it may be exponential, or too noisy , or ... In general, it wont be easily interpretable for
	// generating offsprings with the best probabilities. 
	// And even if it were the  case, the distribution of fitnesses ultimately depends on the population itself.
	// This is why a ranking-fitness should be used in the general case, instead of raw trial scores. Scores may be
	// relevant for certain trials, but do not use them when unsure.
	bool rankingFitness;


	//defaults:
	PopulationEvolutionParameters() {
		selectionPressure = 0.0f;
		regularizationFactor = 0.1f;
		nichingNorm = 0.0f;
		useSameTrialInit = false;
		normalizedScoreGradients = false;
		saturationFactor = .05f;
		targetNSpecimens = 0;
		rankingFitness = true;
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

	Population(int IN_SIZE, int OUT_SIZE, int nSpecimens);

	// No requirement on avgScorePerSpecimen, other that a higher score = a better specimen.
	void computeFitnesses(std::vector<float>& avgScorePerSpecimen);

	void createOffsprings();
	void setEvolutionParameters(PopulationEvolutionParameters params) {
		this->regularizationFactor = params.regularizationFactor;
		this->selectionPressure = params.selectionPressure;
		this->nichingNorm = params.nichingNorm;
		this->useSameTrialInit = params.useSameTrialInit;
		this->normalizedScoreGradients = params.normalizedScoreGradients;
		this->saturationFactor = params.saturationFactor;
		this->targetNSpecimens = params.targetNSpecimens;
		this->rankingFitness = params.rankingFitness;
	}


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

	// DLL util only:
	void mutatePopulation() {
		for (int i = 0; i < nSpecimens; i++) {
			networks[i]->mutate();
		}
	};


	// I dont like getters and setters, but it seems to me like a justified use here. Meant to be called from the DLL part.
	int get_nSpecimens() { return nSpecimens; };
	Network* getSpecimenPointer(int i) { return networks[i]; };

	// Indice in the networks list of the fittest specimen at this step.
	int fittestSpecimen;

	// Current number of generations since initialization.
	int evolutionStep;

private:

	void threadLoop(const int i0, const int subArraySize);
	void evaluate(const int i0, const int subArraySize, Trial* trial, float* scores);

	// Current size of the networks and fitness arrays. Must be a multiple of N_THREADS.
	int nSpecimens;
		
	// Constant, until call to destroy_threads !		
	int N_THREADS;

	// Size nSpeciemns, subject to change at each step.
	std::vector<Network*> networks;
	
	// The vector of fitness per specimen, >0. Fitness 0 = probability 0 of generating offspring.
	// Size nSpeciemns, subject to change at each step.
	std::vector<float> fitnesses;

	// Pointer to the score matrix (a vector local to step(), so make sure it is not out of scope). 
	float* pScores;



	// EVOLUTION PARAMETERS: 
	
	// Set with a PopulationEvolutionParameters struct. Description in the struct definition.
	int targetNSpecimens;

	// Set with a PopulationEvolutionParameters struct. Description in the struct definition.
	float regularizationFactor, nichingNorm, selectionPressure, saturationFactor;

	// Set with a PopulationEvolutionParameters struct. Description in the struct definition.
	bool useSameTrialInit, normalizedScoreGradients, rankingFitness;



	// Threading utils, used only if N_THREADS > 1 (i.e. multithreading enabled):

	std::vector<std::thread> threads;
	std::vector<Trial*> globalTrials;
	int threadIteration;
	bool mustTerminate;
};