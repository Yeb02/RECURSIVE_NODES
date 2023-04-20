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


	// Both values should be < 1.0 , safe value is 0.0 .  
	// ".first " influences the probability of each specimen to be present once in the next generation. 
	// ".second" influences the probability of each specimen to take a spot left empty by a specimen that did not make it
	std::pair<float, float> selectionPressure;


	// If set to true, each trial of the vector passed to the step function will be reset to the same initial values for
	// each specimen. This means that all specimens are evaluated on the exact same tasks.
	bool useSameTrialInit;


	// Experimental, default=0.0. Only used when SATURATION_PENALIZING is defined, as it slows down forward() (a bit).
	// The higher, the stronger the penalty for saturated activations. It may be important to use it when GUIDED_MUTATIONS
	// is defined, as with it networks are prone to oversaturation.
	float saturationFactor;


	// EvaluateFitness() is supposed to receive a vector of fitnesses, 1 value per specimen, more or less normally 
	// distributed (the function handles centering and reducing). However, many trials have no such measure
	// of the fitness : it may be exponential, or discontinuous , or ... In general, it wont be easily interpretable 
	// for generating offsprings with the best probabilities. 
	// And even if it were the  case, the normalizity of the fitness distribution ultimately depends on the 
	// relative performances of the networks. This is why a ranking fitness should be used in the general case, 
	// instead of raw trial scores. When unsure,  use rankingFitness = true.
	bool rankingFitness;


	// Disabled when = 0. Recommended range: [0, .3]. Useless after the maximum score has been reached.
	// Induces a term in the fitness which compares score at this step with score of the parent on the 
	// corresponding trial, at the previous step. (Therefore on a different random initialization.) 
	// Makes sense to use only if trials within a step are semantically different, and the random initialization
	// does not influence the score "too much". See note in Network.postTrialUpdate.
	float competitionFactor;


	//defaults:
	PopulationEvolutionParameters() {
		selectionPressure = { -10.0f, 0.0f };
		regularizationFactor = 0.1f;
		useSameTrialInit = false;
		saturationFactor = .05f;
		rankingFitness = true;
		competitionFactor = .1f;
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

	// nTrials is used in Network.parentData if available.
	void createOffsprings();

	void setEvolutionParameters(PopulationEvolutionParameters params) {
		this->regularizationFactor = params.regularizationFactor;
		this->selectionPressure = params.selectionPressure;
		this->useSameTrialInit = params.useSameTrialInit;
		this->saturationFactor = params.saturationFactor;
		this->rankingFitness = params.rankingFitness;
		this->competitionFactor = params.competitionFactor;
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

	// A util.
	int nTrialsAtThisStep;
	
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
	float regularizationFactor, saturationFactor, competitionFactor;

	// Set with a PopulationEvolutionParameters struct. Description in the struct definition.
	std::pair<float, float> selectionPressure;

	// Set with a PopulationEvolutionParameters struct. Description in the struct definition.
	bool useSameTrialInit, rankingFitness;



	// Threading utils, used only if N_THREADS > 1 (i.e. multithreading enabled):

	std::vector<std::thread> threads;
	std::vector<Trial*> globalTrials;
	int threadIteration;
	bool mustTerminate;
};