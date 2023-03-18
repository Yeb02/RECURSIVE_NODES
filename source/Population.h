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
diminishes noise and speeds up convergence. The scores are saved into a matrix. 

- An original, simple niching algorithm is applied, based on ressource sharing principles. After computing
raw scores, we have a matrix of values, one axis corresponds to the trials, the other to the specimens.
The vectors of scores per trial are linearly transformed to have mean 0 and variance 1.
They are then linearly transformed again to have min = 0 and max = 1. Both steps are needed, and are not
redundant. Then compute the score per specimen, a p-norm of its score vector.
Increasing p beyond 1 fosters specialisation, p=inf being the extreme with only the best trial taken into consideration
when computing fitnesses. p < 1 is allowed because at this point all values are positive (in [0, 1]). 
p=1 encourages digging where other's do not. Decreasing p furthermore weakens the "outwards push".
Which is better is up to the problem's setting and its hyperparameters. TODO experiment.

- Fitness is computed, as a linear combination of trial score and regularization score.

- A roulettte wheel selection with parametrized pressure is applied to pick a specimen, which is simply 
copied to be in the next generation. The selected specimens is not eliminated from the potential parents pool.
The process is repeated as many times as their are specimens, we then proceed to the next step.
*/




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
	void mutatePopulation() { for (int i = 0; i < N_SPECIMENS; i++) networks[i]->mutate(); };
	void setEvolutionParameters(float f0 = .0f, float regularizationFactor = .1f, float nichingNorm=1.0f) {
		this->regularizationFactor = regularizationFactor;
		this->f0 = f0;
		this->nichingNorm = nichingNorm;
	}

	// TODO (exposed):
	std::string save() { 
		return std::string("");
	}; 
	void load(std::string fileName) {}; 
	void defragmentate() {
		std::string fileName = save();
		// delete population
		load(fileName);
		// delete file
	}  

	// I dont like getters and setters, but it seems to me like a justified use here. Meant to be called from the DLL part.
	int get_N_SPECIMENS() { return N_SPECIMENS; };
	Network* getSpecimenPointer(int i) { return networks[i]; };

	int fittestSpecimen;

private:

	void threadLoop(const int i0, const int subArraySize);
	void evaluate(const int i0, const int subArraySize, std::vector<std::unique_ptr<Trial>>& localTrials);
	int N_SPECIMENS, N_THREADS;
	std::vector<Network*> networks;
	// Indice in the networks list of the fittest specimen at this step.

	// The relative importance of the normalized regularization factor to the score. 0 means no regularization,
	// 1 means regularization is as important as score. Recommended value varies with the task, .03f is a safe baseline.
	float regularizationFactor;

	// Enables artificial niching, when > 0. Value of pNorm's p, but can be < 1 because applied to positive values.
	// Requires (mathematically) all trials at a time step have similar distributions of scores,
	// and equal weight in the global fitness.
	float nichingNorm;

	// The higher f0, the lower the selection pressure. When f0 = 0, the least fit individual has a probability of 0
	// to have children. It can be set at each step with one of the following formulas:
	// f0 = 1.0f + UNIFORM_01 * 1.0f; OR f0 = .5f * (1.0f + sinf((float)iteration / 3.0f)); OR ...
	float f0;

	std::vector<float> fitnesses;

	// unused if N_THREADS = 0 or 1:
	std::vector<std::thread> threads;
	std::vector<Trial*> globalTrials;
	float* pScores;
	int iteration;
	bool mustTerminate;
};