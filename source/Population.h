#pragma once

#include <memory>
#include <cmath>
#include <string>
#include <thread>

#include "Network.h"
#include "Trial.h"
/*
The optimizer is a non canon version of the genetic algorithm, this is how the loop goes:

- All specimen are mutated, which is handled by their own class. 
(Mutations first in the loop makes more sense.)

- Each individual is evaluated a certain number of times on each trial of the set.
The trial's initialization is random at each generation, but the same values are kept within 
a generation. The scores are saved into a vector for each individual. 
(The score on a trial is in [0,1], 1 corresponding to the perfect run.)

- Then, a clustering algorithm is applied on the individuals. The fitness of a specimen is
a combination of the norm of its fitness vector (L2 ? l1 ? other ?), of how different it has 
performed compared to the others, i.e. how bad its representation is in the clustering, and of 
a regularization term that penalizes overgrowth (computed by the Network class). 
Potential clustering methods: K-Means (too hign dim...), PCA, ICA, ?

- A roulettte wheel selection with low pressure is applied to pick a specimen, which is simply copied to be in the next generation.
The process is repeated as many times as their are specimens.
*/

// A group of a fixed number of individuals, optimized with a genetic algorithm
class Population {

public:	
	// Not exposed to the DLL interface:
	~Population();
	void startThreads(int N_THREADS);
	void stopThreads();
	void step(std::vector<std::unique_ptr<Trial>>& trials);

	// Exposed to the DLL interface:
	Population(int IN_SIZE, int OUT_SIZE, int N_SPECIMENS);

	void computeFitnesses(std::vector<float> avgScorePerSpecimen);
	void createOffsprings();
	void mutatePopulation() { for (int i = 0; i < N_SPECIMENS; i++) networks[i]->mutate(); };
	Network* getFittestSpecimenPointer() {return networks[fittestSpecimen];}
	void setEvolutionParameters(float f0 = .2f, float regularizationFactor = .03f) {
		this->regularizationFactor = regularizationFactor;
		this->f0 = f0;
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

private:

	void threadLoop(const int i0, const int subArraySize);
	void evaluate(const int i0, const int subArraySize, std::vector<std::unique_ptr<Trial>>& localTrials);
	int N_SPECIMENS, N_THREADS;
	std::vector<Network*> networks;
	// Indice in the networks list of the fittest specimen at this step.
	int fittestSpecimen;

	// The relative importance of the normalized regularization factor to the score. 0 means no regularization,
	// 1 means regularization is as important as score. Recommended value varies with the task, .03f is a safe baseline.
	float regularizationFactor;

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