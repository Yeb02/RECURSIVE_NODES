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
	Population(int IN_SIZE, int OUT_SIZE, int N_SPECIMENS);
	~Population();
	void startThreads(int N_THREADS);
	void step(std::vector<Trial*> trials);
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
	Network* getFittestSpecimenPointer() {
		return networks[fittestSpecimen];
	}

private:
	void mutateNevaluateThreaded(const int i0, const int subArraySize);
	int N_SPECIMENS, N_THREADS;
	std::vector<Network*> networks;
	int fittestSpecimen;

	// unused if N_THREADS = 0.
	std::vector<std::thread> threads;
	std::vector<Trial*> globalTrials;
	float* pScores;
	int iteration;
};