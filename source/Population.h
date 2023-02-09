#pragma once

#include "Network.h"
#include "Trial.h"


/*
The genetic optimizer is a peculiar version of the genetic algorithm: besides the unusual mutations the
topology requires, the speciation, evaluation and next gen constitution are unorthodox:

- Each individual is evaluated a certain number of times on each trial of a given set,
with exactly the same random init for each individual. The scores are saved into a vector 
for each individual. 
(The score on a trial is in [0,1], 1 corresponding to the perfect run.)

- Then, a clustering algorithm is applied on the individuals. The fitness of a specimen is
a combination of the norm of its fitness vector (L2 ? l1 ? other ?) and of how different it has 
performed compared to the others, i.e. how bad its representation is in the clustering. 
*/
class Population {

public:
	Population(int IN_SIZE, int OUT_SIZE, int N_SPECIMENS);
	~Population();
	void step(std::vector<Trial*>);

private:
	int N_SPECIMENS;
	std::vector<Network*> networks;
	void mutate();
	void evaluate();
};