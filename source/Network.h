#pragma once

#include <vector>
#include <memory>
#include <cmath>

#include "Random.h"
#include "Phenotype.h"
#include "Genotype.h"

//#include <boost/archive/text_iarchive.hpp>


class Network {
	friend class Drawer;
public:
	Network(int inputSize, int outputSize);

	// Does NOT create the phenotype tree ! No "topNodeP = new PhenotypeNode(&genome[genome.size()-1]);"
	Network(Network* n);
	~Network() {};

	std::vector<float> getOutput();
	void step(const std::vector<float>& obs);
	void mutate();
	
	void createPhenotype() {
		if (topNodeP.get() == NULL) topNodeP.reset(new PhenotypeNode(genome[genome.size() - 1].get()));
	};

	// sets to 0 the dynamic elements of the phenotype
	void intertrialReset() {
		topNodeP->interTrialReset();
	};

	// a positive float, increasing with the networks number of parameters and their amplitudes. Ignores biases.
	float getRegularizationLoss();

	int inputSize, outputSize;

private:
	int nSimpleNeurons;
	std::vector<std::unique_ptr<GenotypeNode>> genome;
	std::unique_ptr<PhenotypeNode> topNodeP;

	// necessary because otherwise, nodes could get bigger than the top node. TODO change this
	// stupid "end of list" architecture.
	void removeUnusedNodes();

	void updateDepths();
	// assumes the nodes' depths are up to date.
	void sortGenome();
};