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
		if (topNodeP.get() == NULL) topNodeP.reset(new PhenotypeNode(topNodeG.get()));
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
	std::unique_ptr<GenotypeNode> topNodeG;
	std::vector<std::unique_ptr<GenotypeNode>> genome;
	std::unique_ptr<PhenotypeNode> topNodeP;

	// Requires genome be sorted by ascending depth !!
	// Called with a small probability in mutations. Deletes all nodes that do not show up in the phenotype.
	void removeUnusedNodes();

	// Requires depths to be up to date
	bool hasChild(GenotypeNode* parent, GenotypeNode* potentialChild);

	// Update the depths of the genome and the top node. Requires every node to have a valid position !
	// i.e. in [0, genome.size() - 1], and no two nodes share the same.
	void updateDepths();

	// Requires the nodes' depths be up to date !
	void sortGenome();
};