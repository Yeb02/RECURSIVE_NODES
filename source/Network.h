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

	// Since output returns a float*, application must either use it before any other call to step(),
	// or to destroyPhenotype(), preTrialReset(), ... or Network destruction, either deep copy the
	// pointee immediatly when getOutput() returns. If unsure, deep copy.
	float* getOutput();

	void step(const std::vector<float>& obs);
	void mutate();
	
	void createPhenotype();
	void destroyPhenotype();

	// Sets to 0 the dynamic elements of the phenotype. 
	void preTrialReset();

	// #if defined GUIDED_MUTATIONS && defined CONTINUOUS_LEARNING
	// In some cases, it is not a good idea to keep learned weights between trials, for instance when learned knowledge does
	// not transpose from one to the other. The fitness  argument can either be relative to other individuals in the genotype,
	// or absolute, in which case the Network instance must keep in memory the fitness of its parent on the same trial.
	void postTrialUpdate(float score);

	// a positive float, increasing with the networks number of parameters and their amplitudes. Ignores biases.
	float getRegularizationLoss();

	int inputSize, outputSize;

private:
	int nSimpleNeurons;
	std::unique_ptr<GenotypeNode> topNodeG;
	std::vector<std::unique_ptr<GenotypeNode>> genome;
	std::unique_ptr<PhenotypeNode> topNodeP;

	// Arrays for plasticity based updates. Contain ALL values from the phenotype.
	// Must be : reset to all 0s at the start of each trial;
	// created alongside PhenotypeNode creation; freed alongside PhenotypeNode deletion.
	std::unique_ptr<float[]> previousOutputs, currentOutputs, previousInputs, currentInputs;

	int phenotypeInArraySize, phenotypeOutArraySize;

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