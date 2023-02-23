#pragma once

#include <vector>
#include <memory>
#include <cmath>

#include "Random.h"
#include "Phenotype.h"
#include "Genotype.h"

//#include <boost/archive/text_iarchive.hpp>


class Network {

public:
	Network(int inputSize, int outputSize);

	// Does NOT create the phenotype tree ! No "topNodeP = new PhenotypeNode(&genome[genome.size()-1]);"
	Network(Network* n);
	~Network() {};

	std::vector<float> getOutput();
	void step(std::vector<float> obs);
	void save(std::string path);
	void mutate();

	// sets to 0 the dynamic elements of the phenotype
	void intertrialReset();

	// a positive float, increasing with the networks number of parameters.
	float getSizeRegularizationLoss();
	// a positive float, increasing with the amplitude of the network's parameters. 
	float getAmplitudeRegularizationLoss();

private:
	int nSimpleNeurons;
	int inputSize, outputSize;
	std::vector<std::unique_ptr<GenotypeNode>> genome;
	std::unique_ptr<PhenotypeNode> topNodeP;

	void removeUnusedNodes();
};