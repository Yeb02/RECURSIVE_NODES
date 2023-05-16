#pragma once

#include <vector>
#include <memory>
#include <cmath>

#include "Random.h"
#include "ComplexNode_P.h"
#include "ComplexNode_G.h"

//#include <boost/archive/text_iarchive.hpp>

// Stores data on the parent's performances, for use in the selection process and potentially 
// in GUIDED_MUTATIONS. Is not used (for now) in the dll, only the exe.
struct ParentData {
	bool isAvailable;

	// raw scores post ranking transformation, if ranking fitness, post normalization otherwise.
	float* scores;

	//  = trials.size(), and not nTrialsEvaluated.
	int scoreSize;

	ParentData() {
		isAvailable = false;
		scores = nullptr;
	}

	~ParentData() {
		delete scores;
	}
};

class Network {
	friend class Drawer;
public:
	Network(int inputSize, int outputSize);

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

#ifdef GUIDED_MUTATIONS
	// Sets at 0 the accumulators of both internal connexions for complex nodes and key+query for memory nodes.
	void zeroAccumulators();
#endif

	// Sets to 0 the dynamic elements of the phenotype. 
	void preTrialReset();

	// #if defined GUIDED_MUTATIONS && defined CONTINUOUS_LEARNING
	// In some cases, it is not a good idea to keep learned weights between trials, for instance when learned knowledge does
	// not transpose from one to the other. The fitness  argument can either be relative to other individuals in the genotype,
	// or absolute, in which case the Network instance must keep in memory the fitness of its parent on the same trial.
	void postTrialUpdate(float score, int trialID);

	// a positive float, increasing with the networks number of parameters and their amplitudes. Ignores biases.
	float getRegularizationLoss();

	float getSaturationPenalization();

	int inputSize, outputSize;

	ParentData parentData;

private:
	std::unique_ptr<ComplexNode_G> topNodeG;

	std::vector<std::unique_ptr<ComplexNode_G>> complexGenome;
	std::vector<std::unique_ptr<MemoryNode_G>> memoryGenome;

	// The phenotype is expected to be created and destroyed only once per Network, but it could happen several times.
	std::unique_ptr<ComplexNode_P> topNodeP;



	// Arrays for plasticity based updates. Contain all presynaptic and postSynaptic activities.
	// Must be : - reset to all 0s at the start of each trial;
	//			 - created alongside ComplexNode_P creation; 
	//           - freed alongside ComplexNode_P deletion.

	std::unique_ptr<float[]> previousPostSynActs;
	std::unique_ptr<float[]> currentPostSynActs;
	std::unique_ptr<float[]> preSynActs;

	// size of previousPostSynActs and currentPostSynActs
	int postSynActArraySize;

	// How many inferences were performed since last call to preTrialReset by the phenotype.
	int nInferencesOverTrial;

	// How many inferences were performed since phenotype creation.
	int nInferencesOverLifetime;

	// How many trials the phenotype has experimented.
	int nExperiencedTrials;

#ifdef SATURATION_PENALIZING
	// Sum over all the phenotype's activations, over the lifetime, of powf(activations[i], 2*n), n=typically 10.
	float saturationPenalization;

	// Follows the same usage pattern as the 4 arrays for plasticity updates. Size averageActivationArraySize.
	// Used to store, for each activation function of the phenotype, its average output over lifetime, for use in
	// getSaturationPenalization(). So set to 0 at phenotype creation, and never touched again.
	std::unique_ptr<float[]> averageActivation;

	// size of the averageActivation array.
	int averageActivationArraySize;
#endif

	// Sets phenotypicMultiplicity for each node of the genome, and the topNode. Should be called after each
	// structural modification of the genotype, i.e. after Network creation or mutations. Not necessary at copy 
	// construction.
	void updatePhenotypicMultiplicities();

	// Called with a small probability in mutations. Deletes all nodes that do not show up in the phenotype.
	void removeUnusedNodes();

	// Update the depths of the genome and the top node. Requires every node to have a valid position !
	// i.e. in [0, genome.size() - 1], and no two nodes share the same.
	void updateDepths();

	// Requires the nodes' depths be up to date !
	void sortGenome();
};