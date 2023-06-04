#pragma once

#include <vector>
#include <memory>
#include <cmath>
#include <unordered_map>
#include <fstream>

#include "Random.h"
#include "ComplexNode_P.h"
#include "ComplexNode_G.h"


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

	Network(std::ifstream& is);
	
	void save(std::ofstream& os);

	// TODO as of now, any topological difference in a node, between a secondary parent and 
	// the primary parent, nullifies the contribution of the secondary parent to this node.
	// For complex nodes, a topological difference is a difference in either of inputSize,
	// outputSize, complexChild.inputSize, complexChild.outputSize, memoryChild.inputSize, 
	// memoryChild.outputSize.
	// For memory nodes, inputSize, outputSize, kernelDimension.
	// This could happen too frequently as networks grow larger.
	static Network* combine(std::vector<Network*>& parents, std::vector<float>& rawWeights);


	// Since getOutput returns a float*, application must either use it before any other call to step(),
	// destroyPhenotype(), preTrialReset(), ... or Network destruction, either deep copy the
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
	// when a new node is created, node->nodeID = currentNodeID++;
	int currentMemoryNodeID, currentComplexNodeID;

	// Overkill for small network sizes. To be (re)created after network creation or topological mutation
	// Maps valid complexNodeID s to their complexNode*, FOR PHENOTYPICALLY ACTIVE COMPLEX NODES. i.e. iff
	// node->phenotypicMultiplicity = 0, complexIDmap[node->complexNodeID] = complexIDmap.end().
	std::unordered_map<int, ComplexNode_G*> complexIDmap;

	// Overkill for small network sizes. To be (re)created after network creation or topological mutation
	// Maps valid memoryNodeID s to their memoryNode*, FOR PHENOTYPICALLY ACTIVE MEMORY NODES. i.e. iff
	// node->phenotypicMultiplicity = 0, memoryIDmap[node->memoryNodeID] = memoryIDmap.end().
	std::unordered_map<int, MemoryNode_G*> memoryIDmap;

	// To be called after network creation or mutation. Requires up to date phenotypic multiplicities.
	void createIDMaps();

	std::unique_ptr<ComplexNode_G> topNodeG;

	std::vector<std::unique_ptr<ComplexNode_G>> complexGenome;
	std::vector<std::unique_ptr<MemoryNode_G>> memoryGenome;

	// The phenotype is expected to be created and destroyed only once per Network, but it could happen several times.
	std::unique_ptr<ComplexNode_P> topNodeP;



	// Arrays for plasticity based updates. Contain all presynaptic and postSynaptic activities.
	// Must be : - reset to all 0s at the start of each trial;
	//			 - created alongside ComplexNode_P creation; 
	//           - freed alongside ComplexNode_P deletion.
	// Layout detailed in the Phenotype structs.
	std::unique_ptr<float[]> postSynActs, preSynActs;

#ifdef STDP
	// same size and layout that of preSynActs.
	std::unique_ptr<float[]> accumulatedPreSynActs;
#endif

	// size of postSynActs
	int postSynActArraySize;

	// size of preSynActs
	int preSynActsArraySize;

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

	// WARNING: DOES NOT UPDATE POSITIONS, NOR DEPTHS, NOR MULTIPLICITIES.
	// Use in network.mutate(): it is used to erase some of the non-phenotypically active nodes. However, 
	// this function itself does not require any of the updates above. So it can be called several times in
	// a row, and positions, depths etc can be updated only once at the end.
	void eraseComplexNode(int genomeID);

	// everything is still up to date after a call to this function.
	void eraseMemoryNode(int genomeID);

	// Update the depths of the genome and the top node. Requires every node to have a valid position !
	// i.e. in [0, genome.size() - 1], and no two nodes share the same.
	void updateDepths();

	// Requires the nodes' depths be up to date !
	void sortGenome();
};