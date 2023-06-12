#pragma once

#include <memory>
#include <cmath>
#include <string>
#include <thread>

#include "Network.h"
#include "Trial.h"

#include <fstream>
#include <chrono> // time since 1970

const enum SCORE_BATCH_TRANSFORMATION { NONE = 0, NORMALIZE = 1, RANK = 2};


// Contains the parameters for population evolution. They can be changed at each step.
struct PopulationEvolutionParameters {

	// The importance of the (normalized) regularization factor relative to the score. 0 means no regularization,
	// 1 means regularization is as important as score. Recommended value depends on the task.
	float regularizationFactor;


	// Both values should be < 1.0 , safe value is 0.0 .  
	// ".first " influences the probability of each specimen to be present once in the next generation. 
	// ".second" influences the probability of each specimen to take a spot left empty by a specimen that did not make it
	std::pair<float, float> selectionPressure;


	// If set to true, each trial of the vector passed to the step function will be reset to the same initial values for
	// each specimen. This means that all specimens are evaluated on the exact same tasks.
	bool useSameTrialInit;


	// Experimental, default=0.0. Only used when SATURATION_PENALIZING is defined, as it slows down forward() (a bit).
	// The higher, the stronger the penalty for saturated activations. It may be important to use it when GUIDED_MUTATIONS
	// is defined, as with it networks are prone to oversaturation.
	float saturationFactor;


	// EvaluateFitness() is supposed to receive a vector of fitnesses, 1 value per specimen, more or less normally 
	// distributed (the function handles centering and reducing). However, many trials have no such measure
	// of the fitness : it may be exponential, or discontinuous , or ... In general, it wont be easily interpretable 
	// for generating offsprings with the best probabilities. 
	// And even if it were the  case, the normalizity of the fitness distribution ultimately depends on the 
	// relative performances of the networks. This is why a ranking fitness should be used in the general case, 
	// instead of raw trial scores. When unsure,  use rankingFitness = true.
	bool rankingFitness;


	// Disabled when = 0. Recommended range: [0, .3]. Useless after the maximum score has been reached.
	// Induces a term in the fitness which compares score at this step with score of the parent on the 
	// corresponding trial, at the previous step. (Therefore on a different random initialization.) 
	// Makes sense to use only if trials within a step are semantically different, and the random initialization
	// does not influence the score "too much". See note in Network.postTrialUpdate.
	float competitionFactor;


	// Minimum at 1. If = 1, mutated clone of the parent. No explicit maximum, but bounded by nSpecimens, 
	// and MAX_MATING_DEPTH implicitly. Cost O( n log(n) ).
	int nParents;

	// Usually, raw scores as output from the trials are not very informative about how specimens compare. This
	// parameter specifies a transformation of the scores of the whole population PER TRIAL. It is not redundant 
	// with rankingFitness, it makes sense to have both enabled. 
	SCORE_BATCH_TRANSFORMATION scoreBatchTransformation;

	//defaults:
	PopulationEvolutionParameters() {
		selectionPressure = { -10.0f, 0.0f };
		regularizationFactor = 0.1f;
		useSameTrialInit = false;
		saturationFactor = .05f;
		rankingFitness = true;
		competitionFactor = .1f;
		scoreBatchTransformation = NONE;
		nParents = 10;
	}
};


struct PhylogeneticNode
{
	PhylogeneticNode* parent;
	int networkIndice;
	std::vector<PhylogeneticNode*> children;

	// TODO be careful, it wont work as intended if there are multiple Population . 
	// Should not happen. Set in Population::setEvolutionParameters.
	static int maxListSize;

	PhylogeneticNode() {};

	PhylogeneticNode(PhylogeneticNode* parent, int networkIndice) :
		networkIndice(networkIndice), parent(parent) {};

	bool addToList(std::vector<int>& list, int depth) {
		if (depth == 0) {
			if (list.size() == maxListSize) {
				return false;
			}
			list.push_back(networkIndice);
			return true;
		}
		else {
			for (int i = 0; i < children.size(); i++) {
				if (!children[i]->addToList(list, depth - 1)) {
					return false;
				}
			}
			return true;
		}
	}

	void erase(int depth) {
		if (depth == MAX_MATING_DEPTH) return;

		if (parent->children.size() == 1) {
			parent->children.resize(0);
			parent->erase(depth + 1);
		}
		else {
			int s = (int) parent->children.size() - 1;
			for (int i = 0; i < s; i++) {
				if (parent->children[i] == this) {
					parent->children[i] = parent->children[s];
				}
			}
			parent->children.pop_back();
		}
	}
};

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

	Population(int IN_SIZE, int OUT_SIZE, int nSpecimens, bool fromDLL = false);

	// No requirement on avgScorePerSpecimen, other that a higher score = a better specimen.
	void computeFitnesses(std::vector<float>& avgScorePerSpecimen);

	// nTrials is used in Network.parentData if available.
	void createOffsprings();

	void setEvolutionParameters(PopulationEvolutionParameters params) {
		this->regularizationFactor = params.regularizationFactor;
		this->selectionPressure = params.selectionPressure;
		this->useSameTrialInit = params.useSameTrialInit;
		this->saturationFactor = params.saturationFactor;
		this->rankingFitness = params.rankingFitness;
		this->competitionFactor = params.competitionFactor;
		this->scoreBatchTransformation = params.scoreBatchTransformation;
		this->nParents = params.nParents;

		PhylogeneticNode::maxListSize = params.nParents;
	}

	
	void saveFittestSpecimen() const 
	{
		uint64_t ms = std::chrono::duration_cast<std::chrono::milliseconds>(
			std::chrono::system_clock::now().time_since_epoch()).count();

		std::ofstream os("models\\topNet_" + std::to_string(ms) + "_" + std::to_string(evolutionStep) + ".renon", std::ios::binary);
		networks[fittestSpecimen]->save(os);
	}


	// DLL util only:
	void mutatePopulation() {
		for (int i = 0; i < nSpecimens; i++) {
			networks[i]->mutate();
		}
	};


	// I dont like getters and setters, but it seems to me like a justified use here. Meant to be called from the DLL part.
	int get_nSpecimens() { return nSpecimens; };
	Network* getSpecimenPointer(int i) { return networks[i]; };

	// Indice in the networks list of the fittest specimen at this step.
	int fittestSpecimen;

	// Current number of generations since initialization.
	int evolutionStep;

private:

	bool fromDLL;

	// evaluates Networks[i0 -> i0 + subArraySize]
	void evaluate(const int i0, const int subArraySize, Trial* trial, float* scores);

	//  = PhylogeneticNode[MAX_N_PARENTS][nSpecimens]
	PhylogeneticNode* phylogeneticTree;

	// Finds the secondary parents, computes the coefficients, and creates the interpolated child.
	Network* createChild(PhylogeneticNode* primaryParent);

	// A util.
	int nTrialsAtThisStep;
	
	// Current size of the networks and fitness arrays. Must be a multiple of N_THREADS.
	int nSpecimens;
		
	// Constant between startThreads and stopThreads 		
	int N_THREADS;

	// Size nSpeciemns
	std::vector<Network*> networks;
	
	// The vector of fitness per specimen.
	std::vector<float> fitnesses;

	// The scores of the specimens at this step, as output by the trials.
	std::vector<float> rawScores;

	// Raw scores after a transformation that depends on the whole population's performance. 
	// Like ranking, or normalization.
	std::vector<float> batchTransformedScores;


	// EVOLUTION PARAMETERS: 
	
	// Set with a PopulationEvolutionParameters struct. Description in the struct definition.
	float regularizationFactor, saturationFactor, competitionFactor;

	// Set with a PopulationEvolutionParameters struct. Description in the struct definition.
	std::pair<float, float> selectionPressure;

	// Set with a PopulationEvolutionParameters struct. Description in the struct definition.
	bool useSameTrialInit, rankingFitness;

	// Set with a PopulationEvolutionParameters struct. Description in the struct definition.
	SCORE_BATCH_TRANSFORMATION scoreBatchTransformation;

	// Set with a PopulationEvolutionParameters struct. Description in the struct definition.
	int nParents;


	// THREADING UTILS:
	// used only if N_THREADS > 1 (i.e. multithreading enabled):
	
	// Per thread evolution loop. As of now, handles evaluation and mutation.
	void threadLoop(const int i0, const int subArraySize);

	std::vector<std::thread> threads;
	std::vector<Trial*> globalTrials;
	int threadIteration;
	bool mustTerminate;
};