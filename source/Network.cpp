#pragma once

#include "Network.h"
#include <iostream>

Network::Network(Network* n) {
	inputSize = n->inputSize;
	outputSize = n->outputSize;


	// The complex genome must be copied after the simple and the memory ones, for its pointers to be valid..
	
	// Simple genome
	simpleGenome.resize(n->simpleGenome.size());
	for (int i = 0; i < n->simpleGenome.size(); i++) {
		simpleGenome[i] = std::make_unique<SimpleNode_G>(n->simpleGenome[i].get());
	}

	// Memory genome
	memoryGenome.resize(n->memoryGenome.size());
	for (int i = 0; i < n->memoryGenome.size(); i++) {
		memoryGenome[i] = std::make_unique<MemoryNode_G>(n->memoryGenome[i].get());
	}
	for (int i = 0; i < n->memoryGenome.size(); i++) {
		memoryGenome[i]->closestNode = n->memoryGenome[i]->closestNode == NULL ?
			NULL :
			memoryGenome[n->memoryGenome[i]->closestNode->position].get();
	}

	// Complex genome
	complexGenome.resize(n->complexGenome.size());
	for (int i = 0; i < n->complexGenome.size(); i++) {
		complexGenome[i] = std::make_unique<ComplexNode_G>(n->complexGenome[i].get());
	}
	for (int i = 0; i < n->complexGenome.size(); i++) {
		// Setting pointers to children.
		// It can only be done once the genome has been constructed:

		complexGenome[i]->complexChildren.resize(n->complexGenome[i]->complexChildren.size());
		for (int j = 0; j < n->complexGenome[i]->complexChildren.size(); j++) {
			complexGenome[i]->complexChildren[j] = complexGenome[n->complexGenome[i]->complexChildren[j]->position].get();
		}
		complexGenome[i]->simpleChildren.resize(n->complexGenome[i]->simpleChildren.size());
		for (int j = 0; j < n->complexGenome[i]->simpleChildren.size(); j++) {
			complexGenome[i]->simpleChildren[j] = simpleGenome[n->complexGenome[i]->simpleChildren[j]->position].get();
		}
		complexGenome[i]->memoryChildren.resize(n->complexGenome[i]->memoryChildren.size());
		for (int j = 0; j < n->complexGenome[i]->memoryChildren.size(); j++) {
			complexGenome[i]->memoryChildren[j] = memoryGenome[n->complexGenome[i]->memoryChildren[j]->position].get();
		}
		
		complexGenome[i]->closestNode = n->complexGenome[i]->closestNode == NULL ?
			NULL :
			complexGenome[n->complexGenome[i]->closestNode->position].get();
	}
	

	topNodeG = std::make_unique<ComplexNode_G>(n->topNodeG.get());

	// Setting pointers to children:
	topNodeG->complexChildren.resize(n->topNodeG->complexChildren.size());
	for (int j = 0; j < n->topNodeG->complexChildren.size(); j++) {
		topNodeG->complexChildren[j] = complexGenome[n->topNodeG->complexChildren[j]->position].get();
	}
	topNodeG->simpleChildren.resize(n->topNodeG->simpleChildren.size());
	for (int j = 0; j < n->topNodeG->simpleChildren.size(); j++) {
		topNodeG->simpleChildren[j] = simpleGenome[n->topNodeG->simpleChildren[j]->position].get();
	}
	topNodeG->memoryChildren.resize(n->topNodeG->memoryChildren.size());
	for (int j = 0; j < n->topNodeG->memoryChildren.size(); j++) {
		topNodeG->memoryChildren[j] = memoryGenome[n->topNodeG->memoryChildren[j]->position].get();
	}
	topNodeG->closestNode = NULL;
	
	// The problem this solves is that we call mutate() on a newborn child and not on the parent that has finished
	// its lifetime (it would not make sense otherwise). Thats why some data on the parent's lifetime has to be passed 
	// to its children, when GUIDED_MUTATIONS is defined.
	nInferencesOverLifetime = n->nInferencesOverLifetime;
	nExperiencedTrials = n->nExperiencedTrials;

	topNodeP.reset(NULL);
}


Network::Network(int inputSize, int outputSize) :
	inputSize(inputSize), outputSize(outputSize)
{
	// positions are set at the end.

	simpleGenome.reserve(2);
	simpleGenome.emplace_back(new SimpleNode_G(TANH));
	simpleGenome.emplace_back(new SimpleNode_G(GAUSSIAN));

	complexGenome.resize(0);
	ComplexNode_G* baseComplexNode = new ComplexNode_G();
	baseComplexNode->inputSize = 1;
	baseComplexNode->outputSize = 1;
	baseComplexNode->computeInternalBiasSize();
	baseComplexNode->internalBias.resize(baseComplexNode->internalBiasSize);
	complexGenome.emplace_back(baseComplexNode);



	memoryGenome.resize(0);
	memoryGenome.emplace_back(new MemoryNode_G(inputSize, outputSize, inputSize + outputSize));  

	// Top node
	{
		topNodeG = std::make_unique<ComplexNode_G>();

		topNodeG->inputSize = inputSize;
		topNodeG->outputSize = outputSize;
		topNodeG->position = (int)complexGenome.size();
		topNodeG->closestNode = NULL;
		topNodeG->mutationalDistance = 0;

		topNodeG->complexChildren.resize(0);
		topNodeG->simpleChildren.resize(0);
		topNodeG->memoryChildren.resize(0);

		topNodeG->internalConnexions.resize(0);
		topNodeG->internalConnexions.reserve(4);
		topNodeG->internalConnexions.emplace_back(
			INPUT_NODE, OUTPUT, -1, -1, outputSize, inputSize, GenotypeConnexion::RANDOM
		);
		topNodeG->internalConnexions.emplace_back(
			INPUT_NODE, MODULATION, -1, -1, MODULATION_VECTOR_SIZE, inputSize, GenotypeConnexion::RANDOM
		);

		for (int i = 0; i < MODULATION_VECTOR_SIZE; i++) {
			topNodeG->modulationBias[i] = 0.0f;
		}

		topNodeG->computeInternalBiasSize();

		topNodeG->internalBias.resize(topNodeG->internalBiasSize);
	}



	for (int i = 0; i < simpleGenome.size(); i++) {
		simpleGenome[i]->position = i;
	}
	for (int i = 0; i < complexGenome.size(); i++) {
		complexGenome[i]->position = i;
	}
	for (int i = 0; i < memoryGenome.size(); i++) {
		memoryGenome[i]->position = i;
	}

	topNodeP.reset(NULL);

	computeMemoryUtils();
	updateDepths();
	updatePhenotypicMultiplicities();
}


float* Network::getOutput() {
	return topNodeP->currentPostSynAct + topNodeP->type->inputSize;
}


void Network::postTrialUpdate(float score, int trialID) {

	if (nInferencesOverTrial != 0) {
		
#ifndef CONTINUOUS_LEARNING
		topNodeP->updateWatTrialEnd(1.0f / (float)nInferencesOverTrial); // the argument averages H.
#endif

#if defined GUIDED_MUTATIONS  	
		//float trialLengthFactor = 1.0f / logf((float)nInferencesOverTrial);
		//float trialLengthFactor = powf((float)nInferencesOverTrial, -.5f);
		float trialLengthFactor = 1.0f / (float)nInferencesOverTrial;

		float argument = 5.0f;

		// One and only one of these two options for argument can be uncommented, or 
		// the default value can be kept:

		// 1- 
		// Constant factor between 5 and 100 recommended.
		//argument *= score;			 


		// 2-
		// wellll... the logic behind this is that different trials within a step are fundamentally different, 
		// and trials at the same position in the trials[] list but at different steps (or even for different 
		// specimens) are fundamentally similar but have a different random initialization.
		// So this option is not recommended in the simplest cases, when trials[] is made of copies of a single
		// trial, with different inits.
		if (parentData.isAvailable) {
			if (trialID >= parentData.scoreSize) {
				std::cerr <<
					"ERROR : TRIAL LIST CHANGED BETWEEN STEPS WHEN IT WAS NOT EXPECTED"
				<< std::endl;
			} else {
				argument *= (score - parentData.scores[trialID]);
			}
		}
		 

#ifdef CONTINUOUS_LEARNING
		argument *= trialLengthFactor;  
#endif 


		topNodeP->accumulateW(argument);

#endif //def GUIDED_MUTATIONS
	}
	else {
		std::cerr << "ERROR : postTrialUpdate WAS CALLED BEFORE EVALUATION ON A TRIAL !!" << std::endl;
	}

}


void Network::destroyPhenotype() {
	topNodeP.reset(NULL);
	previousPostSynActs.reset(NULL);
	currentPostSynActs.reset(NULL);
	preSynActs.reset(NULL);
#ifdef SATURATION_PENALIZING
	averageActivation.reset(NULL);
#endif
}

#ifdef GUIDED_MUTATIONS
void Network::zeroAccumulators() {

	for (int i = 0; i < complexGenome.size(); i++) {
		if (complexGenome[i]->phenotypicMultiplicity != 0) {
			for (int j = 0; j < complexGenome[i]->internalConnexions.size(); j++) {
				complexGenome[i]->internalConnexions[j].zeroAccumulator();
			}
		}
	}

	for (int i = 0; i < memoryGenome.size(); i++) {
		if (memoryGenome[i]->phenotypicMultiplicity != 0) {
			memoryGenome[i]->link.zeroAccumulator();
		}
	}

}
#endif

void Network::createPhenotype() {
	if (topNodeP.get() == NULL) {
		topNodeP.reset(new ComplexNode_P(topNodeG.get()));


		std::vector<int> genomeState(complexGenome.size() + 1);
		topNodeG->position = (int)complexGenome.size();

		topNodeG->computeActivationArraySize(genomeState);
		activationArraySize = genomeState[(int)complexGenome.size()];

		previousPostSynActs = std::make_unique<float[]>(activationArraySize);
		currentPostSynActs  = std::make_unique<float[]>(activationArraySize);
		preSynActs          = std::make_unique<float[]>(activationArraySize);

#ifdef SATURATION_PENALIZING
		std::fill(genomeState.begin(), genomeState.end(), 0);
		topNodeG->computeSaturationArraySize(genomeState);
		averageActivationArraySize = genomeState[(int)complexGenome.size()];
		averageActivation = std::make_unique<float[]>(averageActivationArraySize);
		float* temp_averageActivation = averageActivation.get();

		saturationPenalization = 0.0f;
		topNodeP->setglobalSaturationAccumulator(&saturationPenalization);
		std::fill(averageActivation.get(), averageActivation.get() + averageActivationArraySize, 0.0f);
#endif
		
		// The following values will be modified by each node of the phenotype as the pointers are set.
		float* temp_previousPostSynActs = previousPostSynActs.get();
		float* temp_currentPostSynActs = currentPostSynActs.get();
		float* temp_preSynActs = preSynActs.get();
		topNodeP->setArrayPointers(
			&temp_previousPostSynActs,
			&temp_currentPostSynActs,
			&temp_preSynActs
#ifdef SATURATION_PENALIZING
			, &temp_averageActivation
#endif
		);

		nInferencesOverTrial = 0;
		nInferencesOverLifetime = 0;
		nExperiencedTrials = 0;
	}
};


void Network::preTrialReset() {
	nInferencesOverTrial = 0;
	nExperiencedTrials++;
	std::fill(previousPostSynActs.get(), previousPostSynActs.get() + activationArraySize, 0.0f); 
	std::fill(currentPostSynActs.get(), currentPostSynActs.get() + activationArraySize, 0.0f);
	//std::fill(preSynActs.get(), preSynActs.get() + activationArraySize, 0.0f); // is already set to the biases.

	topNodeP->preTrialReset();
};


void Network::step(const std::vector<float>& obs) {
	nInferencesOverLifetime++;
	nInferencesOverTrial++;

	std::copy(obs.begin(), obs.end(), topNodeP->currentPostSynAct);
	for (int i = 0; i < MODULATION_VECTOR_SIZE; i++) {
		topNodeP->totalM[i] = 0.0f;
	}
	topNodeP->forward();
	std::copy(currentPostSynActs.get(), currentPostSynActs.get() + activationArraySize, previousPostSynActs.get());
}


void Network::mutate() {

	// The second constexpr value in each pair should not be neglible when compared to
	// the first, to introduce some kind of spontaneous regularization.

	constexpr float createConnexionProbability = .01f;
	constexpr float deleteConnexionProbability = .002f;

	constexpr float incrementComplexInputSizeProbability = .003f;
	constexpr float decrementComplexInputSizeProbability = .003f;

	constexpr float incrementComplexOutputSizeProbability = .003f;
	constexpr float decrementComplexOutputSizeProbability = .003f;

	constexpr float incrementMemoryInputSizeProbability = .003f;
	constexpr float decrementMemoryInputSizeProbability = .003f;

	constexpr float incrementMemoryOutputSizeProbability = .003f;
	constexpr float decrementMemoryOutputSizeProbability = .003f;

	constexpr float incrementMemoryKernelSizeProbability = .003f;
	constexpr float decrementMemoryKernelSizeProbability = .003f;

	constexpr float addSimpleChildProbability = .01f;
	constexpr float removeSimpleChildProbability = .002f;

	constexpr float addComplexChildProbability = .01f;
	constexpr float removeComplexChildProbability = .002f;

	constexpr float addMemoryChildProbability = .01f;
	constexpr float removeMemoryChildProbability = .002f;

	constexpr float replaceSimpleChildProbability = .05f;
	constexpr float replaceComplexChildProbability = .05f;
	constexpr float replaceMemoryChildProbability = .05f;

	constexpr float duplicateComplexChildProbability = .01f;
	constexpr float duplicateMemoryChildProbability = .01f;

	float r;


	/*
	
	Each kind of mutation has the critical responsability of making sure that, once it is done, the following 
	rules are verified: 

	- The topNode and every node in the genome have up-to-date ->depth.
	- The topNode and every node in the genome have up-to-date ->position.
	- The topNode and every node in the genome have up-to-date ->phenotypicMultiplicity.
	- The complex genome is sorted by ascending depths.
	
	So that were the different types of mutations to be rearranged, deleted, or if other were added, the function 
	would still work properly. Mutations are computationnaly insignificant when compared to inferences so some
	of performance can be sacrificed to that end.
	
	*/

	// The number of mutations is a function of two concepts: the number of parameters in the genotype, and their
	// multiplicity in the phenotype. Empirical studies suggest a negative log-correlation between genome size and
	// mutation rates, i.e. mRate = b - a * log(gSize). TODO
	int adjustedGenomeSize;

	// The effective number of mutations of each kind.
	int nMutations;

	// used in all structural mutations. probability distribution over complex nodes to undergo a certain mutation.
	std::vector<float> probabilities(complexGenome.size() + 1);

	// given a criterion lambda that takes a complex node as input and returns its probablity (unnormalized) as output,
	// populates the "probabilities" array. Returns false if no node was eligible, true otherwise.
	auto computeProbabilities = [this, &probabilities](auto& criterion) {
		float sum = 0.0f;
		for (int i = 0; i < complexGenome.size() + 1; i++) {
			ComplexNode_G* n = i != complexGenome.size() ? complexGenome[i].get() : topNodeG.get();
			probabilities[i] = criterion(n);
			sum += probabilities[i];
		}
		if (sum != 0.0f)
		{
			probabilities[0] = probabilities[0] / sum;
			for (int i = 1; i < complexGenome.size() + 1; i++) {
				probabilities[i] = probabilities[i - 1] + probabilities[i] / sum;
			}
			return true;
		}
		return false;
	};


	// Floating point mutations.
	{
		for (int i = 0; i < complexGenome.size(); i++) {
			if (complexGenome[i]->phenotypicMultiplicity > 0)
			{
				complexGenome[i]->mutateFloats();
			}
		}
		topNodeG->mutateFloats();

		for (int i = 0; i < memoryGenome.size(); i++) {
			if (memoryGenome[i]->phenotypicMultiplicity > 0)
			{
				memoryGenome[i]->mutateFloats();
			}
		}
	}


	// Add or remove connexions in complex genome.
	{
		for (int i = 0; i < complexGenome.size(); i++) {
			r = UNIFORM_01;
			if (r < deleteConnexionProbability) complexGenome[i]->removeConnexion();
			r = UNIFORM_01;
			if (r < createConnexionProbability) complexGenome[i]->addConnexion();
		}
		r = UNIFORM_01;
		if (r < deleteConnexionProbability) topNodeG->removeConnexion();
		r = UNIFORM_01;
		if (r < createConnexionProbability) topNodeG->addConnexion();
	}


	// Complex nodes input/output sizes increase/decrease.
	{
		for (int i = 0; i < complexGenome.size(); i++) {
			if (complexGenome[i]->phenotypicMultiplicity == 0) {
				continue;
			}
			r = UNIFORM_01;
			if (r < incrementComplexInputSizeProbability) {
				bool success = complexGenome[i]->incrementInputSize();
				if (success) {
					for (int j = 0; j < complexGenome.size(); j++) {
						complexGenome[j]->onChildInputSizeIncremented(complexGenome[i]->position, COMPLEX);
					}
					topNodeG->onChildInputSizeIncremented(complexGenome[i]->position, COMPLEX);
				}
			}

			r = UNIFORM_01;
			if (r < incrementComplexOutputSizeProbability) {
				bool success = complexGenome[i]->incrementOutputSize();
				if (success) {
					for (int j = 0; j < complexGenome.size(); j++) {
						complexGenome[j]->onChildOutputSizeIncremented(complexGenome[i]->position, COMPLEX);
					}
					topNodeG->onChildOutputSizeIncremented(complexGenome[i]->position, COMPLEX);
				}
			}

			int rID;
			r = UNIFORM_01;
			if (r < decrementComplexInputSizeProbability) {
				rID = INT_0X(complexGenome[i]->inputSize);
				bool success = complexGenome[i]->decrementInputSize(rID);
				if (success) {
					for (int j = 0; j < complexGenome.size(); j++) {
						complexGenome[j]->onChildInputSizeDecremented(complexGenome[i]->position, COMPLEX, rID);
					}
					topNodeG->onChildInputSizeDecremented(complexGenome[i]->position, COMPLEX, rID);
				}
			}

			r = UNIFORM_01;
			if (r < decrementComplexOutputSizeProbability) {
				rID = INT_0X(complexGenome[i]->outputSize);
				bool success = complexGenome[i]->decrementOutputSize(rID);
				if (success) {
					for (int j = 0; j < complexGenome.size(); j++) {
						complexGenome[j]->onChildOutputSizeDecremented(complexGenome[i]->position, COMPLEX, rID);
					}
					topNodeG->onChildOutputSizeDecremented(complexGenome[i]->position, COMPLEX, rID);
				}
			}
		}
	}

	// Memory nodes input/output/kernel sizes increase/decrease.
	{
		for (int i = 0; i < memoryGenome.size(); i++) {
			if (memoryGenome[i]->phenotypicMultiplicity == 0) {
				continue;
			}
			r = UNIFORM_01;
			if (r < incrementMemoryInputSizeProbability) {
				bool success = memoryGenome[i]->incrementInputSize();
				if (success) {
					for (int j = 0; j < complexGenome.size(); j++) {
						complexGenome[j]->onChildInputSizeIncremented(memoryGenome[i]->position, MEMORY);
					}
					topNodeG->onChildInputSizeIncremented(memoryGenome[i]->position, MEMORY);
				}
			}

			r = UNIFORM_01;
			if (r < incrementMemoryOutputSizeProbability) {
				bool success = memoryGenome[i]->incrementOutputSize();
				if (success) {
					for (int j = 0; j < complexGenome.size(); j++) {
						complexGenome[j]->onChildOutputSizeIncremented(memoryGenome[i]->position, MEMORY);
					}
					topNodeG->onChildOutputSizeIncremented(memoryGenome[i]->position, MEMORY);
				}
			}

			r = UNIFORM_01;
			if (r < decrementMemoryInputSizeProbability) {
				int rID = INT_0X(memoryGenome[i]->inputSize);
				bool success = memoryGenome[i]->decrementInputSize(rID);
				if (success) {
					for (int j = 0; j < complexGenome.size(); j++) {
						complexGenome[j]->onChildInputSizeDecremented(memoryGenome[i]->position, MEMORY, rID);
					}
					topNodeG->onChildInputSizeDecremented(memoryGenome[i]->position, MEMORY, rID);
				}
			}

			r = UNIFORM_01;
			if (r < decrementMemoryOutputSizeProbability) {
				int rID = INT_0X(memoryGenome[i]->outputSize);
				bool success = memoryGenome[i]->decrementOutputSize(rID);
				if (success) {
					for (int j = 0; j < complexGenome.size(); j++) {
						complexGenome[j]->onChildOutputSizeDecremented(memoryGenome[i]->position, MEMORY, rID);
					}
					topNodeG->onChildOutputSizeDecremented(memoryGenome[i]->position, MEMORY, rID);
				}
			}

			r = UNIFORM_01;
			if (r < incrementMemoryKernelSizeProbability) {
				bool success = memoryGenome[i]->incrementKernelDimension();
			}

			r = UNIFORM_01;
			if (r < decrementMemoryKernelSizeProbability) {
				int rID = INT_0X(memoryGenome[i]->kernelDimension);
				bool success = memoryGenome[i]->decrementKernelDimension(rID);
			}
		}
	}


	// Adding child nodes. 
	{
		ComplexNode_G* parent = nullptr;

		// Complex
		SET_BINOMIAL((int)complexGenome.size(), addComplexChildProbability);
		nMutations = BINOMIAL;
		if (nMutations > 0) {
			auto criterion = [](ComplexNode_G* n) {
				float p = (float)n->phenotypicMultiplicity ;

				// The more children a node has, the less likely it is to gain one. Also zeros p when max children reached.
				p *= (float)(MAX_COMPLEX_CHILDREN_PER_COMPLEX - n->complexChildren.size());

				// The shallower a node, the less likely it is to gain a child.
				p *= 1.0f - powf(.9f, (float)n->depth);

				return p;
			};
			
			if (!computeProbabilities(criterion))
			{
				nMutations = 0;
			}
			for (int _unused = 0; _unused < nMutations; _unused++)
			{
				// The fewer children a node already has, and higher its depth, the more likely it is to gain one.
				// "higher its depth" and not "deeper", as nodes closer to the top node are more likely to gain children.
				// (for parallelization bias)

				// choose parent:
				float r = UNIFORM_01;
				int parentID = binarySearch(probabilities, r);
				parent = parentID != complexGenome.size() ? complexGenome[parentID].get() : topNodeG.get();

				// Choose a child at random among all nodes that dont have the parent as direct or indirect parent.
				// (to avoid infinite recursion)

				// -1 if parent is not a direct or indirect child, 1 if it is, 0 if not known yet.
				std::vector<int> hasParentAsChild(complexGenome.size());

				for (int i = 0; i < complexGenome.size(); i++) {
					hasParentAsChild[i] = -1;
					for (int j = 0; j < complexGenome[i]->complexChildren.size(); j++) {
						if (complexGenome[i]->complexChildren[j] == parent) {
							hasParentAsChild[i] = 1;
							break;
						}
						if (hasParentAsChild[complexGenome[i]->complexChildren[j]->position] == 1) {
							hasParentAsChild[i] = 1;
							break;
						}
					}
				}
				if (parent->position != complexGenome.size()) { hasParentAsChild[parent->position] = 1; }

				int nPotentialChildren = 0;
				for (int i = 0; i < complexGenome.size(); i++) {
					if (hasParentAsChild[i] == -1) {
						nPotentialChildren++;
					}
				}

				if (nPotentialChildren == 0) {
					continue;
				}

				ComplexNode_G* child = nullptr;
				int childID = INT_0X(nPotentialChildren);
				int id = -1;
				for (int i = 0; i < complexGenome.size(); i++) {
					if (hasParentAsChild[i] == -1) {
						id++;
						if (id == childID) {
							child = complexGenome[i].get();
							break;
						}
					}
				}

				parent->addComplexChild(child);

				updatePhenotypicMultiplicities();
				updateDepths();
				sortGenome();
			}
		}
		

		// Simple
		SET_BINOMIAL((int)complexGenome.size(), addSimpleChildProbability);
		nMutations = BINOMIAL;
		if (nMutations > 0) {
			auto criterion = [](ComplexNode_G* n) {
				float p = (float)n->phenotypicMultiplicity * (n->simpleChildren.size() < MAX_SIMPLE_CHILDREN_PER_COMPLEX);

				// The more children a node has, the less likely it is to gain one.
				p *= (float)(2 * MAX_SIMPLE_CHILDREN_PER_COMPLEX - n->simpleChildren.size());

				// The shallower a node, the less likely it is to gain a child.
				p *= 1.0f - powf(.9f, (float)n->depth);

				return p;
			};

			if (!computeProbabilities(criterion))
			{
				nMutations = 0;
			}

			for (int _unused = 0; _unused < nMutations; _unused++)
			{
				// The fewer children a node already has, and higher its depth, the more likely it is to gain one.
				// "higher its depth" and not "deeper", as nodes closer to the top node are more likely to gain children.
				// (for parallelization bias)

				// choose parent:
				float r = UNIFORM_01;
				int parentID = binarySearch(probabilities, r);
				parent = parentID != complexGenome.size() ? complexGenome[parentID].get() : topNodeG.get();

				// child is simply a random simple neuron
				SimpleNode_G* child = simpleGenome[INT_0X((int)simpleGenome.size())].get();

				parent->addSimpleChild(child);
				updateDepths();
				sortGenome();

			}
		}


		// Memory
		SET_BINOMIAL((int)complexGenome.size(), addMemoryChildProbability);
		nMutations = BINOMIAL;
		if (nMutations > 0 && memoryGenome.size() > 0) {
			auto criterion = [](ComplexNode_G* n) {
				float p = (float)n->phenotypicMultiplicity * (n->memoryChildren.size() < MAX_MEMORY_CHILDREN_PER_COMPLEX);

				// The more children a node has, the less likely it is to gain one.
				p *= (float)(2 * MAX_MEMORY_CHILDREN_PER_COMPLEX - n->memoryChildren.size());

				// The shallower a node, the less likely it is to gain a child.
				p *= 1.0f - powf(.9f, (float)n->depth);

				return p;
			};

			if (!computeProbabilities(criterion))
			{
				nMutations = 0;
			}

			for (int _unused = 0; _unused < nMutations; _unused++)
			{
				// The fewer children a node already has, and higher its depth, the more likely it is to gain one.
				// "higher its depth" and not "deeper", as nodes closer to the top node are more likely to gain children.
				// (for parallelization bias)

				// choose parent:
				float r = UNIFORM_01;
				int parentID = binarySearch(probabilities, r);
				parent = parentID != complexGenome.size() ? complexGenome[parentID].get() : topNodeG.get();

				MemoryNode_G* child = memoryGenome[INT_0X((int)memoryGenome.size())].get();					

				parent->addMemoryChild(child);
				child->phenotypicMultiplicity += parent->phenotypicMultiplicity;

			}
		}
	}
	

	// Replacing child nodes.
	{
		ComplexNode_G* parent = nullptr;

		// Complex
		SET_BINOMIAL((int)complexGenome.size(), replaceComplexChildProbability);
		nMutations = BINOMIAL;
		if (nMutations > 0) {
			auto criterion = [](ComplexNode_G* n) {
				float p = (float)n->phenotypicMultiplicity * n->complexChildren.size();
				return p;
			};

			if (!computeProbabilities(criterion))
			{
				nMutations = 0;
			}
			for (int _unused = 0; _unused < nMutations; _unused++)
			{

				// choose parent:
				float r = UNIFORM_01;
				int parentID = binarySearch(probabilities, r);
				parent = parentID != complexGenome.size() ? complexGenome[parentID].get() : topNodeG.get();

				// pick a child:
				int inParentID = INT_0X((int)parent->complexChildren.size());
				ComplexNode_G* toBeReplacedChild = parent->complexChildren[inParentID];

				// Determine candidates to replacement. A valid candidate is a node that:
				//		- either has toBeReplacedChild as its closestNode or, which is toBeReplacedChild's closestNode.
				//      - does not have toBeReplacedChild among its direct or indirect children

				// isPotentialReplacement: 1 if the node at position i in the complex genome does not have toBeReplacedChild 
				// among its direct or indirect children, -1 if it has, 0 if not known yet.
				std::vector<int> isPotentialReplacement(complexGenome.size());
				for (int i = 0; i < complexGenome.size(); i++) {
					isPotentialReplacement[i] = 1;
					for (int j = 0; j < complexGenome[i]->complexChildren.size(); j++) {
						if (complexGenome[i]->complexChildren[j] == toBeReplacedChild) {
							isPotentialReplacement[i] = -1;
							break;
						}
						if (isPotentialReplacement[complexGenome[i]->complexChildren[j]->position] == -1) {
							isPotentialReplacement[i] = -1;
							break;
						}
					}
				}
					
				float sumInvDistances = 0.0f;
				std::vector<ComplexNode_G*> candidates;
				std::vector<float> invDistances;
				if (toBeReplacedChild->closestNode != NULL && 
					isPotentialReplacement[toBeReplacedChild->closestNode->position] == 1) {

					candidates.push_back(toBeReplacedChild->closestNode);
					float invDistance = 1.0f / (10.0f + (float)toBeReplacedChild->mutationalDistance);
					invDistances.push_back(invDistance);
					sumInvDistances += invDistance;
				}
				for (int j = 0; j < complexGenome.size(); j++) {
					if (complexGenome[j]->closestNode == toBeReplacedChild && 
						isPotentialReplacement[j] == 1) {

						candidates.push_back(complexGenome[j].get());
						float invDistance = 1.0f / (10.0f + (float)complexGenome[j]->mutationalDistance);
						invDistances.push_back(invDistance);
						sumInvDistances += invDistance;
					}
				}

				// replacement is happening.
				if (candidates.size() != 0) {

					std::vector<float> probabilities(invDistances.size());
					probabilities[0] = invDistances[0] / sumInvDistances;
					for (int i = 1; i < invDistances.size(); i++) {
						probabilities[i] = invDistances[i - 1] + invDistances[i] / sumInvDistances;
					}
					float r = UNIFORM_01;
					int rID = binarySearch(probabilities, r);

					parent->complexChildren[inParentID] = candidates[rID];

					// adjust connexion sizes. TODO better.

					for (int i = 0; i < parent->internalConnexions.size(); i++) {
						if (parent->internalConnexions[i].destinationType == COMPLEX &&
							parent->internalConnexions[i].destinationID == inParentID && 
							candidates[rID]->inputSize != toBeReplacedChild->inputSize)
						{
							GenotypeConnexion gc(
								parent->internalConnexions[i].originType,
								parent->internalConnexions[i].destinationType,
								parent->internalConnexions[i].originID,
								parent->internalConnexions[i].destinationID,
								candidates[rID]->inputSize,
								parent->internalConnexions[i].nColumns,
								GenotypeConnexion::RANDOM
							);
							parent->internalConnexions[i] = gc;
						}
						if (parent->internalConnexions[i].originType == COMPLEX &&
							parent->internalConnexions[i].originID == inParentID &&
							candidates[rID]->outputSize != toBeReplacedChild->outputSize)
						{
							GenotypeConnexion gc(
								parent->internalConnexions[i].originType,
								parent->internalConnexions[i].destinationType,
								parent->internalConnexions[i].originID,
								parent->internalConnexions[i].destinationID,
								parent->internalConnexions[i].nLines,
								candidates[rID]->outputSize,
								GenotypeConnexion::RANDOM
							);
							parent->internalConnexions[i] = gc;
						}
					}
				
					


					// one could lower the mutational distance between the swapped children. But this is a recursive
					// mutation, and I have never managed to make those improve results.
					updatePhenotypicMultiplicities();
					updateDepths();
					sortGenome();
				}

			}

		}


		// Simple
		SET_BINOMIAL((int)complexGenome.size(), replaceSimpleChildProbability);
		nMutations = BINOMIAL;
		if (nMutations > 0) {
			auto criterion = [](ComplexNode_G* n) {
				float p = (float)n->phenotypicMultiplicity * n->simpleChildren.size();
				return p;
			};

			if (!computeProbabilities(criterion))
			{
				nMutations = 0;
			}
			for (int _unused = 0; _unused < nMutations; _unused++)
			{

				// choose parent:
				float r = UNIFORM_01;
				int parentID = binarySearch(probabilities, r);
				parent = parentID != complexGenome.size() ? complexGenome[parentID].get() : topNodeG.get();

				// pick a child:
				int inParentID = INT_0X((int)parent->simpleChildren.size());
				SimpleNode_G* toBeReplacedChild = parent->simpleChildren[inParentID];


				parent->simpleChildren[inParentID] = simpleGenome[INT_0X((int)simpleGenome.size())].get();

			}

		}


		// Memory
		SET_BINOMIAL((int)complexGenome.size(), replaceMemoryChildProbability);
		nMutations = BINOMIAL;
		if (nMutations > 0) {
			auto criterion = [](ComplexNode_G* n) {
				float p = (float)n->phenotypicMultiplicity * n->memoryChildren.size();
				return p;
			};

			if (!computeProbabilities(criterion))
			{
				nMutations = 0;
			}
			for (int _unused = 0; _unused < nMutations; _unused++)
			{

				// choose parent:
				float r = UNIFORM_01;
				int parentID = binarySearch(probabilities, r);
				parent = parentID != complexGenome.size() ? complexGenome[parentID].get() : topNodeG.get();

				// pick a child:
				int inParentID = INT_0X((int)parent->memoryChildren.size());
				MemoryNode_G* toBeReplacedChild = memoryGenome[INT_0X((int)memoryGenome.size())].get();


				// determine candidates to replacement
				float sumInvDistances = 0.0f;
				std::vector<MemoryNode_G*> candidates;
				std::vector<float> invDistances;
				if (toBeReplacedChild->closestNode != NULL) {

					candidates.push_back(toBeReplacedChild->closestNode);
					float invDistance = 1.0f / (10.0f + (float)toBeReplacedChild->mutationalDistance);
					invDistances.push_back(invDistance);
					sumInvDistances += invDistance;
				}
				for (int j = 0; j < memoryGenome.size(); j++) {
					if (memoryGenome[j]->closestNode == toBeReplacedChild) {

						candidates.push_back(memoryGenome[j].get());
						float invDistance = 1.0f / (10.0f + (float)memoryGenome[j]->mutationalDistance);
						invDistances.push_back(invDistance);
						sumInvDistances += invDistance;
					}
				}

				// replacement is happening.
				if (candidates.size() != 0) {

					std::vector<float> probabilities(invDistances.size());
					probabilities[0] = invDistances[0] / sumInvDistances;
					for (int i = 1; i < invDistances.size(); i++) {
						probabilities[i] = invDistances[i - 1] + invDistances[i] / sumInvDistances;
					}
					float r = UNIFORM_01;
					int rID = binarySearch(probabilities, r);

					parent->memoryChildren[inParentID]->phenotypicMultiplicity -= parent->phenotypicMultiplicity;
					candidates[rID]->phenotypicMultiplicity += parent->phenotypicMultiplicity;
					parent->memoryChildren[inParentID] = candidates[rID];


					// adjust connexion sizes. TODO better.

					for (int i = 0; i < parent->internalConnexions.size(); i++) {
						if (parent->internalConnexions[i].destinationType == MEMORY &&
							parent->internalConnexions[i].destinationID == inParentID &&
							candidates[rID]->inputSize != toBeReplacedChild->inputSize)
						{
							GenotypeConnexion gc(
								parent->internalConnexions[i].originType,
								parent->internalConnexions[i].destinationType,
								parent->internalConnexions[i].originID,
								parent->internalConnexions[i].destinationID,
								candidates[rID]->inputSize,
								parent->internalConnexions[i].nColumns,
								GenotypeConnexion::RANDOM
							);
							parent->internalConnexions[i] = gc;
						}
						if (parent->internalConnexions[i].originType == MEMORY &&
							parent->internalConnexions[i].originID == inParentID &&
							candidates[rID]->outputSize != toBeReplacedChild->outputSize)
						{
							GenotypeConnexion gc(
								parent->internalConnexions[i].originType,
								parent->internalConnexions[i].destinationType,
								parent->internalConnexions[i].originID,
								parent->internalConnexions[i].destinationID,
								parent->internalConnexions[i].nLines,
								candidates[rID]->outputSize,
								GenotypeConnexion::RANDOM
							);
							parent->internalConnexions[i] = gc;
						}
					}

					// one could lower the mutational distance between the swapped children. But this is a recursive
					// mutation, and I have never managed to make those improve results.

				}

			}

		}
	}


	// Removing child nodes. 
	{
		ComplexNode_G* parent = nullptr;

		// Complex
		SET_BINOMIAL((int)complexGenome.size(), removeComplexChildProbability);
		nMutations = BINOMIAL;
		if (nMutations > 0) {
			auto criterion = [](ComplexNode_G* n) {
				float p = (float)n->phenotypicMultiplicity * n->complexChildren.size();
				return p;
			};

			if (!computeProbabilities(criterion))
			{
				nMutations = 0;
			}
			for (int _unused = 0; _unused < nMutations; _unused++)
			{
				// choose parent:
				float r = UNIFORM_01;
				int parentID = binarySearch(probabilities, r);
				parent = parentID != complexGenome.size() ? complexGenome[parentID].get() : topNodeG.get();

				// pick a child:
				int inParentID = INT_0X((int)parent->complexChildren.size());
				parent->removeComplexChild(inParentID);

				updatePhenotypicMultiplicities();
				updateDepths();
				sortGenome();
			}

		}


		// Simple
		SET_BINOMIAL((int)complexGenome.size(), removeSimpleChildProbability);
		nMutations = BINOMIAL;
		if (nMutations > 0) {
			auto criterion = [](ComplexNode_G* n) {
				float p = (float)n->phenotypicMultiplicity * n->simpleChildren.size();
				return p;
			};

			if (!computeProbabilities(criterion))
			{
				nMutations = 0;
			}
			for (int _unused = 0; _unused < nMutations; _unused++)
			{
				// choose parent:
				float r = UNIFORM_01;
				int parentID = binarySearch(probabilities, r);
				parent = parentID != complexGenome.size() ? complexGenome[parentID].get() : topNodeG.get();

				// pick a child:
				int inParentID = INT_0X((int)parent->simpleChildren.size());
				parent->removeSimpleChild(inParentID);
			}

		}


		// Memory
		SET_BINOMIAL((int)complexGenome.size(), removeMemoryChildProbability);
		nMutations = BINOMIAL;
		if (nMutations > 0) {
			auto criterion = [](ComplexNode_G* n) {
				float p = (float)n->phenotypicMultiplicity * n->memoryChildren.size();
				return p;
			};

			if (!computeProbabilities(criterion))
			{
				nMutations = 0;
			}
			for (int _unused = 0; _unused < nMutations; _unused++)
			{
				// choose parent:
				float r = UNIFORM_01;
				int parentID = binarySearch(probabilities, r);
				parent = parentID != complexGenome.size() ? complexGenome[parentID].get() : topNodeG.get();

				// pick a child:
				int inParentID = INT_0X((int)parent->memoryChildren.size());
				parent->memoryChildren[inParentID]->phenotypicMultiplicity -= parent->phenotypicMultiplicity;
				parent->removeMemoryChild(inParentID);
			}

		}
	}
	
	// Duplicating child nodes.
	{
		// Chooses a node in the genotype, clones it, and replaces it with the clone in one of the nodes
		// that has it as a child. The shallower and the more common the node, the more likely it is
		// to be selected.

		ComplexNode_G* parent = nullptr;

		// Complex
		SET_BINOMIAL((int)complexGenome.size(), duplicateComplexChildProbability);
		nMutations = BINOMIAL;
		if (nMutations > 0) {
			auto criterion = [](ComplexNode_G* n) {
				float p = (float)n->phenotypicMultiplicity * n->complexChildren.size();
				p *= 1.0f - powf(.8f, (float)n->depth);
				return p;
			};

			if (!computeProbabilities(criterion))
			{
				nMutations = 0;
			}
			for (int _unused = 0; _unused < nMutations; _unused++)
			{
				// choose parent:
				float r = UNIFORM_01;
				int parentID = binarySearch(probabilities, r);
				parent = parentID != complexGenome.size() ? complexGenome[parentID].get() : topNodeG.get();
				if (parent->complexChildren.size() == 0) {
					break; // can be removed, i think. problem was probabilities was not updated.
				}
				// pick a child
				int inParentID = INT_0X((int)parent->complexChildren.size());
				ComplexNode_G* clonedNode = parent->complexChildren[inParentID];


				ComplexNode_G* n = new ComplexNode_G(clonedNode);

				n->closestNode = clonedNode;
				n->mutationalDistance = 0;
				
				complexGenome.emplace(complexGenome.begin() + clonedNode->position + 1, n);

				parent->complexChildren[inParentID] = n;

				for (int j = 0; j < complexGenome.size(); j++) {
					complexGenome[j]->position = j;
				}
				topNodeG->position = (int)complexGenome.size();

				updatePhenotypicMultiplicities();

				// probability have to be re-computed since a node was added to the complex genome.
				probabilities.resize(complexGenome.size()+1);
				if (!computeProbabilities(criterion))
				{
					break;
				}

				// Not needed:
				//updateDepths();
				//sortGenome();
			}

		}

		// Memory
		SET_BINOMIAL((int)complexGenome.size(), duplicateMemoryChildProbability);
		nMutations = BINOMIAL;
		if (nMutations > 0) {
			auto criterion = [](ComplexNode_G* n) {
				float p = (float)n->phenotypicMultiplicity * n->memoryChildren.size();
				p *= 1.0f - powf(.8f, (float)n->depth);
				return p;
			};

			if (!computeProbabilities(criterion))
			{
				nMutations = 0;
			}
			for (int _unused = 0; _unused < nMutations; _unused++)
			{
				// choose parent:
				float r = UNIFORM_01;
				int parentID = binarySearch(probabilities, r);
				parent = parentID != complexGenome.size() ? complexGenome[parentID].get() : topNodeG.get();

				// pick a child
				int inParentID = INT_0X((int)parent->memoryChildren.size());
				MemoryNode_G* clonedNode = parent->memoryChildren[inParentID];

				MemoryNode_G* n = new MemoryNode_G(clonedNode);

				n->closestNode = clonedNode;
				n->mutationalDistance = 0;

				memoryGenome.emplace_back(n);
				n->position = (int)memoryGenome.size() - 1;

				parent->memoryChildren[inParentID] = n;
				n->phenotypicMultiplicity = parent->phenotypicMultiplicity;
				clonedNode->phenotypicMultiplicity -= parent->phenotypicMultiplicity;
			}

		}
	}

	// the order of the following calls must be respected.
	
	// these 3 are unnecessary:
	/*
	updateDepths();
	sortGenome();
	updatePhenotypicMultiplicities(); 
	*/
	

	// Removing unused nodes. Find a better solution, TODO 
	if (UNIFORM_01 < .03f) {
		removeUnusedNodes(); 
	}


	// Update mutational distances.
	for (int i = 0; i < complexGenome.size(); i++) {
		if (complexGenome[i]->phenotypicMultiplicity > 0) {
			complexGenome[i]->mutationalDistance++;
		}
		complexGenome[i]->computeInternalBiasSize();
	}
	topNodeG->computeInternalBiasSize();
	for (int i = 0; i < memoryGenome.size(); i++) {
		if (memoryGenome[i]->phenotypicMultiplicity > 0) {
			memoryGenome[i]->mutationalDistance++;
		}
	}

	computeMemoryUtils(); // ready memory nodes for inference
	
	// Phenotype is destroyed, as it may have become outdated. It will have to be recreated
	// before next inference.
	topNodeP.reset(NULL);
}


void Network::updateDepths() {
	std::vector<int> genomeState(complexGenome.size()+1);

	topNodeG->updateDepth(genomeState);
	for (int i = (int)complexGenome.size() - 1; i >= 0; i--) {
		if(genomeState[i] == 0) complexGenome[i]->updateDepth(genomeState);
	}
}


void Network::sortGenome() {

	std::vector<std::pair<int, int>> depthXposition;
	int size = (int)complexGenome.size();
	depthXposition.resize(size);

	for (int i = 0; i < size; i++) {
		depthXposition[i] = std::make_pair(complexGenome[i]->depth, i);
	}

	std::sort(depthXposition.begin(), depthXposition.end(), 
		[](std::pair<int, int> a, std::pair<int, int> b)
		{
			return a.first < b.first;  //ascending order
		});

	std::vector<ComplexNode_G*> tempStorage(size); 
	for (int i = 0; i < size; i++) {
		tempStorage[i] = complexGenome[i].release();
	}

	for (int i = 0; i < size; i++) {
		complexGenome[i].reset(tempStorage[depthXposition[i].second]);
	}

	for (int i = 0; i < complexGenome.size(); i++) {
		complexGenome[i]->position = i;
	}
	return;
}


void Network::updatePhenotypicMultiplicities() {
	topNodeG->phenotypicMultiplicity = 1;
	std::vector<int> multiplicities(complexGenome.size());

	// Init
	for (int i = 0; i < memoryGenome.size(); i++) {
		memoryGenome[i]->phenotypicMultiplicity = 0;
	}
	for (int i = 0; i < topNodeG->complexChildren.size(); i++) {
		multiplicities[topNodeG->complexChildren[i]->position]++;
	}
	for (int i = 0; i < topNodeG->memoryChildren.size(); i++) {
		topNodeG->memoryChildren[i]->phenotypicMultiplicity++;
	}

	// Main loop
	for (int i = (int)complexGenome.size() - 1; i >= 0; i--) { 
		for (int j = 0; j < complexGenome[i]->complexChildren.size(); j++) {
			multiplicities[complexGenome[i]->complexChildren[j]->position] += multiplicities[complexGenome[i]->position];
		}
		for (int j = 0; j < complexGenome[i]->memoryChildren.size(); j++) {
			complexGenome[i]->memoryChildren[j]->phenotypicMultiplicity += multiplicities[complexGenome[i]->position];
		}
	}

	// Update
	for (int i = 0; i < complexGenome.size(); i++) {
		complexGenome[i]->phenotypicMultiplicity = multiplicities[i];
	}
}


// requires the genome be sorted by ascending (and up to date) depths, positions and multiplicities be correct.
void Network::removeUnusedNodes() {
	std::vector<int> occurences(complexGenome.size());
	for (int i = 0; i < (int)topNodeG->complexChildren.size(); i++) {
		occurences[topNodeG->complexChildren[i]->position]++;
	}

	for (int i = (int)complexGenome.size() - 1; i >= 0; i--) {

		if (complexGenome[i]->phenotypicMultiplicity == 0) { 

			// firstly, handle the replacement pointers. Could do better, TODO .
			if (complexGenome[i]->closestNode != NULL)
			{
				for (int j = 0; j < complexGenome.size(); j++) {
					if (complexGenome[j]->closestNode == complexGenome[i].get()) {
						complexGenome[j]->closestNode = complexGenome[i]->closestNode;
						complexGenome[j]->mutationalDistance += complexGenome[i]->mutationalDistance;
					}
				}
			}
			else {
				int dMin = 1000000000; 
				ComplexNode_G* newRoot = nullptr;
				for (int j = 0; j < complexGenome.size(); j++) {
					if (complexGenome[j]->closestNode == complexGenome[i].get()) {
						if (complexGenome[j]->mutationalDistance < dMin) {
							dMin = complexGenome[j]->mutationalDistance;
							newRoot = complexGenome[j].get();
						}
					}
				}
				if (newRoot != nullptr) {
					newRoot->closestNode = NULL;
					newRoot->mutationalDistance = 0;
					for (int j = 0; j < complexGenome.size(); j++) {
						if (complexGenome[j]->closestNode == complexGenome[i].get()) {
							complexGenome[j]->mutationalDistance += dMin;
							complexGenome[j]->closestNode = newRoot;
						}
					}
				}
			}

			// erase it from the genome list.
			complexGenome.erase(complexGenome.begin() + i);
			continue;
		}
	}

	for (int i = 0; i < complexGenome.size(); i++) complexGenome[i]->position = i;
	topNodeG->position = (int)complexGenome.size();

	for (int i = (int)memoryGenome.size()-1; i >= 0; i--) {

		if (memoryGenome[i]->phenotypicMultiplicity == 0) {

			// firstly, handle the replacement pointers. Could do better, TODO .
			if (memoryGenome[i]->closestNode != NULL)
			{
				for (int j = 0; j < memoryGenome.size(); j++) {
					if (memoryGenome[j]->closestNode == memoryGenome[i].get()) {
						memoryGenome[j]->closestNode = memoryGenome[i]->closestNode;
						memoryGenome[j]->mutationalDistance += memoryGenome[i]->mutationalDistance;
					}
				}
			}
			else {
				int dMin = 1000000000;
				MemoryNode_G* newRoot = nullptr;
				for (int j = 0; j < memoryGenome.size(); j++) {
					if (memoryGenome[j]->closestNode == memoryGenome[i].get()) {
						if (memoryGenome[j]->mutationalDistance < dMin) {
							dMin = memoryGenome[j]->mutationalDistance;
							newRoot = memoryGenome[j].get();
						}
					}
				}
				if (newRoot != nullptr) {
					newRoot->closestNode = NULL;
					newRoot->mutationalDistance = 0;
					for (int j = 0; j < memoryGenome.size(); j++) {
						if (memoryGenome[j]->closestNode == memoryGenome[i].get()) {
							memoryGenome[j]->mutationalDistance += dMin;
							memoryGenome[j]->closestNode = newRoot;
						}
					}
				}
			}

			// erase it from the genome list.
			memoryGenome.erase(memoryGenome.begin() + i);

			continue;
		}
	}
	for (int i = 0; i < memoryGenome.size(); i++) memoryGenome[i]->position = i;
}


#ifdef SATURATION_PENALIZING
float Network::getSaturationPenalization()
{
	if (nInferencesOverLifetime == 0) {
		std::cerr <<
			"ERROR : getSaturationPenalization() WAS CALLED, BUT THE PHENOTYPE HAS NEVER BEEN USED BEFORE !"
			<< std::endl;
		return 0.0f;
	}

	
	float p1 = averageActivationArraySize != 0 ? saturationPenalization / (nInferencesOverLifetime * averageActivationArraySize) : 0.0f;


	float p2 = 0.0f;
	float invNInferencesN = 1.0f / nInferencesOverLifetime;
	for (int i = 0; i < averageActivationArraySize; i++) {
		p2 += powf(averageActivation[i] * invNInferencesN, 6.0f);
	}
	p2 /= (float) averageActivationArraySize;
	


	constexpr float µ = .5f;
	return µ * p1 + (1 - µ) * p2;
}
#endif


// TODO take genome.size() and maybe internalBias into consideration.
float Network::getRegularizationLoss() {
	constexpr int nArrays = 6; // eta and gamma's amplitudes are irrelevant here.

	// key and query matrix not considered for now.
	std::vector<int> nMemoryParams(memoryGenome.size() + 1);
	std::vector<float> memoryAmplitudes(memoryGenome.size() + 1);
	for (int i = 0; i < memoryGenome.size(); i++) {
		int size = memoryGenome[i]->link.nLines * memoryGenome[i]->link.nColumns;
		nMemoryParams[i] = size;
		memoryAmplitudes[i] = 0.0f;
		for (int k = 0; k < size; k++) {
			memoryAmplitudes[i] += abs(memoryGenome[i]->link.A[k]);
			memoryAmplitudes[i] += abs(memoryGenome[i]->link.B[k]);
			memoryAmplitudes[i] += abs(memoryGenome[i]->link.C[k]);
			memoryAmplitudes[i] += abs(memoryGenome[i]->link.D[k]);
			memoryAmplitudes[i] += abs(memoryGenome[i]->link.alpha[k]);
			memoryAmplitudes[i] += abs(memoryGenome[i]->link.w[k]);
		}
	}

	std::vector<int> nParams(complexGenome.size() + 1);
	std::vector<float> amplitudes(complexGenome.size() + 1);
	for (int i = 0; i < complexGenome.size() + 1; i++) {
		ComplexNode_G* n;
		n = i != complexGenome.size() ? complexGenome[i].get() : topNodeG.get();

		nParams[i] = 0;
		amplitudes[i] = 0.0f;
		for (int j = 0; j < n->internalConnexions.size(); j++) {
			int size = n->internalConnexions[j].nLines * n->internalConnexions[j].nColumns;
			nParams[i] += size;
			for (int k = 0; k < size; k++) {
				amplitudes[i] += abs(n->internalConnexions[j].A[k]);
				amplitudes[i] += abs(n->internalConnexions[j].B[k]);
				amplitudes[i] += abs(n->internalConnexions[j].C[k]);
				amplitudes[i] += abs(n->internalConnexions[j].D[k]); 
				amplitudes[i] += abs(n->internalConnexions[j].alpha[k]);
				amplitudes[i] += abs(n->internalConnexions[j].w[k]);
			}
		}
		for (int j = 0; j < n->complexChildren.size(); j++) {
			nParams[i] += nParams[n->complexChildren[j]->position];
			amplitudes[i] += amplitudes[n->complexChildren[j]->position];
		}
		for (int j = 0; j < n->memoryChildren.size(); j++) {
			nParams[i] += nMemoryParams[n->memoryChildren[j]->position];
			amplitudes[i] += memoryAmplitudes[n->memoryChildren[j]->position];
		}
	}

	float a = (float)amplitudes[complexGenome.size()] / (float)(nParams[complexGenome.size()]*nArrays); // amplitude term
	float b = logf(inputSize + outputSize + (float)nParams[complexGenome.size()]);						// size term
	return 0.0f*a + .5f * a * b + .5f*b; // normalized, so multiplying by a constant does nothing.
}
