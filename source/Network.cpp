#pragma once

#include "Network.h"
#include <iostream>

Network::Network(Network* n) {
	inputSize = n->inputSize;
	outputSize = n->outputSize;


	// The complex genome must be copied after the memory one, for its pointers to be valid..
	

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

	complexGenome.resize(0);

	ComplexNode_G* baseComplexNode = new ComplexNode_G(1, 1);
	complexGenome.emplace_back(baseComplexNode);


	memoryGenome.resize(0);
	memoryGenome.emplace_back(new MemoryNode_G(inputSize, outputSize, inputSize + outputSize));  

	
	topNodeG = std::make_unique<ComplexNode_G>(inputSize, outputSize);
	topNodeG->phenotypicMultiplicity = 1;
	topNodeG->complexChildren.resize(0);
	topNodeG->memoryChildren.resize(0);
		
	// enabled for memory tests.
	if (false)
	{
		topNodeG->memoryChildren.resize(1);
		topNodeG->memoryChildren[0] = memoryGenome[0].get();
	}


	for (int i = 0; i < complexGenome.size(); i++) {
		complexGenome[i]->position = i;
		complexGenome[i]->computeBiasSizes();
		complexGenome[i]->createInternalConnexions();
		complexGenome[i]->computeMemoryPreSynOffset();
	}
	topNodeG->position = (int) complexGenome.size();
	topNodeG->computeBiasSizes();
	topNodeG->createInternalConnexions();
	topNodeG->computeMemoryPreSynOffset();


	for (int i = 0; i < memoryGenome.size(); i++) {
		memoryGenome[i]->position = i;
	}

	topNodeP.reset(NULL);

	updateDepths();
	updatePhenotypicMultiplicities();
}


float* Network::getOutput() {
	return topNodeP->preSynAct;
}

// TODO parametrize whether to use rawScores or batchTransformedScores.
void Network::postTrialUpdate(float score, int trialID) {

	if (nInferencesOverTrial != 0) {
		
#ifndef CONTINUOUS_LEARNING
		topNodeP->updateWatTrialEnd(1.0f / (float)nInferencesOverTrial); // the argument averages H.
#endif

#if defined GUIDED_MUTATIONS  	

		float argument = 1.0f;

		// One and only one of these two options for "argument" can be uncommented, or 
		// the default constant value can be used:

		// 1- 
		// The batch transformed score mesures the success relative to others, and hopefully how right the network was in its learning.
		//argument *= score;			 


		// 2-
		// Assumes negligible stochasticity. Can work with either of batch transformed and raw scores.
		if (parentData.isAvailable) {
			if (trialID >= parentData.scoreSize) {
				std::cerr <<
					"ERROR : TRIAL LIST CHANGED BETWEEN STEPS WHEN IT WAS NOT EXPECTED" 
					// and as of yet if this is enabled it is never expected to.
				<< std::endl;
			} else {
				argument *= (score - parentData.scores[trialID]);
			}
		}
		 

#ifdef CONTINUOUS_LEARNING
		//float trialLengthFactor = 1.0f / logf((float)nInferencesOverTrial);
		//float trialLengthFactor = powf((float)nInferencesOverTrial, -.5f);
		float trialLengthFactor = 1.0f / (float)nInferencesOverTrial;

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
			complexGenome[i]->toComplex.zeroAccumulator();
			complexGenome[i]->toMemory.zeroAccumulator();
			complexGenome[i]->toModulation.zeroAccumulator();
			complexGenome[i]->toOutput.zeroAccumulator();
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

		topNodeG->computePreSynActArraySize(genomeState);
		int size = genomeState[(int)complexGenome.size()];
		preSynActs = std::make_unique<float[]>(size);

		std::fill(genomeState.begin(), genomeState.end(), 0);

		topNodeG->computePostSynActArraySize(genomeState);
		postSynActArraySize = genomeState[(int)complexGenome.size()];
		previousPostSynActs = std::make_unique<float[]>(postSynActArraySize);
		currentPostSynActs = std::make_unique<float[]>(postSynActArraySize);

#ifdef SATURATION_PENALIZING
		std::fill(genomeState.begin(), genomeState.end(), 0);
		topNodeG->computeSaturationArraySize(genomeState);
		averageActivationArraySize = genomeState[(int)complexGenome.size()];
		averageActivation = std::make_unique<float[]>(averageActivationArraySize);
		float* ptr_averageActivation = averageActivation.get();

		saturationPenalization = 0.0f;
		topNodeP->setglobalSaturationAccumulator(&saturationPenalization);
		std::fill(averageActivation.get(), averageActivation.get() + averageActivationArraySize, 0.0f);
#endif
		
		// The following values will be modified by each node of the phenotype as the pointers are set.
		float* ptr_previousPostSynActs = previousPostSynActs.get();
		float* ptr_currentPostSynActs = currentPostSynActs.get();
		float* ptr_preSynActs = preSynActs.get();
		topNodeP->setArrayPointers(
			&ptr_previousPostSynActs,
			&ptr_currentPostSynActs,
			&ptr_preSynActs
#ifdef SATURATION_PENALIZING
			, &ptr_averageActivation
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
	std::fill(previousPostSynActs.get(), previousPostSynActs.get() + postSynActArraySize, 0.0f);
	std::fill(currentPostSynActs.get(), currentPostSynActs.get() + postSynActArraySize, 0.0f);
	//std::fill(preSynActs.get(),0.0f); // is already set to the biases.

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

	// this lambda is a copy of the one in ComplexNode_P::forward.
	auto applyNonLinearities = [](float* src, float* dst, ACTIVATION* fcts, int size)
	{
		for (int i = 0; i < size; i++) {
			switch (fcts[i]) {
			case TANH:
				dst[i] = tanhf(src[i]);
				break;
			case GAUSSIAN:
				dst[i] = 2.0f * expf(-src[i] * src[i]) - 1.0f; // technically the bias is not correctly put in. Does it matter ?
				break;
			case SINE:
				dst[i] = sinf(src[i]);
				break;
			case CENTERED_SINE:
				constexpr float z = 1.0f / .375261f; // to map to [-1, 1]
				dst[i] = tanhf(src[i]) * expf(-powf(src[i], 2.0f)) * z;
			}
		}
	};

	applyNonLinearities(
		topNodeP->preSynAct,
		topNodeP->preSynAct,
		topNodeG->outputActivations.data(),
		topNodeG->outputSize
	);

	std::copy(currentPostSynActs.get(), currentPostSynActs.get() + postSynActArraySize, previousPostSynActs.get());
}


void Network::mutate() {

	// The second constexpr value in each pair should not be neglible when compared to
	// the first, to introduce some kind of spontaneous regularization.

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


	constexpr float addComplexChildProbability = .004f;
	constexpr float removeComplexChildProbability = .002f;

	constexpr float addMemoryChildProbability = .004f; // .01f
	constexpr float removeMemoryChildProbability = .002f; // .002f


	constexpr float replaceComplexChildProbability = .05f;
	constexpr float replaceMemoryChildProbability = .05f;

	constexpr float duplicateComplexChildProbability = .005f;
	constexpr float duplicateMemoryChildProbability = .002f; //.01f

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


	// Biases, hebbian weights and activations mutations.
	{
		for (int i = 0; i < complexGenome.size(); i++) {
			if (complexGenome[i]->phenotypicMultiplicity > 0)
			{
				complexGenome[i]->mutateFloats();
				complexGenome[i]->mutateActivations();
			}
		}
		topNodeG->mutateFloats();
		topNodeG->mutateActivations();

		for (int i = 0; i < memoryGenome.size(); i++) {
			if (memoryGenome[i]->phenotypicMultiplicity > 0)
			{
				memoryGenome[i]->mutateFloats();
			}
		}
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
						complexGenome[j]->onChildInputSizeIncremented(complexGenome[i].get(), true);
					}
					topNodeG->onChildInputSizeIncremented(complexGenome[i].get(), true);
				}
			}

			r = UNIFORM_01;
			if (r < incrementComplexOutputSizeProbability) {
				bool success = complexGenome[i]->incrementOutputSize();
				if (success) {
					for (int j = 0; j < complexGenome.size(); j++) {
						complexGenome[j]->onChildOutputSizeIncremented(complexGenome[i].get(), true);
					}
					topNodeG->onChildOutputSizeIncremented(complexGenome[i].get(), true);
				}
			}

			int rID;
			r = UNIFORM_01;
			if (r < decrementComplexInputSizeProbability) {
				rID = INT_0X(complexGenome[i]->inputSize);
				bool success = complexGenome[i]->decrementInputSize(rID);
				if (success) {
					for (int j = 0; j < complexGenome.size(); j++) {
						complexGenome[j]->onChildInputSizeDecremented(complexGenome[i].get(), true, rID);
					}
					topNodeG->onChildInputSizeDecremented(complexGenome[i].get(), true, rID);
				}
			}

			r = UNIFORM_01;
			if (r < decrementComplexOutputSizeProbability) {
				rID = INT_0X(complexGenome[i]->outputSize);
				bool success = complexGenome[i]->decrementOutputSize(rID);
				if (success) {
					for (int j = 0; j < complexGenome.size(); j++) {
						complexGenome[j]->onChildOutputSizeDecremented(complexGenome[i].get(), true, rID);
					}
					topNodeG->onChildOutputSizeDecremented(complexGenome[i].get(), true, rID);
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
						complexGenome[j]->onChildInputSizeIncremented(memoryGenome[i].get(), false);
					}
					topNodeG->onChildInputSizeIncremented(memoryGenome[i].get(), false);
				}
			}

			r = UNIFORM_01;
			if (r < incrementMemoryOutputSizeProbability) {
				bool success = memoryGenome[i]->incrementOutputSize();
				if (success) {
					for (int j = 0; j < complexGenome.size(); j++) {
						complexGenome[j]->onChildOutputSizeIncremented(memoryGenome[i].get(), false);
					}
					topNodeG->onChildOutputSizeIncremented(memoryGenome[i].get(), false);
				}
			}

			r = UNIFORM_01;
			if (r < decrementMemoryInputSizeProbability) {
				int rID = INT_0X(memoryGenome[i]->inputSize);
				bool success = memoryGenome[i]->decrementInputSize(rID);
				if (success) {
					for (int j = 0; j < complexGenome.size(); j++) {
						complexGenome[j]->onChildInputSizeDecremented(memoryGenome[i].get(), false, rID);
					}
					topNodeG->onChildInputSizeDecremented(memoryGenome[i].get(), false, rID);
				}
			}

			r = UNIFORM_01;
			if (r < decrementMemoryOutputSizeProbability) {
				int rID = INT_0X(memoryGenome[i]->outputSize);
				bool success = memoryGenome[i]->decrementOutputSize(rID);
				if (success) {
					for (int j = 0; j < complexGenome.size(); j++) {
						complexGenome[j]->onChildOutputSizeDecremented(memoryGenome[i].get(), false, rID);
					}
					topNodeG->onChildOutputSizeDecremented(memoryGenome[i].get(), false, rID);
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
		SET_BINOMIAL((int)complexGenome.size() + 1, addComplexChildProbability);
		nMutations = BINOMIAL;
		if (nMutations > 0 && complexGenome.size() > 0) {
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
				// The fewer children a node already has, and deeper it is (high depth), the more likely it is to gain one.

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


		// Memory
		SET_BINOMIAL((int)complexGenome.size() + 1, addMemoryChildProbability);
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
		SET_BINOMIAL((int)complexGenome.size() + 1, replaceComplexChildProbability);
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
				//      - has same inputSize and outputSize that of toBeReplacedChild 

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
					isPotentialReplacement[toBeReplacedChild->closestNode->position] == 1 &&
					toBeReplacedChild->inputSize == toBeReplacedChild->closestNode->inputSize &&
					toBeReplacedChild->outputSize == toBeReplacedChild->closestNode->outputSize) {

					candidates.push_back(toBeReplacedChild->closestNode);
					float invDistance = 1.0f / (10.0f + (float)toBeReplacedChild->mutationalDistance);
					invDistances.push_back(invDistance);
					sumInvDistances += invDistance;
				}
				for (int i = 0; i < complexGenome.size(); i++) {
					if (complexGenome[i]->closestNode == toBeReplacedChild && 
						isPotentialReplacement[i] == 1 &&
						toBeReplacedChild->inputSize == complexGenome[i]->inputSize &&
						toBeReplacedChild->outputSize == complexGenome[i]->outputSize) {

						candidates.push_back(complexGenome[i].get());
						float invDistance = 1.0f / (10.0f + (float)complexGenome[i]->mutationalDistance);
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

					ComplexNode_G* replacement = candidates[rID];
					parent->complexChildren[inParentID] = replacement;

					// one could lower the mutational distance between the swapped children. But this is a recursive
					// mutation, and I have never managed to make those improve results.
					updatePhenotypicMultiplicities();
					updateDepths();
					sortGenome();
				}

			}

		}


		// Memory
		SET_BINOMIAL((int)complexGenome.size() + 1, replaceMemoryChildProbability);
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
				MemoryNode_G* toBeReplacedChild = parent->memoryChildren[inParentID];


				// determine candidates to replacement
				float sumInvDistances = 0.0f;
				std::vector<MemoryNode_G*> candidates;
				std::vector<float> invDistances;
				if (toBeReplacedChild->closestNode != NULL &&
					toBeReplacedChild->inputSize == toBeReplacedChild->closestNode->inputSize &&
					toBeReplacedChild->outputSize == toBeReplacedChild->closestNode->outputSize) {

					candidates.push_back(toBeReplacedChild->closestNode);
					float invDistance = 1.0f / (10.0f + (float)toBeReplacedChild->mutationalDistance);
					invDistances.push_back(invDistance);
					sumInvDistances += invDistance;
				}
				for (int j = 0; j < memoryGenome.size(); j++) {
					if (memoryGenome[j]->closestNode == toBeReplacedChild &&
						toBeReplacedChild->inputSize == memoryGenome[j]->inputSize &&
						toBeReplacedChild->outputSize == memoryGenome[j]->outputSize) {

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

					MemoryNode_G* replacement = candidates[rID];
					toBeReplacedChild->phenotypicMultiplicity -= parent->phenotypicMultiplicity;
					replacement->phenotypicMultiplicity += parent->phenotypicMultiplicity;


					// one could lower the mutational distance between the swapped children. But this is a recursive
					// mutation, and I have never managed to make those improve results.
					parent->memoryChildren[inParentID] = replacement;
				}

			}

		}
	}


	// Removing child nodes. 
	{
		ComplexNode_G* parent = nullptr;

		// Complex
		SET_BINOMIAL((int)complexGenome.size() + 1, removeComplexChildProbability);
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


		// Memory
		SET_BINOMIAL((int)complexGenome.size() + 1, removeMemoryChildProbability);
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
		SET_BINOMIAL((int)complexGenome.size() + 1, duplicateComplexChildProbability);
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
		SET_BINOMIAL((int)complexGenome.size() + 1, duplicateMemoryChildProbability);
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
	
	// these 3 are unnecessary, because a correct state is guaranteed by each mutation.
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
		complexGenome[i]->computeBiasSizes();
		complexGenome[i]->computeMemoryPreSynOffset();
	}
	topNodeG->computeBiasSizes();
	topNodeG->computeMemoryPreSynOffset();
	for (int i = 0; i < memoryGenome.size(); i++) {
		if (memoryGenome[i]->phenotypicMultiplicity > 0) {
			memoryGenome[i]->mutationalDistance++;
		}
	}


	// Phenotype is destroyed, as it may have become outdated. It will have to be recreated
	// before next inference. (The phenotype should not exist at this stage anyway)
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
	for (int i = inputSize+outputSize; i < averageActivationArraySize; i++) {
		p2 += powf(averageActivation[i] * invNInferencesN, 6.0f);
	}
	p2 /= (float) (averageActivationArraySize - inputSize - outputSize);
	


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

		// biases are not regulated.
		nParams[i] = 0;
		amplitudes[i] = 0.0f;
		
		auto accumulate = [] (InternalConnexion_G& co, float* valueAcc, int* sizeAcc) {
			int s = co.nLines * co.nColumns;
			*sizeAcc += s;
			for (int k = 0; k < s; k++) {
				*valueAcc += abs(co.A[k]);
				*valueAcc += abs(co.B[k]);
				*valueAcc += abs(co.C[k]);
				*valueAcc += abs(co.D[k]);
				*valueAcc += abs(co.alpha[k]);
				*valueAcc += abs(co.w[k]);
			}
		};
		
		accumulate(n->toComplex, &amplitudes[i], &nParams[i]);
		accumulate(n->toMemory, &amplitudes[i], &nParams[i]);
		accumulate(n->toModulation, &amplitudes[i], &nParams[i]);
		accumulate(n->toOutput, &amplitudes[i], &nParams[i]);

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
	return 0.0f*a + .5f * a * b + .5f*b; // normalized / ranked, so multiplying by a constant does nothing.
}
