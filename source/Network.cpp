#pragma once

#include "Network.h"
#include <iostream>

Network::Network(Network* n) {
	nSimpleNeurons = n->nSimpleNeurons;
	inputSize = n->inputSize; 
	outputSize = n->outputSize;

	genome.reserve(n->genome.size());
	for (int i = 0; i < n->genome.size(); i++) genome.emplace_back(new GenotypeNode());

	for (int i = 0; i < nSimpleNeurons; i++) {
		genome[i]->copyParameters(n->genome[i].get());
	}

	for (int i = nSimpleNeurons; i < n->genome.size(); i++) {
		genome[i]->copyParameters(n->genome[i].get());

		// The line below is valid only if any non top and non simple neuron necessarily has a closest node. Should be the case.
		genome[i]->closestNode = genome[n->genome[i]->closestNode->position].get();

		genome[i]->children.resize(n->genome[i]->children.size());
		for (int j = 0; j < n->genome[i]->children.size(); j++) { 
			genome[i]->children[j] = genome[n->genome[i]->children[j]->position].get();
		}
	}

	topNodeG = std::make_unique<GenotypeNode>();
	topNodeG->copyParameters(n->topNodeG.get());
	topNodeG->closestNode = NULL;
	topNodeG->children.resize(n->topNodeG->children.size());
	for (int j = 0; j < n->topNodeG->children.size(); j++) {
		topNodeG->children[j] = genome[n->topNodeG->children[j]->position].get();
	}

	topNodeP.reset(NULL);

#ifdef SATURATION_PENALIZING
	saturationPenalization = 0.0f;
	nInferencesN = 0;
#endif
}


Network::Network(int inputSize, int outputSize) :
inputSize(inputSize), outputSize(outputSize)
{
	nSimpleNeurons = 2;
	genome.reserve(nSimpleNeurons + 3);
	for (int i = 0; i < nSimpleNeurons; i++) genome.emplace_back(new GenotypeNode());

	int i = 0;
	// 0: TanH simple neuron
	genome[i]->nodeType = GenotypeNode::TANH;
	genome[i]->inputSize = 1;
	genome[i]->outputSize = 1;
	genome[i]->depth = 0;
	genome[i]->position = i;
	genome[i]->closestNode = NULL;

	i++;
	// 1: Derivator simple neuron : outputs the differences between its input at this step and at the previous step.
	genome[i]->nodeType = GenotypeNode::DERIVATOR;
	genome[i]->inputSize = 1;
	genome[i]->outputSize = 1;
	genome[i]->depth = 0;
	genome[i]->position = i;
	genome[i]->closestNode = NULL;



	// Top node
	topNodeG = std::make_unique<GenotypeNode>();

	topNodeG->nodeType = GenotypeNode::COMPLEX;
	topNodeG->inputSize = inputSize;
	topNodeG->outputSize = outputSize;
	topNodeG->sumChildrenInputSizes = outputSize;
	topNodeG->depth = 1;
	topNodeG->position = (int)genome.size();
	topNodeG->closestNode = NULL;
	topNodeG->mutationalDistance = 0;
	topNodeG->children.resize(0);
	topNodeG->childrenConnexions.resize(0);
	topNodeG->childrenConnexions.reserve(4);
	topNodeG->childrenConnexions.emplace_back(
		INPUT_ID, (int)genome[i]->children.size(), outputSize, inputSize, GenotypeConnexion::RANDOM
	);
	topNodeG->childrenConnexions.emplace_back(
		INPUT_ID, MODULATION_ID, 2, inputSize, GenotypeConnexion::RANDOM
	);
	topNodeG->biasM[0] = 0.0f;
	topNodeG->biasM[1] = 0.0f;
	topNodeG->computeBeacons();
	topNodeG->childrenInBias.resize(topNodeG->sumChildrenInputSizes + topNodeG->outputSize);

	topNodeP.reset(NULL);

#ifdef SATURATION_PENALIZING
	saturationPenalization = 0.0f;
	nInferencesN = 0;
#endif

	computePhenotypicMultiplicities();
}


float* Network::getOutput() {
	return topNodeP->currentOutput;
}


void Network::postTrialUpdate(float score) {
#ifndef CONTINUOUS_LEARNING
	if (topNodeP->nInferencesP != 0) {
		float invNInferencesP = 1.0f / (float)topNodeP->nInferencesP;
		topNodeP->updateWatTrialEnd(invNInferencesP);
	}
	else {
		std::cerr << "ERROR : postTrialUpdate WAS CALLED BEFORE EVALUATION ON A TRIAL !!" << std::endl;
	}
	
#endif

#if defined GUIDED_MUTATIONS && defined CONTINUOUS_LEARNING
	if (topNodeP->nInferencesP != 0) {
		//float trialLengthFactor = 1.0f / logf((float)topNodeP->nInferencesP);
		float trialLengthFactor = 1.0f / (float)topNodeP->nInferencesP;

		// One and only one of these three calls to accumulateW MUST be active:
		

		// Requires population.normalizedScoreGradients = true. Constant factor between 5 and 50 recommended.
		//topNodeP->accumulateW(1.0f*score*trialLengthFactor);			 

		// Drastically improves perfs when learning is "too simple". Constant factor between 5 and 100 recommended.
		topNodeP->accumulateW(5.0f * trialLengthFactor);

		// Not implemented yet.
		//topNodeP->accumulateW((score-parentScore) * trialLengthFactor);  
	}
	else {
		std::cerr << "ERROR : postTrialUpdate WAS CALLED BEFORE EVALUATION ON A TRIAL !!" << std::endl;
	}
#endif
}


void Network::destroyPhenotype() {
	topNodeP.reset(NULL);
	previousOutputs.reset(NULL);
	currentOutputs.reset(NULL);
	previousInputs.reset(NULL);
	currentInputs.reset(NULL); 
#ifdef SATURATION_PENALIZING
	averageActivation.reset(NULL);
#endif
}


void Network::createPhenotype() {
	if (topNodeP.get() == NULL) {
		topNodeP.reset(new PhenotypeNode(topNodeG.get()));


		std::vector<int> genomeState(genome.size() + 1);
		topNodeG->position = (int)genome.size();
		for (int i = 0; i < nSimpleNeurons; i++) genomeState[i] = 1;

		topNodeG->computeInArraySize(genomeState);
		phenotypeInArraySize = genomeState[(int)genome.size()];

		for (int i = nSimpleNeurons; i < genome.size() + 1; i++) genomeState[i] = 0;

		topNodeG->computeOutArraySize(genomeState);
		phenotypeOutArraySize = genomeState[(int)genome.size()];


		previousOutputs = std::make_unique<float[]>(phenotypeOutArraySize);
		currentOutputs =  std::make_unique<float[]>(phenotypeOutArraySize);
		previousInputs =  std::make_unique<float[]>(phenotypeInArraySize);
		currentInputs =   std::make_unique<float[]>(phenotypeInArraySize);

#ifdef SATURATION_PENALIZING
		for (int i = nSimpleNeurons; i < genome.size() + 1; i++) genomeState[i] = 0;
		topNodeG->computeSaturationArraySize(genomeState);
		phenotypeSaturationArraySize = genomeState[(int)genome.size()];
		averageActivation = std::make_unique<float[]>(phenotypeSaturationArraySize);
#endif
		

		topNodeP->setArrayPointers(
			previousOutputs.get(),
			currentOutputs.get(),
			previousInputs.get(),
			currentInputs.get()
#ifdef SATURATION_PENALIZING
			, averageActivation.get()
#endif
		);

#ifdef SATURATION_PENALIZING
		topNodeP->setSaturationPenalizationPtr(&saturationPenalization);
#endif
	}
};


void Network::preTrialReset() {
	// Set to currenInputs at each Network::step, so no need to fill.
	//std::fill(previousInputs.get(), previousInputs.get() + phenotypeInArraySize, 0.0f);

	std::fill(previousOutputs.get(), previousOutputs.get() + phenotypeOutArraySize, 0.0f); 
	std::fill(currentOutputs.get(), currentOutputs.get() + phenotypeOutArraySize, 0.0f);
	std::fill(currentInputs.get(), currentInputs.get() + phenotypeInArraySize, 0.0f); 
#ifdef SATURATION_PENALIZING
	std::fill(averageActivation.get(), averageActivation.get() + phenotypeSaturationArraySize, 0.0f);
#endif


	topNodeP->preTrialReset();
};


void Network::step(const std::vector<float>& obs) {
#ifdef SATURATION_PENALIZING
	nInferencesN++;
#endif
	std::copy(currentInputs.get(), currentInputs.get() + phenotypeInArraySize, previousInputs.get());
	std::copy(obs.begin(), obs.end(), topNodeP->currentInput);
	topNodeP->forward();
}


// TODO mutation rates should be functions of the network size. (Higher Prob when smaller Net)
void Network::mutate() {

	// The first constexpr value in each pair should be greater than the second, 
	// or at worst equal, to introduce some kind of spontaneous regularization.

	constexpr float createConnexionProbability = .01f;
	constexpr float deleteConnexionProbability = .002f;

	constexpr float incrementInputSizeProbability = .003f;
	constexpr float decrementInputSizeProbability = .002f;

	constexpr float incrementOutputSizeProbability = .003f;
	constexpr float decrementOutputSizeProbability = .002f;

	constexpr float addChildProbability = .01f;
	constexpr float removeChildProbability = .002f;

	constexpr float childReplacementProbability = .05f;

	constexpr float nodeBoxingProbability = .005f;

	constexpr float nodeDuplicationProbability = .01f;

	float r;


#if defined GUIDED_MUTATIONS && not defined CONTINUOUS_LEARNING
	if (topNodeP.get() != NULL && topNodeP->nInferencesP != 0) {
		topNodeP->accumulateW(.5f);
	}
#endif


	/*
	* 
	Each kind of mutations has the critical responsability of making sure that, once it is done, the following 
	rules are verified: 

	- The topNode and every node in the genome have up-to-date ->depth.
	- The topNode and every node in the genome have up-to-date ->position.
	- The topNode and every node in the genome have up-to-date ->phenotypicMultiplicity.
	- The genome is sorted by ascending depths.
	
	So that were the different types of mutations to be rearranged, deleted, or if other were added, the function 
	would still work properly. Mutations are computationnaly insignificant when compared to inferences.
	
	*/


	// Floating point mutations.
	{
		for (int i = nSimpleNeurons; i < genome.size(); i++) {
			if (genome[i]->phenotypicMultiplicity > 0)
			{
				genome[i]->mutateFloats();
			}
		}
		topNodeG->mutateFloats();
	}


	// Add or remove connection.
	{
		for (int i = nSimpleNeurons; i < genome.size(); i++) {
			r = UNIFORM_01;
			if (r < deleteConnexionProbability) genome[i]->removeConnexion();
			r = UNIFORM_01;
			if (r < createConnexionProbability) genome[i]->addConnexion();
		}
		r = UNIFORM_01;
		if (r < deleteConnexionProbability) topNodeG->removeConnexion();
		r = UNIFORM_01;
		if (r < createConnexionProbability) topNodeG->addConnexion();
	}


	// Input and output sizes mutations.
	{
		for (int i = nSimpleNeurons; i < genome.size(); i++) {
			r = UNIFORM_01;
if (r < incrementInputSizeProbability) {
	bool success = genome[i]->incrementInputSize();
	if (success) {
		for (int j = 0; j < genome.size(); j++) {
			genome[j]->onChildInputSizeIncremented(genome[i].get());
		}
		topNodeG->onChildInputSizeIncremented(genome[i].get());
	}
}

r = UNIFORM_01;
if (r < incrementOutputSizeProbability) {
	bool success = genome[i]->incrementOutputSize();
	if (success) {
		for (int j = 0; j < genome.size(); j++) {
			genome[j]->onChildOutputSizeIncremented(genome[i].get());
		}
		topNodeG->onChildOutputSizeIncremented(genome[i].get());
	}
}

int rID;
r = UNIFORM_01;
if (r < decrementInputSizeProbability) {
	rID = INT_0X(genome[i]->inputSize);
	bool success = genome[i]->decrementInputSize(rID);
	if (success) {
		for (int j = 0; j < genome.size(); j++) {
			genome[j]->onChildInputSizeDecremented(genome[i].get(), rID);
		}
		topNodeG->onChildInputSizeDecremented(genome[i].get(), rID);
	}
}

r = UNIFORM_01;
if (r < decrementOutputSizeProbability) {
	rID = INT_0X(genome[i]->outputSize);
	bool success = genome[i]->decrementOutputSize(rID);
	if (success) {
		for (int j = 0; j < genome.size(); j++) {
			genome[j]->onChildOutputSizeDecremented(genome[i].get(), rID);
		}
		topNodeG->onChildOutputSizeDecremented(genome[i].get(), rID);
	}
}
		}
	}



	// Adding a child node. 
	if (UNIFORM_01 < addChildProbability) {
		// The fewer children a node already has, and higher its depth, the more likely it is to gain one.
		// "higher its depth" and not "deeper", as nodes closer to the top node are more likely to gain children.
		// (for parallelization bias)

		GenotypeNode* parent = nullptr;
		bool aborted = false;

		// Look for a node in which to insert a child
		{
			std::vector<float> probabilities(genome.size() + 1);
			float sum = 0.0f;
			for (int i = nSimpleNeurons; i < genome.size() + 1; i++) {
				GenotypeNode* n = i != genome.size() ? genome[i].get() : topNodeG.get();
				if (n->phenotypicMultiplicity != 0 && n->children.size() < MAX_CHILDREN_PER_BLOCK) {

					// twice more likely when no children that when full
					float baseProbability = (float)(2 * MAX_CHILDREN_PER_BLOCK - n->children.size());

					// the shallower, the least likely it is to gain children.
					float depthFactor = powf(.9f, (float)(topNodeG->depth - n->depth));

					probabilities[i] = baseProbability * depthFactor;
					sum += probabilities[i];
				}
				else {
					probabilities[i] = 0.0f;
				}
			}
			if (sum != 0.0f)
			{
				probabilities[0] = probabilities[0] / sum;
				for (int i = 1; i < genome.size(); i++) {
					probabilities[i] = probabilities[i - 1] + probabilities[i] / sum;
				}
				float r = UNIFORM_01;
				int parentID = binarySearch(probabilities, r);
				parent = parentID != genome.size() ? genome[parentID].get() : topNodeG.get();
			}
			else { aborted = true; }
		}

		// Chooses a new child for the parent, among all nodes that dont have the parent as direct
		// or indirect child, so as to avoid loops.
		if (!aborted) {

			std::vector<int> hasParentAsChild(genome.size()); // -1 if is potential child, 1 if is not, 0 if not known yet.
			for (int i = 0; i < nSimpleNeurons; i++) {
				hasParentAsChild[i] = -1;
			}
			for (int i = nSimpleNeurons; i < genome.size(); i++) {
				hasParentAsChild[i] = -1;
				for (int j = 0; j < genome[i]->children.size(); j++) {
					if (genome[i]->children[j] == parent) {
						hasParentAsChild[i] = 1;
						break;
					}
					if (hasParentAsChild[genome[i]->children[j]->position]==1) {
						hasParentAsChild[i] = 1;
						break;
					}
				}
			}
			if (parent->position != genome.size()) { hasParentAsChild[parent->position] = 1; }

			int nPotentialChildren = 0;
			for (int i = 0; i < genome.size(); i++) {
				if (hasParentAsChild[i] == -1) {
					nPotentialChildren++;
				}
			}

			GenotypeNode* child = nullptr;
			int childID = INT_0X(nPotentialChildren);
			int id = -1;
			for (int i = 0; i < genome.size(); i++) {
				if (hasParentAsChild[i] == -1) {
					id++;
					if (id == childID) {
						child = genome[i].get();
						break;
					}
				}
			}

			parent->addChild(child);
			child->phenotypicMultiplicity += parent->phenotypicMultiplicity;
			updateDepths();
			sortGenome();
		}
	}


	// Replaces a child node.
	if (UNIFORM_01 < childReplacementProbability) {

		GenotypeNode* replacedNode = nullptr;
		GenotypeNode* parent = nullptr;
		int inParentID;
		bool aborted = false;

		// Establish which node is to be replaced.
		{
			std::vector<int> nDifferentUses(genome.size());
			for (int i = nSimpleNeurons; i < genome.size() + 1; i++) {
				GenotypeNode* n = i != genome.size() ? genome[i].get() : topNodeG.get();
				for (int j = 0; j < n->children.size(); j++) {
					nDifferentUses[n->children[j]->position]++;
				}
			}
			std::vector<float> probabilities(genome.size());
			float sum = 0.0f;
			for (int i = nSimpleNeurons; i < genome.size(); i++) {
				probabilities[i] = (float)nDifferentUses[i]; // *powf(X, (float)genome[i]->depth);
				sum += probabilities[i];
			}
			if (sum != 0.0f)
			{
				probabilities[0] = probabilities[0] / sum;
				for (int i = 1; i < genome.size(); i++) {
					probabilities[i] = probabilities[i - 1] + probabilities[i] / sum;
				}
				float r = UNIFORM_01;
				int replacedNodeID = binarySearch(probabilities, r);
				replacedNode = genome[replacedNodeID].get();

				int replacedApparition = INT_0X(nDifferentUses[replacedNodeID]);
				int currentApparition = -1;
				for (int i = nSimpleNeurons; i < genome.size() + 1; i++) {
					GenotypeNode* n = i != genome.size() ? genome[i].get() : topNodeG.get();
					if (replacedApparition == currentApparition) {
						break;
					}
					for (int j = 0; j < n->children.size(); j++) {
						if (n->children[j] == replacedNode) {
							currentApparition++;
							if (currentApparition == replacedApparition) {
								parent = n;
								inParentID = j;
								break;
							}
						}
					}
				}
			}
			else { aborted = true; }
		}

		// Look for a candidate to replace the chosen node. If one is found, it replaces the chosen node in the parent.
		// If not, the operation is aborted.
		if (!aborted) {
			float sumInvDistances = 0.0f;
			std::vector<GenotypeNode*> candidates;
			std::vector<float> invDistances;
			if (replacedNode->closestNode != NULL &&
				replacedNode->closestNode->inputSize ==  replacedNode->inputSize &&
				replacedNode->closestNode->outputSize == replacedNode->outputSize &&
				!hasChild(replacedNode->closestNode, parent)) {

				candidates.push_back(replacedNode->closestNode);
				float invDistance = 1.0f / (10.0f + (float)replacedNode->mutationalDistance);
				invDistances.push_back(invDistance);
				sumInvDistances += invDistance;
			}
			for (int j = nSimpleNeurons; j < genome.size(); j++) {
				if (genome[j]->closestNode == replacedNode &&
					genome[j]->inputSize  ==  replacedNode->inputSize &&
					genome[j]->outputSize ==  replacedNode->outputSize &&
					!hasChild(genome[j].get(), parent)) {

					candidates.push_back(genome[j].get());
					float invDistance = 1.0f / (10.0f + (float)genome[j]->mutationalDistance);
					invDistances.push_back(invDistance);
					sumInvDistances += invDistance;
				}
			}

			if (candidates.size() != 0) {

				std::vector<float> probabilities(invDistances.size());
				probabilities[0] = invDistances[0] / sumInvDistances;
				for (int i = 1; i < invDistances.size(); i++) {
					invDistances[i] = invDistances[i - 1] + invDistances[i] / sumInvDistances;
				}
				float r = UNIFORM_01;
				int rID = binarySearch(probabilities, r);
				
				parent->children[inParentID] = candidates[rID];
				
				updateDepths();
				sortGenome();
			}
		}
	}


	// Removes a child node. 
	if (UNIFORM_01 < removeChildProbability) {
		GenotypeNode* removedNode = nullptr;
		GenotypeNode* parent = nullptr;
		int inParentID;
		bool aborted = false;

		// Establish which node is to be removed.
		{
			std::vector<int> nDifferentUses(genome.size());
			for (int i = nSimpleNeurons; i < genome.size() + 1; i++) {
				GenotypeNode* n = i != genome.size() ? genome[i].get() : topNodeG.get();
				for (int j = 0; j < n->children.size(); j++) {
					nDifferentUses[n->children[j]->position]++;
				}
			}
			std::vector<float> probabilities(genome.size());
			float sum = 0.0f;
			for (int i = nSimpleNeurons; i < genome.size(); i++) {
				probabilities[i] = (float)nDifferentUses[i]; // *powf(X, (float)genome[i]->depth);
				sum += probabilities[i];
			}
			if (sum != 0.0f)
			{
				probabilities[0] = probabilities[0] / sum;
				for (int i = 1; i < genome.size(); i++) {
					probabilities[i] = probabilities[i - 1] + probabilities[i] / sum;
				}
				float r = UNIFORM_01;
				int replacedNodeID = binarySearch(probabilities, r);
				removedNode = genome[replacedNodeID].get();

				int replacedApparition = INT_0X(nDifferentUses[replacedNodeID]);
				int currentApparition = -1;
				for (int i = nSimpleNeurons; i < genome.size() + 1; i++) {
					GenotypeNode* n = i != genome.size() ? genome[i].get() : topNodeG.get();
					if (replacedApparition == currentApparition) {
						break;
					}
					for (int j = 0; j < n->children.size(); j++) {
						if (n->children[j] == removedNode) {
							currentApparition++;
							if (currentApparition == replacedApparition) {
								parent = n;
								inParentID = j;
								break;
							}
						}
					}
				}
			}
			else { aborted = true; }
		}

		// Removal.
		if (!aborted) {
			parent->removeChild(inParentID);

			updateDepths();
			sortGenome();
		}
	}
	

	// Duplicate a child node.
	if (UNIFORM_01 < nodeDuplicationProbability){
		// Chooses a node in the genotype, clones it, and replaces it with the clone in one of the nodes
		// that has it as a child. The shallower and the more common the node, the more likely it is
		// to be selected.

		GenotypeNode* clonedNode = nullptr;
		GenotypeNode* parent = nullptr;
		int inParentID;
		bool aborted = false;

		// Establish which node is to be cloned.
		{
			std::vector<int> nDifferentUses(genome.size());
			for (int i = nSimpleNeurons; i < genome.size()+1; i++) {
				GenotypeNode* n = i != genome.size() ? genome[i].get() : topNodeG.get();
				for (int j = 0; j < n->children.size(); j++) {
					nDifferentUses[n->children[j]->position]++;
				}
			}
			std::vector<float> probabilities(genome.size());
			float sum = 0.0f;
			for (int i = nSimpleNeurons; i < genome.size(); i++) {
				probabilities[i] = (float)nDifferentUses[i] * powf(.7f, (float)genome[i]->depth);
				sum += probabilities[i];
			}
			if (sum != 0.0f) 
			{
				probabilities[0] = probabilities[0] / sum;
				for (int i = 1; i < genome.size(); i++) {
					probabilities[i] = probabilities[i - 1] + probabilities[i] / sum;
				}
				float r = UNIFORM_01;
				int clonedNodeID = binarySearch(probabilities, r);
				clonedNode = genome[clonedNodeID].get();

				int replacedApparition = INT_0X(nDifferentUses[clonedNodeID]);
				int currentApparition = -1;
				for (int i = nSimpleNeurons; i < genome.size() + 1; i++) {
					GenotypeNode* n = i != genome.size() ? genome[i].get() : topNodeG.get();
					if (replacedApparition == currentApparition) {
						break;
					}
					for (int j = 0; j < n->children.size(); j++) {
						if (n->children[j] == clonedNode) {
							currentApparition++;
							if (currentApparition == replacedApparition) {
								parent = n;
								inParentID = j;
								break;
							}
						}
					}
				}
			}
			else { aborted = true; }
		}

		// Create the clone, and replace the original node with it at the chosen spot.
		// No need to sort the genome nor update depths.
		if (!aborted) {
			GenotypeNode* n = new GenotypeNode();
			

			n->copyParameters(clonedNode);

			n->closestNode = clonedNode;
			n->children.assign(clonedNode->children.begin(), clonedNode->children.end());

			genome.emplace(genome.begin() + clonedNode->position + 1, n);

			parent->children[inParentID] = n;

			for (int j = 0; j < genome.size(); j++) {
				genome[j]->position = j;
			}
			topNodeG->position = (int)genome.size();
		}

	}


	// Boxes a child node.
	if (UNIFORM_01 < nodeBoxingProbability) {
		// Chooses a node in the genotype, creates a node that boxes it, and replaces the original node
		// with the box in one of the nodes that has it as a child. The shallower and the more common the node,
		// the more likely it is to be selected.

		GenotypeNode* boxedNode = nullptr;
		GenotypeNode* parent = nullptr;
		int inParentID;
		bool aborted = false;

		// Establish which node is to be boxed.
		{
			std::vector<int> nDifferentUses(genome.size());
			for (int i = nSimpleNeurons; i < genome.size() + 1; i++) {
				GenotypeNode* n = i != genome.size() ? genome[i].get() : topNodeG.get();
				for (int j = 0; j < n->children.size(); j++) {
					nDifferentUses[n->children[j]->position]++;
				}
			}
			std::vector<float> probabilities(genome.size());
			float sum = 0.0f;
			for (int i = 0; i < genome.size(); i++) {
				probabilities[i] = (float)nDifferentUses[i] * powf(.8f, (float)genome[i]->depth);
				sum += probabilities[i];
			}
			if (sum != 0.0f)
			{
				probabilities[0] = probabilities[0] / sum;
				for (int i = 1; i < genome.size(); i++) {
					probabilities[i] = probabilities[i - 1] + probabilities[i] / sum;
				}
				float r = UNIFORM_01;
				int boxedNodeID = binarySearch(probabilities, r);
				boxedNode = genome[boxedNodeID].get();

				int replacedApparition = INT_0X(nDifferentUses[boxedNodeID]);
				int currentApparition = -1;
				for (int i = nSimpleNeurons; i < genome.size()+1; i++) {
					GenotypeNode* n = i != genome.size() ? genome[i].get() : topNodeG.get();
					for (int j = 0; j < n->children.size(); j++) {
						if (n->children[j] == boxedNode) {
							currentApparition++;
							if (currentApparition == replacedApparition) {
								parent = n;
								inParentID = j;
								break;
							}
						}
					}
				}
			}
			else { aborted = true; }
		}
		
		// create the box and replace the boxed child in the parent at inParentId
		if (!aborted) {
			GenotypeNode* box = new GenotypeNode();

			box->nodeType = GenotypeNode::COMPLEX;
			box->inputSize = boxedNode->inputSize;
			box->outputSize = boxedNode->outputSize;
			
			box->biasM[0] = NORMAL_01 * .2f;
			box->biasM[1] = NORMAL_01 * .2f;
				
			box->depth = boxedNode->depth + 1;
			box->closestNode = boxedNode;
			box->mutationalDistance = 0;
			box->position = (int)genome.size();

			box->children.resize(1);
			box->children[0] = boxedNode;

			box->computeBeacons();
			box->childrenInBias.resize(box->outputSize + box->sumChildrenInputSizes);
			for (int j = 0; j < box->childrenInBias.size(); j++) {
				box->childrenInBias[j] = NORMAL_01 * .2f;
			}

			box->childrenConnexions.reserve(4);
			box->childrenConnexions.emplace_back(INPUT_ID, MODULATION_ID, 2, box->inputSize, GenotypeConnexion::RANDOM);
			box->childrenConnexions.emplace_back(INPUT_ID, 0, box->inputSize, box->inputSize, GenotypeConnexion::RANDOM);
			box->childrenConnexions.emplace_back(0, 1, box->outputSize, box->outputSize, GenotypeConnexion::RANDOM);

			parent->children[inParentID] = box;

			genome.emplace_back(box);
			updateDepths();
			sortGenome();
		}

		// No need to sort the genome nor update depths.
	}
	
	


	// THE FOLLOWING CALL ORDER MUST BE RESPECTED: 
	//  removeUnusedNodes -> computePhenotypicMultiplicities -> Update mutational distances
	

	// Removing unused nodes. Find a better solution, TODO 
	if (UNIFORM_01 < .03f) {
		removeUnusedNodes(); // already computes phenotypic multiplicities for itself.
	}

	computePhenotypicMultiplicities();

	// Update mutational distances.
	for (int i = nSimpleNeurons; i < genome.size(); i++) {
		if (genome[i]->phenotypicMultiplicity > 0) {
			genome[i]->mutationalDistance++;
		}
	}

	// Phenotype may have become outdated, and is no longer needed for updates. It must be recreated before
	// next inference.
	topNodeP.reset(NULL);
}


void Network::updateDepths() {
	std::vector<int> genomeState(genome.size()+1);
	for (int i = 0; i < nSimpleNeurons; i++) genomeState[i] = 1;
	topNodeG->position = (int) genome.size();
	topNodeG->updateDepth(genomeState);
	for (int i = (int)genome.size() - 1; i >= nSimpleNeurons; i--) {
		if(genomeState[i] == 0) genome[i]->updateDepth(genomeState);
	}
}


void Network::sortGenome() {

	std::vector<std::pair<int, int>> depthXposition;
	int size = (int)genome.size() - nSimpleNeurons;
	depthXposition.resize(size);

	for (int i = 0; i < size; i++) {
		depthXposition[i] = std::make_pair(genome[i+nSimpleNeurons]->depth, i + nSimpleNeurons);
	}

	std::sort(depthXposition.begin(), depthXposition.end(), 
		[](std::pair<int, int> a, std::pair<int, int> b)
		{
			return a.first < b.first;  //ascending order
		});

	std::vector<GenotypeNode*> tempStorage(size); // store all nodes after the simple neurons
	for (int i = 0; i < size; i++) {
		tempStorage[i] = genome[i + nSimpleNeurons].release();
	}

	genome.resize(nSimpleNeurons);
	for (int i = 0; i < size; i++) {
		genome.emplace_back(tempStorage[depthXposition[i].second - nSimpleNeurons]);
	}

	for (int i = 0; i < genome.size(); i++) {
		genome[i]->position = i;
	}
	topNodeG->position = (int)genome.size();
	return;
}


void Network::computePhenotypicMultiplicities() {
	topNodeG->phenotypicMultiplicity = 1;
	std::vector<int> multiplicities(genome.size());

	for (int i = 0; i < (int)topNodeG->children.size(); i++) {
		multiplicities[topNodeG->children[i]->position]++;
	}
	for (int i = (int)genome.size() - 1; i >= nSimpleNeurons; i--) { // requires the genome be sorted by ascending depths.
		for (int j = 0; j < genome[i]->children.size(); j++) {
			multiplicities[genome[i]->children[j]->position] += multiplicities[genome[i]->position];
		}
	}
	for (int i = 0; i < genome.size(); i++) {
		genome[i]->phenotypicMultiplicity = multiplicities[i];
	}
}


// requires the genome be sorted by ascending (and up to date) depths.
void Network::removeUnusedNodes() {
	std::vector<int> occurences(genome.size());
	for (int i = 0; i < (int)topNodeG->children.size(); i++) {
		occurences[topNodeG->children[i]->position]++;
	}

	for (int i = (int)genome.size()-1; i >= nSimpleNeurons; i--) { 

		if (occurences[i] == 0) { // genome[i] is unused. It must be removed.

			// firstly, handle the replacement pointers. Could do better, TODO .
			for (int j = nSimpleNeurons; j < genome.size(); j++) {
				if (genome[j]->closestNode == genome[i].get()) {
					genome[j]->closestNode = genome[i]->closestNode;
					genome[j]->mutationalDistance += genome[i]->mutationalDistance;
				}
			}

			// erase it from the genome list.
			genome[i].reset(NULL); // next line does it already, i think.
			genome.erase(genome.begin() + i);
			for (int j = i; j < genome.size(); j++) genome[j]->position = j;

			continue;
		}

		for (int j = 0; j < genome[i]->children.size(); j++) {
			occurences[genome[i]->children[j]->position] += occurences[i];
		}
	}
}


bool Network::hasChild(GenotypeNode* parent, GenotypeNode* potentialChild) {
	if (parent == potentialChild) return true;
	std::vector<int> checked(genome.size());
	return parent->hasChild(checked, potentialChild);
}

#ifdef SATURATION_PENALIZING
float Network::getSaturationPenalization()
{
	if (nInferencesN == 0) {
		std::cerr <<
			"ERROR : getSaturationPenalization() WAS CALLED, BUT THE NETWORK HAS NEVER BEEN USED BEFORE !"
			<< std::endl;
		return 0.0f;
	}

	std::vector<int> genomeState(genome.size() + 1);
	for (int i = 0; i < nSimpleNeurons; i++) {
		genomeState[i] = 1; // simple neurons cost only 1 function evaluation each.
	}
	topNodeG->position = (int)genome.size();
	topNodeG->getNnonLinearities(genomeState);
	int nNonLinearities = genomeState[genome.size()] - inputSize - outputSize;
	float p1 = nNonLinearities != 0 ? saturationPenalization / (nInferencesN * nNonLinearities) : 0.0f;


	float p2 = 0.0f;
	float invNInferencesN = 1.0f / nInferencesN;
	for (int i = inputSize; i < phenotypeSaturationArraySize; i++) {
		p2 += powf(averageActivation[i] * invNInferencesN, 6.0f);
	}
	p2 /= (float) (phenotypeSaturationArraySize - inputSize);
	


	constexpr float µ = .5f;
	return µ * p1 + (1 - µ) * p2;
}
#endif


// TODO take genome.size() and maybe childrenInBias into consideration.
float Network::getRegularizationLoss() {
	std::vector<int> nParams(genome.size() + 1);
	std::vector<float> amplitudes(genome.size() + 1);
	int size;
	constexpr int nArrays = 6; // eta and gamma's amplitudes are irrelevant here.

	for (int i = nSimpleNeurons; i < genome.size() + 1; i++) {
		GenotypeNode* n;
		n = i != genome.size() ? genome[i].get() : topNodeG.get();

		nParams[i] = 0;
		amplitudes[i] = 0.0f;
		for (int j = 0; j < n->childrenConnexions.size(); j++) {
			size = n->childrenConnexions[j].nLines * n->childrenConnexions[j].nColumns;
			nParams[i] += size;
			for (int k = 0; k < size; k++) {
				amplitudes[i] += abs(n->childrenConnexions[j].A[k]);
				amplitudes[i] += abs(n->childrenConnexions[j].B[k]);
				amplitudes[i] += abs(n->childrenConnexions[j].C[k]);
				amplitudes[i] += abs(n->childrenConnexions[j].D[k]); 
				amplitudes[i] += abs(n->childrenConnexions[j].alpha[k]);
				amplitudes[i] += abs(n->childrenConnexions[j].w[k]);
			}
		}
		for (int j = 0; j < n->children.size(); j++) {
			nParams[i] += nParams[n->children[j]->position];
			amplitudes[i] += amplitudes[n->children[j]->position];
		}
	}

	float a = (float)amplitudes[genome.size()] / (float)(nParams[genome.size()]*nArrays); // amplitude term
	float b = logf(inputSize + outputSize + (float)nParams[genome.size()]);				  // size term
	return 0.2f*a*(1.0f + b) + 1.0f * b; // The whole vector will be reduced, so multiplying all terms by a common factor does nothing.
}
