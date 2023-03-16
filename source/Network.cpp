#pragma once

#include "Network.h"


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

		//genome[i]->closestNode = n->genome[i]->closestNode != NULL ? genome[n->genome[i]->closestNode->position].get();
		// The line below is only valid if any non top and non simple neuron necessarily has a closest node. TODO change boxing.
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
}


Network::Network(int inputSize, int outputSize) :
inputSize(inputSize), outputSize(outputSize)
{
	nSimpleNeurons = 1;
	genome.reserve(nSimpleNeurons + 3);
	for (int i = 0; i < nSimpleNeurons; i++) genome.emplace_back(new GenotypeNode());

	int i = 0;

	// 0: TanH simple neuron
	genome[i]->isSimpleNeuron = true;
	genome[i]->inputSize = 1;
	genome[i]->outputSize = 1;
	genome[i]->f = *tanhf;
	genome[i]->depth = 0;
	genome[i]->position = 0;
	genome[i]->closestNode = NULL;
	
	// 1: ReLu, or cos, or whatever could make sense with hebbian rules. (cos probably does not =\ )

	
	topNodeG = std::make_unique<GenotypeNode>();

	topNodeG->isSimpleNeuron = false;
	topNodeG->inputSize = inputSize;
	topNodeG->outputSize = outputSize;
	topNodeG->concatenatedChildrenInputLength = outputSize;
	topNodeG->f = NULL;
	topNodeG->depth = 1;
	topNodeG->position = -1;
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
	topNodeG->biasMplus = NORMAL_01;
	topNodeG->biasMminus = NORMAL_01;
	topNodeG->inBias.resize(inputSize);
	topNodeG->outBias.resize(outputSize);
	topNodeG->computeBeacons();
}


std::vector<float> Network::getOutput() {
	return topNodeP->currentOutput;
}


void Network::step(const std::vector<float>& obs) {
	topNodeP->neuromodulatorySignal = 1.0f; // other nodes have it set by their parent
	topNodeP->forward(obs.data());
}


void Network::mutate() {

	// value in pairs should be equal, or at least the first greater than the second, to introduce some kind of regularization.

	constexpr float createConnexionProbability = .01f;
	constexpr float deleteConnexionProbability = .004f;

	constexpr float incrementInputSizeProbability = .001f;
	constexpr float decrementInputSizeProbability = .001f;

	constexpr float incrementOutputSizeProbability = .001f;
	constexpr float decrementOutputSizeProbability = .001f;

	constexpr float addChildProbability = .01f;
	constexpr float removeChildProbability = .004f;

	constexpr float childReplacementProbability = .005f;

	constexpr float nodeBoxingProbability = .003f;
	
	constexpr float nodeDuplicationProbability = .004f;

	float r;


	// Floating point mutations.
	{
		for (int i = nSimpleNeurons; i < genome.size(); i++) {
			genome[i]->mutateFloats();
		}
		topNodeG->mutateFloats();
		topNodeG->zeroInBias();
	}
	

	// Add or remove connection.
	{
		for (int i = nSimpleNeurons; i < genome.size(); i++) {
			r = UNIFORM_01;
			if (r < deleteConnexionProbability) genome[i]->disconnect();
			r = UNIFORM_01;
			if (r < createConnexionProbability) genome[i]->connect();
		}
		r = UNIFORM_01;
		if (r < deleteConnexionProbability) topNodeG->disconnect();
		r = UNIFORM_01;
		if (r < createConnexionProbability) topNodeG->connect();
	}

	// Input and output sizes mutations.
	{
		for (int i = nSimpleNeurons; i < genome.size(); i++) {
			r = UNIFORM_01;
			if (r < incrementInputSizeProbability) {
				genome[i]->incrementInputSize();
				for (int j = 0; j < genome.size(); j++) {
					genome[j]->onChildInputSizeIncremented(genome[i].get());
				}
			}

			r = UNIFORM_01;
			if (r < incrementOutputSizeProbability) {
				genome[i]->incrementOutputSize();
				for (int j = 0; j < genome.size(); j++) {
					genome[j]->onChildOutputSizeIncremented(genome[i].get());
				}
			}

			int rID;
			r = UNIFORM_01;
			if (r < decrementInputSizeProbability && genome[i]->inputSize > 0) {
				rID = INT_0X(genome[i]->inputSize);
				genome[i]->decrementInputSize(rID);
				for (int j = 0; j < genome.size(); j++) {
					genome[j]->onChildInputSizeDecremented(genome[i].get(), rID);
				}
			}

			r = UNIFORM_01;
			if (r < decrementOutputSizeProbability && genome[i]->outputSize > 0) {
				rID = INT_0X(genome[i]->outputSize);
				genome[i]->decrementOutputSize(rID);
				for (int j = 0; j < genome.size(); j++) {
					genome[j]->onChildOutputSizeDecremented(genome[i].get(), rID);
				}
			}
		}
	}


	// WARNING THE FOLLOWING ORDER MUST BE RESPECTED IN THE FUNCTION !!!   
	// adding a child --> replacing a child --> removing a child --> duplicating a child --> removing unused nodes
	
	
	// Adding a child node. TODO modify active alternative to lower probability as depth increases.
	{
		for (int i = nSimpleNeurons; i < genome.size(); i++) {
			if (UNIFORM_01 < addChildProbability && genome[i]->children.size() < MAX_CHILDREN_PER_BLOCK) {

				//////
				//Alternative. either this, which requires the genome to be sorted by ascending depths
				
				/*int childID = i;
				while (childID < genome.size() && genome[childID]->depth <= genome[i]->depth) childID++;
				childID = INT_0X(childID-1);
				if (childID >= i) childID++;*/

				// Or this: 
				int childID = INT_0X(genome.size());
				// make sure we do not create a loop:
				if (hasChild(genome[childID].get(), genome[i].get())) continue; 
				///////

				genome[i]->addChild(genome[childID].get());
				updateDepths();
				sortGenome();
			}
		}
		r = UNIFORM_01;
		if (r < addChildProbability && topNodeG->children.size() < MAX_CHILDREN_PER_BLOCK) {
			int childID = INT_0X(genome.size());
			topNodeG->addChild(genome[childID].get());
		}
	}

	// Child replacement				TODO  propagate further ?
	{
		int rChildID, replacementID;
		for (int i = nSimpleNeurons; i < genome.size()+1; i++) {
			GenotypeNode* n;
			n = i != genome.size() ? genome[i].get() : topNodeG.get();

			if (UNIFORM_01 > childReplacementProbability || n->children.size() == 0) continue;

			rChildID = INT_0X(n->children.size());

			GenotypeNode* child = n->children[rChildID];
			std::vector<GenotypeNode*> candidates;
			std::vector<int> distances;
			if (child->closestNode != NULL &&
				child->closestNode->inputSize == child->inputSize &&
				child->closestNode->outputSize == child->outputSize &&
				//child->closestNode->depth < n->depth) { 
				!hasChild(child->closestNode, n)) { 
				candidates.push_back(child->closestNode);
				distances.push_back(child->mutationalDistance);
			}
			for (int j = nSimpleNeurons; j < genome.size(); j++) {
				if (genome[j]->closestNode == child &&
					genome[j]->inputSize == child->inputSize &&
					genome[j]->outputSize == child->outputSize &&
					//genome[j]->depth < n->depth) {  
					!hasChild(genome[j].get(), n)) {
					candidates.push_back(genome[j].get());
					distances.push_back(genome[j]->mutationalDistance);
				}
			}

			if (candidates.size() == 0) continue;

			std::vector<int> sumDistances(distances.size());
			int sum = 0;
			for (int j = 0; j < distances.size(); j++) {
				sumDistances[j] = sum;
				sum += distances[j];
			}
			replacementID = INT_0X(sum);
			int j = 0;
			while (j < distances.size() - 1 && sumDistances[j + 1] <= replacementID) j++;

			n->children[rChildID] = candidates[j];

			updateDepths();
			sortGenome();

			break;
		}
	}

	// Removing a child node. 
	{
		for (int i = nSimpleNeurons; i < genome.size(); i++) { // proceeding by ascending depth changes removal probabilities.
			r = UNIFORM_01;
			if (r < removeChildProbability && genome[i]->children.size() > 0) {

				int prevDepth = genome[i]->depth;
				int rID = INT_0X(genome[i]->children.size());
				genome[i]->removeChild(rID);

				updateDepths();
				sortGenome();

				break;
			}
		}
		if (UNIFORM_01 < removeChildProbability && topNodeG->children.size() > 0) {
			int rID = INT_0X(topNodeG->children.size());
			topNodeG->removeChild(rID);
		}
	}

	// Duplicate a node : chooses a node in the genotype, and clones one of its children, replacing it with the clone.
	{
		// the simpler the node, and the most used it is, the higher the chance of being duplicated
		float positionMultiplicator;
		int rID;
		for (int i = nSimpleNeurons; i < genome.size()+1; i++) {
			GenotypeNode* parentNode = i != genome.size() ? genome[i].get() : topNodeG.get();
			positionMultiplicator = powf(.7f, (float)parentNode->depth);
			r = UNIFORM_01;
			if (r > nodeDuplicationProbability * positionMultiplicator || parentNode->children.size() == 0) continue;

			rID = INT_0X(parentNode->children.size());
			if (parentNode->children[rID]->isSimpleNeuron) continue;
			GenotypeNode* n = new GenotypeNode();
			GenotypeNode* base = parentNode->children[rID];

			n->isSimpleNeuron = false;
			n->f = NULL;
			n->inputSize = base->inputSize;
			n->outputSize = base->outputSize;
			n->inBias.assign(base->inBias.begin(), base->inBias.end());
			n->outBias.assign(base->outBias.begin(), base->outBias.end());
			n->biasMplus = base->biasMplus;
			n->biasMminus = base->biasMminus;
			n->concatenatedChildrenInputLength = base->concatenatedChildrenInputLength;
			n->depth = base->depth;
			n->closestNode = base;
			n->mutationalDistance = 0;

			n->concatenatedChildrenInputBeacons.assign(base->concatenatedChildrenInputBeacons.begin(), base->concatenatedChildrenInputBeacons.end());
			n->children.assign(base->children.begin(), base->children.end());

			n->childrenConnexions.reserve((int)((float)n->childrenConnexions.size() * 1.5f));
			for (int j = 0; j < n->childrenConnexions.size(); j++) {
				n->childrenConnexions.emplace_back(base->childrenConnexions[j]);
			}

			genome.emplace(genome.begin() + base->position + 1, n);

			parentNode->children[rID] = n;

			for (int j = 0; j < genome.size(); j++) {
				genome[j]->position = j;
			}
		}
	}

	// Removing unused nodes. Find a better solution.
	if (UNIFORM_01 < .03f) {
		removeUnusedNodes();   
	}
	

	// Node Boxing : boxes one of the children. The shallower the child, the more likely it is to succeed.
	{
		for (int i = 0; i < genome.size()+1; i++) {
			GenotypeNode* parentNode = i != genome.size() ? genome[i].get() : topNodeG.get();
			if (parentNode->children.size() == 0) continue;

			int rID = INT_0X(parentNode->children.size());
			float depthFactor = powf(.7f, (float)parentNode->children[rID]->depth);
			if (UNIFORM_01 < nodeBoxingProbability * depthFactor) {
				
				GenotypeNode* child = parentNode->children[rID];
				GenotypeNode* box = new GenotypeNode();

				box->isSimpleNeuron = false;
				box->f = NULL;
				box->inputSize = child->inputSize;
				box->outputSize = child->outputSize;
				box->inBias.resize(child->inputSize);
				box->outBias.resize(child->outputSize);

				if (child->isSimpleNeuron) {
					for (int j = 0; j < box->inputSize; j++) {
						box->inBias[j] = 0.0f;
					}
					for (int j = 0; j < box->outputSize; j++) {
						box->outBias[j] = 0.0f;
					}
					box->biasMplus = NORMAL_01*.2f;
					box->biasMminus = NORMAL_01 * .2f;
				}
				else {
					for (int j = 0; j < box->inputSize; j++) {
						box->inBias[j] = child->inBias[j];
					}
					for (int j = 0; j < box->outputSize; j++) {
						box->outBias[j] = child->outBias[j];
					}
					box->biasMplus = child->biasMplus;
					box->biasMminus = child->biasMminus;
				}
				
				box->depth = child->depth + 1;
				box->closestNode = child;
				box->mutationalDistance = 0;
				box->position = (int)genome.size();

				box->children.resize(1);
				box->children[0] = child;

				box->childrenConnexions.reserve(4);
				box->childrenConnexions.emplace_back(INPUT_ID, MODULATION_ID, 2, box->inputSize, GenotypeConnexion::RANDOM);
				box->childrenConnexions.emplace_back(INPUT_ID, 0, box->inputSize, box->inputSize, GenotypeConnexion::IDENTITY);
				box->childrenConnexions.emplace_back(0, 1, box->outputSize, box->outputSize, GenotypeConnexion::IDENTITY);

				parentNode->children[rID] = box;

				genome.emplace_back(box);
				updateDepths();
				sortGenome();
			}
		}
	}
	
	
	// Update beacons for forward(), and update mutational distances.
	for (int i = nSimpleNeurons; i < genome.size(); i++) {
		genome[i]->computeBeacons();
		genome[i]->mutationalDistance++;
	}
	topNodeG->computeBeacons();
	topNodeG->mutationalDistance++;

	// Phenotype may have become outdated, we destroy it.
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
	return;
}


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
			genome[i].reset(NULL);
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


float Network::getRegularizationLoss() {
	std::vector<int> nParams(genome.size()+1);
	std::vector<float> amplitudes(genome.size()+1);
	int size;
	constexpr int nArrays = 5; // eta and gamma's amplitudes are irrelevant here.

	for (int i = nSimpleNeurons; i < genome.size()+1; i++) {
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
				amplitudes[i] += abs(n->childrenConnexions[j].alpha[k]);
				amplitudes[i] += abs(n->childrenConnexions[j].w[k]);
			}
		}
		nParams[i] *= nArrays;
		for (int j = 0; j < n->children.size(); j++) {
			nParams[i] += nParams[n->children[j]->position]; 
			amplitudes[i] += amplitudes[n->children[j]->position];
		}
	}

	// the lower the exponent, the stronger the size regularization
	constexpr float exponent = .7f;
	return amplitudes[genome.size()] * powf((float) nParams[genome.size()], -exponent); 
}
