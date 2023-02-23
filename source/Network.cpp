#pragma once

#include "Network.h"

// TODO , using boost serialize.
void Network::save(std::string path) {

}


Network::Network(Network* n) {
	nSimpleNeurons = n->nSimpleNeurons;
	inputSize = n->inputSize; 
	outputSize = n->outputSize;

	genome.reserve(n->genome.size());
	for (int i = 0; i < n->genome.size(); i++) genome.emplace_back(new GenotypeNode());

	for (int i = 0; i < nSimpleNeurons; i++) {
		genome[i]->isSimpleNeuron = true;
		genome[i]->f = n->genome[i]->f; 
		genome[i]->inputSize = n->genome[i]->inputSize;
		genome[i]->outputSize = n->genome[i]->outputSize; 
		genome[i]->bias.resize(0);   
		genome[i]->children.reserve(0);
		genome[i]->childrenConnexions.reserve(0);
#ifdef USING_NEUROMODULATION
		genome[i]->wNeuromodulation.reserve(0);
		genome[i]->neuromodulationBias = 0.0f;
#endif 
		genome[i]->concatenatedChildrenInputLength = 0;
		genome[i]->concatenatedChildrenInputBeacons.reserve(0);
		genome[i]->depth = 0;
		genome[i]->position = n->genome[i]->position;
		genome[i]->closestNode = NULL;
		genome[i]->mutationalDistance = 0;
	}

	for (int i = nSimpleNeurons; i < n->genome.size(); i++) {
		genome[i]->isSimpleNeuron = false;
		genome[i]->f = NULL;
		genome[i]->inputSize = n->genome[i]->inputSize;
		genome[i]->outputSize = n->genome[i]->outputSize;
#ifdef USING_NEUROMODULATION
		genome[i]->wNeuromodulation.reserve(0);
		genome[i]->neuromodulationBias = n->genome[i]->neuromodulationBias;
		genome[i]->wNeuromodulation.assign(n->genome[i]->wNeuromodulation.begin(), n->genome[i]->wNeuromodulation.end());
#endif 
		genome[i]->concatenatedChildrenInputLength = n->genome[i]->concatenatedChildrenInputLength;
		genome[i]->depth = n->genome[i]->depth;
		genome[i]->position = n->genome[i]->position;
		genome[i]->closestNode = n->genome[i]->closestNode != NULL ? genome[n->genome[i]->closestNode->position].get() : NULL; // topNode case
		genome[i]->mutationalDistance = n->genome[i]->mutationalDistance;

		genome[i]->bias.assign(n->genome[i]->bias.begin(), n->genome[i]->bias.end());
		genome[i]->concatenatedChildrenInputBeacons.assign(n->genome[i]->concatenatedChildrenInputBeacons.begin(), n->genome[i]->concatenatedChildrenInputBeacons.end());

		genome[i]->children.resize(n->genome[i]->children.size());
		for (int j = 0; j < n->genome[i]->children.size(); j++) { 
			genome[i]->children[j] = genome[n->genome[i]->children[j]->position].get();
		}

		genome[i]->childrenConnexions.reserve((int)((float)n->genome[i]->childrenConnexions.size()*1.5f));
		for (int j = 0; j < n->genome[i]->childrenConnexions.size(); j++) {
			genome[i]->childrenConnexions.emplace_back(n->genome[i]->childrenConnexions[j]);
		}
	}

}


Network::Network(int inputSize, int outputSize) :
inputSize(inputSize), outputSize(outputSize)
{
	nSimpleNeurons = 2;
	genome.reserve(nSimpleNeurons + 1);
	for (int i = 0; i < nSimpleNeurons + 1; i++) genome.emplace_back(new GenotypeNode());

	int i = 0;

	// 0: TanH 
	{
		genome[i]->isSimpleNeuron = true;
		genome[i]->inputSize = 1;
		genome[i]->outputSize = 1;
		genome[i]->f = *tanhf;
		genome[i]->bias.resize(0);
		genome[i]->children.reserve(0);
		genome[i]->childrenConnexions.reserve(0);
		genome[i]->concatenatedChildrenInputLength = 0;
		genome[i]->concatenatedChildrenInputBeacons.reserve(0);
		genome[i]->depth = 0;
		genome[i]->position = 0;
		genome[i]->closestNode = NULL;
		genome[i]->mutationalDistance = 0;
	}

	i++;
	// 1: ReLU
	{
		genome[i]->isSimpleNeuron = true;
		genome[i]->inputSize = 1;
		genome[i]->outputSize = 1;
		genome[i]->f = *ReLU;
		genome[i]->bias.resize(0);
		genome[i]->children.reserve(0);
		genome[i]->childrenConnexions.reserve(0);
		genome[i]->concatenatedChildrenInputLength = 0;
		genome[i]->concatenatedChildrenInputBeacons.reserve(0);
		genome[i]->depth = 0;
		genome[i]->position = 1;
		genome[i]->closestNode = NULL;
		genome[i]->mutationalDistance = 0;
	}

	i++;
	// 3: The initial top node
	{
		genome[i]->isSimpleNeuron = false;
		genome[i]->inputSize = inputSize;
		genome[i]->outputSize = outputSize;
		genome[i]->concatenatedChildrenInputLength = outputSize;
		genome[i]->f = NULL;
		genome[i]->depth = 1;
		genome[i]->position = 2;
		genome[i]->closestNode = NULL;
		genome[i]->mutationalDistance = 0;
		genome[i]->bias.resize(outputSize); 
		genome[i]->children.resize(0); 
		genome[i]->childrenConnexions.resize(0);
		genome[i]->childrenConnexions.emplace_back(
			INPUT_ID, genome[i]->children.size(), inputSize, outputSize, GenotypeConnexion::RANDOM
		);
#ifdef USING_NEUROMODULATION
		genome[i]->neuromodulationBias = 0.0f;
		genome[i]->wNeuromodulation.resize(outputSize);
#endif 
		genome[i]->computeBeacons();
	}

	//topNodeP.reset(new PhenotypeNode(genome[nSimpleNeurons].get()));
}


std::vector<float> Network::getOutput() {
	return topNodeP->currentOutput;
}


void Network::step(std::vector<float> obs) {
#ifdef USING_NEUROMODULATION
	topNodeP->neuromodulatorySignal = 1.0f; // other nodes have it set by their parent
#endif 
	topNodeP->forward(&(obs[0]));
}


void Network::mutate() {
	float r;
	// Floating point mutations.
	for (int i = nSimpleNeurons; i < genome.size(); i++) {
		genome[i]->mutateFloats();
	}

	// Children interconnection mutations.
	for (int i = nSimpleNeurons; i < genome.size(); i++) {
		r = UNIFORM_01;
		if (r < .01) genome[i]->disconnect();
		r = UNIFORM_01;
		if (r < .01) genome[i]->connect();
	}

	// Input and output sizes mutations.
	{
		for (int i = nSimpleNeurons; i < genome.size() - 1; i++) {
			r = UNIFORM_01;
			if (r < .01) {
				genome[i]->incrementInputSize();
				for (int j = 0; j < genome.size(); j++) {
					genome[j]->onChildInputSizeIncremented(genome[i].get());
				}
			}

			r = UNIFORM_01;
			if (r < .01) {
				genome[i]->incrementOutputSize();
				for (int j = 0; j < genome.size(); j++) {
					genome[j]->onChildOutputSizeIncremented(genome[i].get());
				}
			}

			int rID;
			r = UNIFORM_01;
			if (r < .01 && genome[i]->inputSize > 0) {
				rID = (int)(UNIFORM_01 * (float)inputSize);
				genome[i]->decrementInputSize(rID);
				for (int j = 0; j < genome.size(); j++) {
					genome[j]->onChildInputSizeDecremented(genome[i].get(), rID);
				}
			}

			r = UNIFORM_01;
			if (r < .01 && genome[i]->outputSize > 0) {
				rID = (int)(UNIFORM_01 * (float)outputSize);
				genome[i]->decrementOutputSize(rID);
				for (int j = 0; j < genome.size(); j++) {
					genome[j]->onChildOutputSizeDecremented(genome[i].get(), rID);
				}
			}
		}
	}

	// THE FOLLOWING ORDER MUST BE RESPECTED !!!   
	// adding child node --> replacing a child --> removing a child --> removing unused nodes
	
	
	// Adding a child node 
	{
		for (int i = nSimpleNeurons; i < genome.size(); i++) {
			r = UNIFORM_01;
			if (r < .01f && genome[i]->children.size() < MAX_CHILDREN_PER_BLOCK) {
				int prevDepth = genome[i]->depth;

				int childID = genome[i]->position;
				while (genome[childID]->depth <= genome[i]->depth && childID < genome.size() - 1) childID++;
				r = UNIFORM_01;
				childID--;
				childID = (int)(r * (float)childID);
				if (childID >= i) childID++;
				genome[i]->addChild(genome[childID].get());
				for (int j = nSimpleNeurons; j < genome.size(); j++) genome[j]->updateDepth();

				// if the child's depth was equal to this node's, the depth is incremented by one and in must be moved in the genome vector.
				if (prevDepth < genome[i]->depth && i != genome.size() - 1) {
					int id = genome[i]->position + 1;
					while (genome[id]->depth < genome[i]->depth) id++;
					std::unique_ptr<GenotypeNode> temp = std::move(genome[i]);
					genome.erase(genome.begin() + i); //inefficient, but insignificant here.
					genome.insert(genome.begin() + id - 1, std::move(temp)); // -1 because of the erasure

					// update positions for affected nodes.
					for (int j = 0; j < genome.size(); j++) genome[j]->position = j;
				}
			}
		}
	}

	// Child replacement				TODO  propagate further ?
	{
		int rChildID, replacementID;
		for (int i = nSimpleNeurons; i < genome.size(); i++) {
			r = UNIFORM_01;
			if (r > .02F || genome[i]->children.size() == 0) continue;

			r = UNIFORM_01;
			rChildID = (int)((float)genome[i]->children.size() * r);

			if (genome[i]->children[rChildID]->isSimpleNeuron) continue;

			GenotypeNode* child = genome[i]->children[rChildID];
			std::vector<GenotypeNode*> candidates;
			std::vector<int> distances;
			if (child->closestNode != NULL &&
				child->closestNode->inputSize == child->inputSize &&
				child->closestNode->outputSize == child->outputSize) {
				candidates.push_back(child->closestNode);
				distances.push_back(child->mutationalDistance);
			}
			for (int j = nSimpleNeurons; j < genome.size(); j++) {
				if (genome[j]->closestNode == child &&
					genome[j]->inputSize == child->inputSize &&
					genome[j]->outputSize == child->outputSize) {
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
			r = UNIFORM_01;
			replacementID = (int)((float)sum * r);
			int j = 0;
			while (j < distances.size() - 1 && sumDistances[j + 1] <= replacementID) j++;

			genome[i]->children[rChildID] = candidates[j];
			break;
		}
	}

	// Removing a child node. 
	{
		for (int i = nSimpleNeurons; i < genome.size(); i++) { // proceeding by ascending depth changes removal probabilities.
			r = UNIFORM_01;
			if (r < .01f && genome[i]->children.size() > 0) {

				int prevDepth = genome[i]->depth;
				int rID = (int)(UNIFORM_01 * (float)genome[i]->children.size());
				GenotypeNode* removedChild = genome[i]->children[rID];
				genome[i]->removeChild(rID);

				for (int j = nSimpleNeurons; j < genome.size(); j++) {
					genome[j]->updateDepth();
				}

				// If the removed child's depth was the highest amongst children, the node's depth is decreased 
				// and it must be moved in the genome vector.
				if (prevDepth > genome[i]->depth && i != genome.size() - 1) {
					int id = genome[i]->position - 1; // == i-1 nominaly  
					while (genome[id]->depth > genome[i]->depth) id--;
					std::unique_ptr<GenotypeNode> temp = std::move(genome[i]);
					genome.erase(genome.begin() + i); //inefficient, but insignificant here.
					genome.insert(genome.begin() + id + 1, std::move(temp));

					// update positions for affected nodes.
					for (int j = 0; j < genome.size(); j++) genome[j]->position = j;
				}
				break;
			}
		}
	}

	// Removing unused nodes.
	removeUnusedNodes();

	// Top Node Boxing
	{
		r = UNIFORM_01;
		if (r < .01f) {
			GenotypeNode* n = new GenotypeNode();
			GenotypeNode* prev = genome[genome.size() - 1].get();

			n->isSimpleNeuron = false;
			n->f = NULL;
			n->inputSize = inputSize;
			n->outputSize = outputSize;
#ifdef USING_NEUROMODULATION
			n->wNeuromodulation.resize(outputSize);
			n->neuromodulationBias = 0.0f;
#endif 
			n->concatenatedChildrenInputLength = outputSize;
			n->depth = prev->depth + 1;
			n->position = (int)genome.size();
			n->closestNode = NULL;
			n->mutationalDistance = 0;

			n->bias.resize(outputSize);
			n->concatenatedChildrenInputBeacons = { 0,inputSize };

			n->children.resize(1);
			n->children[0] = prev;


			n->childrenConnexions.reserve(4);
			n->childrenConnexions.emplace_back(INPUT_ID, 0, inputSize, outputSize, GenotypeConnexion::IDENTITY);
			n->childrenConnexions.emplace_back(0, 1, outputSize, inputSize, GenotypeConnexion::IDENTITY);

			genome.emplace_back(n);
		}
	}


	// Duplication : chooses a node in the genotype, and clones one of its children, replacing it with the clone.
	{
		// the simpler the node, and the most used it is, the higher the chance of being duplicated
		float positionMultiplicator;
		int rID;
		for (int i = nSimpleNeurons; i < genome.size(); i++) {
			positionMultiplicator = powf(.7f, (float)genome[i]->depth);
			r = UNIFORM_01;
			if (r > .02F * positionMultiplicator || genome[i]->children.size() == 0) continue;

			r = UNIFORM_01;
			rID = (int)((float)genome[i]->children.size() * r);
			if (genome[i]->children[rID]->isSimpleNeuron) continue;
			GenotypeNode* n = new GenotypeNode();
			GenotypeNode* base = genome[i]->children[rID];

			n->isSimpleNeuron = false;
			n->f = NULL;
			n->inputSize = base->inputSize;
			n->outputSize = base->outputSize;
#ifdef USING_NEUROMODULATION
			n->wNeuromodulation.assign(base->wNeuromodulation.begin(), base->wNeuromodulation.end());
			n->neuromodulationBias = base->neuromodulationBias;
#endif 
			n->concatenatedChildrenInputLength = base->concatenatedChildrenInputLength;
			n->depth = base->depth;
			n->closestNode = base;
			n->mutationalDistance = 0;

			n->bias.assign(base->bias.begin(), base->bias.end());
			n->concatenatedChildrenInputBeacons.assign(base->concatenatedChildrenInputBeacons.begin(), base->concatenatedChildrenInputBeacons.end());
			n->children.assign(base->children.begin(), base->children.end());

			n->childrenConnexions.reserve((int)((float)n->childrenConnexions.size() * 1.5f));
			for (int j = 0; j < n->childrenConnexions.size(); j++) {
				n->childrenConnexions.emplace_back(base->childrenConnexions[j]);
			}

			genome.emplace(genome.begin() + base->position + 1, n);
			genome[i+1]->children[rID] = genome[base->position + 1].get();

			for (int j = 0; j < genome.size(); j++) {
				genome[j]->position = j;
			}
			break;
		}
	}

	
	
	// Update beacons for forward().
	for (int i = nSimpleNeurons; i < genome.size(); i++) {
		genome[i]->computeBeacons();
		genome[i]->mutationalDistance++;
	}
	topNodeP.reset(new PhenotypeNode(genome[genome.size() - 1].get()));
}


void Network::removeUnusedNodes() {
	std::vector<int> occurences(genome.size());
	occurences[genome.size() - 1] = 1;
	for (int i = genome.size()-1; i >= nSimpleNeurons; i--) {

		if (occurences[i] == 0) { // genome[i] is unused. It must be removed.

			// firstly, handle the replacement pointers. Could do better, TODO .
			for (int j = nSimpleNeurons; j < genome.size(); j++) {
				if (genome[j]->closestNode == genome[i].get()) {
					genome[j]->closestNode = genome[i]->closestNode;
					genome[j]->mutationalDistance += genome[i]->mutationalDistance;
				}
			}

			// erase it from the genome list.
			genome[i].release();
			genome.erase(genome.begin() + i);
			for (int j = i; j < genome.size(); j++) genome[j]->position = j;
			
			continue;
		}

		for (int j = 0; j < genome[i]->children.size(); j++) {
			occurences[genome[i]->children[j]->position]++;
		}
	}
}


void Network::intertrialReset() {
	if (topNodeP.get() == NULL) topNodeP.reset(new PhenotypeNode(genome[genome.size() - 1].get()));;
#if defined RISI_NAJARRO_2020
	topNodeP->randomH();
#elif defined USING_NEUROMODULATION
	topNodeP->zero();
#endif 
}


float Network::getSizeRegularizationLoss() {
	std::vector<int> nParams(genome.size());
	for (int i = nSimpleNeurons; i < genome.size(); i++) {
		nParams[i] = 0;
		for (int j = 0; j < genome[i]->childrenConnexions.size(); j++) {
			nParams[i] += genome[i]->childrenConnexions[j].nLines * genome[i]->childrenConnexions[j].nColumns;
		}
		for (int j = 0; j < genome[i]->children.size(); j++) {
			// TODO can vector elements be non contiguous ? The following line could be problematic.
			nParams[i] += nParams[genome[i]->children[j]->position]; 
		}
		nParams[i] += 2 * genome[i]->outputSize; // to account for the biases and the neuromodulation weights
	}
	return (float) nParams[genome.size()-1];
}


float Network::getAmplitudeRegularizationLoss() {
	float sum = 0.0f;
	int size;
	for (int i = nSimpleNeurons; i < genome.size(); i++) {
		for (int j = 0; j < genome[i]->childrenConnexions.size(); j++) {
			size = genome[i]->childrenConnexions[j].nLines * genome[i]->childrenConnexions[j].nColumns;
			for (int k = 0; k < size; k++) {
				sum += abs(genome[i]->childrenConnexions[j].A[k]);
				sum += abs(genome[i]->childrenConnexions[j].B[k]);
				sum += abs(genome[i]->childrenConnexions[j].C[k]);
				sum += abs(genome[i]->childrenConnexions[j].eta[k]);
#if defined RISI_NAJARRO_2020
				sum += abs(genome[i]->childrenConnexions[j].D[k]);
#elif defined USING_NEUROMODULATION
				sum += abs(genome[i]->childrenConnexions[j].alpha[k]);
				sum += abs(genome[i]->childrenConnexions[j].w[k]);
#endif 
			}
		}
		for (int j = 0; j < genome[i]->outputSize; j++) {
			sum += abs(genome[i]->bias[j]);
#ifdef USING_NEUROMODULATION
			sum += abs(genome[i]->wNeuromodulation[j]);
#endif
		}
	}
	return sum;
}
