#pragma once

#include "Network.h"


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
		genome[i]->depth = 0;
		genome[i]->position = n->genome[i]->position;

	}

	for (int i = nSimpleNeurons; i < n->genome.size(); i++) {
		genome[i]->isSimpleNeuron = false;
		genome[i]->f = NULL;
		genome[i]->inputSize = n->genome[i]->inputSize;
		genome[i]->outputSize = n->genome[i]->outputSize;
		genome[i]->inBias.assign(n->genome[i]->inBias.begin(), n->genome[i]->inBias.end());
		genome[i]->outBias.assign(n->genome[i]->outBias.begin(), n->genome[i]->outBias.end());
		genome[i]->biasMplus = 0.0f;
		genome[i]->biasMminus = 0.0f;
		genome[i]->concatenatedChildrenInputLength = n->genome[i]->concatenatedChildrenInputLength;
		genome[i]->depth = n->genome[i]->depth;
		genome[i]->position = n->genome[i]->position;
		genome[i]->closestNode = n->genome[i]->closestNode != NULL ? genome[n->genome[i]->closestNode->position].get() : NULL; // topNode case
		genome[i]->mutationalDistance = n->genome[i]->mutationalDistance;

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
	nSimpleNeurons = 1;
	genome.reserve(nSimpleNeurons + 1 + 3);
	for (int i = 0; i < nSimpleNeurons + 1; i++) genome.emplace_back(new GenotypeNode());

	int i = 0;

	// 0: TanH 
	{
		genome[i]->isSimpleNeuron = true;
		genome[i]->inputSize = 1;
		genome[i]->outputSize = 1;
		genome[i]->f = *tanhf;
		genome[i]->depth = 0;
		genome[i]->position = 0;
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
		genome[i]->position = i;
		genome[i]->closestNode = NULL;
		genome[i]->mutationalDistance = 0; 
		genome[i]->children.resize(0); 
		genome[i]->childrenConnexions.resize(0); 
		genome[i]->childrenConnexions.reserve(4);
		genome[i]->childrenConnexions.emplace_back(
			INPUT_ID, (int)genome[i]->children.size(), outputSize, inputSize, GenotypeConnexion::RANDOM
		);
		genome[i]->childrenConnexions.emplace_back(
			INPUT_ID, MODULATION_ID, 2, inputSize, GenotypeConnexion::RANDOM
		);
		genome[i]->biasMplus = NORMAL_01;
		genome[i]->biasMminus = NORMAL_01;
		genome[i]->inBias.resize(inputSize);
		genome[i]->outBias.resize(outputSize);
		genome[i]->computeBeacons();
	}

	//topNodeP.reset(new PhenotypeNode(genome[nSimpleNeurons].get())); // instantiation at this point is not necessary.
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
	constexpr float deleteConnexionProbability = .002f;

	constexpr float incrementInputSizeProbability = .0005f;
	constexpr float decrementInputSizeProbability = .0005f;

	constexpr float incrementOutputSizeProbability = .0005f;
	constexpr float decrementOutputSizeProbability = .0005f;

	constexpr float addChildProbability = .007f;
	constexpr float removeChildProbability = .002f;

	constexpr float childReplacementProbability = .005f;

	constexpr float topNodeBoxingProbability = .01f;
	
	constexpr float nodeDuplicationProbability = .003f;

	float r;


	// Floating point mutations.
	for (int i = nSimpleNeurons; i < genome.size(); i++) {
		genome[i]->mutateFloats();
	}
	genome[genome.size() - 1]->zeroInBias();
	

	// Children interconnection mutations.
	for (int i = nSimpleNeurons; i < genome.size(); i++) {
		r = UNIFORM_01;
		if (r < deleteConnexionProbability) genome[i]->disconnect();
		r = UNIFORM_01;
		if (r < createConnexionProbability) genome[i]->connect();
	}

	// Input and output sizes mutations.
	{
		for (int i = nSimpleNeurons; i < genome.size() - 1; i++) {
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
				rID = (int)(UNIFORM_01 * (float)genome[i]->inputSize);
				genome[i]->decrementInputSize(rID);
				for (int j = 0; j < genome.size(); j++) {
					genome[j]->onChildInputSizeDecremented(genome[i].get(), rID);
				}
			}

			r = UNIFORM_01;
			if (r < decrementOutputSizeProbability && genome[i]->outputSize > 0) {
				rID = (int)(UNIFORM_01 * (float)genome[i]->outputSize);
				genome[i]->decrementOutputSize(rID);
				for (int j = 0; j < genome.size(); j++) {
					genome[j]->onChildOutputSizeDecremented(genome[i].get(), rID);
				}
			}
		}
	}


	// WARNING THE FOLLOWING ORDER MUST BE RESPECTED IN THE FUNCTION !!!   
	// adding a child --> replacing a child --> removing a child --> duplicating a child --> removing unused nodes
	
	
	// Adding a child node 
	{
		for (int i = nSimpleNeurons; i < genome.size(); i++) {
			r = UNIFORM_01;
			if (r < addChildProbability && genome[i]->children.size() < MAX_CHILDREN_PER_BLOCK) {
				int prevDepth = genome[i]->depth;

				int childID = genome[i]->position;
				while (genome[childID]->depth <= genome[i]->depth && childID < genome.size() - 1) childID++;
				r = UNIFORM_01;
				childID--;
				childID = (int)(r * (float)childID);
				if (childID >= i) childID++;
				genome[i]->addChild(genome[childID].get());
				
				updateDepths();
				sortGenome();
			}
		}
	}

	// Child replacement				TODO  propagate further ?
	{
		int rChildID, replacementID;
		for (int i = nSimpleNeurons; i < genome.size(); i++) {
			r = UNIFORM_01;
			if (r > childReplacementProbability || genome[i]->children.size() == 0) continue;

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
				int rID = (int)(UNIFORM_01 * (float)genome[i]->children.size());
				GenotypeNode* removedChild = genome[i]->children[rID];
				genome[i]->removeChild(rID);

				updateDepths();
				sortGenome();

				break;
			}
		}
	}

	// Duplicate a node : chooses a node in the genotype, and clones one of its children, replacing it with the clone.
	 {
		// the simpler the node, and the most used it is, the higher the chance of being duplicated
		float positionMultiplicator;
		int rID;
		for (int i = nSimpleNeurons; i < genome.size(); i++) {
			positionMultiplicator = powf(.7f, (float)genome[i]->depth);
			r = UNIFORM_01;
			if (r > nodeDuplicationProbability * positionMultiplicator || genome[i]->children.size() == 0) continue;

			r = UNIFORM_01;
			rID = (int)((float)genome[i]->children.size() * r);
			if (genome[i]->children[rID]->isSimpleNeuron) continue;
			GenotypeNode* n = new GenotypeNode();
			GenotypeNode* base = genome[i]->children[rID];

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
			genome[i + 1]->children[rID] = genome[base->position + 1].get();

			for (int j = 0; j < genome.size(); j++) {
				genome[j]->position = j;
			}
			break;
		}
	}

	// Removing unused nodes.
	removeUnusedNodes();

	// Top Node Boxing
	{
		r = UNIFORM_01;
		float depthFactor = powf(.7f, (float)genome[genome.size() - 1]->depth);
		if (r < topNodeBoxingProbability * depthFactor) {
			GenotypeNode* n = new GenotypeNode();
			GenotypeNode* prev = genome[genome.size() - 1].get();

			n->isSimpleNeuron = false;
			n->f = NULL;
			n->inputSize = inputSize;
			n->outputSize = outputSize;
			n->biasMplus = prev->biasMplus;
			n->biasMminus = prev->biasMminus;
			n->inBias.resize(inputSize);
			n->outBias.resize(outputSize);
			for (int i = 0; i < n->inputSize; i++) {
				n->inBias[i] = prev->inBias[i];
			}
			for (int i = 0; i < n->outputSize; i++) {
				n->outBias[i] = prev->outBias[i];
			}
			n->concatenatedChildrenInputLength = outputSize;
			n->depth = prev->depth + 1;
			n->position = (int)genome.size();
			n->closestNode = NULL;
			n->mutationalDistance = 0;

			n->concatenatedChildrenInputBeacons = { 0,inputSize };

			n->children.resize(1);
			n->children[0] = prev;


			n->childrenConnexions.reserve(4);
			n->childrenConnexions.emplace_back(INPUT_ID, MODULATION_ID, 2, inputSize, GenotypeConnexion::RANDOM);
			n->childrenConnexions.emplace_back(INPUT_ID, 0, inputSize, inputSize, GenotypeConnexion::IDENTITY);
			n->childrenConnexions.emplace_back(0, 1, outputSize, outputSize, GenotypeConnexion::IDENTITY);

			genome.emplace_back(n);
		}
	}
	
	
	// Update beacons for forward().
	for (int i = nSimpleNeurons; i < genome.size(); i++) {
		genome[i]->computeBeacons();
		genome[i]->mutationalDistance++;
	}

	topNodeP.reset(new PhenotypeNode(genome[genome.size() - 1].get()));
}


void Network::updateDepths() {
	std::vector<int> genomeState(genome.size());
	for (int i = 0; i < nSimpleNeurons; i++) genomeState[i] = 1;
	genome[genome.size() - 1]->updateDepth(genomeState);
}


void Network::sortGenome() {
#ifdef _DEBUG
	int dmax = 0;
	for (int i = 0; i < genome.size()-1; i++) if (genome[i]->depth > dmax) dmax = genome[i]->depth;
	if (dmax >= genome[genome.size() - 1]->depth) {

		int problem = 0; // set breakpoint here, should not happen.
	}
#endif // _DEBUG

	std::vector<std::pair<int, int>> depthXposition;
	int size = (int)genome.size() - nSimpleNeurons - 1;
	depthXposition.resize(size);

	for (int i = 0; i < size; i++) {
		depthXposition[i] = std::make_pair(genome[i+nSimpleNeurons]->depth, i + nSimpleNeurons);
	}

	std::sort(depthXposition.begin(), depthXposition.end(), 
		[](std::pair<int, int> a, std::pair<int, int> b)
		{
			return a.first < b.first;  //ascending order
		});

	std::vector<GenotypeNode*> tempStorage(size+1); // store all nodes after the simple neurons
	for (int i = 0; i < size+1; i++) {
		tempStorage[i] = genome[i + nSimpleNeurons].release();
	}

	genome.resize(nSimpleNeurons);
	for (int i = 0; i < size; i++) {
		genome.emplace_back(tempStorage[depthXposition[i].second - nSimpleNeurons]);
	}
	genome.emplace_back(tempStorage[size]);

	for (int i = 0; i < genome.size(); i++) {
		genome[i]->position = i;
	}
	return;
}


void Network::removeUnusedNodes() {
	std::vector<int> occurences(genome.size());
	occurences[genome.size() - 1] = 1;
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
			occurences[genome[i]->children[j]->position]++;
		}
	}
}


void Network::intertrialReset() {
	if (topNodeP.get() == NULL) topNodeP.reset(new PhenotypeNode(genome[genome.size() - 1].get()));;

	topNodeP->zero();
}


float Network::getRegularizationLoss() {
	std::vector<int> nParams(genome.size());
	std::vector<float> amplitudes(genome.size());
	int size;
	constexpr int nArrays = 6;

	for (int i = nSimpleNeurons; i < genome.size(); i++) {
		nParams[i] = 0;
		amplitudes[i] = 0.0f;
		for (int j = 0; j < genome[i]->childrenConnexions.size(); j++) {
			size = genome[i]->childrenConnexions[j].nLines * genome[i]->childrenConnexions[j].nColumns;
			nParams[i] += size;
			for (int k = 0; k < size; k++) {
				amplitudes[i] += abs(genome[i]->childrenConnexions[j].A[k]);
				amplitudes[i] += abs(genome[i]->childrenConnexions[j].B[k]);
				amplitudes[i] += abs(genome[i]->childrenConnexions[j].C[k]);
				amplitudes[i] += abs(genome[i]->childrenConnexions[j].eta[k]);
				amplitudes[i] += abs(genome[i]->childrenConnexions[j].alpha[k]);
				amplitudes[i] += abs(genome[i]->childrenConnexions[j].w[k]);
			}
		}
		nParams[i] *= nArrays;
		for (int j = 0; j < genome[i]->children.size(); j++) {
			nParams[i] += nParams[genome[i]->children[j]->position]; 
			amplitudes[i] += amplitudes[genome[i]->children[j]->position];
		}
	}
	// the lower the exponent, the stronger the size regularization
	constexpr float exponent = .8f;
	return amplitudes[genome.size() - 1] * powf((float) nParams[genome.size()-1] + 10*nArrays, -exponent); 
}
