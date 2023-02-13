#pragma once

#include "Network.h"

#define N_BASE_NEURONS  3				  // ReLu, tanH, cos (or modulo X)
#define MAX_CHILDREN_PER_BLOCK  10
#define MAX_BLOCK_INPUT_SIZE  10          // Does not apply to the top one, which is the network itself
#define MAX_BLOCK_OUTPUT_SIZE  10         // Does not apply to the top one, which is the network itself

// In the genotype, denotes the fact that a child is connected
// to the input 
#define INPUT_ID -1     

#include "Random.h"

inline float ReLU(float x) { return x > 0 ? x : 0; }

GenotypeConnexion::GenotypeConnexion(int oID, int dID, int nLines, int nColumns, bool zeroInit) :
	originID(oID), destinationID(dID), nLines(nLines), nColumns(nColumns)
{
	alpha = new float[nLines * nColumns];
	eta = new float[nLines * nColumns];
	w = new float[nLines * nColumns];
	A = new float[nLines * nColumns];
	B = new float[nLines * nColumns];
	C = new float[nLines * nColumns];

	if (zeroInit) {
		for (int i = 0; i < nLines * nColumns; i++) {
			alpha[i] = 0.0f;
			eta[i] = .8f;
			w[i] = 0.0f;
			A[i] = 0.0f;
			B[i] = 0.0f ;
			C[i] = 0.0f;
		}
	} else {
		for (int i = 0; i < nLines * nColumns; i++) {
			alpha[i] = NORMAL_01;
			eta[i] = UNIFORM_01 *.5f + .5f;
			w[i] = NORMAL_01;
			A[i] = NORMAL_01;
			B[i] = NORMAL_01 * .2f;
			C[i] = NORMAL_01 *.2f;
		}
	}
}

void GenotypeNode::computeBeacons() {
	concatenatedChildrenInputBeacons.resize(children.size() + 1);
	concatenatedChildrenInputBeacons[0] = 0;
	int s = 0;
	for (int i = 0; i < children.size(); i++) {
		s += children[i]->inputSize;
		concatenatedChildrenInputBeacons[i + 1] = s;
	}
	concatenatedChildrenInputLength = s;
}

void GenotypeNode::mutateFloats() {
	int rID, listID, matrixID; 
	const float pMutation = .3f; // TODO
	const float K = .2f;
	float r;

	// Mutate int(6*Pmutation*nParam) parameters in the inter-children connexions.

	std::vector<int> ids;
	int l = 0;
	for (int i = 0; i < childrenConnexions.size(); i++) {
		ids.push_back(l);
		l += childrenConnexions[i].nLines * childrenConnexions[i].nColumns * 6;
	}
	ids.push_back(l);

	float _nParams = (float)l;
	int _nMutations = (int)std::floor(_nParams * pMutation);

	for (int i = 0; i < _nMutations; i++) {
		rID = (int) (UNIFORM_01 * (float) _nParams);

		int j = 0;
		while (rID >= ids[j+1]) { j++; }
		listID = j;
		j = (rID - ids[listID]) % 6;
		matrixID = (rID - listID - j) / 6;
		
		r = .2f * NORMAL_01;
		switch (j) {
		case 0: childrenConnexions[listID].A[matrixID] += r;
		case 1: childrenConnexions[listID].B[matrixID] += r;
		case 2: childrenConnexions[listID].C[matrixID] += r;
		case 3: childrenConnexions[listID].alpha[matrixID] += r;
		case 4: childrenConnexions[listID].w[matrixID] += r;
		case 5: 
			float eta = childrenConnexions[listID].eta[matrixID];
			childrenConnexions[listID].eta[matrixID] += r>0?
				 K * eta * (1 - eta) :
				-K * eta * (1 - eta) ;
		}
	}

	for (int i = 0; i < outputSize; i++) {
		r = .2f * NORMAL_01;
		bias[i] += r;
		wNeuromodulation[i] += r;
	}

	r = .2f * NORMAL_01;
	neuromodulationBias += r;
}

void GenotypeNode::connect() {
	if (children.size() == 0) return;
	int c1, c2;
	// this implementation makes it less likely to gain connexions when many are already populated
	const int maxAttempts = 10; 
	bool alreadyExists;
	for (int i = 0; i < maxAttempts; i++) {
		alreadyExists = false;
		c1 = (int) (UNIFORM_01 * (float) (children.size()+1)) - 1; // in [-1, children.size() - 1]
		c2 = (c1 + 2 + (int)(UNIFORM_01 * (float)children.size())) % children.size(); //guarantees c1 != c2. in [0, children.size()]
		if (c1 == -1) c1 = INPUT_ID; 
		
		for (const GenotypeConnexion c : childrenConnexions) {
			if (c.originID == c1 && c.destinationID == c2) {
				alreadyExists = true;
				break;
			}
		}

		if (alreadyExists) continue;

		// zeroInit is set to true to minimize disturbance of the network
		childrenConnexions.emplace_back(c1, c2, children[c2]->inputSize, children[c1]->outputSize, true);
		break;
	}
}
void GenotypeNode::disconnect() {
	if (childrenConnexions.size() == 0) return;
	if (childrenConnexions.size() == 1 && children.size() == 0) return;
	int id = (int)(UNIFORM_01 * (float) childrenConnexions.size());
	childrenConnexions.erase(childrenConnexions.begin() + id);
}

void GenotypeNode::incrementInputSize() {
	inputSize++;
	for (int i = 0; i < childrenConnexions.size(); i++) {
		if (childrenConnexions[i].originID == INPUT_ID) {
			incrementDestinationInputSize(i);
		}
	}
}
void GenotypeNode::onChildInputSizeIncremented(GenotypeNode* modifiedType) {
	for (int i = 0; i < childrenConnexions.size(); i++) {
		if (children[childrenConnexions[i].destinationID] == modifiedType) {
			incrementDestinationInputSize(i);
		}
	}
}
void GenotypeNode::incrementDestinationInputSize(int i) {
	GenotypeConnexion newConnexion = GenotypeConnexion(
		childrenConnexions[i].originID,
		childrenConnexions[i].destinationID,
		childrenConnexions[i].nLines,
		childrenConnexions[i].nColumns + 1
	);

	int idNew=0, idOld=0;
	for (int j = 0; j < childrenConnexions[i].nLines; j++) {
		for (int k = 0; k < childrenConnexions[i].nColumns; k++) {

			newConnexion.A[idNew] = childrenConnexions[i].A[idOld];
			newConnexion.B[idNew] = childrenConnexions[i].B[idOld];
			newConnexion.C[idNew] = childrenConnexions[i].C[idOld];
			newConnexion.alpha[idNew] = childrenConnexions[i].alpha[idOld];
			newConnexion.eta[idNew] = childrenConnexions[i].eta[idOld];
			newConnexion.w[idNew] = childrenConnexions[i].w[idOld];

			idNew++;
			idOld++;
		}

		newConnexion.A[idNew] = 0.0f;
		newConnexion.B[idNew] = 0.0f;
		newConnexion.C[idNew] = 0.0f;
		newConnexion.alpha[idNew] = 0.0f;
		newConnexion.eta[idNew] = 0.0f;
		newConnexion.w[idNew] = 0.0f;
		idNew++;
	}

	childrenConnexions[i] = newConnexion;
}


void GenotypeNode::incrementOutputSize(){
	outputSize++;
	for (int i = 0; i < childrenConnexions.size(); i++) {
		if (childrenConnexions[i].destinationID == children.size()) {
			incrementOriginOutputSize(i);
		}
	}
	wNeuromodulation.push_back(0);
}
void GenotypeNode::onChildOutputSizeIncremented(GenotypeNode* modifiedType) {
	for (int i = 0; i < childrenConnexions.size(); i++) {
		if (children[childrenConnexions[i].originID] == modifiedType) {
			incrementOriginOutputSize(i);
		}
	}
}
void GenotypeNode::incrementOriginOutputSize(int i) {
	GenotypeConnexion newConnexion = GenotypeConnexion(
		childrenConnexions[i].originID,
		childrenConnexions[i].destinationID,
		childrenConnexions[i].nLines + 1,
		childrenConnexions[i].nColumns
	);

	int idNew=0, idOld=0;
	for (int j = 0; j < childrenConnexions[i].nLines; j++) {
		for (int k = 0; k < childrenConnexions[i].nColumns; k++) {

			newConnexion.A[idNew] = childrenConnexions[i].A[idOld];
			newConnexion.B[idNew] = childrenConnexions[i].B[idOld];
			newConnexion.C[idNew] = childrenConnexions[i].C[idOld];
			newConnexion.alpha[idNew] = childrenConnexions[i].alpha[idOld];
			newConnexion.eta[idNew] = childrenConnexions[i].eta[idOld];
			newConnexion.w[idNew] = childrenConnexions[i].w[idOld];

			idNew++;
			idOld++;
		}
	}
	for (int k = 0; k < childrenConnexions[i].nColumns; k++) {
		newConnexion.A[idNew] = 0.0f;
		newConnexion.B[idNew] = 0.0f;
		newConnexion.C[idNew] = 0.0f;
		newConnexion.alpha[idNew] = 0.0f;
		newConnexion.eta[idNew] = 0.0f;
		newConnexion.w[idNew] = 0.0f;
		idNew++;
	}
	childrenConnexions[i] = newConnexion;
}


void GenotypeNode::decrementInputSize(int id){
	if (inputSize == 1) return;
	inputSize--;

	for (int i = 0; i < childrenConnexions.size(); i++) {
		if (childrenConnexions[i].originID == INPUT_ID) {
			decrementDestinationInputSize(i, id);
		}
	}
}
void GenotypeNode::onChildInputSizeDecremented(GenotypeNode* modifiedType, int id) {
	for (int i = 0; i < childrenConnexions.size(); i++) {
		if (children[childrenConnexions[i].destinationID] == modifiedType) {
			decrementDestinationInputSize(i, id);
		}
	}
}
void GenotypeNode::decrementDestinationInputSize(int i, int id) {
	GenotypeConnexion newConnexion = GenotypeConnexion(
		childrenConnexions[i].originID,
		childrenConnexions[i].destinationID,
		childrenConnexions[i].nLines,
		childrenConnexions[i].nColumns - 1
	);

	int idNew=0, idOld=0;
	for (int j = 0; j < childrenConnexions[i].nLines; j++) {
		for (int k = 0; k < childrenConnexions[i].nColumns - 1; k++) {

			if (k == id) {
				idOld++;
				continue;
			}
			newConnexion.A[idNew] = childrenConnexions[i].A[idOld];
			newConnexion.B[idNew] = childrenConnexions[i].B[idOld];
			newConnexion.C[idNew] = childrenConnexions[i].C[idOld];
			newConnexion.alpha[idNew] = childrenConnexions[i].alpha[idOld];
			newConnexion.eta[idNew] = childrenConnexions[i].eta[idOld];
			newConnexion.w[idNew] = childrenConnexions[i].w[idOld];

			idNew++;
			idOld++;
		}
	}

	childrenConnexions[i] = newConnexion;
}


void GenotypeNode::decrementOutputSize(int id){
	if (outputSize == 1) return;
	outputSize--;

	for (int i = 0; i < childrenConnexions.size(); i++) {
		if (childrenConnexions[i].originID == childrenConnexions.size()) {
			decrementDestinationInputSize(i, id);
		}
	}

	wNeuromodulation.erase(wNeuromodulation.begin() + id);
}
void GenotypeNode::onChildOutputSizeDecremented(GenotypeNode* modifiedType, int id){
	for (int i = 0; i < childrenConnexions.size(); i++) {
		if (children[childrenConnexions[i].destinationID] == modifiedType) {
			decrementDestinationInputSize(i, id);
		}
	}
}
void GenotypeNode::decrementOriginOutputSize(int i, int id) {
	GenotypeConnexion newConnexion = GenotypeConnexion(
		childrenConnexions[i].originID,
		childrenConnexions[i].destinationID,
		childrenConnexions[i].nLines - 1,
		childrenConnexions[i].nColumns
	);

	int idNew=0, idOld=0;
	for (int j = 0; j < childrenConnexions[i].nLines; j++) {
		if (j == id) {
			idOld += childrenConnexions[i].nColumns;
			continue;
		}
		for (int k = 0; k < childrenConnexions[i].nColumns - 1; k++) {
			newConnexion.A[idNew] = childrenConnexions[i].A[idOld];
			newConnexion.B[idNew] = childrenConnexions[i].B[idOld];
			newConnexion.C[idNew] = childrenConnexions[i].C[idOld];
			newConnexion.alpha[idNew] = childrenConnexions[i].alpha[idOld];
			newConnexion.eta[idNew] = childrenConnexions[i].eta[idOld];
			newConnexion.w[idNew] = childrenConnexions[i].w[idOld];

			idNew++;
			idOld++;
		}
	}

	childrenConnexions[i] = newConnexion;
}


void PhenotypeNode::forward(float* input) {
	float* _childInputs = new float[type->concatenatedChildrenInputLength + type->outputSize];
	for (int i = 0; i < type->concatenatedChildrenInputLength + type->outputSize; i++) _childInputs[i] = 0;
	
	
	// propagate the previous steps's outputs, by iterating over the connexions between children

	int i0, originID, destinationID;  // for readability, to be optimized away
	for (int id = 0; id < childrenConnexions.size(); id++) {
		originID = type->childrenConnexions[id].originID;
		destinationID = type->childrenConnexions[id].destinationID;
		i0 = type->concatenatedChildrenInputBeacons[destinationID];

		int nl = type->childrenConnexions[id].nLines;
		int nc = type->childrenConnexions[id].nColumns;
		if (originID == INPUT_ID) {
			for (int i = 0; i < nl; i++) {
				for (int j = 0; j < nc; j++) {
					// += (H * alpha + w) * input
					_childInputs[i0 + i] += (childrenConnexions[id].H[i*nc+j] * type->childrenConnexions[id].alpha[i*nc+j]
						+ type->childrenConnexions[id].w[i*nc+j])
						* input[j];
				}
			}
		}
		else {
			for (int i = 0; i < nl; i++) {
				for (int j = 0; j < nc; j++) {
					// += (H * alpha + w) * prevAct
					_childInputs[i0 + i] += (childrenConnexions[id].H[i*nc+j] * type->childrenConnexions[id].alpha[i*nc+j]
						+ type->childrenConnexions[id].w[i*nc+j])
						* children[originID]->previousOutput[j];
				}
			}
		}	
	}
	
	// apply children's forwards:

	int _childID = 0;
	int _inputListID = 0;
	for (PhenotypeNode* child : children) {
		child->neuromodulatorySignal = this->neuromodulatorySignal;

		// Depending on the child's nature, we have 2 cases:
		//  - the child is a bloc
		//  - the child is a simple neuron

		if (child->type->isSimpleNeuron) {
			child->previousOutput[0] = child->currentOutput[0];
			child->currentOutput[0] = child->type->f(_childInputs[_inputListID]);
		} else {						
			child->forward(&_childInputs[_inputListID]);
		}

		_inputListID += child->type->inputSize;
		_childID++;
	}

	// process this node's output, stored in the input of the virtual output node:
	previousOutput.assign(currentOutput.begin(), currentOutput.end()); // save the previous activations
	for (int i = 0; i < type->outputSize; i++) {
		currentOutput[i] = tanh(_childInputs[_inputListID + i] + type->bias[i]);
	}

	// compute the neuromodulatory output
	float temp = type->neuromodulationBias;
	for (int i = 0; i < type->outputSize; i++) {
		temp += type->wNeuromodulation[i] * currentOutput[i];
	}
	neuromodulatorySignal *= 1.41f * tanh(temp); 
	


	// Update hebbian and eligibility traces
	// int originID, destinationID;  // For readability only, compiler will optimize them away
	for (int id = 0; id < childrenConnexions.size(); id++) {
		originID = type->childrenConnexions[id].originID;
		destinationID = type->childrenConnexions[id].destinationID;

		
		float eta, A, B, C, yi, yj; // For readability only, compiler will optimize them away
		int nl = type->childrenConnexions[id].nLines;
		int nc = type->childrenConnexions[id].nColumns;
		for (int i = 0; i < nl; i++) {
			for (int j = 0; j < nc; j++) {
				eta = type->childrenConnexions[id].eta[i*nc+j];
				A = type->childrenConnexions[id].A[i*nc+j];
				B = type->childrenConnexions[id].B[i*nc+j];
				C = type->childrenConnexions[id].C[i*nc+j];
				yi = destinationID != children.size() ?
					children[destinationID]->currentOutput[i] :
					currentOutput[i];
				yj = originID != INPUT_ID ? 
					children[originID]->previousOutput[j] :
					previousInput[j];

				childrenConnexions[id].E[i*nc+j] = (1 - eta) * childrenConnexions[id].E[i*nc+j]
					+ eta * (A * yi * yj + B * yi + C * yj);

				childrenConnexions[id].H[i*nc+j] += childrenConnexions[id].E[i*nc+j] * neuromodulatorySignal;
				childrenConnexions[id].H[i*nc+j] = std::max(-1.0f, std::min(childrenConnexions[id].H[i*nc+j], 1.0f));
			}
		}
	}

	for (int i = 0; i < type->inputSize; i++) {
		previousInput[i] = input[i];
	}
	delete[] _childInputs;
}

// TODO , using boost serialize.
void Network::save(std::string path) {

}

Network::Network(Network* n) {
	nSimpleNeurons = n->nSimpleNeurons;
	inputSize = n->inputSize; 
	outputSize = n->outputSize;

	genome.resize(n->genome.size());
	for (int i = 0; i < nSimpleNeurons; i++) {
		genome[i].isSimpleNeuron = true;
		genome[i].f = n->genome[i].f; 
		genome[i].inputSize = n->genome[i].inputSize;
		genome[i].outputSize = n->genome[i].outputSize; 
		genome[i].bias.resize(0);   
		genome[i].children.reserve(0);
		genome[i].childrenConnexions.reserve(0);
		genome[i].wNeuromodulation.reserve(0);
		genome[i].neuromodulationBias=0.0f;
		genome[i].concatenatedChildrenInputLength = 0;
		genome[i].concatenatedChildrenInputBeacons.reserve(0);
	}


	for (int i = nSimpleNeurons; i < n->genome.size(); i++) {
		genome[i].isSimpleNeuron = false;
		genome[i].f = NULL;
		genome[i].inputSize = n->genome[i].inputSize;
		genome[i].outputSize = n->genome[i].outputSize;
		genome[i].neuromodulationBias = n->genome[i].neuromodulationBias;
		genome[i].concatenatedChildrenInputLength = n->genome[i].concatenatedChildrenInputLength;

		genome[i].bias.assign(n->genome[i].bias.begin(), n->genome[i].bias.end());
		genome[i].wNeuromodulation.assign(n->genome[i].wNeuromodulation.begin(), n->genome[i].wNeuromodulation.end());
		genome[i].concatenatedChildrenInputBeacons.assign(n->genome[i].concatenatedChildrenInputBeacons.begin(), n->genome[i].concatenatedChildrenInputBeacons.end());

		genome[i].children.resize(n->genome[i].children.size());
		for (int j = 0; j < n->genome[i].children.size(); j++) { // TODO problematic if pointers are not equally spaced
			genome[i].children[j] = &genome[0] + (n->genome[i].children[j] - &n->genome[0]);
		}

		genome[i].childrenConnexions.resize(n->genome[i].childrenConnexions.size());
		for (int j = 0; j < n->genome[i].childrenConnexions.size(); j++) {
			genome[i].childrenConnexions[j].destinationID = n->genome[i].childrenConnexions[j].destinationID;
			genome[i].childrenConnexions[j].originID = n->genome[i].childrenConnexions[j].originID;
			genome[i].childrenConnexions[j].nLines = n->genome[i].childrenConnexions[j].nLines;
			genome[i].childrenConnexions[j].nColumns = n->genome[i].childrenConnexions[j].nColumns;

			int s = genome[i].childrenConnexions[j].nLines * genome[i].childrenConnexions[j].nColumns;
			genome[i].childrenConnexions[j].eta = new float[s];
			genome[i].childrenConnexions[j].alpha = new float[s];
			genome[i].childrenConnexions[j].A = new float[s];
			genome[i].childrenConnexions[j].B = new float[s];
			genome[i].childrenConnexions[j].C = new float[s];
			genome[i].childrenConnexions[j].w = new float[s];

			memcpy(genome[i].childrenConnexions[j].eta, n->genome[i].childrenConnexions[j].eta, sizeof(float) * s);
			memcpy(genome[i].childrenConnexions[j].alpha, n->genome[i].childrenConnexions[j].alpha, sizeof(float) * s);
			memcpy(genome[i].childrenConnexions[j].A, n->genome[i].childrenConnexions[j].A, sizeof(float) * s);
			memcpy(genome[i].childrenConnexions[j].B, n->genome[i].childrenConnexions[j].B, sizeof(float) * s);
			memcpy(genome[i].childrenConnexions[j].C, n->genome[i].childrenConnexions[j].C, sizeof(float) * s);
			memcpy(genome[i].childrenConnexions[j].w, n->genome[i].childrenConnexions[j].w, sizeof(float) * s);
		}
	}

	topNodeP = NULL;
}

Network::Network(int inputSize, int outputSize) :
inputSize(inputSize), outputSize(outputSize)
{
	nSimpleNeurons = 2;
	genome.resize(nSimpleNeurons + 1);

	int i = 0;
	// 0: TanH 
	{
		genome[i].isSimpleNeuron = true;
		genome[i].inputSize = 1;
		genome[i].outputSize = 1;
		genome[i].f = *tanhf;
		genome[i].bias.resize(0);
		genome[i].children.reserve(0);
		genome[i].childrenConnexions.reserve(0);
		genome[i].wNeuromodulation.reserve(0);
		genome[i].neuromodulationBias = 0.0f;
		genome[i].concatenatedChildrenInputLength = 0;
		genome[i].concatenatedChildrenInputBeacons.reserve(0);
	}

	i++;
	// 1: ReLU
	{
		genome[i].isSimpleNeuron = true;
		genome[i].inputSize = 1;
		genome[i].outputSize = 1;
		genome[i].f = *ReLU;
		genome[i].bias.resize(0);
		genome[i].children.reserve(0);
		genome[i].childrenConnexions.reserve(0);
		genome[i].wNeuromodulation.reserve(0);
		genome[i].neuromodulationBias = 0.0f;
		genome[i].concatenatedChildrenInputLength = 0;
		genome[i].concatenatedChildrenInputBeacons.reserve(0);
	}

	i++;
	// 3: The initial top node, 0 initialised
	{
		genome[i].isSimpleNeuron = false;
		genome[i].inputSize = inputSize;
		genome[i].outputSize = outputSize;
		genome[i].concatenatedChildrenInputLength = outputSize;
		genome[i].f = NULL;
		genome[i].bias.resize(outputSize); 
		genome[i].children.resize(0); 
		genome[i].childrenConnexions.resize(0);
		genome[i].childrenConnexions.emplace_back(
			INPUT_ID, genome[i].children.size(), inputSize, outputSize
		);
		genome[i].neuromodulationBias = 0.0f;
		genome[i].wNeuromodulation.resize(outputSize);
		genome[i].computeBeacons();
	}

	topNodeP = new PhenotypeNode(&genome[nSimpleNeurons]); 

}

Network::~Network() {
   	delete topNodeP;
}

std::vector<float> Network::getOutput() {
	return topNodeP->currentOutput;
}

void Network::step(std::vector<float> obs) {
	topNodeP->neuromodulatorySignal = 1.0f; // other nodes have it set by their parent
	topNodeP->forward(&obs[0]);
}

void Network::mutate() {
	delete topNodeP;
	float r;

	// Floating point mutations.
	for (int i = nSimpleNeurons; i < genome.size(); i++) {
		genome[i].mutateFloats();
	}
	
	// Children interconnection mutations.
	for (int i = nSimpleNeurons; i < genome.size(); i++) {
		r = UNIFORM_01;
		if (r < .001) genome[i].disconnect();
		r = UNIFORM_01;
		if (r < .01) genome[i].connect();
	}
	/*
	// Input and output sizes mutations.
	for (int i = nSimpleNeurons; i < genome.size() - 1; i++) {
		r = UNIFORM_01;
		if (r < .002) {
			genome[i].incrementInputSize();
			for (int j = nSimpleNeurons; j < i; j++) {
				genome[j].onChildInputSizeIncremented(&genome[i]);
			}
		}

		r = UNIFORM_01;
		if (r < .002) {
			genome[i].incrementOutputSize();
			for (int j = nSimpleNeurons; j < i; j++) {
				genome[j].onChildOutputSizeIncremented(&genome[i]);
			}
		}

		int rID;
		r = UNIFORM_01;
		if (r < .0003) {
			rID = (int) (UNIFORM_01 * (float)inputSize);
			genome[i].decrementInputSize(rID);
			for (int j = i+1; j < genome.size(); j++) {
				genome[j].onChildInputSizeDecremented(&genome[i], rID);
			}
		}

		r = UNIFORM_01;
		if (r < .0003) {
			rID = (int)(UNIFORM_01 * (float)outputSize);
			genome[i].decrementOutputSize(rID); 
			for (int j = i+1; j < genome.size(); j++) {
				genome[j].onChildOutputSizeDecremented(&genome[i], rID);
			}
		}
	}
	*/
	// TODO 's:

	// Add child 
	// Remove child
	// Top Node Boxing
	// Duplication
	// Child replacement

	
	// Update beacons for forward().
	for (int i = nSimpleNeurons; i < genome.size(); i++) {
		genome[i].computeBeacons();
	}
	topNodeP = new PhenotypeNode(&genome[genome.size() - 1]);
}

void Network::intertrialReset() {
	topNodeP->zero();
}

float Network::getRegularizationLoss() {
	std::vector<int> nParams(genome.size());
	for (int i = nSimpleNeurons; i < genome.size(); i++) {
		nParams[i] = 0;
		for (int j = 0; j < genome[i].childrenConnexions.size(); j++) {
			nParams[i] += genome[i].childrenConnexions[j].nLines * genome[i].childrenConnexions[j].nColumns;
		}
		for (int j = 0; j < genome[i].children.size(); j++) {
			// TODO can vector elements be non contiguous ? The following line could be problematic.
			nParams[i] += nParams[(int) (genome[i].children[j] - &genome[0])]; 
		}
	}
	return nParams[genome.size()-1];
}

