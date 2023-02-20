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
	int s = nLines * nColumns;
	eta = std::make_unique<float[]>(s);
	A = std::make_unique<float[]>(s);
	B = std::make_unique<float[]>(s);
	C = std::make_unique<float[]>(s);

#if defined RISI_NAJARRO_2020
	D = std::make_unique<float[]>(s);
#elif defined USING_NEUROMODULATION
	alpha = std::make_unique<float[]>(s);
	w = std::make_unique<float[]>(s);
#endif 

	if (zeroInit) {
		for (int i = 0; i < nLines * nColumns; i++) {
			eta[i] = .8f;
			A[i] = 0.0f;
			B[i] = 0.0f ;
			C[i] = 0.0f;

#if defined RISI_NAJARRO_2020
			D[i] = 0.0f;
#elif defined USING_NEUROMODULATION
			alpha[i] = 0.0f;
			w[i] = 0.0f;
#endif 
		}
	} else {
		for (int i = 0; i < nLines * nColumns; i++) {

#if defined RISI_NAJARRO_2020
			A[i] = UNIFORM_01;
			B[i] = UNIFORM_01;
			C[i] = UNIFORM_01;
			D[i] = UNIFORM_01;
			eta[i] = UNIFORM_01;
#elif defined USING_NEUROMODULATION
			A[i] = NORMAL_01;
			B[i] = NORMAL_01 * .2f;
			C[i] = NORMAL_01 * .2f;
			alpha[i] = NORMAL_01;
			eta[i] = UNIFORM_01 * .5f + .5f;
			w[i] = NORMAL_01;
#endif 
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
	float r;

#if defined RISI_NAJARRO_2020
	constexpr int nArrays = 5;
#elif defined USING_NEUROMODULATION
	constexpr int nArrays = 6;
#endif 
	
	// Mutate int(nArrays*Pmutation*nParam) parameters in the inter-children connexions.

	std::vector<int> ids;
	int l = 0;
	for (int i = 0; i < childrenConnexions.size(); i++) {
		ids.push_back(l);
		l += childrenConnexions[i].nLines * childrenConnexions[i].nColumns * nArrays;
	}
	ids.push_back(l);

	float _nParams = (float)l;
	int _nMutations = (int)std::floor(_nParams * pMutation);

	for (int i = 0; i < _nMutations; i++) {
		rID = (int) (UNIFORM_01 * (float) _nParams);

		int j = 0;
		while (rID >= ids[j+1]) { j++; }
		listID = j;
		j = (rID - ids[listID]) % nArrays;
		matrixID = (rID - listID - j) / nArrays;
		
		r = .2f * NORMAL_01;

#if defined RISI_NAJARRO_2020
		switch (j) {
		case 0: childrenConnexions[listID].A[matrixID] += r;
		case 1: childrenConnexions[listID].B[matrixID] += r;
		case 2: childrenConnexions[listID].C[matrixID] += r;
		case 3: childrenConnexions[listID].D[matrixID] += r;
		case 4:
			float eta = childrenConnexions[listID].eta[matrixID];
			childrenConnexions[listID].eta[matrixID] += r * .3f * eta * (1 - eta); // guarantees eta in [0,1] and slower variations around
		}
#elif defined USING_NEUROMODULATION
		switch (j) {
		case 0: childrenConnexions[listID].A[matrixID] += r;
		case 1: childrenConnexions[listID].B[matrixID] += r;
		case 2: childrenConnexions[listID].C[matrixID] += r;
		case 3: childrenConnexions[listID].alpha[matrixID] += r;
		case 4: childrenConnexions[listID].w[matrixID] += r;
		case 5:
			float eta = childrenConnexions[listID].eta[matrixID];
			childrenConnexions[listID].eta[matrixID] += r > 0 ?
				K * eta * (1 - eta) :
				-K * eta * (1 - eta);
	}
#endif 
	}

	for (int i = 0; i < outputSize; i++) {
		r = .2f * NORMAL_01;
		bias[i] += r;
	}

#ifdef USING_NEUROMODULATION
	for (int i = 0; i < outputSize; i++) {
		r = .2f * NORMAL_01;
		wNeuromodulation[i] += r;
	}
	r = .2f * NORMAL_01;
	neuromodulationBias += r;
#endif 
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
		
		for (int i = 0; i < childrenConnexions.size(); i++) {
			if (childrenConnexions[i].originID == c1 && childrenConnexions[i].destinationID == c2) {
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

void GenotypeNode::addChild(GenotypeNode* child) {

	// Previously, the output node was at the children.size() position (virtually). It is now being shifted right 1 slot,
	// so the destination IDs of the connexions must be updated when their destination is the output.
	for (int i = 0; i < childrenConnexions.size(); i++) {
		if (childrenConnexions[i].destinationID == children.size()) {
			childrenConnexions[i].destinationID++;
		}
	}

	// There is a bias towards the output node being chosen as dID, and the input as oID. More details in the technical notes.
	constexpr float parallelBias = 2.0f;
	int oID, dID;
	float r;
	int destinationsInputSize, originOutputSize;

	r = UNIFORM_01;
	oID = (int)((float)(children.size() + 1) * parallelBias * r);
	if (oID >= children.size()) { // the incoming connexion comes from the parent's input
		oID = INPUT_ID;
		originOutputSize = inputSize;
	} else { 
		originOutputSize = children[oID]->outputSize;
	}

	r = UNIFORM_01;
	dID = (int)((float)(children.size() + 1) * parallelBias * r);
	if (dID >= children.size()) { // the outgoing connexion goes to the parent's output
		dID = children.size()+1;
		destinationsInputSize = outputSize;
	} else { 
		destinationsInputSize = children[dID]->inputSize;
	}

	childrenConnexions.emplace_back(oID,   children.size(), child->inputSize,      originOutputSize,  true);
	childrenConnexions.emplace_back(children.size(),   dID, destinationsInputSize, child->outputSize, true);
	children.push_back(child);
	updateDepth();
}
void GenotypeNode::removeChild() {
	int rID = (int)(UNIFORM_01 * (float)children.size());
	children.erase(children.begin() + rID);

	// Erase connexions that lead to the removed child. Slow algorithm, but does not matter here.
	int initialSize = childrenConnexions.size();
	int nRemovals = 0;
	int i = 0;
	while (i < initialSize - nRemovals) {
		if (childrenConnexions[i].destinationID == rID || childrenConnexions[i].originID == rID) {
			childrenConnexions.erase(childrenConnexions.begin() + i);
			i--;
		}
		i++;
	}

	// Update the IDs in the connexions
	for (int i = 0; i < childrenConnexions.size(); i++) {
		if (childrenConnexions[i].destinationID >= rID) {
			childrenConnexions[i].destinationID--;
		}
		if (childrenConnexions[i].originID >= rID) {
			childrenConnexions[i].originID--;
		}
	}

	updateDepth();
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
			newConnexion.eta[idNew] = childrenConnexions[i].eta[idOld];
#if defined RISI_NAJARRO_2020
			newConnexion.D[idNew] = childrenConnexions[i].D[idOld];
#elif defined USING_NEUROMODULATION
			newConnexion.w[idNew] = childrenConnexions[i].w[idOld];
			newConnexion.alpha[idNew] = childrenConnexions[i].alpha[idOld];
#endif 

			idNew++;
			idOld++;
		}

		newConnexion.A[idNew] = 0.0f;
		newConnexion.B[idNew] = 0.0f;
		newConnexion.C[idNew] = 0.0f;
		newConnexion.eta[idNew] = 0.0f;
#if defined RISI_NAJARRO_2020
		newConnexion.D[idNew] = 0.0f;
#elif defined USING_NEUROMODULATION
		newConnexion.alpha[idNew] = 0.0f;
		newConnexion.w[idNew] = 0.0f;
#endif 

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
	bias.push_back(0);
#ifdef USING_NEUROMODULATION
	wNeuromodulation.push_back(0);
#endif 
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
			newConnexion.eta[idNew] = childrenConnexions[i].eta[idOld];
#if defined RISI_NAJARRO_2020
			newConnexion.D[idNew] = childrenConnexions[i].D[idOld];
#elif defined USING_NEUROMODULATION
			newConnexion.w[idNew] = childrenConnexions[i].w[idOld];
			newConnexion.alpha[idNew] = childrenConnexions[i].alpha[idOld];
#endif 

			idNew++;
			idOld++;
		}
	}
	for (int k = 0; k < childrenConnexions[i].nColumns; k++) {
		newConnexion.A[idNew] = 0.0f;
		newConnexion.B[idNew] = 0.0f;
		newConnexion.C[idNew] = 0.0f;
		newConnexion.eta[idNew] = 0.0f;
#if defined RISI_NAJARRO_2020
		newConnexion.D[idNew] = 0.0f;
#elif defined USING_NEUROMODULATION
		newConnexion.alpha[idNew] = 0.0f;
		newConnexion.w[idNew] = 0.0f;
#endif 
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
			newConnexion.eta[idNew] = childrenConnexions[i].eta[idOld];
#if defined RISI_NAJARRO_2020
			newConnexion.D[idNew] = childrenConnexions[i].D[idOld];
#elif defined USING_NEUROMODULATION
			newConnexion.w[idNew] = childrenConnexions[i].w[idOld];
			newConnexion.alpha[idNew] = childrenConnexions[i].alpha[idOld];
#endif 

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

#ifdef USING_NEUROMODULATION
	wNeuromodulation.erase(wNeuromodulation.begin() + id);
#endif 
	bias.erase(bias.begin() + id);

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
			newConnexion.eta[idNew] = childrenConnexions[i].eta[idOld];
#if defined RISI_NAJARRO_2020
			newConnexion.D[idNew] = childrenConnexions[i].D[idOld];
#elif defined USING_NEUROMODULATION
			newConnexion.w[idNew] = childrenConnexions[i].w[idOld];
			newConnexion.alpha[idNew] = childrenConnexions[i].alpha[idOld];
#endif 

			idNew++;
			idOld++;
		}
	}

	childrenConnexions[i] = newConnexion;
}

void GenotypeNode::updateDepth() {
	int dmax = 0;
	for (int i = 0; i < children.size(); i++) {
		if (children[i]->depth > dmax) dmax = children[i]->depth;
	}
	depth = dmax + 1;
}

void PhenotypeNode::forward(float* input) {
	float* _childInputs = new float[type->concatenatedChildrenInputLength + type->outputSize];
	for (int i = 0; i < type->concatenatedChildrenInputLength + type->outputSize; i++) _childInputs[i] = 0.0f;
	
	
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

#if defined RISI_NAJARRO_2020
					// The article's w is our H. z += w * yj, y = f(z)
					_childInputs[i0 + i] += childrenConnexions[id].H[i * nc + j] * input[j];
#elif defined USING_NEUROMODULATION
					// += (H * alpha + w) * input
					_childInputs[i0 + i] += (childrenConnexions[id].H[i*nc+j] * type->childrenConnexions[id].alpha[i*nc+j]
						+ type->childrenConnexions[id].w[i*nc+j])
						* input[j];
#endif 
				}
			}
		}
		else {
			for (int i = 0; i < nl; i++) {
				for (int j = 0; j < nc; j++) {

#if defined RISI_NAJARRO_2020
					_childInputs[i0 + i] += childrenConnexions[id].H[i * nc + j] * children[originID]->previousOutput[j];
#elif defined USING_NEUROMODULATION
					// += (H * alpha + w) * prevAct
					_childInputs[i0 + i] += (childrenConnexions[id].H[i * nc + j] * type->childrenConnexions[id].alpha[i * nc + j]
						+ type->childrenConnexions[id].w[i * nc + j])
						* children[originID]->previousOutput[j];
#endif 
				}
			}
		}	
	}
	
	// apply children's forwards:

	int _childID = 0;
	int _inputListID = 0;
	for (const std::unique_ptr<PhenotypeNode>& child : children) {
#ifdef USING_NEUROMODULATION
		child->neuromodulatorySignal = this->neuromodulatorySignal;
#endif 

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

#ifdef RISI_NAJARRO_2020  // TODO experiment. In their feedforward, they do not use a bias.
		currentOutput[i] = tanh(_childInputs[_inputListID + i]);
#else
		currentOutput[i] = tanh(_childInputs[_inputListID + i] + type->bias[i]);
#endif

	}

#ifdef USING_NEUROMODULATION
	// compute the neuromodulatory output
	float temp = type->neuromodulationBias;
	for (int i = 0; i < type->outputSize; i++) {
		temp += type->wNeuromodulation[i] * currentOutput[i];
	}
	neuromodulatorySignal *= 1.41f * tanh(temp); 
#endif	


	// Update hebbian and eligibility traces
	// int originID, destinationID;  // For readability only, compiler will optimize them away
	for (int id = 0; id < childrenConnexions.size(); id++) {
		originID = type->childrenConnexions[id].originID;
		destinationID = type->childrenConnexions[id].destinationID;

		
		float eta, A, B, C, D, yi, yj; // For readability only, compiler will optimize them away
		int nl = type->childrenConnexions[id].nLines;
		int nc = type->childrenConnexions[id].nColumns;
		for (int i = 0; i < nl; i++) {
			for (int j = 0; j < nc; j++) {

				A = type->childrenConnexions[id].A[i * nc + j];
				B = type->childrenConnexions[id].B[i * nc + j];
				C = type->childrenConnexions[id].C[i * nc + j];
				eta = type->childrenConnexions[id].eta[i * nc + j];

#if defined RISI_NAJARRO_2020
				D = type->childrenConnexions[id].D[i * nc + j];
#endif 
				
				yi = destinationID != children.size() ?
					children[destinationID]->currentOutput[i] :
					currentOutput[i];
				yj = originID != INPUT_ID ? 
					children[originID]->previousOutput[j] :
					previousInput[j];

#if defined RISI_NAJARRO_2020
				childrenConnexions[id].H[i * nc + j] += eta * (A * yi * yj + B * yi + C * yj + D);
#elif defined USING_NEUROMODULATION
				childrenConnexions[id].E[i*nc+j] = (1 - eta) * childrenConnexions[id].E[i*nc+j]
					+ eta * (A * yi * yj + B * yi + C * yj);

				childrenConnexions[id].H[i*nc+j] += childrenConnexions[id].E[i*nc+j] * neuromodulatorySignal;
				childrenConnexions[id].H[i*nc+j] = std::max(-1.0f, std::min(childrenConnexions[id].H[i*nc+j], 1.0f));
#endif 
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
	for (int i = 0; i < n->genome.size(); i++) genome[i] = std::make_unique<GenotypeNode>(new GenotypeNode());
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
		genome[i]->neuromodulationBias = 0.0f;
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
		for (int j = 0; j < n->genome[i]->children.size(); j++) { // TODO problematic if pointers are not equally spaced
			genome[i]->children[j] = genome[n->genome[i]->children[j]->position].get();
		}

		genome[i]->childrenConnexions = n->genome[i]->childrenConnexions; // Hopefully vector copy calls copy constructor
	}

	topNodeP = NULL;
}

Network::Network(int inputSize, int outputSize) :
inputSize(inputSize), outputSize(outputSize)
{
	nSimpleNeurons = 2;
	genome.resize(nSimpleNeurons + 1);
	for (int i = 0; i < nSimpleNeurons + 1; i++) genome[i] = std::make_unique< GenotypeNode>(new GenotypeNode());

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
	// 3: The initial top node, 0 initialised
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
			INPUT_ID, genome[i]->children.size(), inputSize, outputSize
		);
#ifdef USING_NEUROMODULATION
		genome[i]->neuromodulationBias = 0.0f;
		genome[i]->wNeuromodulation.resize(outputSize);
#endif 
		genome[i]->computeBeacons();
	}

	topNodeP = std::make_unique<PhenotypeNode>(new PhenotypeNode(genome[nSimpleNeurons].get()));

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
	topNodeP.reset();
	float r;

	// Floating point mutations.
	for (int i = nSimpleNeurons; i < genome.size(); i++) {
		genome[i]->mutateFloats();
	}
	
	// Children interconnection mutations.
	for (int i = nSimpleNeurons; i < genome.size(); i++) {
		r = UNIFORM_01;
		if (r < .001) genome[i]->disconnect();
		r = UNIFORM_01;
		if (r < .01) genome[i]->connect();
	}
	
	// Input and output sizes mutations.
	for (int i = nSimpleNeurons; i < genome.size() - 1; i++) {
		r = UNIFORM_01;
		if (r < .002) {
			genome[i]->incrementInputSize();
			for (int j = nSimpleNeurons; j < i; j++) {
				genome[j]->onChildInputSizeIncremented(genome[i].get());
			}
		}

		r = UNIFORM_01;
		if (r < .002) {
			genome[i]->incrementOutputSize();
			for (int j = nSimpleNeurons; j < i; j++) {
				genome[j]->onChildOutputSizeIncremented(genome[i].get());
			}
		}

		int rID;
		r = UNIFORM_01;
		if (r < .0003) {
			rID = (int) (UNIFORM_01 * (float)inputSize);
			genome[i]->decrementInputSize(rID);
			for (int j = i+1; j < genome.size(); j++) {
				genome[j]->onChildInputSizeDecremented(genome[i].get(), rID);
			}
		}

		r = UNIFORM_01;
		if (r < .0003) {
			rID = (int)(UNIFORM_01 * (float)outputSize);
			genome[i]->decrementOutputSize(rID); 
			for (int j = i+1; j < genome.size(); j++) {
				genome[j]->onChildOutputSizeDecremented(genome[i].get(), rID);
			}
		}
	}
	
	
	// Adding a child node 
	for (int i = nSimpleNeurons; i < genome.size(); i++) {
		r = UNIFORM_01;
		if (r < .005f && genome[i]->children.size() < MAX_CHILDREN_PER_BLOCK) {
			int prevDepth = genome[i]->depth;

			int childID = genome[i]->position;
			while (genome[childID]->depth <= genome[i]->depth && childID < genome.size() - 1) childID++;
			r = UNIFORM_01;
			childID--;
			childID = (int) (r * (float)childID);
			if (childID >= i) childID++; 
			genome[i]->addChild(genome[childID].get());
			
			// if the child's depth was equal to this node's, the depth is incremented by one and in must be moved in the genome vector.
			if (prevDepth < genome[i]->depth && i != genome.size() - 1) {
				int id = genome[i]->position + 1;
				while (genome[id]->depth < genome[i]->depth) id++;
				std::unique_ptr<GenotypeNode> temp = std::move(genome[i]);
				genome.erase(genome.begin() + i); //inefficient, but insignificant here.
				genome.insert(genome.begin() + id - 1, std::move(temp)); // -1 because of the erasure

				// update positions for affected nodes.
				for (int j = i; j < id; j++) genome[j]->position = j;   // TODO check correctness when unbroken mentally
			} 
		}
	}
	
	// Removing a child node
	for (int i = nSimpleNeurons; i < genome.size(); i++) {
		r = UNIFORM_01;
		if (r < .001f && genome[i]->children.size() > 0) {

			int prevDepth = genome[i]->depth;
			genome[i]->removeChild();

			// if the child's depth was this node's - 1, its depth is decreased and it must be moved in the genome vector.
			if (prevDepth > genome[i]->depth && i != genome.size() - 1) {
				int id = genome[i]->position - 1;
				while (genome[id]->depth > genome[i]->depth) id--;
				std::unique_ptr<GenotypeNode> temp = std::move(genome[i]);
				genome.erase(genome.begin() + i); //inefficient, but insignificant here.
				genome.insert(genome.begin() + id, std::move(temp));

				// update positions for affected nodes.
				for (int j = id; j < i+1; j++) genome[j]->position = j;   // TODO check correctness when unbroken mentally
			}
		}
	}

	// TODO 's:

	// Top Node Boxing
	// Duplication
	// Child replacement

	
	// Update beacons for forward().
	for (int i = nSimpleNeurons; i < genome.size(); i++) {
		genome[i]->computeBeacons();
	}
	topNodeP = std::make_unique<PhenotypeNode>(new PhenotypeNode(genome[genome.size() - 1].get()));
}

void Network::intertrialReset() {
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
	return nParams[genome.size()-1];
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
