#include "Network.h"

#define N_BASE_NEURONS  3				  // ReLu, tanH, cos (or modulo X)
#define MAX_CHILDREN_PER_BLOCK  10
#define MAX_BLOCK_INPUT_SIZE  10          // Does not apply to the top one, which is the network itself
#define MAX_BLOCK_OUTPUT_SIZE  10         // Does not apply to the top one, which is the network itself

// In the genotype, denotes the fact that a child is connected
// to the input 
#define INPUT_ID -1     


int g_seed = 10000000;

inline int fastrand() {
	g_seed = (214013 * g_seed + 2531011);
	return (g_seed >> 16) & 0x7FFF;
}


inline float ReLU(float x) { return x > 0 ? x : 0; }

void GenotypeNode::mutateFloats() {
	int rID, listID, matrixID; 
	const float pMutation = .3f; // TODO
	const float K = .2f;
	float r;

	// Mutate int(6*Pmutation*nParam) parameters in the inter-children connexions.

	std::vector<int> ids;
	int l = 0;
	for (const auto c : childrenConnexions) {
		ids.push_back(l);
		l += c.nLines * c.nColumns * 6;
	}
	ids.push_back(l);
	int _nMutations = (int) std::floor((float)l * pMutation);

	for (int i = 0; i < _nMutations; i++) {
		rID = fastrand()%l;

		int j = 1;
		while (rID < ids[j]) { j++; }
		listID = j-1;
		j = (rID - ids[listID]) % 6;
		matrixID = (rID - listID - j) / 6;
		
		r = .2f * (((float)(fastrand() - 16383)) / 16383.0f);
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
		r = .2f * (((float)(fastrand() - 16383)) / 16383.0f);
		bias[i] += r;
		wNeuromodulation[i] += r;
	}

	r = .2f * (((float)(fastrand() - 16383)) / 16383.0f);
	neuromodulationBias += r;

	for (int i = 0; i < children.size(); i++) {
		if (!children[i]->isSimpleNeuron) {
			children[i]->mutateFloats();
		}
	}
}

void GenotypeNode::connect() {
	int c1, c2;
	// this implementation makes it less likely to gain connexions when many are already populated
	const int maxAttempts = 10; 
	bool alreadyExists;
	for (int i = 0; i < maxAttempts; i++) {
		alreadyExists = false;
		c1 = fastrand() % children.size();
		c2 = (c1 + 1 + fastrand() % (children.size()-1) ) % children.size(); //guarantees c1 != c2
		if (fastrand() % children.size() == 0) c1 = INPUT_ID; // so that the input can be linked.
		
		for (const GenotypeConnexion c : childrenConnexions) {
			if (c.originID == c1 && c.destinationID == c2) {
				alreadyExists = true;
				break;
			}
		}

		if (alreadyExists) continue;

		childrenConnexions.emplace_back(c1, c2, children[c2]->inputSize, children[c1]->outputSize);
		break;
	}
}
void GenotypeNode::disconnect() {
	int id = fastrand() % childrenConnexions.size();
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
	float* _childInputs = new float[type->concatenatedChildrenInputLength];
	int l = type->concatenatedChildrenInputLength;
	for (int i = 0; i < l; i++) _childInputs[i] = 0;
	
	
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
			child->currentOutput[0] = child->type->f(_childInputs[_inputListID] + child->type->bias[0]);
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
		genome[i].concatenatedChildrenInputLength = 0;
		genome[i].f = *tanhf;
		genome[i].bias.reserve(1);
		genome[i].bias.resize(1);
		genome[i].bias[0] = 0;
		genome[i].children.reserve(0);
		genome[i].childrenConnexions.reserve(0);
		genome[i].concatenatedChildrenInputBeacons.reserve(0);
	}

	i++;
	// 1: ReLU
	{
		genome[i].isSimpleNeuron = true;
		genome[i].inputSize = 1;
		genome[i].outputSize = 1;
		genome[i].concatenatedChildrenInputLength = 0;
		genome[i].f = *ReLU;
		genome[i].bias.reserve(1);
		genome[i].bias.resize(1);
		genome[i].bias[0] = 0;
		genome[i].children.reserve(0);
		genome[i].childrenConnexions.reserve(0);
		genome[i].concatenatedChildrenInputBeacons.reserve(0);
	}

	i++;
	// 3: The initial top node
	{
		genome[i].isSimpleNeuron = false;
		genome[i].inputSize = inputSize;
		genome[i].outputSize = outputSize;
		genome[i].concatenatedChildrenInputLength = outputSize;
		genome[i].f = NULL;
		genome[i].bias.reserve(outputSize);
		genome[i].bias.resize(outputSize);
		for (int j = 0; j < outputSize; genome[i].bias[j++] = 0);
		genome[i].children.resize(0); 
		genome[i].childrenConnexions.resize(0);
		genome[i].childrenConnexions.emplace_back(
			INPUT_ID, 0, inputSize, outputSize
		);
		genome[i].concatenatedChildrenInputBeacons.resize(1);
		genome[i].concatenatedChildrenInputBeacons[0] = 0;
		genome[i].neuromodulationBias = 0;
		//float* v = new float[inputSize]; ??? TODO understand why this is here
		genome[i].wNeuromodulation.resize(outputSize);
	}

	topNodeP = new PhenotypeNode(&genome[nSimpleNeurons]); 

}

Network::~Network() {
   	delete topNodeP;
}

std::vector<float> Network::step(float* obs) {
	topNodeP->neuromodulatorySignal = 1.0f; // other nodes have it set by their parent
	topNodeP->forward(obs);
	return topNodeP->currentOutput; 
}

void Network::mutate() {
	topNodeP->type->mutateFloats();
	float r;
	for (int i = nSimpleNeurons; i < genome.size(); i++) {
		r = ((float)fastrand()) / 32767.0f;
		if (r < .001) genome[i].disconnect();
		r = ((float)fastrand()) / 32767.0f;
		if (r < .01) genome[i].connect();
	}

	for (int i = nSimpleNeurons; i < genome.size() - 1; i++) {
		r = ((float)fastrand()) / 32767.0f;
		if (r < .002) {
			genome[i].incrementInputSize();
			for (int j = nSimpleNeurons; j < i; j++) {
				genome[j].onChildInputSizeIncremented(&genome[i]);
			}
		}

		r = ((float)fastrand()) / 32767.0f;
		if (r < .002) {
			genome[i].incrementOutputSize();
			for (int j = nSimpleNeurons; j < i; j++) {
				genome[j].onChildOutputSizeIncremented(&genome[i]);
			}
		}

		r = ((float)fastrand()) / 32767.0f;
		if (r < .0003) {
			r = floor((float)fastrand() * inputSize / 32768.0f);
			genome[i].decrementInputSize((int)r);
			for (int j = nSimpleNeurons; j < i; j++) {
				genome[j].onChildInputSizeDecremented(&genome[i], (int)r);
			}
		}

		r = ((float)fastrand()) / 32767.0f;
		if (r < .0003) {
			r = floor((float)fastrand() * inputSize / 32768.0f);
			genome[i].decrementOutputSize((int)r);
			for (int j = nSimpleNeurons; j < i; j++) {
				genome[j].onChildOutputSizeDecremented(&genome[i], (int)r);
			}
		}
	}
}

