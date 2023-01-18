#include "Network.h"

#define N_BASE_NEURONS  3				  // ReLu, tanH, cos (or modulo X)
#define MAX_CHILDREN_PER_BLOCK  10
#define MAX_BLOCK_INPUT_SIZE  10          // Does not apply to the top one, which is the network itself
#define MAX_BLOCK_OUTPUT_SIZE  10         // Does not apply to the top one, which is the network itself

// In the genotype, denotes the fact that a child is connected
// to the input 
#define INPUT_ID -1     

inline float ReLU(float x) { return x > 0 ? x : 0; }

// TODO: neuromodulatory signal.
void PhenotypeNode::forward(float* input) {
	float* _childInputs = new float[type->concatenatedChildrenInputLength];
	int l = type->concatenatedChildrenInputLength;
	for (int i = 0; i < l; i++) _childInputs[i] = 0;
	
	
	// propagate the previous steps's outputs:

	int i0, originID, destinationID;  // for readability, to be optimized away
	for (int id = 0; id < childrenConnexions.size(); id++) {
		originID = type->childrenConnexions[id]->originID;
		destinationID = type->childrenConnexions[id]->destinationID;
		i0 = type->concatenatedChildrenInputBeacons[destinationID];

		int nl = type->childrenConnexions[id]->nLines;
		int nc = type->childrenConnexions[id]->nColumns;
		if (originID == INPUT_ID) {
			for (int i = 0; i < nl; i++) {
				for (int j = 0; j < nc; j++) {
					// += (H * alpha + w) * input
					_childInputs[i0 + i] += (childrenConnexions[id].H[i*nc+j] * type->childrenConnexions[id]->alpha[i*nc+j]
						+ type->childrenConnexions[id]->w[i*nc+j])
						* input[j];
				}
			}
		}
		else {
			for (int i = 0; i < nl; i++) {
				for (int j = 0; j < nc; j++) {
					// += (H * alpha + w) * prevAct
					_childInputs[i0 + i] += (childrenConnexions[id].H[i*nc+j] * type->childrenConnexions[id]->alpha[i*nc+j]
						+ type->childrenConnexions[id]->w[i*nc+j])
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
		if (_childID == children.size() - 1) break;
	}

	// process this node's output, stored in the input of the virtual output node:
	previousOutput.assign(currentOutput.begin(), currentOutput.end()); // save the previous activations
	for (int i = 0; i < type->outputSize - 1; i++) {
		currentOutput[i] = tanh(_childInputs[_inputListID + i] + type->bias[i]);
	}

	neuromodulatorySignal = 1.41f * currentOutput[type->outputSize];
	// neuromodulatorySignal *= currentOutput[type->outputSize]; 
	// *= because it has been set by its parent.


	// Update hebbian and eligibility traces
	// int originID, destinationID;  // For readability only, compiler will optimize them away
	for (int id = 0; id < childrenConnexions.size(); id++) {
		originID = type->childrenConnexions[id]->originID;
		destinationID = type->childrenConnexions[id]->destinationID;

		
		float eta, A, B, C; // For readability only, compiler will optimize them away
		int nl = type->childrenConnexions[id]->nLines;
		int nc = type->childrenConnexions[id]->nColumns;
		for (int i = 0; i < nl; i++) {
			for (int j = 0; j < nc; j++) {
				eta = type->childrenConnexions[id]->eta[i*nc+j];
				A = type->childrenConnexions[id]->A[i*nc+j];
				B = type->childrenConnexions[id]->B[i*nc+j];
				C = type->childrenConnexions[id]->C[i*nc+j];

				childrenConnexions[id].E[i*nc+j] = (1 - eta) * childrenConnexions[id].E[i*nc+j]
					+ eta * (A * children[destinationID]->currentOutput[i] * children[originID]->previousOutput[j]
						+ B * children[destinationID]->currentOutput[i]
						+ C * children[originID]->previousOutput[j]);

				childrenConnexions[id].H[i*nc+j] += childrenConnexions[id].E[i*nc+j] * neuromodulatorySignal;
				childrenConnexions[id].H[i*nc+j] = std::max(-1.0f, std::min(childrenConnexions[id].H[i*nc+j], 1.0f));
			}
		}
	}

	delete[] _childInputs;
}

// TODO , using boost serialize.
void Network::save(std::string path) {

}



Network::Network(int inputSize, int outputSize) :
inputSize(inputSize), outputSize(outputSize)
{
	genome.resize(4);

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
	// 2: Cos
	{
		genome[i].isSimpleNeuron = true;
		genome[i].inputSize = 1;
		genome[i].outputSize = 1;
		genome[i].concatenatedChildrenInputLength = 0;
		genome[i].f = *cosf;
		genome[i].bias.reserve(1);
		genome[i].bias.resize(1);
		genome[i].bias[0] = 0;
		genome[i].children.reserve(0);
		genome[i].childrenConnexions.reserve(0);
		genome[i].concatenatedChildrenInputBeacons.reserve(0);
	}

	i++;
	// 3: The main network
	{
		genome[i].isSimpleNeuron = false;
		genome[i].inputSize = inputSize;
		genome[i].outputSize = outputSize;
		genome[i].concatenatedChildrenInputLength = outputSize;
		genome[i].f = NULL;
		genome[i].bias.reserve(outputSize);
		genome[i].bias.resize(outputSize);
		for (int j = 0; j < outputSize; genome[i].bias[j++] = 0);
		genome[i].children.resize(1); // contains only the virtual output node
		genome[i].childrenConnexions.resize(0);
		genome[i].childrenConnexions.emplace_back(
			new GenotypeConnexion(INPUT_ID, 0, inputSize, outputSize
		));
		genome[i].concatenatedChildrenInputBeacons.resize(1);
		genome[i].concatenatedChildrenInputBeacons[0] = 0;
	}

	topNodeP = PhenotypeNode(&genome[3]);

}

std::vector<float> Network::step(float* obs) {
	topNodeP.forward(obs);
	return topNodeP.currentOutput;
}


