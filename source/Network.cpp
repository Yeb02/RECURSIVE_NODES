#include "Network.h"

#define N_BASE_NEURONS  3				  // ReLu, tanH, cos (or modulo X)
#define MAX_CHILDREN_PER_BLOCK  10
#define MAX_BLOCK_INPUT_SIZE  10          // Does not apply to the top one, which is the network itself
#define MAX_BLOCK_OUTPUT_SIZE  10         // Does not apply to the top one, which is the network itself

// In the genotype, denotes the fact that a child is connected
// to the input 
#define INPUT_ID -1     


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

		if (originID == INPUT_ID) {
			for (int i = 0; i < type->childrenConnexions[id]->nLines; i++) {
				for (int j = 0; j < type->childrenConnexions[id]->nColumns; j++) {
					// += (H * alpha + w) * input
					_childInputs[i0 + i] += (childrenConnexions[id].H[i][j] * type->childrenConnexions[id]->alpha[i][j]
						+ type->childrenConnexions[id]->w[i][j])
						* input[j];
				}
			}
		}
		else {
			for (int i = 0; i < type->childrenConnexions[id]->nLines; i++) {
				for (int j = 0; j < type->childrenConnexions[id]->nColumns; j++) {
					// += (H * alpha + w) * prevAct
					_childInputs[i0 + i] += (childrenConnexions[id].H[i][j] * type->childrenConnexions[id]->alpha[i][j]
						+ type->childrenConnexions[id]->w[i][j])
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

		if (!child->isNeuron) {			    // if the child is a bloc
			child->forward(&_childInputs[_inputListID]);
		}
		else {								// else, it is a base neuron
			child->previousOutput[0] = child->currentOutput[0];
			child->currentOutput[0] = child->type->f(_childInputs[_inputListID] + child->type->bias[0]);
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


	// Update hebbian and eligibility traces
	int originID, destinationID;  // For readability only, compiler will optimize them away
	for (int id = 0; id < childrenConnexions.size(); id++) {
		originID = type->childrenConnexions[id]->originID;
		destinationID = type->childrenConnexions[id]->destinationID;

		
		float eta, A, B, C; // For readability only, compiler will optimize them away
		for (int i = 0; i < type->childrenConnexions[id]->nLines; i++) {
			for (int j = 0; j < type->childrenConnexions[id]->nColumns; j++) {
				eta = type->childrenConnexions[id]->eta[i][j];
				A = type->childrenConnexions[id]->A[i][j];
				B = type->childrenConnexions[id]->B[i][j];
				C = type->childrenConnexions[id]->C[i][j];

				childrenConnexions[id].E[i][j] = (1 - eta) * childrenConnexions[id].E[i][j]
					+ eta * (A * children[destinationID]->currentOutput[i] * children[originID]->previousOutput[j]
						+ B * children[destinationID]->currentOutput[i]
						+ C * children[originID]->previousOutput[j]);

				childrenConnexions[id].H[i][j] += childrenConnexions[id].E[i][j] * neuromodulatorySignal;
				childrenConnexions[id].H[i][j] = std::max(-1.0f, std::min(childrenConnexions[id].H[i][j], 1.0f));
			}
		}
	}

	neuromodulatorySignal = 1.41f * currentOutput[type->outputSize];
	// neuromodulatorySignal *= currentOutput[type->outputSize]; 
	// *= because it has been set by its parent.

	delete[] _childInputs;
}

Network::Network() {

}

std::vector<float> Network::step(std::vector<float> obs) {

}


