#include "Phenotype.h"



PhenotypeConnexion::PhenotypeConnexion(int s)
{
	H = std::make_unique<float[]>(s);
	E = std::make_unique<float[]>(s);
	zero(s);
}

void PhenotypeConnexion::zero(int s) {
	for (int i = 0; i < s; i++) {
		H[i] = 0.0f;
		E[i] = 0.0f;
	}
}

PhenotypeNode::PhenotypeNode(GenotypeNode* type) : type(type)
{
	previousInput.resize(type->inputSize);
	previousOutput.resize(type->outputSize);
	currentOutput.resize(type->outputSize);

	// create children recursively 
	children.reserve(type->children.size());
	for (int i = 0; i < type->children.size(); i++) {
		children.emplace_back(type->children[i]);
	}

	// create connexions structs
	childrenConnexions.reserve(type->childrenConnexions.size());
	for (int i = 0; i < type->childrenConnexions.size(); i++) {
		childrenConnexions.emplace_back(
			type->childrenConnexions[i].nLines *
			type->childrenConnexions[i].nColumns
		);
	}

	neuromodulatorySignal = 0.0f;
	M[0] = 0.0f;
	M[1] = 0.0f;
};

void PhenotypeNode::zero() {
	std::fill(previousInput.begin(), previousInput.end(), 0.0f);
	std::fill(previousOutput.begin(), previousOutput.end(), 0.0f);
	std::fill(currentOutput.begin(), currentOutput.end(), 0.0f);
	for (int i = 0; i < children.size(); i++) {
		if (!children[i].type->isSimpleNeuron) children[i].zero();
		else {
			children[i].previousInput[0] = 0.0f;
			children[i].previousOutput[0] = 0.0f;
			children[i].currentOutput[0] = 0.0f;
		}
	}
	for (int i = 0; i < childrenConnexions.size(); i++) {
		int s = type->childrenConnexions[i].nLines * type->childrenConnexions[i].nColumns;
		childrenConnexions[i].zero(s);
	}
	neuromodulatorySignal = 0.0f;
	M[0] = 0.0f;
	M[1] = 0.0f;
}

void PhenotypeNode::forward(const float* input) {
	int i0, originID, destinationID;
	int nc, nl, matID;

	// set up the array where every's child input is accumulated before applying their forward.
	float* _childInputs = new float[type->concatenatedChildrenInputLength + type->outputSize];
	for (int i = 0; i < type->concatenatedChildrenInputLength + type->outputSize; i++) _childInputs[i] = 0.0f;

	// save previous step's M.
	float previousM[2] = { M[0], M[1] };
	M[0] = type->biasMplus;
	M[1] = type->biasMminus;

	// propagate the previous steps's outputs, by iterating over the connexions between children
	for (int id = 0; id < childrenConnexions.size(); id++) {
		originID = type->childrenConnexions[id].originID;
		destinationID = type->childrenConnexions[id].destinationID;
		i0 = type->concatenatedChildrenInputBeacons[destinationID];

		nl = type->childrenConnexions[id].nLines;
		nc = type->childrenConnexions[id].nColumns;
		matID = 0;

		float* H = childrenConnexions[id].H.get();
		float* alpha = type->childrenConnexions[id].alpha.get();
		float* w = type->childrenConnexions[id].w.get();

		if (originID == INPUT_ID) {
			if (destinationID == MODULATION_ID) {
				for (int i = 0; i < nl; i++) {
					for (int j = 0; j < nc; j++) {
						// += (H * alpha + w) * prevAct
						M[i] += (H[matID] * alpha[matID] + w[matID]) * input[j];
						matID++;
					}
				}
			}
			else {
				for (int i = 0; i < nl; i++) {
					for (int j = 0; j < nc; j++) {
						// += (H * alpha + w) * prevAct
						_childInputs[i0 + i] += (H[matID] * alpha[matID] + w[matID]) * input[j];
						matID++;
					}
				}
			}
		}
		else {
			for (int i = 0; i < nl; i++) {
				for (int j = 0; j < nc; j++) {
					// += (H * alpha + w) * prevAct
					_childInputs[i0 + i] += (H[matID] * alpha[matID] + w[matID]) * children[originID].previousOutput[j];
					matID++;
				}
			}
		}
	}

	// neuromodulation
	M[0] = tanhf(M[0]);
	M[1] = tanhf(M[1]);
	neuromodulatorySignal *= (M[0] - M[1]) * .707f; // 1/sqrt(2)

	// apply children's forward, after a tanh for non-simple neurons. 
	int _inputListID = 0;
	for (int i = 0; i < children.size(); i++) {
		children[i].neuromodulatorySignal = this->neuromodulatorySignal;

		// Depending on the child's nature, we have 2 cases:
		//  - the child is a simple neuron, and we handle everything for him.
		//  - the child is a bloc, and it handles its own forward and activation saving.
		

		if (children[i].type->isSimpleNeuron) {
			children[i].previousOutput[0] = children[i].currentOutput[0];
			_childInputs[_inputListID] = children[i].type->f(_childInputs[_inputListID]); // a simple neuron has no bias.
			children[i].currentOutput[0] = _childInputs[_inputListID];
		}
		else {
			int maxJ = _inputListID + children[i].type->inputSize;
			for (int j = _inputListID; j < maxJ; j++) {
				_childInputs[j] = tanhf(_childInputs[j] + children[i].type->inBias[j - _inputListID]);
			}
			children[i].forward(&_childInputs[_inputListID]);
		}

		_inputListID += children[i].type->inputSize;
	}

	// process this node's output, stored in the input of the virtual output node:
	previousOutput.assign(currentOutput.begin(), currentOutput.end()); // save the previous activations
	for (int i = 0; i < type->outputSize; i++) {
		currentOutput[i] = tanhf(_childInputs[_inputListID + i] + type->outBias[i]);
	}


	// Update hebbian and eligibility traces
	for (int id = 0; id < childrenConnexions.size(); id++) {
		originID = type->childrenConnexions[id].originID;
		destinationID = type->childrenConnexions[id].destinationID;

		float* A = type->childrenConnexions[id].A.get();
		float* B = type->childrenConnexions[id].B.get();
		float* C = type->childrenConnexions[id].C.get();
		float* eta = type->childrenConnexions[id].eta.get();
		float* H = childrenConnexions[id].H.get();
		float* E = childrenConnexions[id].E.get();


		float* iArray;
		if (destinationID == children.size()) {
			iArray = currentOutput.data();
		}
		else if (destinationID == MODULATION_ID) {
			iArray = M;
		}
		else {
			iArray = children[destinationID].previousInput.data();
		}
			
		float* jArray = originID != INPUT_ID ?
			children[originID].previousOutput.data():
			previousInput.data();

		nl = type->childrenConnexions[id].nLines;
		nc = type->childrenConnexions[id].nColumns;
		matID = 0;  // = i*nc+j
		for (int i = 0; i < nl; i++) {
			for (int j = 0; j < nc; j++) {
				E[matID] = (1 - eta[matID]) * E[matID] + 
					eta[matID] * (A[matID] * iArray[i] * jArray[j] + B[matID] * iArray[i] + C[matID] * jArray[j]);

				H[matID] += E[matID] * neuromodulatorySignal;
				H[matID] = std::max(-1.0f, std::min(H[matID], 1.0f));
				matID++;
			}
		}
	}

	for (int i = 0; i < type->inputSize; i++) {
		previousInput[i] = input[i];
	}
	delete[] _childInputs;
}