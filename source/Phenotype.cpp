#include "Phenotype.h"

PhenotypeConnexion::PhenotypeConnexion(int s)
{
	H = std::make_unique<float[]>(s);

#ifdef USING_NEUROMODULATION
	E = std::make_unique<float[]>(s);
#endif 
	zero(s);
}

void PhenotypeConnexion::zero(int s) {
	for (int i = 0; i < s; i++) {
		H[i] = 0.0f;
#ifdef USING_NEUROMODULATION
		E[i] = 0.0f;
#endif 
	}
}

void PhenotypeConnexion::randomH(int s) {
	for (int i = 0; i < s; i++) {
		H[i] = NORMAL_01 * .1f;
	}
}



PhenotypeNode::PhenotypeNode(GenotypeNode* type) : type(type)
{
#ifdef USING_NEUROMODULATION
	neuromodulatorySignal = 1.0f;
#endif 
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
}

void PhenotypeNode::randomH() {
	for (int i = 0; i < children.size(); i++) {
		if (!children[i].type->isSimpleNeuron) children[i].randomH();
	}
	for (int i = 0; i < childrenConnexions.size(); i++) {
		int s = type->childrenConnexions[i].nLines * type->childrenConnexions[i].nColumns;
		childrenConnexions[i].randomH(s);
	}
}

void PhenotypeNode::reset() {
#ifdef USING_NEUROMODULATION
	neuromodulatorySignal = 1.0f;
#endif 
	for (int i = 0; i < previousInput.size(); i++) {
		previousInput[i] = 0;
	}
	for (int i = 0; i < previousOutput.size(); i++) {
		previousInput[i] = 0;
	}

	for (int i = 0; i < type->children.size() - 1; i++) {
		children[i].reset();
	}
	for (int i = 0; i < type->childrenConnexions.size(); i++) {
		childrenConnexions[i].zero(
			type->childrenConnexions[i].nLines *
			type->childrenConnexions[i].nColumns
		);
	}
}

void PhenotypeNode::forward(const float* input) {
	float* _childInputs = new float[type->concatenatedChildrenInputLength + type->outputSize];
	for (int i = 0; i < type->concatenatedChildrenInputLength + type->outputSize; i++) _childInputs[i] = 0.0f;

	float* _destPrevs = new float[type->concatenatedChildrenInputLength + type->outputSize]; // util for hebbian updates
	for (int i = 0; i < children.size(); i++) {
		for (int j = type->concatenatedChildrenInputBeacons[i]; j < type->concatenatedChildrenInputBeacons[i + 1]; j++) {
			_destPrevs[j] = children[i].previousInput[j - type->concatenatedChildrenInputBeacons[i]];
		}
	}

#ifdef USING_NEUROMODULATION
	// compute the neuromodulatory output
	float temp = type->neuromodulationBias;
	for (int i = 0; i < type->outputSize; i++) {
		temp += type->wNeuromodulation[i] * currentOutput[i];
	}
	neuromodulatorySignal *= 1.41f * tanh(temp);
#endif	

	// propagate the previous steps's outputs, by iterating over the connexions between children

	int i0, originID, destinationID;  // for readability, to be optimized away
	for (int id = 0; id < childrenConnexions.size(); id++) {
		originID = type->childrenConnexions[id].originID;
		destinationID = type->childrenConnexions[id].destinationID;
		i0 = type->concatenatedChildrenInputBeacons[destinationID];

		float* H = childrenConnexions[id].H.get();
#if defined USING_NEUROMODULATION
		float* alpha = type->childrenConnexions[id].alpha.get();
		float* w = type->childrenConnexions[id].w.get();
#endif
		int nl = type->childrenConnexions[id].nLines;
		int nc = type->childrenConnexions[id].nColumns;
		int matID = 0;
		if (originID == INPUT_ID) {
			for (int i = 0; i < nl; i++) {
				for (int j = 0; j < nc; j++) {

#if defined RISI_NAJARRO_2020
					// The article's w is our H. z += w * yj, y = f(z)
					_childInputs[i0 + i] += H[matID] * input[j];
#elif defined USING_NEUROMODULATION
					_childInputs[i0 + i] += (H[matID] * alpha[matID] + w[matID]) * input[j];
#endif 
					matID++;
				}
			}
		}
		else {
			for (int i = 0; i < nl; i++) {
				for (int j = 0; j < nc; j++) {

#if defined RISI_NAJARRO_2020
					// The article's w is our H. z += w * yj, y = f(z)
					_childInputs[i0 + i] += H[matID] * children[originID].previousOutput[j];
#elif defined USING_NEUROMODULATION
					// += (H * alpha + w) * prevAct
					_childInputs[i0 + i] += (H[matID] * alpha[matID] + w[matID]) * children[originID].previousOutput[j];
#endif 
					matID++;
				}
			}
		}
	}


	// apply children's forward, after a tanh for non-simple neurons. 

	int _childID = 0;
	int _inputListID = 0;
	for (int i = 0; i < children.size(); i++) {
#ifdef USING_NEUROMODULATION
		children[i].neuromodulatorySignal = this->neuromodulatorySignal;
#endif 

		// Depending on the child's nature, we have 2 cases:
		//  - the child is a bloc
		//  - the child is a simple neuron

		if (children[i].type->isSimpleNeuron) {
			children[i].previousOutput[0] = children[i].currentOutput[0];
			children[i].currentOutput[0] = children[i].type->f(_childInputs[_inputListID]);
		}
		else {
			int maxJ = _inputListID + children[i].type->inputSize;
#ifdef RISI_NAJARRO_2020  // they do not use the bias
			for (int j = _inputListID; j < maxJ; j++) _childInputs[j] = tanh(_childInputs[j]);
#elif defined USING_NEUROMODULATION
			for (int j = _inputListID; j < maxJ; j++) _childInputs[j] = tanh(_childInputs[j] + children[i].type->inBias[j - _inputListID]);
#endif
			children[i].forward(&_childInputs[_inputListID]);
		}

		_inputListID += children[i].type->inputSize;
		_childID++;
	}

	// process this node's output, stored in the input of the virtual output node:
	previousOutput.assign(currentOutput.begin(), currentOutput.end()); // save the previous activations
	for (int i = 0; i < type->outputSize; i++) {
#ifdef RISI_NAJARRO_2020  // they do not use the bias
		currentOutput[i] = tanh(_childInputs[_inputListID + i]);
#elif defined USING_NEUROMODULATION
		currentOutput[i] = tanh(_childInputs[_inputListID + i] + type->outBias[i]);
#endif
	}


	// Update hebbian and eligibility traces
	float destCur, destPrev, orCur, orPrev; // For readability only, compiler will optimize them away
	for (int id = 0; id < childrenConnexions.size(); id++) {
		originID = type->childrenConnexions[id].originID;
		destinationID = type->childrenConnexions[id].destinationID;

		float* A = type->childrenConnexions[id].A.get();
		float* B = type->childrenConnexions[id].B.get();
		float* C = type->childrenConnexions[id].C.get();
		float* eta = type->childrenConnexions[id].eta.get();
		float* H = childrenConnexions[id].H.get();
#if defined RISI_NAJARRO_2020
		float* D = type->childrenConnexions[id].D.get();
#elif defined USING_NEUROMODULATION
		float* E = childrenConnexions[id].E.get();
#endif
		int nl = type->childrenConnexions[id].nLines;
		int nc = type->childrenConnexions[id].nColumns;
		int matID = 0;  // = i*nc+j
		for (int i = 0; i < nl; i++) {
			for (int j = 0; j < nc; j++) {

				// If the child's output size = input size, we could link through it... 
				// Probably a stupid idea though.

				destCur = destinationID != children.size() ?
					children[destinationID].previousInput[i] :
					currentOutput[i];

				destPrev = destinationID != children.size() ?
					_destPrevs[type->concatenatedChildrenInputBeacons[destinationID]+i] :
					previousOutput[i];

				orPrev = originID != INPUT_ID ?
					children[originID].previousOutput[j] :
					previousInput[j];

				orCur = originID != INPUT_ID ?
					children[originID].currentOutput[j] :
					input[j];


#if defined RISI_NAJARRO_2020
				H[matID] += eta[matID] * (A[matID] * yi * aj + B[matID] * yi + C[matID] * aj + D[matID]);
#elif defined USING_NEUROMODULATION
				E[matID] = (1 - eta[matID]) * E[matID] + eta[matID] * (A[matID] * destCur * orPrev + B[matID] * destCur + C[matID] * orPrev);
				/*E[matID] = (1 - eta[matID]) * E[matID] + eta[matID] *
					(A[matID] * (destCur * orPrev  - destPrev * orCur) 
				     + B[matID] * destCur + C[matID] * orPrev);*/

				H[matID] += E[matID] * neuromodulatorySignal;
				H[matID] = std::max(-1.0f, std::min(H[matID], 1.0f));
#endif 
				matID++;
			}
		}
	}

	for (int i = 0; i < type->inputSize; i++) {
		previousInput[i] = input[i];
	}
	delete[] _childInputs;
	delete[] _destPrevs;
}