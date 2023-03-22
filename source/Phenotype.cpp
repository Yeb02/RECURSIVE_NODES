#include "Phenotype.h"



PhenotypeConnexion::PhenotypeConnexion(int s)
{
	H = std::make_unique<float[]>(s);
	E = std::make_unique<float[]>(s);
	wLifetime = std::make_unique<float[]>(s);

#ifndef CONTINUOUS_LEARNING
	avgH = std::make_unique<float[]>(s);
#endif

	zeroWlifetime(s);

	zero(s); // not necessary, because in theory PhenotypeNode::preTrialReset() should be called before any computation. TODO
}

#ifndef CONTINUOUS_LEARNING
void PhenotypeConnexion::updateWatTrialEnd(int s, float factor, float* alpha) {
	for (int i = 0; i < s; i++) {
		wLifetime[i] += alpha[i] * avgH[i] * factor;
	}
}
#endif

void PhenotypeConnexion::zero(int s) {
	for (int i = 0; i < s; i++) {
		H[i] = 0.0f;
		E[i] = 0.0f;
#ifndef CONTINUOUS_LEARNING
		avgH[i] = 0.0f;
#endif
	}
}

void PhenotypeConnexion::zeroWlifetime(int s) {
	for (int i = 0; i < s; i++) {
		wLifetime[i] = 0.0f;
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

	// LocalM and TotalM are not instantiated here because a call to preTrialReset() 
	// must be made before any forward pass. (Or call to postTrialUpdate). nInferences is 
	// in the same situation but is instantiated to detect those unexpected calls.
	nInferences = 0;
};

#if defined GUIDED_MUTATIONS && defined CONTINUOUS_LEARNING
void PhenotypeNode::accumulateW(float factor) {

	for (int i = 0; i < childrenConnexions.size(); i++) {
		type->nApparitions++;
		int s = type->childrenConnexions[i].nLines * type->childrenConnexions[i].nColumns;
		for (int j = 0; j < s; j++) {
			type->childrenConnexions[i].accumulator[j] += factor * childrenConnexions[i].wLifetime[j];
			childrenConnexions[i].wLifetime[j] = 0.0f;
		}
	}

	for (int i = 0; i < children.size(); i++) {
		if (!children[i].type->isSimpleNeuron) {
			children[i].accumulateW(factor);
		}
	}
}
#endif

#ifndef CONTINUOUS_LEARNING
void PhenotypeNode::updateWatTrialEnd(float invNInferences) {
	if (nInferences == 0) return; // should not have been called in the first place.

	for (int i = 0; i < children.size(); i++) {
		if (!children[i].type->isSimpleNeuron) children[i].updateWatTrialEnd(invNInferences);
	}

	for (int i = 0; i < childrenConnexions.size(); i++) {
		int s = type->childrenConnexions[i].nLines * type->childrenConnexions[i].nColumns;
		childrenConnexions[i].updateWatTrialEnd(s, invNInferences, type->childrenConnexions[i].alpha.get());
	}
}
#endif

void PhenotypeNode::preTrialReset() {
	std::fill(previousInput.begin(), previousInput.end(), 0.0f);
	std::fill(previousOutput.begin(), previousOutput.end(), 0.0f);
	std::fill(currentOutput.begin(), currentOutput.end(), 0.0f);
	for (int i = 0; i < children.size(); i++) {
		if (!children[i].type->isSimpleNeuron) children[i].preTrialReset();
		else {
			children[i].previousInput[0] = 0.0f;
			children[i].previousOutput[0] = 0.0f;
			children[i].currentOutput[0] = 0.0f;
		}
	}

	for (int i = 0; i < childrenConnexions.size(); i++) {
		int s = type->childrenConnexions[i].nLines * type->childrenConnexions[i].nColumns;
		childrenConnexions[i].zero(s); // zeros E, H, AVG_H, 
	}
	totalM[0] = 0.0f;
	totalM[1] = 0.0f;
	localM[0] = 0.0f;
	localM[1] = 0.0f;
	nInferences = 0;
}

void PhenotypeNode::forward(const float* input) {
	nInferences++;
	int i0, originID, destinationID;
	int nc, nl, matID;

	// set up the array where every's child input is accumulated before applying their forward.
	float* _childInputs = new float[type->sumChildrenInputSizes + type->outputSize];
	for (int i = 0; i < type->sumChildrenInputSizes + type->outputSize; i++) {
		_childInputs[i] = type->childrenInBias[i];
	}

	// save previous step's M.
	float previousLocalM[2];
	previousLocalM[0] = localM[0];
	previousLocalM[1] = localM[1];
	localM[0] = type->biasM[0];
	localM[1] = type->biasM[1];

	// propagate the previous steps's outputs, by iterating over the connexions between children
	for (int id = 0; id < childrenConnexions.size(); id++) {
		originID = type->childrenConnexions[id].originID;
		destinationID = type->childrenConnexions[id].destinationID;
		if (destinationID != MODULATION_ID)
			i0 = type->concatenatedChildrenInputBeacons[destinationID];

		nl = type->childrenConnexions[id].nLines;
		nc = type->childrenConnexions[id].nColumns;
		matID = 0;

		float* H = childrenConnexions[id].H.get();
		float* wLifetime = childrenConnexions[id].wLifetime.get();
		float* alpha = type->childrenConnexions[id].alpha.get();
		float* w = type->childrenConnexions[id].w.get();

		if (originID == INPUT_ID) {
			// TODO when input to modulation will be changed to be more than parent input, process connexions 
			// to modulation and modulation calculus AFTER children's forward, way below.   TODO TODO TODO
			if (destinationID == MODULATION_ID) { 
				for (int i = 0; i < nl; i++) {
					for (int j = 0; j < nc; j++) {
						// += (H * alpha + w) * prevAct
						localM[i] += (H[matID] * alpha[matID] + w[matID] + wLifetime[matID]) * input[j];
						matID++;
					}
				}
			}
			else {
				for (int i = 0; i < nl; i++) {
					for (int j = 0; j < nc; j++) {
						// += (H * alpha + w) * prevAct
						_childInputs[i0 + i] += (H[matID] * alpha[matID] + w[matID] + wLifetime[matID]) * input[j];
						matID++;
					}
				}
			}
		}
		else {
			for (int i = 0; i < nl; i++) {
				for (int j = 0; j < nc; j++) {
					// += (H * alpha + w) * prevAct
					_childInputs[i0 + i] += (H[matID] * alpha[matID] + w[matID] + wLifetime[matID]) * 
						children[originID].previousOutput[j];
					matID++;
				}
			}
		}
	}

	// neuromodulation
	localM[0] = tanhf(localM[0]);
	localM[1] = tanhf(localM[1]);
	totalM[0] += localM[0];
	totalM[1] += localM[1];

	// apply children's forward, after a tanh for non-simple neurons. 
	int _inputListID = 0;
	for (int i = 0; i < children.size(); i++) {
	
		// Depending on the child's nature, we have 2 cases:
		//  - the child is a simple neuron, and we handle everything for him.
		//  - the child is a bloc, and it handles its own forward and activation saving.
		

		if (children[i].type->isSimpleNeuron) {
			// a simple neuron has no internal neuromodulation, so children[i].neuromodulatorySignal is untouched.
			children[i].previousOutput[0] = children[i].currentOutput[0];
			_childInputs[_inputListID] = children[i].type->f(_childInputs[_inputListID]); // a simple neuron has no bias.
			children[i].currentOutput[0] = _childInputs[_inputListID];
		}
		else {
			children[i].totalM[0] = this->totalM[0];
			children[i].totalM[1] = this->totalM[1];
			int maxJ = _inputListID + children[i].type->inputSize;
			for (int j = _inputListID; j < maxJ; j++) {
				_childInputs[j] = tanhf(_childInputs[j]);
			}
			children[i].forward(&_childInputs[_inputListID]);
		}

		_inputListID += children[i].type->inputSize;
	}

	// process this node's output, stored in the input of the virtual output node:
	previousOutput.assign(currentOutput.begin(), currentOutput.end()); // save the previous activations
	for (int i = 0; i < type->outputSize; i++) {
		currentOutput[i] = tanhf(_childInputs[_inputListID + i]);
	}


	// Update hebbian and eligibility traces
	for (int id = 0; id < childrenConnexions.size(); id++) {
		originID = type->childrenConnexions[id].originID;
		destinationID = type->childrenConnexions[id].destinationID;

		float* A = type->childrenConnexions[id].A.get();
		float* B = type->childrenConnexions[id].B.get();
		float* C = type->childrenConnexions[id].C.get();
		float* D = type->childrenConnexions[id].D.get();
		float* eta = type->childrenConnexions[id].eta.get();
		float* H = childrenConnexions[id].H.get();
		float* E = childrenConnexions[id].E.get();

#ifdef CONTINUOUS_LEARNING
		float* wLifetime = childrenConnexions[id].wLifetime.get();
		float* gamma = type->childrenConnexions[id].gamma.get();
		float* alpha = type->childrenConnexions[id].alpha.get();
#else
		float* avgH = childrenConnexions[id].avgH.get();
#endif
		


		float* iArray;
		if (destinationID == children.size()) {
			iArray = currentOutput.data();
		}
		else if (destinationID == MODULATION_ID) {
			iArray = localM;
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
#ifdef CONTINUOUS_LEARNING
				wLifetime[matID] += alpha[matID] * H[matID] * gamma[matID] * totalM[1]; 
#endif
				E[matID] = (1.0f - eta[matID]) * E[matID] + eta[matID] *
					(A[matID] * iArray[i] * jArray[j] + B[matID] * iArray[i] + C[matID] * jArray[j] + D[matID]);

				H[matID] += E[matID] * totalM[0];
				H[matID] = std::max(-1.0f, std::min(H[matID], 1.0f));
#ifndef CONTINUOUS_LEARNING
				avgH[matID] += H[matID];
#endif
				matID++;

			}
		}
	}

	// input update
	for (int i = 0; i < type->inputSize; i++) {
		previousInput[i] = input[i];
	}

	delete[] _childInputs;
}
