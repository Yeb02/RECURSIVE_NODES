#include "Phenotype.h"



PhenotypeConnexion::PhenotypeConnexion(int s)
{
	H = std::make_unique<float[]>(s);
	E = std::make_unique<float[]>(s);
	wLifetime = std::make_unique<float[]>(s);

#ifndef CONTINUOUS_LEARNING
	avgH = std::make_unique<float[]>(s);
#endif

	zero(s);
	zeroWlifetime(s);
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
	nInferences = 1; // not 0 to avoid division by 0 if intertrialReset is called before going through any trial 
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

	totalM = 0.0f;
	localM = 0.0f;
};

void PhenotypeNode::interTrialReset() {
	std::fill(previousInput.begin(), previousInput.end(), 0.0f);
	std::fill(previousOutput.begin(), previousOutput.end(), 0.0f);
	std::fill(currentOutput.begin(), currentOutput.end(), 0.0f);
	for (int i = 0; i < children.size(); i++) {
		if (!children[i].type->isSimpleNeuron) children[i].interTrialReset();
		else {
			children[i].previousInput[0] = 0.0f;
			children[i].previousOutput[0] = 0.0f;
			children[i].currentOutput[0] = 0.0f;
		}
	}

	for (int i = 0; i < childrenConnexions.size(); i++) {
		int s = type->childrenConnexions[i].nLines * type->childrenConnexions[i].nColumns;
#ifndef CONTINUOUS_LEARNING
		float factor = 1.0f / (float)nInferences; // here for readability, compiler will put it outside the loop
		childrenConnexions[i].updateWatTrialEnd(s, factor, type->childrenConnexions[i].alpha.get());
#endif
		childrenConnexions[i].zero(s);
	}
	totalM = 0.0f;
	localM = 0.0f;
	nInferences = 1; // not 0 to avoid division by 0 if intertrialReset is called again before going through any new trial 
}

void PhenotypeNode::forward(const float* input) {
	nInferences++;
	int i0, originID, destinationID;
	int nc, nl, matID;

	// set up the array where every's child input is accumulated before applying their forward.
	float* _childInputs = new float[type->concatenatedChildrenInputLength + type->outputSize];
	for (int i = 0; i < type->concatenatedChildrenInputLength + type->outputSize; i++) _childInputs[i] = 0.0f;

	// save previous step's M.
	float previousLocalM = localM;
	localM = type->biasM;

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
			if (destinationID == MODULATION_ID) {
				for (int i = 0; i < nl; i++) {
					for (int j = 0; j < nc; j++) {
						// += (H * alpha + w) * prevAct
						localM += (H[matID] * alpha[matID] + w[matID] + wLifetime[matID]) * input[j];
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
	localM = tanhf(localM);
	totalM += localM; 

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
			children[i].totalM = this->totalM;
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
			iArray = &localM;
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
				// On simple examples (cartpole), these supposedly exclusive lines worsen avgFitness. 
				// Does the explanation reside solely in gamma's mutations ?  
				//wLifetime[matID] += alpha[matID] * E[matID] * gamma[matID] * totalM; 
				//wLifetime[matID] += alpha[matID] * H[matID] * gamma[matID] * totalM; 
				//wLifetime[matID] += alpha[matID] * H[matID] * gamma[matID];
				//wLifetime[matID] += alpha[matID] * E[matID] * gamma[matID];
#endif
				E[matID] = (1.0f - eta[matID]) * E[matID] + eta[matID] *
					(A[matID] * iArray[i] * jArray[j] + B[matID] * iArray[i] + C[matID] * jArray[j] + D[matID]);

				H[matID] += E[matID] * totalM;
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