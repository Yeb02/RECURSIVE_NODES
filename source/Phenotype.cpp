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
	// must be made before any forward pass. (Or call to postTrialUpdate). nInferencesP is 
	// in the same situation but is instantiated to detect those unexpected calls.
	nInferencesP = 0;
};

#if defined GUIDED_MUTATIONS
void PhenotypeNode::accumulateW(float factor) {
	type->nAccumulations++;
	for (int i = 0; i < childrenConnexions.size(); i++) {
		int s = type->childrenConnexions[i].nLines * type->childrenConnexions[i].nColumns;
		for (int j = 0; j < s; j++) {
			type->childrenConnexions[i].accumulator[j] += factor * childrenConnexions[i].wLifetime[j];
			childrenConnexions[i].wLifetime[j] = 0.0f;
		}
	}

	for (int i = 0; i < children.size(); i++) {
		if (children[i].type->nodeType == GenotypeNode::COMPLEX) {
			children[i].accumulateW(factor);
		}
	}
}
#endif

#ifndef CONTINUOUS_LEARNING
void PhenotypeNode::updateWatTrialEnd(float invnInferencesP) {
	if (nInferencesP == 0) return; // should not have been called in the first place.

	for (int i = 0; i < children.size(); i++) {
		if (children[i].type->nodeType == GenotypeNode::COMPLEX) children[i].updateWatTrialEnd(invnInferencesP);
	}

	for (int i = 0; i < childrenConnexions.size(); i++) {
		int s = type->childrenConnexions[i].nLines * type->childrenConnexions[i].nColumns;
		childrenConnexions[i].updateWatTrialEnd(s, invnInferencesP, type->childrenConnexions[i].alpha.get());
	}
}
#endif

void PhenotypeNode::setArrayPointers(float* po, float* co, float* pi, float* ci, float* aa) {

	previousOutput = po;
	currentOutput = co;
	previousInput = pi;
	currentInput = ci;
#ifdef SATURATION_PENALIZING
	averageActivation = aa;
#endif

	po += type->outputSize;
	co += type->outputSize;
	pi += type->inputSize;
	ci += type->inputSize;
#ifdef SATURATION_PENALIZING
	aa += type->nodeType != GenotypeNode::COMPLEX ? 
		1 : type->inputSize + type->outputSize + 2; // and in this order in the array.
#endif


	for (int i = 0; i < children.size(); i++) {
		children[i].setArrayPointers(po, co, pi, ci, aa);
	}
}


void PhenotypeNode::preTrialReset() {

	for (int i = 0; i < children.size(); i++) {
		if (children[i].type->nodeType == GenotypeNode::COMPLEX) {
			children[i].preTrialReset();
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
	nInferencesP = 0;
}


#ifdef SATURATION_PENALIZING
void PhenotypeNode::setSaturationPenalizationPtr(float* saturationPenalizationPtr) {
	this->saturationPenalizationPtr = saturationPenalizationPtr;
	for (int i = 0; i < children.size(); i++) {
		if (children[i].type->nodeType == GenotypeNode::COMPLEX) {
			children[i].setSaturationPenalizationPtr(saturationPenalizationPtr);
		}
	}
}
#endif


void PhenotypeNode::forward() {
	int id, originID, destinationID;
	int nc, nl, matID;

#ifdef SATURATION_PENALIZING
	constexpr float modulationMultiplier = 0.0f; // must be set to the same value in Genotype::getNnonLinearities. TODO cleaner.
	constexpr float saturationExponent = 20.0f;  // one could try lower values... But they must be 2*integer.
#endif

	nInferencesP++;


	// STEP 1: SAVE THIS NODE'S PREVIOUS OUTPUT AND PREVIOUS LOCAL M, FOR HEBBIAN RULES LATER ON.

	std::copy(currentOutput, currentOutput + type->outputSize, previousOutput);
	float previousLocalM[2];
	float localMInput[2];
	previousLocalM[0] = localM[0];
	previousLocalM[1] = localM[1];

#ifdef SATURATION_PENALIZING
	// Update the input activation.
	for (int i = 0; i < type->inputSize; i++) {
		averageActivation[i] += currentInput[i];
	}
#endif

	// STEP 2: INITIALIZE ALL PRE-SYNAPTIC INPUTS WITH THE ASSOCIATED WEIGHT

	localMInput[0] = type->biasM[0];
	localMInput[1] = type->biasM[1];
	id = 0;
	for (int i = 0; i < children.size(); i++) {
		for (int j = 0; j < children[i].type->inputSize; j++) {
			children[i].currentInput[j] = type->childrenInBias[id];
			id++;
		}
	}
	for (int i = 0; i < type->outputSize; i++) {
		currentOutput[i] = type->childrenInBias[id + i];
	}


	// STEP 3: ADD THE PREVIOUS STEPS'S OUTPUTS TO THE PRE-SYNAPTIC INPUTS, AND APPLY CHILDREN'S FORWARD. ALL CHILDREN 
	// COULD BE UPDATED SIMULTANEOUSLY, BUT TO SPEED UP INFORMATION TRANSMITION IT HAPPENS "TYPE" BY "TYPE, IN THE 
	// FOLLOWING ORDER:  neuromodulation node -> complex children -> simple children -> output node.

	// #ifdef SATURATION_PENALIZING, also update saturationPenalizationPtr and averageActivation at each
	// evaluation of an activation function. (complex children handle their part)


	// STEP 3A: neuromodulation node.
	{
	// propagate.
		for (int id = 0; id < childrenConnexions.size(); id++) {
			destinationID = type->childrenConnexions[id].destinationID;
			if (destinationID != MODULATION_ID) {
				continue;
			}

			originID = type->childrenConnexions[id].originID;
			nl = type->childrenConnexions[id].nLines;
			nc = type->childrenConnexions[id].nColumns;
			matID = 0;

			float* H = childrenConnexions[id].H.get();
			float* wLifetime = childrenConnexions[id].wLifetime.get();
			float* alpha = type->childrenConnexions[id].alpha.get();
			float* w = type->childrenConnexions[id].w.get();

			float* originArray;
			if (originID == INPUT_ID) {
				originArray = currentInput;
			}
			else if (originID == MODULATION_ID) {
				originArray = localM;
			}
			else {
				originArray = children[originID].currentOutput;
			}

			for (int i = 0; i < nl; i++) {
				for (int j = 0; j < nc; j++) {
					// += (H * alpha + w) * prevAct
					localMInput[i] += (H[matID] * alpha[matID] + w[matID] + wLifetime[matID]) * originArray[j];
					matID++;
				}
			}
		}

		// Apply forward.
		localM[0] = tanhf(localMInput[0]);
		localM[1] = tanhf(localMInput[1]);
		totalM[0] += localM[0];
		totalM[1] += localM[1];

#ifdef SATURATION_PENALIZING
		// neuromodulation is weighted stronger because more important.
		* saturationPenalizationPtr += modulationMultiplier * powf(localM[0], saturationExponent);
		* saturationPenalizationPtr += modulationMultiplier * powf(localM[1], saturationExponent);
		averageActivation[type->inputSize + type->outputSize + 0] += localM[0];
		averageActivation[type->inputSize + type->outputSize + 1] += localM[1];
#endif
	}

	// STEP 3B: complex children.
	{
		// propagate.
		for (int id = 0; id < childrenConnexions.size(); id++) {
			destinationID = type->childrenConnexions[id].destinationID;
			if (destinationID == children.size() ||
				destinationID == MODULATION_ID ||
				children[destinationID].type->nodeType != GenotypeNode::COMPLEX) {

				continue;
			}

			originID = type->childrenConnexions[id].originID;
			nl = type->childrenConnexions[id].nLines;
			nc = type->childrenConnexions[id].nColumns;
			matID = 0;

			float* H = childrenConnexions[id].H.get();
			float* wLifetime = childrenConnexions[id].wLifetime.get();
			float* alpha = type->childrenConnexions[id].alpha.get();
			float* w = type->childrenConnexions[id].w.get();

			float* originArray;
			if (originID == INPUT_ID) {
				originArray = currentInput;
			}
			else if (originID == MODULATION_ID) {
				originArray = localM;
			}
			else {
				originArray = children[originID].currentOutput;
			}

			for (int i = 0; i < nl; i++) {
				for (int j = 0; j < nc; j++) {
					// += (H * alpha + w) * prevAct
					children[destinationID].currentInput[i] += (H[matID] * alpha[matID] + w[matID] + wLifetime[matID]) * originArray[j];
					matID++;
				}
			}
		}

		// transmit data and apply forward:
		for (int i = 0; i < children.size(); i++) {
			if (children[i].type->nodeType == GenotypeNode::COMPLEX) {
				children[i].totalM[0] = this->totalM[0];
				children[i].totalM[1] = this->totalM[1];
				for (int j = 0; j < children[i].type->inputSize; j++) {
					children[i].currentInput[j] = tanhf(children[i].currentInput[j]);
#ifdef SATURATION_PENALIZING
					* saturationPenalizationPtr += powf(children[i].currentInput[j], saturationExponent);
#endif
				}
				children[i].forward();
			}
		}
	}

	// STEP 3C: simple children.
	{
		// propagate.
		for (int id = 0; id < childrenConnexions.size(); id++) {
			destinationID = type->childrenConnexions[id].destinationID;
			if (destinationID == children.size() ||
				destinationID == MODULATION_ID ||
				children[destinationID].type->nodeType == GenotypeNode::COMPLEX) {

				continue;
			}

			originID = type->childrenConnexions[id].originID;
			nl = type->childrenConnexions[id].nLines;
			nc = type->childrenConnexions[id].nColumns;
			matID = 0;

			float* H = childrenConnexions[id].H.get();
			float* wLifetime = childrenConnexions[id].wLifetime.get();
			float* alpha = type->childrenConnexions[id].alpha.get();
			float* w = type->childrenConnexions[id].w.get();

			float* originArray;
			if (originID == INPUT_ID) {
				originArray = currentInput;
			}
			else if (originID == MODULATION_ID) {
				originArray = localM;
			}
			else {
				originArray = children[originID].currentOutput;
			}

			for (int i = 0; i < nl; i++) { // nl = 1 in this case
				for (int j = 0; j < nc; j++) {
					// += (H * alpha + w) * prevAct
					children[destinationID].currentInput[i] += (H[matID] * alpha[matID] + w[matID] + wLifetime[matID]) * originArray[j];
					matID++;
				}
			}
		}

		// Apply child's forward function and manage its I-O:
		for (int i = 0; i < children.size(); i++) {
			GenotypeNode::NODE_TYPE type = children[i].type->nodeType;
			if (type != GenotypeNode::COMPLEX) {
				children[i].previousOutput[0] = children[i].currentOutput[0];

				if (type == GenotypeNode::TANH) {
					children[i].currentInput[0] = tanhf(children[i].currentInput[0]);
				}
				else if (type == GenotypeNode::DERIVATOR) {
					children[i].currentInput[0] = children[i].currentInput[0] - children[i].previousInput[0];
				}
				
				children[i].currentOutput[0] = children[i].currentInput[0];
#ifdef SATURATION_PENALIZING
				* saturationPenalizationPtr += powf(children[i].currentInput[0], saturationExponent);
				children[i].averageActivation[0] += children[i].currentInput[0];
#endif
			}
		}
	}

	// STEP 3D: output node.
	{
		// propagate.
		for (int id = 0; id < childrenConnexions.size(); id++) {
			destinationID = type->childrenConnexions[id].destinationID;

			// TODO TODO TODO
			// How does it converge when this block is on 
			if (destinationID == children.size() ||
				destinationID == MODULATION_ID ||
				children[destinationID].type->nodeType == GenotypeNode::COMPLEX) {

				continue;
			}
			// instead of this one ?????? TODO TODO TODO
			/*if (destinationID != children.size() ) {
				continue;
			}*/

			originID = type->childrenConnexions[id].originID;
			nl = type->childrenConnexions[id].nLines;
			nc = type->childrenConnexions[id].nColumns;
			matID = 0;

			float* H = childrenConnexions[id].H.get();
			float* wLifetime = childrenConnexions[id].wLifetime.get();
			float* alpha = type->childrenConnexions[id].alpha.get();
			float* w = type->childrenConnexions[id].w.get();

			float* originArray;
			if (originID == INPUT_ID) {
				originArray = currentInput;
			}
			else if (originID == MODULATION_ID) {
				originArray = localM;
			}
			else {
				originArray = children[originID].currentOutput;
			}

			for (int i = 0; i < nl; i++) { // nl = 1 in this case
				for (int j = 0; j < nc; j++) {
					// += (H * alpha + w) * prevAct
					currentOutput[i] += (H[matID] * alpha[matID] + w[matID] + wLifetime[matID]) * originArray[j];
					matID++;
				}
			}
		}

		for (int i = 0; i < type->outputSize; i++) {
			currentOutput[i] = tanhf(currentOutput[i]);
#ifdef SATURATION_PENALIZING
			* saturationPenalizationPtr += powf(currentOutput[i], saturationExponent);
			averageActivation[type->inputSize + i] += currentOutput[i];
#endif
		}
	}
	

	// STEP 4: Update hebbian and eligibility traces.

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
			iArray = currentOutput;
		}
		else if (destinationID == MODULATION_ID) {
			iArray = localM;
		}
		else {
			iArray = children[destinationID].currentInput;
		}
			
		float* jArray;
		if (originID == INPUT_ID) {
			jArray = previousInput;
		}
		else if (originID == MODULATION_ID) {
			jArray = previousLocalM;
		}
		else {
			jArray = children[originID].previousOutput;
		}
			

		nl = type->childrenConnexions[id].nLines;
		nc = type->childrenConnexions[id].nColumns;
		matID = 0;  // = i*nc+j
		for (int i = 0; i < nl; i++) {
			for (int j = 0; j < nc; j++) {
#ifdef CONTINUOUS_LEARNING
				wLifetime[matID] = (1 - gamma[matID]) * wLifetime[matID] + alpha[matID] * H[matID] * gamma[matID] * totalM[1];
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
}
