#include "Genotype.h"


GenotypeConnexion::GenotypeConnexion(int oID, int dID, int nLines, int nColumns, GenotypeConnexion::initType init) :
	originID(oID), destinationID(dID), nLines(nLines), nColumns(nColumns)
{
	int s = nLines * nColumns;
	eta = std::make_unique<float[]>(s);
	A = std::make_unique<float[]>(s);
	B = std::make_unique<float[]>(s);
	C = std::make_unique<float[]>(s);
	D = std::make_unique<float[]>(s);
	alpha = std::make_unique<float[]>(s);
	w = std::make_unique<float[]>(s);
#ifdef CONTINUOUS_LEARNING
	gamma = std::make_unique<float[]>(s);
#endif
#ifdef GUIDED_MUTATIONS
	accumulator = std::make_unique<float[]>(s);
#endif

	SET_BINOMIAL(s, .5f); // for gamma and eta.
	float factor = .3f / (float)s;

	if (init == ZERO ) {
		for (int i = 0; i < nLines * nColumns; i++) {

			A[i] = NORMAL_01 * .2f;
			B[i] = NORMAL_01 * .2f;
			C[i] = NORMAL_01 * .2f;
			D[i] = NORMAL_01 * .2f;
			alpha[i] = 0.0f;
			eta[i] = factor * (float)BINOMIAL + .1f;
			w[i] = 0.0f;
#ifdef CONTINUOUS_LEARNING
			gamma[i] = factor * (float) BINOMIAL + .1f;
#endif
#ifdef GUIDED_MUTATIONS
			accumulator[i] = 0.0f;
#endif
		}
	}
	else if (init == RANDOM) {
		for (int i = 0; i < nLines * nColumns; i++) {
			A[i] = NORMAL_01 * .2f;
			B[i] = NORMAL_01 * .2f;
			C[i] = NORMAL_01 * .2f;
			D[i] = NORMAL_01 * .2f;
			alpha[i] = NORMAL_01 *.2f;
			eta[i] = factor * (float) BINOMIAL + .1f;
			w[i] = NORMAL_01*.2f;
#ifdef CONTINUOUS_LEARNING
			gamma[i] = factor * (float) BINOMIAL + .1f;
#endif
#ifdef GUIDED_MUTATIONS
			accumulator[i] = 0.0f;
#endif
		}
	}
}

GenotypeConnexion::GenotypeConnexion(GenotypeConnexion&& gc) noexcept {

	originID = gc.originID;
	destinationID = gc.destinationID;
	nLines = gc.nLines;
	nColumns = gc.nColumns;

	// move() removes ownership from the original pointer. Its use here is kind of an hacky workaround the fact
	// vector reallocation calls move constructor AND destructor. So the pointee would be destroyed otherwise.
	// https://stackoverflow.com/questions/41864544/stdvector-calls-contained-objects-destructor-when-reallocating-no-matter-what
	A = std::move(gc.A);
	B = std::move(gc.B);
	C = std::move(gc.C);
	D = std::move(gc.D);
	eta = std::move(gc.eta);
	w = std::move(gc.w);
	alpha = std::move(gc.alpha);
#ifdef CONTINUOUS_LEARNING
	gamma = std::move(gc.gamma);
#endif
#ifdef GUIDED_MUTATIONS
	accumulator = std::move(gc.accumulator);
#endif

}

GenotypeConnexion::GenotypeConnexion(const GenotypeConnexion& gc) {

	destinationID = gc.destinationID;
	originID = gc.originID;
	nLines = gc.nLines;
	nColumns = gc.nColumns;

	int s = nLines * nColumns;

	eta = std::make_unique<float[]>(s);
	A = std::make_unique<float[]>(s);
	B = std::make_unique<float[]>(s);
	C = std::make_unique<float[]>(s);
	D = std::make_unique<float[]>(s);
	alpha = std::make_unique<float[]>(s);
	w = std::make_unique<float[]>(s);

	std::copy(gc.eta.get(), gc.eta.get() + s, eta.get());
	std::copy(gc.A.get(), gc.A.get() + s, A.get());
	std::copy(gc.B.get(), gc.B.get() + s, B.get());
	std::copy(gc.C.get(), gc.C.get() + s, C.get());
	std::copy(gc.D.get(), gc.D.get() + s, D.get());
	std::copy(gc.w.get(), gc.w.get() + s, w.get());
	std::copy(gc.alpha.get(), gc.alpha.get() + s, alpha.get());

#ifdef CONTINUOUS_LEARNING
	gamma = std::make_unique<float[]>(s);
	std::copy(gc.gamma.get(), gc.gamma.get() + s, gamma.get());
#endif

#ifdef GUIDED_MUTATIONS
	accumulator = std::make_unique<float[]>(s);
	std::copy(gc.accumulator.get(), gc.accumulator.get() + s, accumulator.get());
#endif
}

GenotypeConnexion GenotypeConnexion::operator=(const GenotypeConnexion& gc) {

	destinationID = gc.destinationID;
	originID = gc.originID;
	nLines = gc.nLines;
	nColumns = gc.nColumns;

	int s = nLines * nColumns;

	eta = std::make_unique<float[]>(s);
	A = std::make_unique<float[]>(s);
	B = std::make_unique<float[]>(s);
	C = std::make_unique<float[]>(s);
	D = std::make_unique<float[]>(s);
	alpha = std::make_unique<float[]>(s);
	w = std::make_unique<float[]>(s);

	std::copy(gc.eta.get(), gc.eta.get() + s, eta.get());
	std::copy(gc.A.get(), gc.A.get() + s, A.get());
	std::copy(gc.B.get(), gc.B.get() + s, B.get());
	std::copy(gc.C.get(), gc.C.get() + s, C.get());
	std::copy(gc.D.get(), gc.D.get() + s, D.get());
	std::copy(gc.w.get(), gc.w.get() + s, w.get());
	std::copy(gc.alpha.get(), gc.alpha.get() + s, alpha.get());

#ifdef CONTINUOUS_LEARNING
	gamma = std::make_unique<float[]>(s);
	std::copy(gc.gamma.get(), gc.gamma.get() + s, gamma.get());
#endif

#ifdef GUIDED_MUTATIONS
	accumulator = std::make_unique<float[]>(s);
	std::copy(gc.accumulator.get(), gc.accumulator.get() + s, accumulator.get());
#endif

	return *this;
}


void GenotypeNode::computeBeacons() {
	concatenatedChildrenInputBeacons.resize(children.size() + 1);
	concatenatedChildrenInputBeacons[0] = 0;
	int s = 0;
	for (int i = 0; i < children.size(); i++) {
		s += children[i]->inputSize;
		concatenatedChildrenInputBeacons[i + 1] = s;
	}
	sumChildrenInputSizes = s;
}

void GenotypeNode::mutateFloats() {
#ifdef CONTINUOUS_LEARNING
	constexpr int nArrays = 8;  // 7+gamma
#else 
	constexpr int nArrays = 7;
#endif
	//param(t+1) = (.95+a*N1)*param(t) + b*N2
	constexpr float a = .2f; // .5f ??
	constexpr float b = .2f; // .5f ??

	constexpr float pMutation = .4f; // .2f ??

#ifdef GUIDED_MUTATIONS
	// w += clip[-deltaWclipRange,deltaWclipRange](deltaW)
	constexpr float deltaWclipRange = .3f;
#endif

	// Mutate int(nArrays*Pmutation*nParam) parameters in the inter-children connexions.
	float* aPtr = nullptr;
	for (int listID = 0; listID < childrenConnexions.size(); listID++) {
		int size = childrenConnexions[listID].nLines * childrenConnexions[listID].nColumns;
		int _nParams =  size * nArrays; 
		SET_BINOMIAL(_nParams, pMutation);
		int _nMutations = BINOMIAL;

		for (int k = 0; k <  _nMutations; k++) { 

			int arrayN = INT_0X(nArrays);
			int matrixID = INT_0X(size);

			switch (arrayN) {
			case 0: aPtr = childrenConnexions[listID].A.get(); break;
			case 1: aPtr = childrenConnexions[listID].B.get(); break;
			case 2: aPtr = childrenConnexions[listID].C.get(); break;
			case 3: aPtr = childrenConnexions[listID].D.get(); break;
			case 4: aPtr = childrenConnexions[listID].alpha.get(); break;
			case 5: aPtr = childrenConnexions[listID].w.get(); break;
			case 6:  aPtr = childrenConnexions[listID].eta.get(); break;
#ifdef CONTINUOUS_LEARNING
			case 7:  aPtr = childrenConnexions[listID].gamma.get(); break;
#endif
			}

			if (arrayN < 6) { // A, B, C, D, w, alpha
				aPtr[matrixID] *= .9f + NORMAL_01 * a;
				aPtr[matrixID] += NORMAL_01 * b;
			}
			else { // eta, gamma
				if (UNIFORM_01 > .05f) [[likely]] {
					aPtr[matrixID] += aPtr[matrixID] * (1 - aPtr[matrixID]) * (UNIFORM_01 - .5f);
				}
				else [[unlikely]] {
					aPtr[matrixID] = aPtr[matrixID] * .6f + UNIFORM_01 * .4f;
				}
			}
		}

#ifdef GUIDED_MUTATIONS
		if (nAccumulations == 0) continue;
		float invFactor = 1.0f / (float)nAccumulations; // should be outside the loop, here for readability
		for (int k = 0; k < size; k++) {
			float rawDelta = childrenConnexions[listID].accumulator[k] * invFactor;
			childrenConnexions[listID].w[k] += std::max(std::min(rawDelta, deltaWclipRange), -deltaWclipRange);
			childrenConnexions[listID].accumulator[k] = 0.0f;
		}
#endif
	}

	// Bias mutations
	int nBiases = (int)childrenInBias.size();
	SET_BINOMIAL(nBiases, pMutation);
	int _nMutations = BINOMIAL;
	for (int i = 0; i < _nMutations; i++) {
		int id = INT_0X(nBiases);
		childrenInBias[id] *= .8f + NORMAL_01 * a;
		childrenInBias[id] += NORMAL_01 * b *.5f;
	}

	for (int i = 0; i < 2; i++) {
		if (UNIFORM_01 > pMutation) { continue; }
		biasM[i] *= .8f + NORMAL_01 * a;
		biasM[i] += NORMAL_01 * b * .5f;
	}
	

#ifdef GUIDED_MUTATIONS
	nAccumulations = 0;
#endif
}

// This implementation makes it less likely to gain connexions when many are already populated.
// It also favors connecting from the input or to the output, to encourage parallelism.
// Note that a child node can be connected to itself.
void GenotypeNode::addConnexion() {
	if (children.size() == 0) return;
	int _inputBias = (int)((float)children.size() / 1.5f);
	int _outputBias = (int)((float)children.size() / 1.5f);
	int c1, c2;
	const int maxAttempts = 3;
	bool alreadyExists;
	int dInSize, oOutSize;
	for (int i = 0; i < maxAttempts; i++) {
		c1 = INT_0X(children.size() + 2 + _inputBias) - 1 - _inputBias; // in [-1-inputBias, children.size()]
		c2 = INT_0X(children.size() + 2 + _outputBias); // in [0, children.size() + outputBias+1]

		if (c1 <= -1) {
			c1 = INPUT_ID; 
			oOutSize = inputSize;
		}
		else if (c1 == children.size()) {
			c1 = MODULATION_ID; 
			oOutSize = 2;
		} 
		else oOutSize = children[c1]->outputSize;

		if (c2 > children.size()) {
			dInSize = outputSize;
			c2 = (int)children.size();
		}
		else if (c2 == children.size()) {
			dInSize = 2;
			c2 = MODULATION_ID;
		}
		else dInSize = children[c2]->inputSize;

		alreadyExists = false;
		for (int j = 0; j < childrenConnexions.size(); j++) {
			if (childrenConnexions[j].originID == c1 && childrenConnexions[j].destinationID == c2) {
				alreadyExists = true;
				break;
			}
		}

		if (alreadyExists) continue;

		// ZERO initialization to minimize disturbance of the network
		childrenConnexions.emplace_back(c1, c2, dInSize, oOutSize, GenotypeConnexion::ZERO);
		break;
	}
}
void GenotypeNode::removeConnexion() {
	if (childrenConnexions.size() <= 1) return; 
	int id = INT_0X(childrenConnexions.size());
	//if (origin = input && childrenConnexions[id].destinationID == MODULATION_ID) return; // not allowed to disconnect input->neuromodulation.
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
	int inputBias = (int)((float)children.size() / 1.5f);
	int outputBias = (int)((float)children.size() / 1.5f);
	int oID, dID;
	int destinationsInputSize, originOutputSize;

	oID = INT_0X((int)children.size() + inputBias + 1);
	if (oID >= children.size()) { // the incoming connexion comes from the parent's input
		oID = INPUT_ID;
		originOutputSize = inputSize;
	}
	else {
		originOutputSize = children[oID]->outputSize;
	}

	dID = INT_0X((int)children.size() + outputBias + 1);
	if (dID >= children.size()) { // the outgoing connexion goes to the parent's output
		dID = (int)children.size() + 1;
		destinationsInputSize = outputSize;
	}
	else {
		destinationsInputSize = children[dID]->inputSize;
	}

	int newChildID = (int)children.size();

	childrenConnexions.emplace_back(oID, newChildID, child->inputSize,      originOutputSize, GenotypeConnexion::ZERO);
	childrenConnexions.emplace_back(newChildID, dID, destinationsInputSize, child->outputSize, GenotypeConnexion::ZERO);
	
	children.emplace_back(child);

	int i0 = concatenatedChildrenInputBeacons[newChildID];
	childrenInBias.insert(childrenInBias.begin() + i0, child->inputSize, 0.0f);
	for (int i = i0; i < i0 + child->inputSize; i++) {
		childrenInBias[i] = NORMAL_01 * .2f;
	}

	computeBeacons();
}
void GenotypeNode::removeChild(int rID) {

	// Erase connexions that lead to the removed child. Very slow algorithm, but does not matter here.
	// Should instead copy connexions to keep, and redirect the vector there.
	int initialSize = (int)childrenConnexions.size();
	int nRemovals = 0;
	int i = 0;
	while (i < initialSize - nRemovals) {
		if (childrenConnexions[i].destinationID == rID || childrenConnexions[i].originID == rID) {
			childrenConnexions.erase(childrenConnexions.begin() + i);
			i--;
			nRemovals++;
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
	auto i0 = childrenInBias.begin() + concatenatedChildrenInputBeacons[rID];
	childrenInBias.erase(i0, i0 + children[rID]->inputSize);

	children.erase(children.begin() + rID);
	computeBeacons();
}

bool GenotypeNode::incrementInputSize() {
	if (inputSize >= MAX_BLOCK_INPUT_SIZE) return false;
	inputSize++;
	for (int i = 0; i < childrenConnexions.size(); i++) {
		if (childrenConnexions[i].originID == INPUT_ID) {
			incrementOriginOutputSize(i);
		}
	}
	return true;
}
void GenotypeNode::onChildInputSizeIncremented(GenotypeNode* modifiedType) {
	int id;
	for (int i = 0; i < childrenConnexions.size(); i++) {
		id = childrenConnexions[i].destinationID;
		if (id != children.size() && id != MODULATION_ID && children[id] == modifiedType) {
			incrementDestinationInputSize(i);
		}
	}
	int nInsertions = 0;
	for (int i = 0; i < children.size(); i++) {
		if (children[i] == modifiedType) {
			childrenInBias.insert(childrenInBias.begin() + concatenatedChildrenInputBeacons[i] + nInsertions, NORMAL_01 * .2f);
			nInsertions++;
		}
	}
	if (nInsertions > 0) {
		computeBeacons();
	}
}
// nColumns++;
void GenotypeNode::incrementOriginOutputSize(int i) {
	int nColumns;

	if (childrenConnexions[i].originID == INPUT_ID) {
		nColumns = inputSize;
	}
	else {
		nColumns = children[childrenConnexions[i].originID]->outputSize;
	}

	GenotypeConnexion newConnexion = GenotypeConnexion(
		childrenConnexions[i].originID,
		childrenConnexions[i].destinationID,
		childrenConnexions[i].nLines,
		nColumns,
		GenotypeConnexion::ZERO
	);

	int idNew = 0, idOld = 0;
	for (int j = 0; j < childrenConnexions[i].nLines; j++) {
		for (int k = 0; k < childrenConnexions[i].nColumns; k++) {

			newConnexion.A[idNew] = childrenConnexions[i].A[idOld];
			newConnexion.B[idNew] = childrenConnexions[i].B[idOld];
			newConnexion.C[idNew] = childrenConnexions[i].C[idOld];
			newConnexion.D[idNew] = childrenConnexions[i].D[idOld];
			newConnexion.eta[idNew] = childrenConnexions[i].eta[idOld];
			newConnexion.w[idNew] = childrenConnexions[i].w[idOld];
			newConnexion.alpha[idNew] = childrenConnexions[i].alpha[idOld];
#ifdef CONTINUOUS_LEARNING
			newConnexion.gamma[idNew] = childrenConnexions[i].gamma[idOld];
#endif

			idNew++;
			idOld++;
		}

		idNew++;
	}

	childrenConnexions[i] = newConnexion;
}


bool GenotypeNode::incrementOutputSize() {
	if (outputSize >= MAX_BLOCK_OUTPUT_SIZE) return false;
	outputSize++;
	for (int i = 0; i < childrenConnexions.size(); i++) {
		if (childrenConnexions[i].destinationID == children.size()) {
			incrementDestinationInputSize(i);
		}
	}
	childrenInBias.push_back(NORMAL_01*.2f);
	return true;
}
void GenotypeNode::onChildOutputSizeIncremented(GenotypeNode* modifiedType) {
	int id;
	for (int i = 0; i < childrenConnexions.size(); i++) {
		id = childrenConnexions[i].originID;
		if (id != INPUT_ID && id != MODULATION_ID && children[id] == modifiedType) {
			incrementOriginOutputSize(i);
		}
	}
}
// nLines++;
void GenotypeNode::incrementDestinationInputSize(int i) {
	int nLines;

	if (childrenConnexions[i].destinationID == children.size()) {
		nLines = outputSize;
	}
	else {
		nLines = children[childrenConnexions[i].destinationID]->inputSize;
	}
	GenotypeConnexion newConnexion = GenotypeConnexion(
		childrenConnexions[i].originID,
		childrenConnexions[i].destinationID,
		nLines,
		childrenConnexions[i].nColumns,
		GenotypeConnexion::ZERO
	);

	int idNew = 0, idOld = 0;
	for (int j = 0; j < childrenConnexions[i].nLines; j++) {
		for (int k = 0; k < childrenConnexions[i].nColumns; k++) {

			newConnexion.A[idNew] = childrenConnexions[i].A[idOld];
			newConnexion.B[idNew] = childrenConnexions[i].B[idOld];
			newConnexion.C[idNew] = childrenConnexions[i].C[idOld];
			newConnexion.D[idNew] = childrenConnexions[i].D[idOld];
			newConnexion.eta[idNew] = childrenConnexions[i].eta[idOld];
			newConnexion.w[idNew] = childrenConnexions[i].w[idOld];
			newConnexion.alpha[idNew] = childrenConnexions[i].alpha[idOld];
#ifdef CONTINUOUS_LEARNING
			newConnexion.gamma[idNew] = childrenConnexions[i].gamma[idOld];
#endif

			idNew++;
			idOld++;
		}
	}

	idNew += childrenConnexions[i].nColumns;
	childrenConnexions[i] = newConnexion; // yeah... (deep) copy assignement.
}


bool GenotypeNode::decrementInputSize(int id) {
	if (inputSize <= 1) return false;
	inputSize--;

	for (int i = 0; i < childrenConnexions.size(); i++) {
		if (childrenConnexions[i].originID == INPUT_ID) {
			decrementOriginOutputSize(i, id);
		}
	}
	return true;
}
void GenotypeNode::onChildInputSizeDecremented(GenotypeNode* modifiedType, int id) {
	int nID;
	for (int i = 0; i < childrenConnexions.size(); i++) {
		nID = childrenConnexions[i].destinationID;
		if (nID != children.size() && nID != MODULATION_ID && children[nID] == modifiedType) {
			decrementDestinationInputSize(i, id);
		}
	}
	int nErasures = 0;
	for (int i = 0; i < children.size(); i++) {
		if (children[i] == modifiedType) {
			childrenInBias.erase(childrenInBias.begin() + concatenatedChildrenInputBeacons[i] + id - nErasures);
			nErasures++;
		}
	}
	if (nErasures > 0) {
		computeBeacons();
	}
}
// nColumns--;
void GenotypeNode::decrementOriginOutputSize(int i, int id) {
	int nColumns;

	if (childrenConnexions[i].originID == INPUT_ID) {
		nColumns = inputSize;
	}
	else {
		nColumns = children[childrenConnexions[i].originID]->outputSize;
	}

	GenotypeConnexion newConnexion = GenotypeConnexion(
		childrenConnexions[i].originID,
		childrenConnexions[i].destinationID,
		childrenConnexions[i].nLines,
		nColumns,
		GenotypeConnexion::ZERO
	);

	int idNew = 0, idOld = 0;
	for (int j = 0; j < childrenConnexions[i].nLines; j++) {
		for (int k = 0; k < childrenConnexions[i].nColumns - 1; k++) {

			if (k == id) {
				idOld++;
				continue;
			}
			newConnexion.A[idNew] = childrenConnexions[i].A[idOld];
			newConnexion.B[idNew] = childrenConnexions[i].B[idOld];
			newConnexion.C[idNew] = childrenConnexions[i].C[idOld];
			newConnexion.D[idNew] = childrenConnexions[i].D[idOld];
			newConnexion.eta[idNew] = childrenConnexions[i].eta[idOld];
			newConnexion.w[idNew] = childrenConnexions[i].w[idOld];
			newConnexion.alpha[idNew] = childrenConnexions[i].alpha[idOld];
#ifdef CONTINUOUS_LEARNING
			newConnexion.gamma[idNew] = childrenConnexions[i].gamma[idOld];
#endif
			idNew++;
			idOld++;
		}
	}

	childrenConnexions[i] = newConnexion;
}


bool GenotypeNode::decrementOutputSize(int id) {
	if (outputSize <= 1) return false;
	outputSize--;

	for (int i = 0; i < childrenConnexions.size(); i++) {
		if (childrenConnexions[i].destinationID == children.size()) {
			decrementDestinationInputSize(i, id);
		}
	}
	childrenInBias.erase(childrenInBias.begin() + concatenatedChildrenInputBeacons[children.size()] + id);
	return true;
}
void GenotypeNode::onChildOutputSizeDecremented(GenotypeNode* modifiedType, int id) {
	int nID;
	for (int i = 0; i < childrenConnexions.size(); i++) {
		nID = childrenConnexions[i].originID;
		if (nID != INPUT_ID && nID != MODULATION_ID && children[nID] == modifiedType) {
			decrementOriginOutputSize(i, id);
		}
	}
}
// nLines++;
void GenotypeNode::decrementDestinationInputSize(int i, int id) {
	int nLines;

	if (childrenConnexions[i].destinationID == children.size()) {
		nLines = outputSize;
	}
	else {
		nLines = children[childrenConnexions[i].destinationID]->inputSize;
	}
	GenotypeConnexion newConnexion = GenotypeConnexion(
		childrenConnexions[i].originID,
		childrenConnexions[i].destinationID,
		nLines,
		childrenConnexions[i].nColumns,
		GenotypeConnexion::ZERO
	);

	int idNew = 0, idOld = 0;
	for (int j = 0; j < childrenConnexions[i].nLines; j++) {
		if (j == id) {
			idOld += childrenConnexions[i].nColumns;
			continue;
		}
		for (int k = 0; k < childrenConnexions[i].nColumns - 1; k++) {
			newConnexion.A[idNew] = childrenConnexions[i].A[idOld];
			newConnexion.B[idNew] = childrenConnexions[i].B[idOld];
			newConnexion.C[idNew] = childrenConnexions[i].C[idOld];
			newConnexion.D[idNew] = childrenConnexions[i].D[idOld];
			newConnexion.eta[idNew] = childrenConnexions[i].eta[idOld];
			newConnexion.w[idNew] = childrenConnexions[i].w[idOld];
			newConnexion.alpha[idNew] = childrenConnexions[i].alpha[idOld];
#ifdef CONTINUOUS_LEARNING
			newConnexion.gamma[idNew] = childrenConnexions[i].gamma[idOld];
#endif

			idNew++;
			idOld++;
		}
	}

	childrenConnexions[i] = newConnexion;
}


void GenotypeNode::getNnonLinearities(std::vector<int>& genomeState) {
	constexpr int modulationMultiplier = 0; // must be set to the same value in Phenotype::forward. TODO cleaner.
	int n = outputSize + 2 * modulationMultiplier;
	for (int i = 0; i < children.size(); i++) {
		if (genomeState[children[i]->position] == 0) children[i]->getNnonLinearities(genomeState);
		n += genomeState[children[i]->position];
	}
	genomeState[position] = n;
}

void GenotypeNode::updateDepth(std::vector<int>& genomeState) {
	int dmax = 0;
	for (int i = 0; i < children.size(); i++) {
		if (genomeState[children[i]->position] == 0) children[i]->updateDepth(genomeState); // simple neurons state is at 1 already.
		if (children[i]->depth > dmax) dmax = children[i]->depth;
	}
	depth = dmax + 1;
	genomeState[position] = 1;
}

void GenotypeNode::computeInArraySize(std::vector<int>& genomeState) { 
	int s = inputSize;
	for (int i = 0; i < children.size(); i++) {
		if (genomeState[children[i]->position] == 0) {
			children[i]->computeInArraySize(genomeState); 
		}
		s += genomeState[children[i]->position];
	}
	genomeState[position] = s;
}

void GenotypeNode::computeOutArraySize(std::vector<int>& genomeState) {  
	int s = outputSize;
	for (int i = 0; i < children.size(); i++) {
		if (genomeState[children[i]->position] == 0) {
			children[i]->computeOutArraySize(genomeState);
		}
		s += genomeState[children[i]->position];
	}
	genomeState[position] = s;
}

#ifdef SATURATION_PENALIZING
// Used to compute the size of the array containing the average saturations of the phenotype.
void GenotypeNode::computeSaturationArraySize(std::vector<int>& genomeState) {
	int s = inputSize + 2 + outputSize;
	for (int i = 0; i < children.size(); i++) {
		if (genomeState[children[i]->position] == 0) {
			children[i]->computeSaturationArraySize(genomeState);
		}
		s += genomeState[children[i]->position];
	}
	genomeState[position] = s;
}
#endif 

void GenotypeNode::copyParameters(GenotypeNode* n) {
	if (n->nodeType != COMPLEX) {
		nodeType = n->nodeType;
		inputSize = n->inputSize;
		outputSize = n->outputSize;
		depth = 0;
		position = n->position;
		closestNode = NULL;
		phenotypicMultiplicity = n->phenotypicMultiplicity;
	}
	else {
		nodeType = n->nodeType;
		inputSize = n->inputSize;
		outputSize = n->outputSize;
		childrenInBias.assign(n->childrenInBias.begin(), n->childrenInBias.end());
		biasM[0] = n->biasM[0];
		biasM[1] = n->biasM[1];
		sumChildrenInputSizes = n->sumChildrenInputSizes;
		depth = n->depth;
		position = n->position;
		mutationalDistance = n->mutationalDistance;

		concatenatedChildrenInputBeacons.assign(n->concatenatedChildrenInputBeacons.begin(), n->concatenatedChildrenInputBeacons.end());

		childrenConnexions.reserve((int)((float)n->childrenConnexions.size() * 1.5f));
		for (int j = 0; j < n->childrenConnexions.size(); j++) {
			childrenConnexions.emplace_back(n->childrenConnexions[j]);
		}
#ifdef GUIDED_MUTATIONS
		nAccumulations = n->nAccumulations;
#endif
		phenotypicMultiplicity = n->phenotypicMultiplicity; 		
	}
}

bool GenotypeNode::hasChild(std::vector<int>& checked, GenotypeNode* potentialChild) {
	if (depth <= potentialChild->depth) return false;

	for (int i = 0; i < (int)children.size(); i++) {
		if (checked[children[i]->position] == 1) continue;
		if (children[i] == potentialChild) return true;
		if (children[i]->hasChild(checked,potentialChild)) return true;
		checked[children[i]->position] = 1;
	}
	return false;
}