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

	SET_BINOMIAL(s, .5f); // gamma and eta.
	float factor = .3f / (float)s;

	if (init == ZERO || (init == IDENTITY && nLines != nColumns)) {
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
	else { // IDENTITY. Used in top node boxing. Must "invert" tanh in certain cases ! nLines = nColumns.
		int i = 0;
		float v;
		for (int l = 0; l < nLines; l++) {
			for (int c = 0; c < nColumns; c++) {

				// tanh'(0) = 1, so * 1.25 is a correct-ish approximation to invert. 
				v = l == c ? 1.25f : 0.0f;
				A[i] = NORMAL_01 * .2f;
				B[i] = NORMAL_01 * .2f;
				C[i] = NORMAL_01 * .2f;
				D[i] = NORMAL_01 * .2f;
				alpha[i] = 0.0f;
				eta[i] = factor * (float) BINOMIAL + .1f;
				w[i] = v;
#ifdef CONTINUOUS_LEARNING
				gamma[i] = factor * (float) BINOMIAL + .1f;
#endif
#ifdef GUIDED_MUTATIONS
				accumulator[i] = 0.0f;
#endif
				i++;
			}
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

	//memcpy(eta.get(), gc.eta.get(), sizeof(float) * s);
	std::copy(eta.get(), eta.get() + s, gc.eta.get()); // todo replace memcpy by std::copy everywhere
	memcpy(A.get(), gc.A.get(), sizeof(float) * s);
	memcpy(B.get(), gc.B.get(), sizeof(float) * s);
	memcpy(C.get(), gc.C.get(), sizeof(float) * s);
	memcpy(D.get(), gc.D.get(), sizeof(float) * s);
	memcpy(alpha.get(), gc.alpha.get(), sizeof(float) * s);
	memcpy(w.get(), gc.w.get(), sizeof(float) * s);

#ifdef CONTINUOUS_LEARNING
	gamma = std::make_unique<float[]>(s);
	memcpy(gamma.get(), gc.gamma.get(), sizeof(float) * s);
#endif

#ifdef GUIDED_MUTATIONS
	accumulator = std::make_unique<float[]>(s);
	memcpy(accumulator.get(), gc.accumulator.get(), sizeof(float) * s);
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

	memcpy(eta.get(), gc.eta.get(), sizeof(float) * s);
	memcpy(A.get(), gc.A.get(), sizeof(float) * s);
	memcpy(B.get(), gc.B.get(), sizeof(float) * s);
	memcpy(C.get(), gc.C.get(), sizeof(float) * s);
	memcpy(alpha.get(), gc.alpha.get(), sizeof(float) * s);
	memcpy(w.get(), gc.w.get(), sizeof(float) * s);

#ifdef CONTINUOUS_LEARNING
	gamma = std::make_unique<float[]>(s);
	memcpy(gamma.get(), gc.gamma.get(), sizeof(float) * s);
#endif

#ifdef GUIDED_MUTATIONS
	accumulator = std::make_unique<float[]>(s);
	memcpy(accumulator.get(), gc.accumulator.get(), sizeof(float) * s);
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
	float r, r2;
#ifdef CONTINUOUS_LEARNING
	constexpr int nArrays = 8;  // added gamma
#else 
	constexpr int nArrays = 7;
#endif
	constexpr float normalFactor = .5f; // .4f ??
	constexpr float sumFactor = .5f; // .4f ??
	// Lowering those 2 to .1f on cartpole yields intriguing results: not only is convergence
	// slower, but also worse. And average population score, after mutation, is much worse too !
	// The found optimum is less stable.

	constexpr float pMutation = .2f; // .2f ??
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

			r = normalFactor * NORMAL_01;
			r2 = sumFactor * NORMAL_01;

			switch (arrayN) {
			case 0: aPtr = childrenConnexions[listID].A.get(); break;
			case 1: aPtr = childrenConnexions[listID].B.get(); break;
			case 2: aPtr = childrenConnexions[listID].C.get(); break;
			case 3: aPtr = childrenConnexions[listID].D.get(); break;
			case 4: aPtr = childrenConnexions[listID].alpha.get(); break;
			case 5: aPtr = childrenConnexions[listID].w.get(); break;
			case 6:  // eta
				aPtr = childrenConnexions[listID].eta.get(); 
				// this allows for high precision mutations when eta (or 1- eta) is close to 1.
				aPtr[matrixID] = UNIFORM_01 > .05f ?
						aPtr[matrixID] * (1 - aPtr[matrixID]) * (UNIFORM_01-.5f) + aPtr[matrixID] :
						aPtr[matrixID] * .6f + UNIFORM_01 * .4f;
				break;
#ifdef CONTINUOUS_LEARNING
			case 7:  // gamma
				aPtr = childrenConnexions[listID].gamma.get(); 
				aPtr[matrixID] = UNIFORM_01 > .05f ?
					aPtr[matrixID] * (1 - aPtr[matrixID]) * (UNIFORM_01 - .5f) + aPtr[matrixID] :
					aPtr[matrixID] * .6f + UNIFORM_01 * .4f;
				break;
#endif
			}

			if (arrayN < 6) {
				r = normalFactor * NORMAL_01;
				r2 = sumFactor * NORMAL_01;
				aPtr[matrixID] *= .9f + r;
				aPtr[matrixID] += r2;
			}
		}

#ifdef GUIDED_MUTATIONS
		if (nApparitions == 0) continue;
		float invFactor = 1.0f / (float)nApparitions; // should be outside the loop, here for readability
		for (int k = 0; k < size; k++) {
			childrenConnexions[listID].w[k] += .3f * childrenConnexions[listID].accumulator[k] * invFactor;
		}
#endif
	}

	for (int i = 0; i < sumChildrenInputSizes + outputSize; i++) {
		r = normalFactor * NORMAL_01;
		r2 = sumFactor * NORMAL_01;
		childrenInBias[i] *= .95f + r;
		childrenInBias[i] += r2;
	}

	r = normalFactor * NORMAL_01;
	r2 = sumFactor * NORMAL_01;
	biasM[0] *= .95f + r;
	biasM[0] += r2;

	r = normalFactor * NORMAL_01;
	r2 = sumFactor * NORMAL_01;
	biasM[1] *= .95f + r;
	biasM[1] += r2;

#ifdef GUIDED_MUTATIONS
	// Important !
	nApparitions = 0;
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
		c1 = INT_0X(children.size() + 1 + _inputBias) - 1 - _inputBias; // in [-1-inputBias, children.size() - 1]
		c2 = INT_0X(children.size() + 1 + _outputBias); // in [0, children.size() + outputBias]

		if (c1 <= -1) {
			c1 = INPUT_ID; //INPUT_ID could be != -1 in future versions.
			oOutSize = inputSize;
		}
		else oOutSize = children[c1]->outputSize;

		if (c2 >= children.size()) {
			dInSize = outputSize;
			c2 = (int)children.size();
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
	if (childrenConnexions.size() <= 1 + 1) return; // not allowed to fall below 1 connexion + neuromodulation. Still can happen in a convoluted case.
	int id = INT_0X(childrenConnexions.size());
	if (childrenConnexions[id].destinationID == MODULATION_ID) return; // not allowed to disconnect neuromodulation.
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

	childrenConnexions.emplace_back(oID, (int)children.size(), child->inputSize,      originOutputSize, GenotypeConnexion::ZERO);
	childrenConnexions.emplace_back((int)children.size(), dID, destinationsInputSize, child->outputSize, GenotypeConnexion::ZERO);
	children.push_back(child);

	childrenInBias.insert(childrenInBias.begin() + sumChildrenInputSizes, child->inputSize, 0);
	for (int i = sumChildrenInputSizes; i < sumChildrenInputSizes + child->inputSize; i++) {
		childrenInBias[i] = NORMAL_01 * .2f;
	}

	computeBeacons();
}
void GenotypeNode::removeChild(int rID) {

	// Erase connexions that lead to the removed child. Slow algorithm, but does not matter here.
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
	childrenInBias.erase(childrenInBias.begin(), childrenInBias.begin() + children[rID]->inputSize);

	children.erase(children.begin() + rID);
	computeBeacons();
}

void GenotypeNode::incrementInputSize() {
	inputSize++;
	for (int i = 0; i < childrenConnexions.size(); i++) {
		if (childrenConnexions[i].originID == INPUT_ID) {
			incrementOriginOutputSize(i);
		}
	}
}
void GenotypeNode::onChildInputSizeIncremented(GenotypeNode* modifiedType) {
	int id;
	int nInsertions = 0;
	for (int i = 0; i < childrenConnexions.size(); i++) {
		id = childrenConnexions[i].destinationID;
		if (id != children.size() && id != MODULATION_ID && children[id] == modifiedType) {
			incrementDestinationInputSize(i);
			childrenInBias.insert(childrenInBias.begin() + concatenatedChildrenInputBeacons[id] + nInsertions, NORMAL_01 * .2f);
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


void GenotypeNode::incrementOutputSize() {
	outputSize++;
	for (int i = 0; i < childrenConnexions.size(); i++) {
		if (childrenConnexions[i].destinationID == children.size()) {
			incrementDestinationInputSize(i);
		}
	}
	childrenInBias.push_back(NORMAL_01*.2f);
}
void GenotypeNode::onChildOutputSizeIncremented(GenotypeNode* modifiedType) {
	int id;
	for (int i = 0; i < childrenConnexions.size(); i++) {
		id = childrenConnexions[i].originID;
		if (id != INPUT_ID && children[id] == modifiedType) {
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


void GenotypeNode::decrementInputSize(int id) {
	if (inputSize == 1) return;
	inputSize--;

	for (int i = 0; i < childrenConnexions.size(); i++) {
		if (childrenConnexions[i].originID == INPUT_ID) {
			decrementOriginOutputSize(i, id);
		}
	}
}
void GenotypeNode::onChildInputSizeDecremented(GenotypeNode* modifiedType, int id) {
	int nID;
	int nErasures = 0;
	for (int i = 0; i < childrenConnexions.size(); i++) {
		nID = childrenConnexions[i].destinationID;
		if (nID != children.size() && nID != MODULATION_ID && children[nID] == modifiedType) {
			decrementDestinationInputSize(i, id);
			childrenInBias.erase(childrenInBias.begin() + concatenatedChildrenInputBeacons[nID] + id - nErasures);
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


void GenotypeNode::decrementOutputSize(int id) {
	if (outputSize == 1) return;
	outputSize--;

	for (int i = 0; i < childrenConnexions.size(); i++) {
		if (childrenConnexions[i].destinationID == children.size()) {
			decrementDestinationInputSize(i, id);
		}
	}
	childrenInBias.erase(childrenInBias.begin() + concatenatedChildrenInputBeacons[children.size()] + id);
}
void GenotypeNode::onChildOutputSizeDecremented(GenotypeNode* modifiedType, int id) {
	int nID;
	for (int i = 0; i < childrenConnexions.size(); i++) {
		nID = childrenConnexions[i].originID;
		if (nID != INPUT_ID && children[nID] == modifiedType) {
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


void GenotypeNode::updateDepth(std::vector<int>& genomeState) {
	int dmax = 0;
	for (int i = 0; i < children.size(); i++) {
		if (genomeState[children[i]->position] == 0) children[i]->updateDepth(genomeState); // simple neurons state is at 1 already.
		if (children[i]->depth > dmax) dmax = children[i]->depth;
	}
	depth = dmax + 1;
	genomeState[position] = 1;
}

void GenotypeNode::computeInArraySize(std::vector<int>& genomeState) { //// TODO BOTH ARE THE SAME ?
	int s = inputSize;
	for (int i = 0; i < children.size(); i++) {
		if (genomeState[children[i]->position] == 0) {
			children[i]->computeInArraySize(genomeState); 
		}
		s += genomeState[children[i]->position];
	}
	genomeState[position] = s;
}

void GenotypeNode::computeOutArraySize(std::vector<int>& genomeState) {  //// TODO BOTH ARE THE SAME ?
	int s = outputSize;
	for (int i = 0; i < children.size(); i++) {
		if (genomeState[children[i]->position] == 0) {
			children[i]->computeOutArraySize(genomeState);
		}
		s += genomeState[children[i]->position];
	}
	genomeState[position] = s;
}


void GenotypeNode::copyParameters(GenotypeNode* n) {
	if (n->isSimpleNeuron) {
		isSimpleNeuron = true;
		f = n->f;
		inputSize = n->inputSize;
		outputSize = n->outputSize;
		depth = 0;
		position = n->position;
		closestNode = NULL;
	}
	else {
		isSimpleNeuron = false;
		f = NULL;
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
		nApparitions = n->nApparitions;
#endif
		
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