#include "Genotype.h"


GenotypeConnexion::GenotypeConnexion(int oID, int dID, int nLines, int nColumns, GenotypeConnexion::initType init) :
	originID(oID), destinationID(dID), nLines(nLines), nColumns(nColumns)
{
	int s = nLines * nColumns;
	eta = std::make_unique<float[]>(s);
	A = std::make_unique<float[]>(s);
	B = std::make_unique<float[]>(s);
	C = std::make_unique<float[]>(s);

#if defined RISI_NAJARRO_2020
	D = std::make_unique<float[]>(s);
#elif defined USING_NEUROMODULATION
	alpha = std::make_unique<float[]>(s);
	w = std::make_unique<float[]>(s);
#endif 

	if (init == ZERO || (init == IDENTITY && nLines != nColumns)) {
		for (int i = 0; i < nLines * nColumns; i++) {
#if defined RISI_NAJARRO_2020
			eta[i] = 0.1f;
			A[i] = 0.0f;
			B[i] = 0.0f;
			C[i] = 0.0f;
			D[i] = 0.0f;
#elif defined USING_NEUROMODULATION
			eta[i] = UNIFORM_01;
			A[i] = NORMAL_01 * .2f;
			B[i] = NORMAL_01 * .2f;
			C[i] = NORMAL_01 * .2f;
			alpha[i] = 0.0f;
			w[i] = 0.0f;
#endif 
		}
	}
	else if (init == RANDOM) {
		for (int i = 0; i < nLines * nColumns; i++) {

#if defined RISI_NAJARRO_2020
			A[i] = UNIFORM_01 - .5f;
			B[i] = UNIFORM_01 - .5f;
			C[i] = UNIFORM_01 - .5f;
			D[i] = UNIFORM_01 - .5f;
			eta[i] = UNIFORM_01;
#elif defined USING_NEUROMODULATION
			A[i] = NORMAL_01 * .2f;
			B[i] = NORMAL_01 * .2f;
			C[i] = NORMAL_01 * .2f;
			alpha[i] = NORMAL_01 *.2f;
			eta[i] = UNIFORM_01;
			w[i] = NORMAL_01*.2f;
#endif 
		}
	}
	else { // IDENTITY. Used in top node boxing. Must "invert" tanh in certain cases ! nLines = nColumns.
		int i = 0;
		float v;
		for (int l = 0; l < nLines; l++) {
			for (int c = 0; c < nColumns; c++) {
				v = l == c ? 1.0f : 0.0f;
#if defined RISI_NAJARRO_2020 // there is no way to link 2 nodes with identity in this case.
				A[i] = 0.0f;
				B[i] = 0.0f;
				C[i] = v;
				D[i] = 0.0f;
				eta[i] = 0.1f;
#elif defined USING_NEUROMODULATION // tanh'(0) = 1, so * 1.0 is a correct-ish approximation to invert. 
				A[i] = NORMAL_01 * .2f;
				B[i] = NORMAL_01 * .2f;
				C[i] = NORMAL_01 * .2f;
				alpha[i] = 0.0f;
				eta[i] = UNIFORM_01;
				w[i] = v;
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

	// move() removes ownership from the original pointer. Its use here is kind of an hacky workaround 
	// that vector reallocation calls move constructor AND destructor. So the pointee would be destroyed otherwise.
	// https://stackoverflow.com/questions/41864544/stdvector-calls-contained-objects-destructor-when-reallocating-no-matter-what
	A = std::move(gc.A);
	B = std::move(gc.B);
	C = std::move(gc.C);
	eta = std::move(gc.eta);

#if defined RISI_NAJARRO_2020
	D = std::move(gc.D);
#elif defined USING_NEUROMODULATION
	w = std::move(gc.w);
	alpha = std::move(gc.alpha);
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

	memcpy(eta.get(), gc.eta.get(), sizeof(float) * s);
	memcpy(A.get(), gc.A.get(), sizeof(float) * s);
	memcpy(B.get(), gc.B.get(), sizeof(float) * s);
	memcpy(C.get(), gc.C.get(), sizeof(float) * s);

#if defined RISI_NAJARRO_2020
	D = std::make_unique<float[]>(s);
	memcpy(D.get(), gc.D.get(), sizeof(float) * s);

#elif defined USING_NEUROMODULATION
	alpha = std::make_unique<float[]>(s);
	w = std::make_unique<float[]>(s);
	memcpy(alpha.get(), gc.alpha.get(), sizeof(float) * s);
	memcpy(w.get(), gc.w.get(), sizeof(float) * s);
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

	memcpy(eta.get(), gc.eta.get(), sizeof(float) * s);
	memcpy(A.get(), gc.A.get(), sizeof(float) * s);
	memcpy(B.get(), gc.B.get(), sizeof(float) * s);
	memcpy(C.get(), gc.C.get(), sizeof(float) * s);

#if defined RISI_NAJARRO_2020
	D = std::make_unique<float[]>(s);
	memcpy(D.get(), gc.D.get(), sizeof(float) * s);

#elif defined USING_NEUROMODULATION
	alpha = std::make_unique<float[]>(s);
	w = std::make_unique<float[]>(s);
	memcpy(alpha.get(), gc.alpha.get(), sizeof(float) * s);
	memcpy(w.get(), gc.w.get(), sizeof(float) * s);
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
	concatenatedChildrenInputLength = s;
}

void GenotypeNode::mutateFloats() {
	float r, r2;
	
#if defined RISI_NAJARRO_2020
	constexpr int nArrays = 5;
#elif defined USING_NEUROMODULATION
	constexpr int nArrays = 6;
#endif 

	constexpr float normalFactor = .3f; // .3f ??
	constexpr float sumFactor = .4f; // .4f ??
	// Lowering those 2 to .1f on cartpole yields intriguing results: not only is convergence
	// slower, but also worse. And average population score, after mutation, is much worse too !
	// The found optimum is less stable.


	constexpr float pMutation = .2f; // .1f ??
	// Mutate int(nArrays*Pmutation*nParam) parameters in the inter-children connexions.

	int rID, matrixID;
	int _nMutations, _nParams;
	float* aPtr = nullptr;
	for (int listID = 0; listID < childrenConnexions.size(); listID++) { 
		_nParams = childrenConnexions[listID].nLines * childrenConnexions[listID].nColumns * nArrays; 
		SET_BINOMIAL(_nParams, pMutation);
		_nMutations = BINOMIAL;

		for (int k = 0; k <  _nMutations; k++) {
			rID = (int)(UNIFORM_01 * _nParams); 

			int j = rID % nArrays;
			matrixID = rID / nArrays;

			r = normalFactor * NORMAL_01;
			r2 = sumFactor * NORMAL_01;

#if defined RISI_NAJARRO_2020
			switch (j) {
			case 0: childrenConnexions[listID].A[matrixID] += r; break;
			case 1: childrenConnexions[listID].B[matrixID] += r; break;
			case 2: childrenConnexions[listID].C[matrixID] += r; break;
			case 3: childrenConnexions[listID].D[matrixID] += r; break;
			case 4: childrenConnexions[listID].eta[matrixID] += r; break;
			}
#elif defined USING_NEUROMODULATION

			switch (j) {
			case 0: aPtr = childrenConnexions[listID].A.get(); break;
			case 1: aPtr = childrenConnexions[listID].B.get(); break;
			case 2: aPtr = childrenConnexions[listID].C.get(); break;
			case 3: aPtr = childrenConnexions[listID].alpha.get(); break;
			case 4: aPtr = childrenConnexions[listID].w.get(); break;
			case 5: aPtr = childrenConnexions[listID].eta.get(); break;
			}
			if (matrixID == 5) {
				//if (UNIFORM_01 < .05f) {
				//	aPtr[matrixID] = aPtr[matrixID] * .8f + UNIFORM_01 * .2f;
				//}
				//else { // this allows for high precision mutations when eta (or 1- eta) is close to 1.
				//	aPtr[matrixID] += aPtr[matrixID] * (1 - aPtr[matrixID]) * .4f * (UNIFORM_01-.5f);
				//}
				// Both are worth a shot.
				aPtr[matrixID] = aPtr[matrixID] * .8f + UNIFORM_01 * .2f;
			}
			else {
				r = normalFactor * NORMAL_01;
				r2 = sumFactor * NORMAL_01;
				aPtr[matrixID] *= .9f + r;
				aPtr[matrixID] += r2;
			}
		}

#endif 
	}


	for (int i = 0; i < outputSize; i++) {
		r = normalFactor * NORMAL_01;
		r2 = sumFactor * NORMAL_01;
		outBias[i] *= .95f + r;
		outBias[i] += r2;
	}

	for (int i = 0; i < inputSize; i++) {
		r = normalFactor * NORMAL_01;
		r2 = sumFactor * NORMAL_01;
		inBias[i] *= .95f + r;
		inBias[i] += r2;
	}

#ifdef USING_NEUROMODULATION
	for (int i = 0; i < outputSize; i++) {
		r = normalFactor * NORMAL_01;
		r2 = sumFactor * NORMAL_01;
		wNeuromodulation[i] *= .95f + r;
		wNeuromodulation[i] += r2;
	}
	r = normalFactor * NORMAL_01;
	r2 = sumFactor * NORMAL_01;
	neuromodulationBias *= .95f + r;
	neuromodulationBias += r2;
#endif 
}

// This implementation makes it less likely to gain connexions when many are already populated.
// It also favors connecting from the input or to the output, to encourage parallelism.
// Note that a child node can be connected to itself.
void GenotypeNode::connect() {
	if (children.size() == 0) return;
	int inputBias = (int)((float)children.size() / 3.0f);
	int outputBias = (int)((float)children.size() / 3.0f);
	int c1, c2;
	const int maxAttempts = 3;
	bool alreadyExists;
	int dInSize, oOutSize;
	for (int i = 0; i < maxAttempts; i++) {
		c1 = (int)(UNIFORM_01 * (float)(children.size() + 1 + inputBias)) - 1 - inputBias; // in [-1-inputBias, children.size() - 1]
		c2 = (int)(UNIFORM_01 * (float)(children.size() + 1 + outputBias)); // in [0, children.size() + outputBias]

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
		for (int i = 0; i < childrenConnexions.size(); i++) {
			if (childrenConnexions[i].originID == c1 && childrenConnexions[i].destinationID == c2) {
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
void GenotypeNode::disconnect() {
	if (childrenConnexions.size() == 0) return;
	if (childrenConnexions.size() == 1 && children.size() == 0) return;
	int id = (int)(UNIFORM_01 * (float)childrenConnexions.size());
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
	constexpr float parallelBias = 2.0f;
	int oID, dID;
	float r;
	int destinationsInputSize, originOutputSize;

	r = UNIFORM_01;
	oID = (int)((float)(children.size() + 1) * parallelBias * r);
	if (oID >= children.size()) { // the incoming connexion comes from the parent's input
		oID = INPUT_ID;
		originOutputSize = inputSize;
	}
	else {
		originOutputSize = children[oID]->outputSize;
	}

	r = UNIFORM_01;
	dID = (int)((float)(children.size() + 1) * parallelBias * r);
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
}
void GenotypeNode::removeChild(int rID) {
	children.erase(children.begin() + rID);

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
}

void GenotypeNode::incrementInputSize() {
	inputSize++;
	for (int i = 0; i < childrenConnexions.size(); i++) {
		if (childrenConnexions[i].originID == INPUT_ID) {
			incrementOriginOutputSize(i);
		}
	}
#ifdef USING_NEUROMODULATION
	inBias.push_back(0);
#endif 
}
void GenotypeNode::onChildInputSizeIncremented(GenotypeNode* modifiedType) {
	int id;
	for (int i = 0; i < childrenConnexions.size(); i++) {
		id = childrenConnexions[i].destinationID;
		if (id != children.size() && children[id] == modifiedType) {
			incrementDestinationInputSize(i);
		}
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
			newConnexion.eta[idNew] = childrenConnexions[i].eta[idOld];
#if defined RISI_NAJARRO_2020
			newConnexion.D[idNew] = childrenConnexions[i].D[idOld];
#elif defined USING_NEUROMODULATION
			newConnexion.w[idNew] = childrenConnexions[i].w[idOld];
			newConnexion.alpha[idNew] = childrenConnexions[i].alpha[idOld];
#endif 

			idNew++;
			idOld++;
		}

		newConnexion.A[idNew] = 0.0f;
		newConnexion.B[idNew] = 0.0f;
		newConnexion.C[idNew] = 0.0f;
		newConnexion.eta[idNew] = 0.0f;
#if defined RISI_NAJARRO_2020
		newConnexion.D[idNew] = 0.0f;
#elif defined USING_NEUROMODULATION
		newConnexion.alpha[idNew] = 0.0f;
		newConnexion.w[idNew] = 0.0f;
#endif 

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
#ifdef USING_NEUROMODULATION
	outBias.push_back(0);
	wNeuromodulation.push_back(0);
#endif 
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
			newConnexion.eta[idNew] = childrenConnexions[i].eta[idOld];
#if defined RISI_NAJARRO_2020
			newConnexion.D[idNew] = childrenConnexions[i].D[idOld];
#elif defined USING_NEUROMODULATION
			newConnexion.w[idNew] = childrenConnexions[i].w[idOld];
			newConnexion.alpha[idNew] = childrenConnexions[i].alpha[idOld];
#endif 

			idNew++;
			idOld++;
		}
	}
	for (int k = 0; k < childrenConnexions[i].nColumns; k++) {
		newConnexion.A[idNew] = 0.0f;
		newConnexion.B[idNew] = 0.0f;
		newConnexion.C[idNew] = 0.0f;
		newConnexion.eta[idNew] = 0.0f;
#if defined RISI_NAJARRO_2020
		newConnexion.D[idNew] = 0.0f;
#elif defined USING_NEUROMODULATION
		newConnexion.alpha[idNew] = 0.0f;
		newConnexion.w[idNew] = 0.0f;
#endif 
		idNew++;
	}
	childrenConnexions[i] = newConnexion;
}


void GenotypeNode::decrementInputSize(int id) {
	if (inputSize == 1) return;
	inputSize--;

	for (int i = 0; i < childrenConnexions.size(); i++) {
		if (childrenConnexions[i].originID == INPUT_ID) {
			decrementOriginOutputSize(i, id);
		}
	}
#ifdef USING_NEUROMODULATION
	inBias.erase(inBias.begin() + id);
#endif 
}
void GenotypeNode::onChildInputSizeDecremented(GenotypeNode* modifiedType, int id) {
	int nID;
	for (int i = 0; i < childrenConnexions.size(); i++) {
		nID = childrenConnexions[i].destinationID;
		if (nID != children.size() && children[nID] == modifiedType) {
			decrementDestinationInputSize(i, id);
		}
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
			newConnexion.eta[idNew] = childrenConnexions[i].eta[idOld];
#if defined RISI_NAJARRO_2020
			newConnexion.D[idNew] = childrenConnexions[i].D[idOld];
#elif defined USING_NEUROMODULATION
			newConnexion.w[idNew] = childrenConnexions[i].w[idOld];
			newConnexion.alpha[idNew] = childrenConnexions[i].alpha[idOld];
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

#ifdef USING_NEUROMODULATION
	outBias.erase(outBias.begin() + id);
	wNeuromodulation.erase(wNeuromodulation.begin() + id);
#endif 
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
			newConnexion.eta[idNew] = childrenConnexions[i].eta[idOld];
#if defined RISI_NAJARRO_2020
			newConnexion.D[idNew] = childrenConnexions[i].D[idOld];
#elif defined USING_NEUROMODULATION
			newConnexion.w[idNew] = childrenConnexions[i].w[idOld];
			newConnexion.alpha[idNew] = childrenConnexions[i].alpha[idOld];
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
		if (genomeState[children[i]->position] == 0) children[i]->updateDepth(genomeState); // simple neurons state is at 1.
		if (children[i]->depth > dmax) dmax = children[i]->depth;
	}
	depth = dmax + 1;
	genomeState[position] = 1;
}
