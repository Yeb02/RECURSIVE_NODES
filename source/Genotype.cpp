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
			eta[i] = .8f;
			A[i] = 0.0f;
			B[i] = 0.0f;
			C[i] = 0.0f;
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
			A[i] = NORMAL_01;
			B[i] = NORMAL_01 * .2f;
			C[i] = NORMAL_01 * .2f;
			alpha[i] = NORMAL_01;
			eta[i] = UNIFORM_01 * .5f + .5f;
			w[i] = NORMAL_01;
#endif 
		}
	}
	else { // IDENTITY
		float v;
		for (int i = 0; i < nLines * nColumns; i++) {
			v = i % nLines == 0 ? 1.0f : 0.0f;
#if defined RISI_NAJARRO_2020
			A[i] = 0.0f;
			B[i] = 0.0f;
			C[i] = v;
			D[i] = 0.0f;
			eta[i] = 0.1f;
#elif defined USING_NEUROMODULATION
			A[i] = NORMAL_01;
			B[i] = NORMAL_01 * .2f;
			C[i] = NORMAL_01 * .2f;
			alpha[i] = 0.0f;
			eta[i] = .8f;
			w[i] = 1.0f;
#endif 
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
	int rID, listID, matrixID;
	const float pMutation = .5f; // TODO
	float r;

#if defined RISI_NAJARRO_2020
	constexpr int nArrays = 5;
#elif defined USING_NEUROMODULATION
	constexpr int nArrays = 6;
#endif 

	// Mutate int(nArrays*Pmutation*nParam) parameters in the inter-children connexions.

	std::vector<int> ids;
	int l = 0;
	for (int i = 0; i < childrenConnexions.size(); i++) {
		ids.push_back(l);
		l += childrenConnexions[i].nLines * childrenConnexions[i].nColumns * nArrays;
	}
	ids.push_back(l);

	float _nParams = (float)l;
	int _nMutations = (int)std::floor(_nParams * pMutation);

	for (int i = 0; i < _nMutations; i++) {
		rID = (int)(UNIFORM_01 * (float)_nParams);

		int j = 0;
		while (rID >= ids[j + 1]) { j++; }
		listID = j;
		j = (rID - ids[listID]) % nArrays;
		matrixID = (rID - ids[listID] - j) / nArrays;

		r = .2f * NORMAL_01;

#if defined RISI_NAJARRO_2020
		switch (j) {
		case 0: childrenConnexions[listID].A[matrixID] += r;
		case 1: childrenConnexions[listID].B[matrixID] += r;
		case 2: childrenConnexions[listID].C[matrixID] += r;
		case 3: childrenConnexions[listID].D[matrixID] += r;
		case 4: childrenConnexions[listID].eta[matrixID] += r;
		}
#elif defined USING_NEUROMODULATION
		switch (j) {
		case 0: childrenConnexions[listID].A[matrixID] += r;
		case 1: childrenConnexions[listID].B[matrixID] += r;
		case 2: childrenConnexions[listID].C[matrixID] += r;
		case 3: childrenConnexions[listID].alpha[matrixID] += r;
		case 4: childrenConnexions[listID].w[matrixID] += r;
		case 5:
			float eta = childrenConnexions[listID].eta[matrixID];
			childrenConnexions[listID].eta[matrixID] += r * eta * (1 - eta);
		}
#endif 
	}

	for (int i = 0; i < outputSize; i++) {
		r = .2f * NORMAL_01;
		bias[i] += r;
	}

#ifdef USING_NEUROMODULATION
	for (int i = 0; i < outputSize; i++) {
		r = .2f * NORMAL_01;
		wNeuromodulation[i] += r;
	}
	r = .2f * NORMAL_01;
	neuromodulationBias += r;
#endif 
}

void GenotypeNode::connect() {
	if (children.size() == 0) return;
	int c1, c2;
	// this implementation makes it less likely to gain connexions when many are already populated
	const int maxAttempts = 10;
	bool alreadyExists;
	int dInSize, oOutSize;
	for (int i = 0; i < maxAttempts; i++) {
		alreadyExists = false;
		c1 = (int)(UNIFORM_01 * (float)(children.size() + 1)) - 1; // in [-1, children.size() - 1]
		c2 = (c1 + 2 + (int)(UNIFORM_01 * (float)children.size())) % children.size(); //guarantees c1 != c2. in [0, children.size()]

		if (c1 == -1) {
			c1 = INPUT_ID; //INPUT_ID could be != -1
			oOutSize = inputSize;
		}
		else oOutSize = children[c1]->outputSize;

		if (c2 == children.size()) {
			dInSize = outputSize;
		}
		else dInSize = children[c2]->inputSize;

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

	childrenConnexions.emplace_back(oID, children.size(), child->inputSize, originOutputSize, GenotypeConnexion::ZERO);
	childrenConnexions.emplace_back(children.size(), dID, destinationsInputSize, child->outputSize, GenotypeConnexion::ZERO);
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
	bias.push_back(0);
#ifdef USING_NEUROMODULATION
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
	wNeuromodulation.erase(wNeuromodulation.begin() + id);
#endif 
	bias.erase(bias.begin() + id);

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
