#pragma once

#include "GenotypeConnexion.h"

GenotypeConnexion::GenotypeConnexion(NODE_TYPE oType, NODE_TYPE dType, int oID, int dID, int dInSize, int oOutSize, GenotypeConnexion::initType init) :
	originType(oType), destinationType(dType),
	originID(oID), destinationID(dID), 
	nLines(dInSize), nColumns(oOutSize)
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

	if (init == ZERO) {
		for (int i = 0; i < nLines * nColumns; i++) {

			A[i] = NORMAL_01 * .2f;
			B[i] = NORMAL_01 * .2f;
			C[i] = NORMAL_01 * .2f;
			D[i] = NORMAL_01 * .2f;
			alpha[i] = 0.0f;
			eta[i] = factor * (float)BINOMIAL + .1f;
			w[i] = 0.0f;
#ifdef CONTINUOUS_LEARNING
			gamma[i] = factor * (float)BINOMIAL + .1f;
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
			alpha[i] = NORMAL_01 * .2f;
			eta[i] = factor * (float)BINOMIAL + .1f;
			w[i] = NORMAL_01 * .2f;
#ifdef CONTINUOUS_LEARNING
			gamma[i] = factor * (float)BINOMIAL + .1f;
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
	originType = gc.originType;
	destinationType = gc.destinationType;

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
	originType = gc.originType;
	destinationType = gc.destinationType;

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
	originType = gc.originType;
	destinationType = gc.destinationType;

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

void GenotypeConnexion::mutateFloats(float p, float invFactor) {

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

	int size = nLines * nColumns;
	SET_BINOMIAL(size * nArrays, p);
	int _nMutations = BINOMIAL;

	float* aPtr = nullptr;
	for (int k = 0; k < _nMutations; k++) {

		int arrayN = INT_0X(nArrays);
		int matrixID = INT_0X(size);

		switch (arrayN) {
		case 0: aPtr = A.get(); break;
		case 1: aPtr = B.get(); break;
		case 2: aPtr = C.get(); break;
		case 3: aPtr = D.get(); break;
		case 4: aPtr = alpha.get(); break;
		case 5: aPtr = w.get(); break;
		case 6:  aPtr = eta.get(); break;
#ifdef CONTINUOUS_LEARNING
		case 7:  aPtr = gamma.get(); break;
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
	if (invFactor != 0.0f) {
		for (int k = 0; k < size; k++) {
			float rawDelta = accumulator[k] * invFactor;
			w[k] += std::max(std::min(rawDelta, deltaWclipRange), -deltaWclipRange);
			accumulator[k] = 0.0f;
		}
	}
#endif
}

void GenotypeConnexion::incrementDestinationInputSize() {
	int new_size = (nLines + 1) * nColumns;
	float* new_A = new float[new_size];
	float* new_B = new float[new_size];
	float* new_C = new float[new_size];
	float* new_D = new float[new_size];
	float* new_eta = new float[new_size];
	float* new_w = new float[new_size];
	float* new_alpha = new float[new_size];
#ifdef CONTINUOUS_LEARNING
	float* new_gamma = new float[new_size];
#endif
#ifdef GUIDED_MUTATIONS
	float* new_accumulator = new float[new_size];
#endif


	int idNew = 0, idOld = 0;
	for (int j = 0; j < nLines; j++) {
		for (int k = 0; k < nColumns; k++) {

			new_A[idNew] = A[idOld];
			new_B[idNew] = B[idOld];
			new_C[idNew] = C[idOld];
			new_D[idNew] = D[idOld];
			new_eta[idNew] = eta[idOld];
			new_w[idNew] = w[idOld];
			new_alpha[idNew] = alpha[idOld];
#ifdef CONTINUOUS_LEARNING
			new_gamma[idNew] = gamma[idOld];
#endif
#ifdef GUIDED_MUTATIONS
			new_accumulator[idNew] = 0.0f;
#endif


			idNew++;
			idOld++;
		}
	} 
	for (int k = 0; k < nColumns; k++) {
		new_A[idNew] = NORMAL_01 * .2f;
		new_B[idNew] = NORMAL_01 * .2f;
		new_C[idNew] = NORMAL_01 * .2f;
		new_D[idNew] = NORMAL_01 * .2f;
		new_eta[idNew] = UNIFORM_01 * .3f + .1f;
		new_w[idNew] = NORMAL_01 * .2f;
		new_alpha[idNew] = NORMAL_01 * .2f;
#ifdef CONTINUOUS_LEARNING
		new_gamma[idNew] = UNIFORM_01 * .3f + .1f;
#endif
#ifdef GUIDED_MUTATIONS
		new_accumulator[idNew] = 0.0f;
#endif
		idNew++;
	}


	A.reset(new_A);
	B.reset(new_B);
	C.reset(new_C);
	D.reset(new_D);
	eta.reset(new_eta);
	w.reset(new_w);
	alpha.reset(new_A);
#ifdef CONTINUOUS_LEARNING
	gamma.reset(new_gamma);
#endif
#ifdef GUIDED_MUTATIONS
	accumulator.reset(new_accumulator);
#endif

	nLines++;
}

void GenotypeConnexion::incrementOriginOutputSize(){
	int new_size = nLines * (nColumns + 1);
	float* new_A = new float[new_size];
	float* new_B = new float[new_size];
	float* new_C = new float[new_size];
	float* new_D = new float[new_size];
	float* new_eta = new float[new_size];
	float* new_w = new float[new_size];
	float* new_alpha = new float[new_size];
#ifdef CONTINUOUS_LEARNING
	float* new_gamma = new float[new_size];
#endif
#ifdef GUIDED_MUTATIONS
	float* new_accumulator = new float[new_size];
#endif


	int idNew = 0, idOld = 0;
	for (int j = 0; j < nLines; j++) {
		for (int k = 0; k < nColumns; k++) {

			new_A[idNew] = A[idOld];
			new_B[idNew] = B[idOld];
			new_C[idNew] = C[idOld];
			new_D[idNew] = D[idOld];
			new_eta[idNew] = eta[idOld];
			new_w[idNew] = w[idOld];
			new_alpha[idNew] = alpha[idOld];
#ifdef CONTINUOUS_LEARNING
			new_gamma[idNew] = gamma[idOld];
#endif
#ifdef GUIDED_MUTATIONS
			new_accumulator[idNew] = 0.0f;
#endif

			idNew++;
			idOld++;
		}

		new_A[idNew] = NORMAL_01 * .2f;
		new_B[idNew] = NORMAL_01 * .2f;
		new_C[idNew] = NORMAL_01 * .2f;
		new_D[idNew] = NORMAL_01 * .2f;
		new_eta[idNew] = UNIFORM_01 * .3f + .1f;
		new_w[idNew] = NORMAL_01 * .2f;
		new_alpha[idNew] = NORMAL_01 * .2f;
#ifdef CONTINUOUS_LEARNING
		new_gamma[idNew] = UNIFORM_01 * .3f + .1f;
#endif
#ifdef GUIDED_MUTATIONS
		new_accumulator[idNew] = 0.0f;
#endif
		idNew++;
	}

	A.reset(new_A);
	B.reset(new_B);
	C.reset(new_C);
	D.reset(new_D);
	eta.reset(new_eta);
	w.reset(new_w);
	alpha.reset(new_A);
#ifdef CONTINUOUS_LEARNING
	gamma.reset(new_gamma);
#endif
#ifdef GUIDED_MUTATIONS
	accumulator.reset(new_accumulator);
#endif

	nColumns++;
}

void GenotypeConnexion::decrementDestinationInputSize(int id){
	int new_size = (nLines - 1) * nColumns;
	float* new_A = new float[new_size];
	float* new_B = new float[new_size];
	float* new_C = new float[new_size];
	float* new_D = new float[new_size];
	float* new_eta = new float[new_size];
	float* new_w = new float[new_size];
	float* new_alpha = new float[new_size];
#ifdef CONTINUOUS_LEARNING
	float* new_gamma = new float[new_size];
#endif
#ifdef GUIDED_MUTATIONS
	float* new_accumulator = new float[new_size];
#endif


	int idNew = 0, idOld = 0;
	for (int j = 0; j < nLines; j++) {
		if (j == id) {
			idOld += nColumns;
			continue;
		}
		for (int k = 0; k < nColumns; k++) {

			new_A[idNew] = A[idOld];
			new_B[idNew] = B[idOld];
			new_C[idNew] = C[idOld];
			new_D[idNew] = D[idOld];
			new_eta[idNew] = eta[idOld];
			new_w[idNew] = w[idOld];
			new_alpha[idNew] = alpha[idOld];
#ifdef CONTINUOUS_LEARNING
			new_gamma[idNew] = gamma[idOld];
#endif
#ifdef GUIDED_MUTATIONS
			new_accumulator[idNew] = 0.0f;
#endif

			idNew++;
			idOld++;
		}
	}

	A.reset(new_A);
	B.reset(new_B);
	C.reset(new_C);
	D.reset(new_D);
	eta.reset(new_eta);
	w.reset(new_w);
	alpha.reset(new_A);
#ifdef CONTINUOUS_LEARNING
	gamma.reset(new_gamma);
#endif
#ifdef GUIDED_MUTATIONS
	accumulator.reset(new_accumulator);
#endif

	nLines--;
}

void GenotypeConnexion::decrementOriginOutputSize(int id){
	int new_size = nLines * (nColumns - 1);
	float* new_A = new float[new_size];
	float* new_B = new float[new_size];
	float* new_C = new float[new_size];
	float* new_D = new float[new_size];
	float* new_eta = new float[new_size];
	float* new_w = new float[new_size];
	float* new_alpha = new float[new_size];
#ifdef CONTINUOUS_LEARNING
	float* new_gamma = new float[new_size];
#endif
#ifdef GUIDED_MUTATIONS
	float* new_accumulator = new float[new_size];
#endif


	int idNew = 0, idOld = 0;
	for (int j = 0; j < nLines; j++) {
		for (int k = 0; k < nColumns; k++) {

			if (k == id) {
				idOld++;
				continue;
			}
			new_A[idNew] = A[idOld];
			new_B[idNew] = B[idOld];
			new_C[idNew] = C[idOld];
			new_D[idNew] = D[idOld];
			new_eta[idNew] = eta[idOld];
			new_w[idNew] = w[idOld];
			new_alpha[idNew] = alpha[idOld];
#ifdef CONTINUOUS_LEARNING
			new_gamma[idNew] = gamma[idOld];
#endif
#ifdef GUIDED_MUTATIONS
			new_accumulator[idNew] = 0.0f;
#endif

			idNew++;
			idOld++;
		}
	}

	A.reset(new_A);
	B.reset(new_B);
	C.reset(new_C);
	D.reset(new_D);
	eta.reset(new_eta);
	w.reset(new_w);
	alpha.reset(new_A);
#ifdef CONTINUOUS_LEARNING
	gamma.reset(new_gamma);
#endif
#ifdef GUIDED_MUTATIONS
	accumulator.reset(new_accumulator);
#endif

	nColumns--;
}