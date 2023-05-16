#pragma once

#include "InternalConnexion_G.h"

InternalConnexion_G::InternalConnexion_G(int nLines, int nColumns, INITIALIZATION init) :
	nLines(nLines), nColumns(nColumns)
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

InternalConnexion_G::InternalConnexion_G(const InternalConnexion_G& gc) {

	
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

InternalConnexion_G InternalConnexion_G::operator=(const InternalConnexion_G& gc) {

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


void InternalConnexion_G::mutateFloats(float p) {

#ifdef CONTINUOUS_LEARNING
	constexpr int nArrays = 8;  // 7+gamma
#else 
	constexpr int nArrays = 7;
#endif
	//param(t+1) = (.95+a*N1)*param(t) + b*N2
	constexpr float a = .2f; // .5f ??
	constexpr float b = .2f; // .5f ??


#ifdef GUIDED_MUTATIONS
	// w += clip[-accumulatorClipRange,accumulatorClipRange](accumulator)
	constexpr float accumulatorClipRange = .3f;
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
	for (int k = 0; k < size; k++) {;
		w[k] += std::max(std::min(accumulator[k], accumulatorClipRange), -accumulatorClipRange);
		accumulator[k] = 0.0f;
	}
	
#endif
}


#if defined GUIDED_MUTATIONS
void InternalConnexion_G::accumulateW(float factor, float* wLifetime) {
	int s = nLines * nColumns;
	for (int j = 0; j < s; j++) {
		accumulator[j] += factor * wLifetime[j];
		//wLifetime[j] = 0.0f; // TODO ?
	}
}
#endif

void InternalConnexion_G::insertLineRange(int id, int s) {
	int _insertedOffset = id * nColumns;
	int _insertedSize = s * nColumns;
	int _matSize = nLines * nColumns;

	float a = .2f * powf((float)nColumns, -.5f);

	auto f_R = [_insertedOffset, _insertedSize, _matSize, a] (std::unique_ptr<float[]>& m)
	{
		float* new_m = new float[_matSize + _insertedSize];
		std::copy(&m[0], &m[_insertedOffset], new_m);
		std::copy(&m[_insertedOffset], &m[_matSize], &new_m[_insertedOffset+_insertedSize]);
		for (int i = _insertedOffset; i < _insertedOffset + _insertedSize; i++) {
			new_m[i] = NORMAL_01 * a;
		}
		m.reset(new_m);
	};

	auto f_01 = [_insertedOffset, _insertedSize, _matSize](std::unique_ptr<float[]>& m)
	{
		float* new_m = new float[_matSize + _insertedSize];
		std::copy(&m[0], &m[_insertedOffset], new_m);
		std::copy(&m[_insertedOffset], &m[_matSize], &new_m[_insertedOffset + _insertedSize]);
		for (int i = _insertedOffset; i < _insertedOffset + _insertedSize; i++) {
			new_m[i] = UNIFORM_01 * .35f + .05f;
		}
		m.reset(new_m);
	};


	f_R(A);
	f_R(B);
	f_R(C);
	f_R(D);
	f_R(w);
	f_R(alpha);
	f_01(eta);
#ifdef CONTINUOUS_LEARNING
	f_01(gamma);
#endif
#ifdef GUIDED_MUTATIONS
	f_01(accumulator); // set to 0...
#endif

	nLines += s;
}

void InternalConnexion_G::insertColumnRange(int id, int s) {
	int _newNColumns = nColumns + s;
	//int _nLines = nLines, _nColumns = nColumns;
	float a = .2f * powf((float)(nColumns+s), -.5f);


	auto f_R = [&](std::unique_ptr<float[]>& m)
	{
		float* new_m = new float[_newNColumns * nLines];
		for (int i = 0; i < nLines; i++) {
			std::copy(&m[i * nColumns], &m[i * nColumns + id], &new_m[i * _newNColumns]);
			for (int j = i * _newNColumns + id + s; j < (i + 1) * _newNColumns; j++) {
				new_m[i] = NORMAL_01 * a;
			}
			std::copy(&m[i * nColumns + id], &m[(i + 1) * nColumns], &new_m[i * _newNColumns + id + s]);
		}
		m.reset(new_m);
	};

	auto f_01 = [&](std::unique_ptr<float[]>& m)
	{
		float* new_m = new float[_newNColumns * nLines];
		for (int i = 0; i < nLines; i++) {
			std::copy(&m[i * nColumns], &m[i * nColumns + id], &new_m[i * _newNColumns]);
			for (int j = i * _newNColumns + id + s; j < (i + 1) * _newNColumns; j++) {
				new_m[i] = UNIFORM_01 * .35f + .05f;
			}
			std::copy(&m[i * nColumns + id], &m[(i + 1) * nColumns], &new_m[i * _newNColumns + id + s]);
		}
		m.reset(new_m);
	};



	f_R(A);
	f_R(B);
	f_R(C);
	f_R(D);
	f_R(w);
	f_R(alpha);
	f_01(eta);
#ifdef CONTINUOUS_LEARNING
	f_01(gamma);
#endif
#ifdef GUIDED_MUTATIONS
	f_01(accumulator); // set to 0...
#endif

	nColumns += s;
}

void InternalConnexion_G::removeLineRange(int id, int s) {

	auto f = [&](std::unique_ptr<float[]>& m) {
		float* new_m = new float[nColumns * (nLines - s)];
		std::copy(&m[0], &m[id * nColumns], new_m);
		std::copy(&m[(id+s) * nColumns], &m[nLines * nColumns], new_m + id * nColumns);
		m.reset(new_m);
	};

	f(A);
	f(B);
	f(C);
	f(D);
	f(w);
	f(alpha);
	f(eta);
#ifdef CONTINUOUS_LEARNING
	f(gamma);
#endif
#ifdef GUIDED_MUTATIONS
	f(accumulator); // set to 0...
#endif

	nLines -= s;
}

void InternalConnexion_G::removeColumnRange(int id, int s) {
	int _newNColumns = nColumns - s;

	auto f = [&](std::unique_ptr<float[]>& m)
	{
		float* new_m = new float[_newNColumns * nLines];
		for (int i = 0; i < nLines; i++) {
			std::copy(&m[i * nColumns], &m[i * nColumns + id], &new_m[i * _newNColumns]);
			std::copy(&m[i * nColumns + id + s], &m[(i + 1) * nColumns], &new_m[i * _newNColumns + id]);
		}
		m.reset(new_m);
	};

	f(A);
	f(B);
	f(C);
	f(D);
	f(w);
	f(alpha);
	f(eta);
#ifdef CONTINUOUS_LEARNING
	f(gamma);
#endif
#ifdef GUIDED_MUTATIONS
	f(accumulator); // set to 0...
#endif

	nColumns -= s;
}