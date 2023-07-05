#pragma once

#include "InternalConnexion_G.h"



// Not very clean, but I dont want to create an initialization function.
#ifdef CONTINUOUS_LEARNING
#define CONTINUOUS_LEARNING_GAMMA 1
#else
#define CONTINUOUS_LEARNING_GAMMA 0
#endif 

#ifdef RANDOM_WB
#define RANDOM_WB_W -1
#else
#define RANDOM_WB_W 0
#endif 


#ifdef OJA
#define OJA_DELTA 1
#else
#define OJA_DELTA 0
#endif 

int InternalConnexion_G::nEvolvedArrays =   7 + 
											CONTINUOUS_LEARNING_GAMMA +
											RANDOM_WB_W +
											OJA_DELTA;

// Normal mutation in the space of log(half-life constant). m default value is .15f.
inline float mutateDecayParam(float dp, float m) 
{
	float exp_r = exp2f(NORMAL_01 * m);
	float tau = log2f(1.0f - dp);


	float t = 1.0f - exp2f(exp_r * tau);


	if (t < 0.001f) {
		t = 0.001f;
	}
	else if (t > .999f) {
		t = .999f;
	}

	return 1.0f - exp2f(exp_r * tau);
}



InternalConnexion_G::InternalConnexion_G(int nLines, int nColumns, INITIALIZATION init) :
	nLines(nLines), nColumns(nColumns)
{

	int s = nLines * nColumns;

	float f0 = 1.0f;
	if (s != 0) { f0 = powf((float)nColumns, -.5f); }
	 

	eta = std::make_unique<float[]>(s);
	A = std::make_unique<float[]>(s);
	B = std::make_unique<float[]>(s);
	C = std::make_unique<float[]>(s);
	D = std::make_unique<float[]>(s);
	alpha = std::make_unique<float[]>(s);

#ifndef RANDOM_WB
	w = std::make_unique<float[]>(s);
#endif

#ifdef OJA
	delta = std::make_unique<float[]>(s);
#endif
	
#ifdef CONTINUOUS_LEARNING
	gamma = std::make_unique<float[]>(s);
#endif

#ifdef GUIDED_MUTATIONS
	accumulator = std::make_unique<float[]>(s);
#endif

	auto zero = [&s](float* vec) {
		for (int i = 0; i < s; i++) {
			vec[i] = 0.0f;
		}
	};

	auto rand = [&s](float* vec, float b, float f) {
		for (int i = 0; i < s; i++) {
			vec[i] = NORMAL_01 * f + b;
		}
	};

	auto rand01 = [&s](float* vec) {
		for (int i = 0; i < s; i++) {
			vec[i] = mutateDecayParam(DECAY_PARAMETERS_INIT_BIAS, .4f); // Not default m for better initial spread.
		}
	};

	rand(A.get(), 0.0f, f0);
	rand(B.get(), 0.0f, f0);
	rand(C.get(), 0.0f, f0);
	rand(D.get(), 0.0f, f0);
	rand01(eta.get());

#ifdef CONTINUOUS_LEARNING
	rand01(gamma.get());
#endif

#ifdef GUIDED_MUTATIONS
	zero(accumulator.get());
#endif

#ifdef OJA
	rand01(delta.get());
#endif

	if (init == ZERO) {
		zero(alpha.get());
		
#ifndef RANDOM_WB
		zero(w.get());
#endif
	}
	else if (init == RANDOM) {
		rand(alpha.get(), 0.0f, f0);

#ifndef RANDOM_WB
		rand(w.get(), 0.0f, f0);
#endif
	}



	s = nLines;
	activationFunctions = std::make_unique<ACTIVATION[]>(s);
	for (int i = 0; i < s; i++) {
		activationFunctions[i] = static_cast<ACTIVATION>(INT_0X(N_ACTIVATIONS));
	}

#ifdef STDP
	STDP_mu = std::make_unique<float[]>(s);
	STDP_lambda = std::make_unique<float[]>(s);
	rand01(STDP_mu.get());
	rand01(STDP_lambda.get());
#endif

#ifndef RANDOM_WB
	biases = std::make_unique<float[]>(s);

	if (init == ZERO) {
		zero(biases.get());
	}
	else if (init == RANDOM) {
		rand(biases.get(), 0.0f, .2f);
	}
#endif
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
	

	std::copy(gc.eta.get(), gc.eta.get() + s, eta.get());
	std::copy(gc.A.get(), gc.A.get() + s, A.get());
	std::copy(gc.B.get(), gc.B.get() + s, B.get());
	std::copy(gc.C.get(), gc.C.get() + s, C.get());
	std::copy(gc.D.get(), gc.D.get() + s, D.get());
	std::copy(gc.alpha.get(), gc.alpha.get() + s, alpha.get());

#ifndef RANDOM_WB
	w = std::make_unique<float[]>(s);
	std::copy(gc.w.get(), gc.w.get() + s, w.get());
#endif

#ifdef OJA
	delta = std::make_unique<float[]>(s);
	std::copy(gc.delta.get(), gc.delta.get() + s, delta.get());
#endif


#ifdef CONTINUOUS_LEARNING
	gamma = std::make_unique<float[]>(s);
	std::copy(gc.gamma.get(), gc.gamma.get() + s, gamma.get());
#endif

#ifdef GUIDED_MUTATIONS
	accumulator = std::make_unique<float[]>(s);
	std::copy(gc.accumulator.get(), gc.accumulator.get() + s, accumulator.get());
#endif

	s = nLines;
	activationFunctions = std::make_unique<ACTIVATION[]>(s);
	std::copy(gc.activationFunctions.get(), gc.activationFunctions.get() + s, activationFunctions.get());

#ifndef RANDOM_WB
	biases = std::make_unique<float[]>(s);
	std::copy(gc.biases.get(), gc.biases.get() + s, biases.get());
#endif

#ifdef STDP
	STDP_mu = std::make_unique<float[]>(s);
	STDP_lambda = std::make_unique<float[]>(s);
	std::copy(gc.STDP_mu.get(), gc.STDP_mu.get() + s, STDP_mu.get());
	std::copy(gc.STDP_lambda.get(), gc.STDP_lambda.get() + s, STDP_lambda.get());
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

	std::copy(gc.eta.get(), gc.eta.get() + s, eta.get());
	std::copy(gc.A.get(), gc.A.get() + s, A.get());
	std::copy(gc.B.get(), gc.B.get() + s, B.get());
	std::copy(gc.C.get(), gc.C.get() + s, C.get());
	std::copy(gc.D.get(), gc.D.get() + s, D.get());
	std::copy(gc.alpha.get(), gc.alpha.get() + s, alpha.get());

#ifndef RANDOM_WB
	w = std::make_unique<float[]>(s);
	std::copy(gc.w.get(), gc.w.get() + s, w.get());
#endif

#ifdef OJA
	delta = std::make_unique<float[]>(s);
	std::copy(gc.delta.get(), gc.delta.get() + s, delta.get());
#endif

#ifdef CONTINUOUS_LEARNING
	gamma = std::make_unique<float[]>(s);
	std::copy(gc.gamma.get(), gc.gamma.get() + s, gamma.get());
#endif

#ifdef GUIDED_MUTATIONS
	accumulator = std::make_unique<float[]>(s);
	std::copy(gc.accumulator.get(), gc.accumulator.get() + s, accumulator.get());
#endif

	s = nLines;
	activationFunctions = std::make_unique<ACTIVATION[]>(s);
	std::copy(gc.activationFunctions.get(), gc.activationFunctions.get() + s, activationFunctions.get());

#ifndef RANDOM_WB
	biases = std::make_unique<float[]>(s);
	std::copy(gc.biases.get(), gc.biases.get() + s, biases.get());
#endif

#ifdef STDP
	STDP_mu = std::make_unique<float[]>(s);
	STDP_lambda = std::make_unique<float[]>(s);
	std::copy(gc.STDP_mu.get(), gc.STDP_mu.get() + s, STDP_mu.get());
	std::copy(gc.STDP_lambda.get(), gc.STDP_lambda.get() + s, STDP_lambda.get());
#endif

	return *this;
}

InternalConnexion_G::InternalConnexion_G(std::ifstream& is)
{
	READ_4B(nLines, is);
	READ_4B(nColumns, is);

	int s = nLines * nColumns;

	eta = std::make_unique<float[]>(s);
	is.read(reinterpret_cast<char*>(eta.get()), s * sizeof(float));
	A = std::make_unique<float[]>(s);
	is.read(reinterpret_cast<char*>(A.get()), s * sizeof(float));
	B = std::make_unique<float[]>(s);
	is.read(reinterpret_cast<char*>(B.get()), s * sizeof(float));
	C = std::make_unique<float[]>(s);
	is.read(reinterpret_cast<char*>(C.get()), s * sizeof(float));
	D = std::make_unique<float[]>(s);
	is.read(reinterpret_cast<char*>(D.get()), s * sizeof(float));
	alpha = std::make_unique<float[]>(s);
	is.read(reinterpret_cast<char*>(alpha.get()), s * sizeof(float));

#ifndef RANDOM_WB
	w = std::make_unique<float[]>(s);
	is.read(reinterpret_cast<char*>(w.get()), s * sizeof(float));
#endif

#ifdef OJA
	delta = std::make_unique<float[]>(s);
	is.read(reinterpret_cast<char*>(delta.get()), s * sizeof(float));
#endif


#ifdef CONTINUOUS_LEARNING
	gamma = std::make_unique<float[]>(s);
	is.read(reinterpret_cast<char*>(gamma.get()), s * sizeof(float));
#endif

#ifdef GUIDED_MUTATIONS
	accumulator = std::make_unique<float[]>(s);
#endif

	s = nLines;

	activationFunctions = std::make_unique<ACTIVATION[]>(s);
	is.read(reinterpret_cast<char*>(activationFunctions.get()), s * sizeof(ACTIVATION));

#ifndef RANDOM_WB
	biases = std::make_unique<float[]>(s);
	is.read(reinterpret_cast<char*>(biases.get()), s * sizeof(float));
#endif

#ifdef STDP
	STDP_mu = std::make_unique<float[]>(s);
	STDP_lambda = std::make_unique<float[]>(s);
	is.read(reinterpret_cast<char*>(STDP_mu.get()), s * sizeof(float));
	is.read(reinterpret_cast<char*>(STDP_lambda.get()), s * sizeof(float));
#endif
}

void InternalConnexion_G::save(std::ofstream& os) 
{
	WRITE_4B(nLines, os);
	WRITE_4B(nColumns, os);

	int s = nLines * nColumns;

	os.write(reinterpret_cast<const char*>(eta.get()), s * sizeof(float));
	os.write(reinterpret_cast<const char*>(A.get()), s * sizeof(float));
	os.write(reinterpret_cast<const char*>(B.get()), s * sizeof(float));
	os.write(reinterpret_cast<const char*>(C.get()), s * sizeof(float));
	os.write(reinterpret_cast<const char*>(D.get()), s * sizeof(float));
	os.write(reinterpret_cast<const char*>(alpha.get()), s * sizeof(float));
	
#ifndef RANDOM_WB
	os.write(reinterpret_cast<char*>(w.get()), s * sizeof(float));
#endif

#ifdef OJA
	os.write(reinterpret_cast<char*>(delta.get()), s * sizeof(float));
#endif

#ifdef CONTINUOUS_LEARNING
	os.write(reinterpret_cast<const char*>(gamma.get()), s * sizeof(float));
#endif

	s = nLines;

	os.write(reinterpret_cast<const char*>(activationFunctions.get()), s * sizeof(ACTIVATION));

#ifndef RANDOM_WB
	os.write(reinterpret_cast<const char*>(biases.get()), s * sizeof(float));
#endif

#ifdef STDP
	os.write(reinterpret_cast<const char*>(STDP_mu.get()), s * sizeof(float));
	os.write(reinterpret_cast<const char*>(STDP_lambda.get()), s * sizeof(float));
#endif
}

void InternalConnexion_G::mutateFloats(float p) {

	//param(t+1) = (b+a*N1)*param(t) + c*N2
	const float sigma = powf((float)nColumns, -.5f);
	const float a = .3f * sigma;
	const float b = 1.0f-a*.3f;
	const float c = a;

#ifdef GUIDED_MUTATIONS
	// w += clip[-accumulatorClipRange,accumulatorClipRange](accumulator)
	constexpr float accumulatorClipRange = 1.0f;
#endif

	int size = nLines * nColumns;
	SET_BINOMIAL(size, p);

	auto mutateMatrix = [&size, p, a, b, c](float* matrix)
	{

		int _nMutations = BINOMIAL;
		for (int k = 0; k < _nMutations; k++) {
			int matrixID = INT_0X(size);


			matrix[matrixID] *= b + NORMAL_01 * a;
			matrix[matrixID] += NORMAL_01 * c;
		}
	};

	auto mutateDecayMatrix = [&size, p](float* matrix)
	{

		int _nMutations = BINOMIAL;
		for (int k = 0; k < _nMutations; k++) {
			int matrixID = INT_0X(size);
			matrix[matrixID] = mutateDecayParam(matrix[matrixID]);
		}
	};
	
	mutateMatrix(A.get());
	mutateMatrix(B.get());
	mutateMatrix(C.get());
	mutateMatrix(D.get());
	mutateMatrix(alpha.get());
	mutateDecayMatrix(eta.get());
		
#ifndef RANDOM_WB
	mutateMatrix(w.get());
#endif

#ifdef CONTINUOUS_LEARNING
	mutateDecayMatrix(gamma.get());
#endif

#ifdef OJA
	mutateDecayMatrix(delta.get());
#endif

#if defined(GUIDED_MUTATIONS) && !defined(RANDOM_W)
	for (int k = 0; k < size; k++) {;
		w[k] += std::max(std::min(accumulator[k], accumulatorClipRange), -accumulatorClipRange); 
		accumulator[k] = 0.0f;
	}
#endif

	
	SET_BINOMIAL(nLines, p);
	int _nMutations;

#ifndef RANDOM_WB
	_nMutations = BINOMIAL;
	for (int i = 0; i < _nMutations; i++) {
		int id = INT_0X(nLines);
		biases[id] *= .94f + NORMAL_01 * .1f; // .9 < 1 to drive the weight towards 0.
		biases[id] += NORMAL_01 * .2f;
	}
#endif

#ifdef STDP
	size = nLines;
	mutateDecayMatrix(STDP_lambda.get());
	mutateDecayMatrix(STDP_mu.get());
#endif

	SET_BINOMIAL(nLines, .1f * p);
    _nMutations = BINOMIAL;
	for (int i = 0; i < _nMutations; i++) {
		int id = INT_0X(nLines);
		activationFunctions[id] = static_cast<ACTIVATION>(INT_0X(N_ACTIVATIONS));
	}

}


#if defined GUIDED_MUTATIONS
void InternalConnexion_G::accumulateW(float factor, float* wLifetime) {
	int s = nLines * nColumns;
	for (int j = 0; j < s; j++) {
		accumulator[j] += factor * wLifetime[j];
	}
}
#endif

// No evolved elements are initialized at 0, because the elements added by the line
// influence the input of the new node, not its output (with the exception of the output
// node incrementing its size.)
void InternalConnexion_G::insertLineRange(int id, int s) {
	int _insertedOffset = id * nColumns;
	int _insertedSize = s * nColumns;
	int _matSize = nLines * nColumns;

	float a = .2f * powf((float)nColumns, -.5f);

	auto f = [_insertedOffset, _insertedSize, _matSize, a] (std::unique_ptr<float[]>& m, float b)
	{
		float* new_m = new float[_matSize + _insertedSize];
		std::copy(&m[0], &m[_insertedOffset], new_m);
		std::copy(&m[_insertedOffset], &m[_matSize], &new_m[_insertedOffset+_insertedSize]);
		for (int i = _insertedOffset; i < _insertedOffset + _insertedSize; i++) {
			new_m[i] = NORMAL_01 * a + b;
		}
		m.reset(new_m);
	};

	auto fDecay = [_insertedOffset, _insertedSize, _matSize, a](std::unique_ptr<float[]>& m)
	{
		float* new_m = new float[_matSize + _insertedSize];
		std::copy(&m[0], &m[_insertedOffset], new_m);
		std::copy(&m[_insertedOffset], &m[_matSize], &new_m[_insertedOffset + _insertedSize]);
		for (int i = _insertedOffset; i < _insertedOffset + _insertedSize; i++) {
			new_m[i] = mutateDecayParam(DECAY_PARAMETERS_INIT_BIAS, .4f);
		}
		m.reset(new_m);
	};

	auto f_0 = [_insertedOffset, _insertedSize, _matSize, a](std::unique_ptr<float[]>& m)
	{
		float* new_m = new float[_matSize + _insertedSize];
		std::copy(&m[0], &m[_insertedOffset], new_m);
		std::copy(&m[_insertedOffset], &m[_matSize], &new_m[_insertedOffset + _insertedSize]);
		for (int i = _insertedOffset; i < _insertedOffset + _insertedSize; i++) {
			new_m[i] = 0.0f;
		}
		m.reset(new_m);
	};


	f(A, 0.0f);
	f(B, 0.0f);
	f(C, 0.0f);
	f(D, 0.0f);

	f(alpha, 0.0f);

	fDecay(eta);

#ifdef CONTINUOUS_LEARNING
	fDecay(gamma);
#endif

#ifdef OJA
	fDecay(delta);
#endif

#ifndef RANDOM_WB
	f(w, 0.0f);
#endif

#ifdef GUIDED_MUTATIONS
	f_0(accumulator); // set to 0...
#endif


#ifndef RANDOM_WB
	float* newB = new float[nLines + s];
	std::copy(biases.get(), biases.get() + id, newB);
	for (int i = id; i < id + s; i++) {
		newB[i] = NORMAL_01;
	}
	std::copy(biases.get() + id, biases.get() + nLines, newB + id + s);
	biases.reset(newB);
#endif
	

	ACTIVATION* newA = new ACTIVATION[nLines + s];
	std::copy(activationFunctions.get(), activationFunctions.get() + id, newA);
	for (int i = id; i < id + s; i++) {
		newA[i] = static_cast<ACTIVATION>(INT_0X(N_ACTIVATIONS));
	}
	std::copy(activationFunctions.get() + id, activationFunctions.get() + nLines, newA + id + s);
	activationFunctions.reset(newA);


#ifdef STDP
	float* newMu = new float[nLines + s];
	float* newLambda = new float[nLines + s];
	std::copy(STDP_mu.get(), STDP_mu.get() + id, newMu);
	std::copy(STDP_lambda.get(), STDP_lambda.get() + id, newLambda);
	for (int i = id; i < id + s; i++) {
		newMu[i] = mutateDecayParam(DECAY_PARAMETERS_INIT_BIAS, .4f);
		newLambda[i] = mutateDecayParam(DECAY_PARAMETERS_INIT_BIAS, .4f);
	}
	std::copy(STDP_mu.get() + id, STDP_mu.get() + nLines, newMu + id + s);
	std::copy(STDP_lambda.get() + id, STDP_lambda.get() + nLines, newLambda + id + s);
	STDP_mu.reset(newMu);
	STDP_lambda.reset(newLambda);
#endif


	nLines += s;
}

// zeroing some of the inserted evolved parameters, to minimize the impact
// of the new topology on the network.
void InternalConnexion_G::insertColumnRange(int id, int s) {
	int _newNColumns = nColumns + s;
	//int _nLines = nLines, _nColumns = nColumns;
	float a = 1.0f * powf((float)_newNColumns, -.5f);


	auto f = [&](std::unique_ptr<float[]>& m, float b)
	{
		float* new_m = new float[_newNColumns * nLines];
		for (int i = 0; i < nLines; i++) {
			std::copy(&m[i * nColumns], &m[i * nColumns + id], &new_m[i * _newNColumns]);

			for (int j = i * _newNColumns + id; j < i * _newNColumns + id + s; j++) {
				new_m[j] = NORMAL_01 * a + b;
			}

			std::copy(&m[i * nColumns + id], &m[(i + 1) * nColumns], &new_m[i * _newNColumns + id + s]);
		}
		m.reset(new_m);
	};
	
	auto fDecay = [&](std::unique_ptr<float[]>& m)
	{
		float* new_m = new float[_newNColumns * nLines];
		for (int i = 0; i < nLines; i++) {
			std::copy(&m[i * nColumns], &m[i * nColumns + id], &new_m[i * _newNColumns]);

			for (int j = i * _newNColumns + id; j < i * _newNColumns + id + s; j++) {
				new_m[j] = mutateDecayParam(DECAY_PARAMETERS_INIT_BIAS, .4f);
			}

			std::copy(&m[i * nColumns + id], &m[(i + 1) * nColumns], &new_m[i * _newNColumns + id + s]);
		}
		m.reset(new_m);
	};

	auto f_0 = [&](std::unique_ptr<float[]>& m)
	{
		float* new_m = new float[_newNColumns * nLines];
		for (int i = 0; i < nLines; i++) {
			std::copy(&m[i * nColumns], &m[i * nColumns + id], &new_m[i * _newNColumns]);
			
			for (int j = i * _newNColumns + id; j < i * _newNColumns + id + s; j++) {
				new_m[j] = 0.0f;
			}

			std::copy(&m[i * nColumns + id], &m[(i + 1) * nColumns], &new_m[i * _newNColumns + id + s]);
		}
		m.reset(new_m);
	};



	f(A, 0.0f);
	f(B, 0.0f);
	f(C, 0.0f);
	f(D, 0.0f);

	f_0(alpha); // necessite de laisser la nouvelle mémoire non modifiée.

	fDecay(eta);

#ifdef CONTINUOUS_LEARNING
	fDecay(gamma);
#endif

#ifdef OJA
	fDecay(delta);
#endif

#ifndef RANDOM_WB
	f_0(w);
#endif

#ifdef GUIDED_MUTATIONS
	f_0(accumulator); // set to 0...
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
	f(alpha);
	f(eta);

#ifdef CONTINUOUS_LEARNING
	f(gamma);
#endif

#ifndef RANDOM_WB
	f(w);
#endif

#ifdef OJA
	f(delta);
#endif

#ifdef GUIDED_MUTATIONS
	f(accumulator); // set to 0...
#endif

#ifndef RANDOM_WB
	float* newB = new float[nLines - s];
	std::copy(biases.get(), biases.get() + id, newB);
	std::copy(biases.get() + id + s, biases.get() + nLines, newB + id);
	biases.reset(newB);
#endif
	

	ACTIVATION* newA = new ACTIVATION[nLines - s];
	std::copy(activationFunctions.get(), activationFunctions.get() + id, newA);
	std::copy(activationFunctions.get() + id + s, activationFunctions.get() + nLines, newA + id);
	activationFunctions.reset(newA);


#ifdef STDP
	float* newMu = new float[nLines - s];
	float* newLambda = new float[nLines - s];
	std::copy(STDP_mu.get(), STDP_mu.get() + id, newMu);
	std::copy(STDP_lambda.get(), STDP_lambda.get() + id, newLambda);
	std::copy(STDP_mu.get() + id + s, STDP_mu.get() + nLines, newMu + id);
	std::copy(STDP_lambda.get() + id + s, STDP_lambda.get() + nLines, newLambda + id);
	STDP_mu.reset(newMu);
	STDP_lambda.reset(newLambda);
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
	f(alpha);
	f(eta);

#ifdef CONTINUOUS_LEARNING
	f(gamma);
#endif

#ifndef RANDOM_WB
	f(w);
#endif

#ifdef OJA
	f(delta);
#endif

#ifdef GUIDED_MUTATIONS
	f(accumulator); // set to 0...
#endif

	nColumns -= s;
}
