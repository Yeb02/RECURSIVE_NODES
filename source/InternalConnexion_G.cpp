#pragma once

#include "InternalConnexion_G.h"



// Not very clean, but I dont want to create an initialization function.
#ifdef CONTINUOUS_LEARNING
#define CONTINUOUS_LEARNING_GAMMA 1
#else
#define CONTINUOUS_LEARNING_GAMMA 0
#endif 

#ifdef RANDOM_W
#define RANDOM_W_W -1
#else
#define RANDOM_W_W 0
#endif 


#ifdef OJA
#define OJA_DELTA 1
#else
#define OJA_DELTA 0
#endif 

int InternalConnexion_G::nEvolvedArrays =   7 + 
											CONTINUOUS_LEARNING_GAMMA +
											RANDOM_W_W +
											OJA_DELTA;



InternalConnexion_G::InternalConnexion_G(int nLines, int nColumns, INITIALIZATION init) :
	nLines(nLines), nColumns(nColumns)
{

	int s = nLines * nColumns;

	float f0 = 1.0f;
	if (s != 0) { f0 = powf((float)nColumns, -.5f); }
	 

	eta = std::make_unique<float[]>(s);
	storage_eta = std::make_unique<float[]>(s);
	A = std::make_unique<float[]>(s);
	B = std::make_unique<float[]>(s);
	C = std::make_unique<float[]>(s);
	D = std::make_unique<float[]>(s);
	alpha = std::make_unique<float[]>(s);

#ifndef RANDOM_W
	w = std::make_unique<float[]>(s);
#endif

#ifdef OJA
	delta = std::make_unique<float[]>(s);
	storage_delta = std::make_unique<float[]>(s);
#endif
	
#ifdef CONTINUOUS_LEARNING
	gamma = std::make_unique<float[]>(s);
	storage_gamma = std::make_unique<float[]>(s);
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

	rand(A.get(), 0.0f, f0);
	rand(B.get(), 0.0f, f0);
	rand(C.get(), 0.0f, f0);
	rand(D.get(), 0.0f, f0);
	rand(storage_eta.get(), DECAY_PARAMETERS_STORAGE_BIAS, .2f);

#ifdef CONTINUOUS_LEARNING
	rand(storage_gamma.get(), DECAY_PARAMETERS_STORAGE_BIAS, .2f);
#endif

#ifdef GUIDED_MUTATIONS
	zero(accumulator.get());
#endif

#ifdef OJA
	rand(storage_delta.get(), DECAY_PARAMETERS_STORAGE_BIAS, .2f);
#endif

	if (init == ZERO) {
		zero(alpha.get());
		
#ifndef RANDOM_W
		zero(w.get());
#endif
	}
	else if (init == RANDOM) {
		rand(alpha.get(), 0.0f, f0);

#ifndef RANDOM_W
		rand(w.get(), 0.0f, f0);
#endif
	}



	s = nLines;
	biases = std::make_unique<float[]>(s);
	activationFunctions = std::make_unique<ACTIVATION[]>(s);

#ifdef STDP
	STDP_storage_lambda = std::make_unique<float[]>(s);
	STDP_storage_mu = std::make_unique<float[]>(s);
	STDP_mu = std::make_unique<float[]>(s);
	STDP_lambda = std::make_unique<float[]>(s);
	rand(STDP_storage_mu.get(), DECAY_PARAMETERS_STORAGE_BIAS, .2f);
	rand(STDP_storage_lambda.get(), DECAY_PARAMETERS_STORAGE_BIAS, .2f);
#endif

	for (int i = 0; i < s; i++) {
		activationFunctions[i] = static_cast<ACTIVATION>(INT_0X(N_ACTIVATIONS));
	}

	if (init == ZERO) {
		zero(biases.get());
	}
	else if (init == RANDOM) {
		rand(biases.get(), 0.0f, 1.0f);
	}
}

InternalConnexion_G::InternalConnexion_G(const InternalConnexion_G& gc) {

	
	nLines = gc.nLines;
	nColumns = gc.nColumns;

	int s = nLines * nColumns;

	eta = std::make_unique<float[]>(s);
	storage_eta = std::make_unique<float[]>(s);
	A = std::make_unique<float[]>(s);
	B = std::make_unique<float[]>(s);
	C = std::make_unique<float[]>(s);
	D = std::make_unique<float[]>(s);
	alpha = std::make_unique<float[]>(s);
	

	std::copy(gc.storage_eta.get(), gc.storage_eta.get() + s, storage_eta.get());
	std::copy(gc.eta.get(), gc.eta.get() + s, eta.get());
	std::copy(gc.A.get(), gc.A.get() + s, A.get());
	std::copy(gc.B.get(), gc.B.get() + s, B.get());
	std::copy(gc.C.get(), gc.C.get() + s, C.get());
	std::copy(gc.D.get(), gc.D.get() + s, D.get());
	std::copy(gc.alpha.get(), gc.alpha.get() + s, alpha.get());

#ifndef RANDOM_W
	w = std::make_unique<float[]>(s);
	std::copy(gc.w.get(), gc.w.get() + s, w.get());
#endif

#ifdef OJA
	delta = std::make_unique<float[]>(s);
	storage_delta = std::make_unique<float[]>(s);
	std::copy(gc.delta.get(), gc.delta.get() + s, delta.get());
	std::copy(gc.storage_delta.get(), gc.storage_delta.get() + s, storage_delta.get());
#endif


#ifdef CONTINUOUS_LEARNING
	gamma = std::make_unique<float[]>(s);
	storage_gamma = std::make_unique<float[]>(s);
	std::copy(gc.gamma.get(), gc.gamma.get() + s, gamma.get());
	std::copy(gc.storage_gamma.get(), gc.storage_gamma.get() + s, storage_gamma.get());
#endif

#ifdef GUIDED_MUTATIONS
	accumulator = std::make_unique<float[]>(s);
	std::copy(gc.accumulator.get(), gc.accumulator.get() + s, accumulator.get());
#endif

	s = nLines;
	biases = std::make_unique<float[]>(s);
	activationFunctions = std::make_unique<ACTIVATION[]>(s);
	std::copy(gc.biases.get(), gc.biases.get() + s, biases.get());
	std::copy(gc.activationFunctions.get(), gc.activationFunctions.get() + s, activationFunctions.get());

#ifdef STDP
	STDP_storage_lambda = std::make_unique<float[]>(s);
	STDP_storage_mu = std::make_unique<float[]>(s);
	STDP_mu = std::make_unique<float[]>(s);
	STDP_lambda = std::make_unique<float[]>(s);
	std::copy(gc.STDP_mu.get(), gc.STDP_mu.get() + s, STDP_mu.get());
	std::copy(gc.STDP_lambda.get(), gc.STDP_lambda.get() + s, STDP_lambda.get());
	std::copy(gc.STDP_storage_mu.get(), gc.STDP_storage_mu.get() + s, STDP_storage_mu.get());
	std::copy(gc.STDP_storage_lambda.get(), gc.STDP_storage_lambda.get() + s, STDP_storage_lambda.get());
#endif
}

InternalConnexion_G InternalConnexion_G::operator=(const InternalConnexion_G& gc) {

	nLines = gc.nLines;
	nColumns = gc.nColumns;

	int s = nLines * nColumns;

	eta = std::make_unique<float[]>(s);
	storage_eta = std::make_unique<float[]>(s);
	A = std::make_unique<float[]>(s);
	B = std::make_unique<float[]>(s);
	C = std::make_unique<float[]>(s);
	D = std::make_unique<float[]>(s);
	alpha = std::make_unique<float[]>(s);

	std::copy(gc.eta.get(), gc.eta.get() + s, eta.get());
	std::copy(gc.storage_eta.get(), gc.storage_eta.get() + s, storage_eta.get());
	std::copy(gc.A.get(), gc.A.get() + s, A.get());
	std::copy(gc.B.get(), gc.B.get() + s, B.get());
	std::copy(gc.C.get(), gc.C.get() + s, C.get());
	std::copy(gc.D.get(), gc.D.get() + s, D.get());
	std::copy(gc.alpha.get(), gc.alpha.get() + s, alpha.get());

#ifndef RANDOM_W
	w = std::make_unique<float[]>(s);
	std::copy(gc.w.get(), gc.w.get() + s, w.get());
#endif

#ifdef OJA
	delta = std::make_unique<float[]>(s);
	storage_delta = std::make_unique<float[]>(s);
	std::copy(gc.delta.get(), gc.delta.get() + s, delta.get());
	std::copy(gc.storage_delta.get(), gc.storage_delta.get() + s, storage_delta.get());
#endif

#ifdef CONTINUOUS_LEARNING
	gamma = std::make_unique<float[]>(s);
	storage_gamma = std::make_unique<float[]>(s);
	std::copy(gc.gamma.get(), gc.gamma.get() + s, gamma.get());
	std::copy(gc.storage_gamma.get(), gc.storage_gamma.get() + s, storage_gamma.get());
#endif

#ifdef GUIDED_MUTATIONS
	accumulator = std::make_unique<float[]>(s);
	std::copy(gc.accumulator.get(), gc.accumulator.get() + s, accumulator.get());
#endif

	s = nLines;
	biases = std::make_unique<float[]>(s);
	activationFunctions = std::make_unique<ACTIVATION[]>(s);
	std::copy(gc.biases.get(), gc.biases.get() + s, biases.get());
	std::copy(gc.activationFunctions.get(), gc.activationFunctions.get() + s, activationFunctions.get());

#ifdef STDP
	STDP_storage_lambda = std::make_unique<float[]>(s);
	STDP_storage_mu = std::make_unique<float[]>(s);
	STDP_mu = std::make_unique<float[]>(s);
	STDP_lambda = std::make_unique<float[]>(s);
	std::copy(gc.STDP_mu.get(), gc.STDP_mu.get() + s, STDP_mu.get());
	std::copy(gc.STDP_lambda.get(), gc.STDP_lambda.get() + s, STDP_lambda.get());
	std::copy(gc.STDP_storage_mu.get(), gc.STDP_storage_mu.get() + s, STDP_storage_mu.get());
	std::copy(gc.STDP_storage_lambda.get(), gc.STDP_storage_lambda.get() + s, STDP_storage_lambda.get());
#endif

	return *this;
}

InternalConnexion_G::InternalConnexion_G(std::ifstream& is)
{
	READ_4B(nLines, is);
	READ_4B(nColumns, is);

	int s = nLines * nColumns;

	eta = std::make_unique<float[]>(s);
	storage_eta = std::make_unique<float[]>(s);
	is.read(reinterpret_cast<char*>(storage_eta.get()), s * sizeof(float));
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

#ifndef RANDOM_W
	w = std::make_unique<float[]>(s);
	is.read(reinterpret_cast<char*>(w.get()), s * sizeof(float));
#endif

#ifdef OJA
	delta = std::make_unique<float[]>(s);
	storage_delta = std::make_unique<float[]>(s);
	is.read(reinterpret_cast<char*>(storage_delta.get()), s * sizeof(float));
#endif


#ifdef CONTINUOUS_LEARNING
	gamma = std::make_unique<float[]>(s);
	storage_gamma = std::make_unique<float[]>(s);
	is.read(reinterpret_cast<char*>(storage_gamma.get()), s * sizeof(float));
#endif

#ifdef GUIDED_MUTATIONS
	accumulator = std::make_unique<float[]>(s);
#endif

	s = nLines;
	biases = std::make_unique<float[]>(s);
	is.read(reinterpret_cast<char*>(biases.get()), s * sizeof(float));

	activationFunctions = std::make_unique<ACTIVATION[]>(s);
	is.read(reinterpret_cast<char*>(activationFunctions.get()), s * sizeof(ACTIVATION));

#ifdef STDP
	STDP_storage_lambda = std::make_unique<float[]>(s);
	STDP_storage_mu = std::make_unique<float[]>(s);
	STDP_mu = std::make_unique<float[]>(s);
	STDP_lambda = std::make_unique<float[]>(s);
	is.read(reinterpret_cast<char*>(STDP_storage_mu.get()), s * sizeof(float));
	is.read(reinterpret_cast<char*>(STDP_storage_lambda.get()), s * sizeof(float));
#endif
}

void InternalConnexion_G::save(std::ofstream& os) 
{
	WRITE_4B(nLines, os);
	WRITE_4B(nColumns, os);

	int s = nLines * nColumns;

	os.write(reinterpret_cast<const char*>(storage_eta.get()), s * sizeof(float));
	os.write(reinterpret_cast<const char*>(A.get()), s * sizeof(float));
	os.write(reinterpret_cast<const char*>(B.get()), s * sizeof(float));
	os.write(reinterpret_cast<const char*>(C.get()), s * sizeof(float));
	os.write(reinterpret_cast<const char*>(D.get()), s * sizeof(float));
	os.write(reinterpret_cast<const char*>(alpha.get()), s * sizeof(float));
	
#ifndef RANDOM_W
	os.write(reinterpret_cast<char*>(w.get()), s * sizeof(float));
#endif

#ifdef OJA
	os.write(reinterpret_cast<char*>(storage_delta.get()), s * sizeof(float));
#endif

#ifdef CONTINUOUS_LEARNING
	os.write(reinterpret_cast<const char*>(storage_gamma.get()), s * sizeof(float));
#endif

	s = nLines;
	os.write(reinterpret_cast<const char*>(biases.get()), s * sizeof(float));

	os.write(reinterpret_cast<const char*>(activationFunctions.get()), s * sizeof(ACTIVATION));

#ifdef STDP
	os.write(reinterpret_cast<const char*>(STDP_storage_mu.get()), s * sizeof(float));
	os.write(reinterpret_cast<const char*>(STDP_storage_lambda.get()), s * sizeof(float));
#endif
}

void InternalConnexion_G::mutateFloats(float p) {

	//param(t+1) = (b+a*N1)*param(t) + c*N2
	const float sigma = powf((float)nColumns, -.5f);
	const float a = .3f * sigma;
	const float b = 1.0f - .5f * a;
	const float c = a;

#ifdef GUIDED_MUTATIONS
	// w += clip[-accumulatorClipRange,accumulatorClipRange](accumulator)
	constexpr float accumulatorClipRange = 1.0f;
#endif

	int size = nLines * nColumns;
	SET_BINOMIAL(size, p);

	auto mutateMatrix = [size, p, a, b, c](float* matrix)
	{

		int _nMutations = BINOMIAL;
		for (int k = 0; k < _nMutations; k++) {
			int matrixID = INT_0X(size);


			matrix[matrixID] *= b + NORMAL_01 * a;
			matrix[matrixID] += NORMAL_01 * c;
		}
	};
	
	mutateMatrix(A.get());
	mutateMatrix(B.get());
	mutateMatrix(C.get());
	mutateMatrix(D.get());
	mutateMatrix(alpha.get());
	mutateMatrix(storage_eta.get());
		
#ifndef RANDOM_W
	mutateMatrix(w.get());
#endif

#ifdef CONTINUOUS_LEARNING
	mutateMatrix(storage_gamma.get());
#endif

#ifdef OJA
	mutateMatrix(storage_delta.get());
#endif

#if defined(GUIDED_MUTATIONS) && !defined(RANDOM_W)
	for (int k = 0; k < size; k++) {;
		w[k] += std::max(std::min(accumulator[k], accumulatorClipRange), -accumulatorClipRange); 
		accumulator[k] = 0.0f;
	}
#endif

	
	SET_BINOMIAL(nLines, p);

	int _nMutations = BINOMIAL;
	for (int i = 0; i < _nMutations; i++) {
		int id = INT_0X(nLines);
		biases[id] *= .9f + NORMAL_01 * .1f; // .9 < 1 to drive the weight towards 0.
		biases[id] += NORMAL_01 * .2f;
	}


#ifdef STDP
	_nMutations = BINOMIAL;
	for (int i = 0; i < _nMutations; i++) {
		int id = INT_0X(nLines);
		STDP_storage_lambda[id] *= .95f + NORMAL_01 * .1f; // .9 < 1 to drive it towards 0.
		STDP_storage_lambda[id] += NORMAL_01 * .1f;
	}

	_nMutations = BINOMIAL;
	for (int i = 0; i < _nMutations; i++) {
		int id = INT_0X(nLines);
		STDP_storage_mu[id] *= .95f + NORMAL_01 * .1f; // .9 < 1 to drive it towards 0.
		STDP_storage_mu[id] += NORMAL_01 * .1f;
	}
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

	f(storage_eta, DECAY_PARAMETERS_STORAGE_BIAS);
	f_0(eta);

#ifdef CONTINUOUS_LEARNING
	f(storage_gamma, DECAY_PARAMETERS_STORAGE_BIAS);
	f_0(gamma);
#endif

#ifdef OJA
	f(storage_delta, DECAY_PARAMETERS_STORAGE_BIAS);
	f_0(delta);
#endif

#ifndef RANDOM_W
	f(w, 0.0f);
#endif

#ifdef GUIDED_MUTATIONS
	f_0(accumulator); // set to 0...
#endif


	
	float* newB = new float[nLines + s];
	std::copy(biases.get(), biases.get() + id, newB);
	for (int i = id; i < id + s; i++) {
		newB[i] = NORMAL_01;
	}
	std::copy(biases.get() + id, biases.get() + nLines, newB);
	biases.reset(newB);

	ACTIVATION* newA = new ACTIVATION[nLines + s];
	std::copy(activationFunctions.get(), activationFunctions.get() + id, newA);
	for (int i = id; i < id + s; i++) {
		newA[i] = static_cast<ACTIVATION>(INT_0X(N_ACTIVATIONS));
	}
	std::copy(activationFunctions.get() + id, activationFunctions.get() + nLines, newA);
	activationFunctions.reset(newA);


#ifdef STDP
	STDP_mu.reset(new float[nLines + s]);
	STDP_lambda.reset(new float[nLines + s]);

	float* newMu = new float[nLines + s];
	float* newLambda = new float[nLines + s];
	std::copy(STDP_storage_mu.get(), STDP_storage_mu.get() + id, newMu);
	std::copy(STDP_storage_lambda.get(), STDP_storage_lambda.get() + id, newLambda);
	for (int i = id; i < id + s; i++) {
		newMu[i] = NORMAL_01*.2f - DECAY_PARAMETERS_STORAGE_BIAS;
		newLambda[i] = NORMAL_01*.2f - DECAY_PARAMETERS_STORAGE_BIAS;
	}
	std::copy(STDP_storage_mu.get() + id, STDP_storage_mu.get() + nLines, newMu);
	std::copy(STDP_storage_lambda.get() + id, STDP_storage_lambda.get() + nLines, newLambda);
	STDP_storage_mu.reset(newMu);
	STDP_storage_lambda.reset(newLambda);
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

	f(storage_eta, DECAY_PARAMETERS_STORAGE_BIAS);
	f_0(eta);

#ifdef CONTINUOUS_LEARNING
	f(storage_gamma, DECAY_PARAMETERS_STORAGE_BIAS);
	f_0(gamma);
#endif

#ifdef OJA
	f(storage_delta, DECAY_PARAMETERS_STORAGE_BIAS);
	f_0(delta);
#endif

#ifndef RANDOM_W
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
	f(storage_eta);

#ifdef CONTINUOUS_LEARNING
	f(gamma);
	f(storage_gamma);
#endif

#ifndef RANDOM_W
	f(w);
#endif

#ifdef OJA
	f(delta);
	f(storage_delta);
#endif

#ifdef GUIDED_MUTATIONS
	f(accumulator); // set to 0...
#endif

	float* newB = new float[nLines - s];
	std::copy(biases.get(), biases.get() + id, newB);
	std::copy(biases.get() + id + s, biases.get() + nLines, newB);
	biases.reset(newB);

	ACTIVATION* newA = new ACTIVATION[nLines - s];
	std::copy(activationFunctions.get(), activationFunctions.get() + id, newA);
	std::copy(activationFunctions.get() + id + s, activationFunctions.get() + nLines, newA);
	activationFunctions.reset(newA);


#ifdef STDP
	STDP_mu.reset(new float[nLines - s]);
	STDP_lambda.reset(new float[nLines - s]);

	float* newMu = new float[nLines - s];
	float* newLambda = new float[nLines - s];
	std::copy(STDP_storage_mu.get(), STDP_storage_mu.get() + id, newMu);
	std::copy(STDP_storage_lambda.get(), STDP_storage_lambda.get() + id, newLambda);
	std::copy(STDP_storage_mu.get() + id + s, STDP_storage_mu.get() + nLines, newMu);
	std::copy(STDP_storage_lambda.get() + id + s, STDP_storage_lambda.get() + nLines, newLambda);
	STDP_storage_mu.reset(newMu);
	STDP_storage_lambda.reset(newLambda);
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
	f(storage_eta);

#ifdef CONTINUOUS_LEARNING
	f(gamma);
	f(storage_gamma);
#endif

#ifndef RANDOM_W
	f(w);
#endif

#ifdef OJA
	f(delta);
	f(storage_delta);
#endif

#ifdef GUIDED_MUTATIONS
	f(accumulator); // set to 0...
#endif

	nColumns -= s;
}

void InternalConnexion_G::transform01Parameters() {
	int s = nLines * nColumns;
	for (int i = 0; i < s; i++) {
		eta[i] = (tanhf(storage_eta[i]) + 1.0f) * .5f;
#ifdef CONTINUOUS_LEARNING
		gamma[i] = (tanhf(storage_gamma[i]) + 1.0f) * .5f;
#endif
#ifdef OJA
		delta[i] = (tanhf(storage_delta[i]) + 1.0f) * .5f;
#endif
	}

	
#ifdef STDP
	s = nLines;
	for (int i = 0; i < s; i++) {
		// 1 - d because it is viable to use it solely as a decay term and not the time
		// constant of a moving exponential average.
		STDP_mu[i] = 1.0f - (tanhf(STDP_storage_mu[i]) + 1.0f) * .5f; 

		STDP_lambda[i] = (tanhf(STDP_storage_lambda[i]) + 1.0f) * .5f; 
	}
#endif
}