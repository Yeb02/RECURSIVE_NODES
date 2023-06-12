#pragma once

#include "MemoryNode_G.h"
#include "Random.h"


int MemoryNode_G::getNParameters() {

#ifdef QKV_MEMORY
	return kernelDimension * inputSize + // Q
		link.getNParameters();
#endif

#ifdef SRWM
	return inputSize * nLinesW0;
#endif


#ifdef DNN_MEMORY
	int s = 0;
	for (int i = 0; i < nLayers; i++) {
		s += sizes[i] * (sizes[i + 1] + 1); // +1 for the bias
	}
	return s;
#endif
}

void MemoryNode_G::transform01Parameters() {


#ifdef QKV_MEMORY
	decay = (tanhf(storage_decay) + 1.0f) * .5f;
	link.transform01Parameters();
#endif

#ifdef SRWM
	for (int _i = 0; _i < 4; _i++){
		SRWM_decay[_i] = (tanhf(SRWM_storage_decay[_i]) + 1.0f) * .5f;
	}	
#endif

#ifdef DNN_MEMORY
	learningRate = (tanhf(learningRate_Storage) + 1.0f) * .5f;
#endif

}



MemoryNode_G::MemoryNode_G(MemoryNode_G* n) {
	position = n->position;
	inputSize = n->inputSize;
	outputSize = n->outputSize;
	mutationalDistance = n->mutationalDistance;
	memoryNodeID = n->memoryNodeID;
	closestNode = n->closestNode;
	phenotypicMultiplicity = n->phenotypicMultiplicity;
	timeSinceLastUse = n->timeSinceLastUse;


#ifdef QKV_MEMORY
	kernelDimension = n->kernelDimension;
	beta = n->beta;
	decay = n->decay;
	storage_decay = n->storage_decay;


	link = n->link; // deep copy assignement using overloaded = operator of genotypeConnexion

	int s = inputSize * kernelDimension;
	Q = std::make_unique<float[]>(s);
	std::copy(n->Q.get(), n->Q.get() + s, Q.get());
	
#endif


#ifdef SRWM
	for (int _i = 0; _i < 4; _i++) {
		SRWM_storage_decay[_i] = n->SRWM_storage_decay[_i];
	}

	nLinesW0 = n->nLinesW0;

	int s = inputSize * nLinesW0;
	W0 = std::make_unique<float[]>(s);
	std::copy(n->W0.get(), n->W0.get() + s, W0.get());
#endif

#ifdef DNN_MEMORY
	learningRate_Storage = n->learningRate_Storage;
	nLayers = n->nLayers;

	sizes = n->sizes; // deep copy.

	for (int i = 0; i < nLayers; i++) 
	{
		int sW = sizes[i] * sizes[i + 1];
		W0s.emplace_back(new float[sW]);
		std::copy(n->W0s[i].get(), n->W0s[i].get() + sW, W0s[i].get());

		int sB = sizes[i + 1];
		B0s.emplace_back(new float[sizes[i + 1]]);
		std::copy(n->B0s[i].get(), n->B0s[i].get() + sB, B0s[i].get());
	}

#ifdef GUIDED_MUTATIONS
	setAccumulatorSize();
	std::copy(n->accumulator.get(), n->accumulator.get() + accumulatorSize, accumulator.get());
#endif

#endif
}

MemoryNode_G::MemoryNode_G(int inputSize, int outputSize) :
	inputSize(inputSize), outputSize(outputSize)
{
	closestNode = NULL;
	mutationalDistance = 0;
	phenotypicMultiplicity = 0;
	timeSinceLastUse = 0;
	position = -1;
	memoryNodeID = -1;


#ifdef QKV_MEMORY
	storage_decay = NORMAL_01 * .2f + DECAY_PARAMETERS_STORAGE_BIAS;

	kernelDimension = inputSize + outputSize;
	
	setBeta();

	link = InternalConnexion_G(outputSize, inputSize, InternalConnexion_G::RANDOM);

	int s = inputSize * kernelDimension;
	Q = std::make_unique<float[]>(s);
	for (int i = 0; i < s; i++) {
		Q[i] = NORMAL_01 * .2f;
	}
#endif

#ifdef SRWM
	for (int _i = 0; _i < 4; _i++) {
		SRWM_storage_decay[_i] = NORMAL_01 * .2f - DECAY_PARAMETERS_STORAGE_BIAS; // "-" to avoid using 1-decay
	}

	nLinesW0 = 2 * inputSize + outputSize + 4;

	float f = powf((float)inputSize, -.5f);
	int s = inputSize * nLinesW0;
	W0 = std::make_unique<float[]>(s);
	for (int i = 0; i < s; i++) {
		W0[i] = NORMAL_01 * f;
	}
#endif

#ifdef DNN_MEMORY
	learningRate_Storage = NORMAL_01 * .2f - 1.5f;
	nLayers = 1;

	sizes.push_back(inputSize);
	for (int i = 0; i < nLayers - 1; i++) { // TODO
		sizes.push_back(inputSize);
	}
	sizes.push_back(outputSize);

	for (int i = 0; i < nLayers; i++)
	{
		float f = powf((float)sizes[i], -.5f);
		int sW = sizes[i] * sizes[i + 1];
		W0s.emplace_back(new float[sW]);
		for (int j = 0; j < sW; j++) {
			W0s[i][j] = NORMAL_01 * f;
		}

		int sB = sizes[i + 1];
		B0s.emplace_back(new float[sizes[i + 1]]);
		for (int j = 0; j < sizes[i + 1]; j++) {
			B0s[i][j] = NORMAL_01;
		}
	}

#ifdef GUIDED_MUTATIONS
	setAccumulatorSize();
	std::fill(accumulator.get(), accumulator.get()+accumulatorSize, 0.0f);
#endif

#endif
}

MemoryNode_G::MemoryNode_G(MemoryNode_G&& n) noexcept {
	position = n.position;
	inputSize = n.inputSize;
	outputSize = n.outputSize;
	mutationalDistance = n.mutationalDistance;
	closestNode = n.closestNode;
	phenotypicMultiplicity = n.phenotypicMultiplicity;
	timeSinceLastUse = n.timeSinceLastUse;



#ifdef QKV_MEMORY
	kernelDimension = n.kernelDimension;
	beta = n.beta;
	decay = n.decay;
	storage_decay = n.storage_decay;

	link = std::move(n.link);
	Q = std::move(n.Q);
#endif

#ifdef SRWM
	for (int _i = 0; _i < 4; _i++) {
		SRWM_storage_decay[_i] = n.SRWM_storage_decay[_i];
	}
	nLinesW0 = n.nLinesW0;
	W0 = std::move(n.W0);
#endif

#ifdef DNN_MEMORY
	learningRate_Storage = n.learningRate_Storage;
	nLayers = n.nLayers;

	sizes = std::move(n.sizes); 
	W0s = std::move(n.W0s); 
	B0s = std::move(n.B0s); // TODO works as intended ?

#ifdef GUIDED_MUTATIONS
	setAccumulatorSize();
	accumulator = std::move(n.accumulator);
#endif

#endif
}


MemoryNode_G::MemoryNode_G(std::ifstream& is)
{
	READ_4B(inputSize, is);
	READ_4B(outputSize, is);

	READ_4B(mutationalDistance, is);
	READ_4B(timeSinceLastUse, is);
	READ_4B(memoryNodeID, is);


#ifdef QKV_MEMORY
	READ_4B(kernelDimension, is);
	READ_4B(storage_decay, is);


	setBeta();

	link = InternalConnexion_G(is);


	int s = inputSize * kernelDimension;
	Q = std::make_unique<float[]>(s);
	is.read(reinterpret_cast<char*>(Q.get()), s * sizeof(float));
#endif


#ifdef SRWM
	for (int _i = 0; _i < 4; _i++) {
		READ_4B(SRWM_storage_decay[_i], is);
	}

	nLinesW0 = 2 * inputSize + outputSize + 4;

	int s = inputSize * nLinesW0;
	W0 = std::make_unique<float[]>(s);
	is.read(reinterpret_cast<char*>(W0.get()), s * sizeof(float));
#endif


#ifdef DNN_MEMORY
	READ_4B(learningRate_Storage, is);
	READ_4B(nLayers, is);

	sizes.resize(nLayers + 1);
	is.read(reinterpret_cast<char*>(sizes.data()), (nLayers+1) * sizeof(float));
	

	for (int i = 0; i < nLayers; i++)
	{
		int sW = sizes[i] * sizes[i + 1];
		W0s[i] = std::make_unique<float[]>(sW);
		is.read(reinterpret_cast<char*>(W0s[i].get()), sW * sizeof(float));

		int sB = sizes[i + 1];
		B0s[i] = std::make_unique<float[]>(sB);
		is.read(reinterpret_cast<char*>(B0s[i].get()), sB * sizeof(float));
	}
#ifdef GUIDED_MUTATIONS
	setAccumulatorSize();
#endif
#endif


}

void MemoryNode_G::save(std::ofstream& os)
{
	WRITE_4B(inputSize, os);
	WRITE_4B(outputSize, os);

	WRITE_4B(mutationalDistance, os);
	WRITE_4B(timeSinceLastUse, os);
	WRITE_4B(memoryNodeID, os);
	
#ifdef QKV_MEMORY
	WRITE_4B(kernelDimension, os);
	WRITE_4B(storage_decay, os);

	link.save(os);

	int s = inputSize * kernelDimension;
	os.write(reinterpret_cast<const char*>(Q.get()), s * sizeof(float));
#endif

#ifdef SRWM
	for (int _i = 0; _i < 4; _i++) {
		WRITE_4B(SRWM_storage_decay[_i], os);
	}

	int s = inputSize * nLinesW0;
	os.write(reinterpret_cast<const char*>(W0.get()), s * sizeof(float));
#endif

#ifdef DNN_MEMORY
	WRITE_4B(learningRate_Storage, os);
	WRITE_4B(nLayers, os);

	os.write(reinterpret_cast<char*>(sizes.data()), (nLayers + 1) * sizeof(float));


	for (int i = 0; i < nLayers; i++)
	{
		int sW = sizes[i] * sizes[i + 1];
		os.write(reinterpret_cast<char*>(W0s[i].get()), sW * sizeof(float));

		int sB = sizes[i + 1];
		os.write(reinterpret_cast<char*>(B0s[i].get()), sB * sizeof(float));
	}
#endif

}


void MemoryNode_G::mutateFloats(float adjustedFMutationP) {
	float p = adjustedFMutationP * log2f((float)phenotypicMultiplicity + 1.0f) / (float)phenotypicMultiplicity;


#ifdef QKV_MEMORY
	link.mutateFloats(p);

	if (UNIFORM_01 < p) {
		storage_decay *= .9f + NORMAL_01 * .1f;
		storage_decay += NORMAL_01 * .1f;
	}


	float f = powf((float)inputSize, -.5f);
	int s = inputSize * kernelDimension;
	SET_BINOMIAL(s, p);
	int nMutations = BINOMIAL;
	for (int i = 0; i < nMutations; i++) {
		int id = INT_0X(s);
		Q[id] = Q[id] * (.95f + .3f * f * NORMAL_01) + NORMAL_01 * f;
	}
#endif

#ifdef SRWM
	float f = powf((float)inputSize, -.5f);
	int s = inputSize * nLinesW0;
	SET_BINOMIAL(s, p);
	int nMutations = BINOMIAL;
	for (int i = 0; i < nMutations; i++) {
		int id = INT_0X(s);
		W0[id] = W0[id] * (.95f + .3f * f * NORMAL_01) + NORMAL_01 * f;
	}

	for (int _i = 0; _i < 4; _i++) {
		if (UNIFORM_01 < p) {
			SRWM_storage_decay[_i] *= .9f + NORMAL_01 * .1f;
			SRWM_storage_decay[_i] += NORMAL_01 * .1f;
		}
	}
#endif


#ifdef DNN_MEMORY
	if (UNIFORM_01 < p) {
		learningRate_Storage *= .9f + NORMAL_01 * .1f;
		learningRate_Storage += NORMAL_01 * .1f;
	}

	int accId = 0; // only used with GUIDED_MUTATIONS
	for (int i = 0; i < nLayers; i++)
	{
		int sW = sizes[i] * sizes[i + 1];
		SET_BINOMIAL(sW, p);
		int nMutations = BINOMIAL;
		float f = powf((float)sizes[i], -.5f);
		for (int j = 0; j < nMutations; j++) {
			int id = INT_0X(sW);
			W0s[i][id] = W0s[i][id] * (.95f + .3f * f * NORMAL_01) + NORMAL_01 * f;
		}

		int sB = sizes[i + 1];
		SET_BINOMIAL(sB, p);
		nMutations = BINOMIAL;
		for (int j = 0; j < nMutations; j++) {
			int id = INT_0X(sB);
			B0s[i][id] = B0s[i][id] * (.95f + .1f * NORMAL_01) + NORMAL_01 * .2f;
		}

#ifdef GUIDED_MUTATIONS
		for (int j = 0; j < sW; j++) {
			W0s[i][j] += std::max(std::min(accumulator[accId], 1.0f), -1.0f);
			accId++;
		}
		for (int j = 0; j < sB; j++) {
			B0s[i][j] += std::max(std::min(accumulator[accId], 1.0f), -1.0f);
			accId++;
		}
#endif

	}

#if defined GUIDED_MUTATIONS
	zeroAccumulator();
#endif

#endif

}

#ifdef GUIDED_MUTATIONS
void MemoryNode_G::zeroAccumulator() 
{
#ifdef QKV_MEMORY
	link.zeroAccumulator();
#endif
	 
#ifdef DNN_MEMORY
	std::fill(accumulator.get(), accumulator.get() + accumulatorSize, 0.0f);
#endif
}
#endif

bool MemoryNode_G::incrementInputSize() {
	if (inputSize == MAX_MEMORY_INPUT_SIZE) { return false; }


#ifdef QKV_MEMORY
	// increment Q's number of columns
	float f = powf((float)inputSize, -.5f);
	int newSize = (inputSize + 1) * kernelDimension;
	float* newQ = new float[newSize];
	int idOld = 0, idNew = 0;
	for (int j = 0; j < kernelDimension; j++) {
		for (int k = 0; k < inputSize; k++) {

			newQ[idNew] = Q[idOld];

			idNew++;
			idOld++;
		}
		newQ[idNew] = NORMAL_01 * f;
		idNew++;
	}
	Q.reset(newQ);

	inputSize++;

	setBeta();
	link.insertColumnRange(link.nColumns, 1);
#endif
	

#ifdef SRWM
	float f = powf((float)inputSize, -.5f);

	// insert 1 column and 2 lines in W0. Could be done in one go, but unnecessary complexity.

	// First the column. It goes in the last position.
	int newSize = (inputSize + 1) * nLinesW0;
	float* newW0 = new float[newSize];
	int idOld = 0, idNew = 0;
	for (int j = 0; j < nLinesW0; j++) {
		for (int k = 0; k < inputSize; k++) {

			newW0[idNew] = W0[idOld];

			idNew++;
			idOld++;
		}
		newW0[idNew] = NORMAL_01 * f;
		idNew++;
	}
	W0.reset(newW0);


	// Then the 2 lines. They go in position p1 and p2.
	int newInputSize = inputSize + 1;
	int p1 = outputSize + inputSize;
	int p2 = outputSize + 2 * inputSize + 1;

	newSize = newInputSize * (nLinesW0 + 2);
	newW0 = new float[newSize];
	idOld = 0;
	idNew = 0;
	for (int j = 0; j < nLinesW0+2; j++) {
		if (j == p1 || j == p2) 
		{
			idNew += newInputSize;
			continue;
		}
		for (int k = 0; k < newInputSize; k++) {

			newW0[idNew] = W0[idOld];

			idNew++;
			idOld++;
		}
	}
	idNew = p1 * newInputSize;
	for (int k = 0; k < newInputSize; k++) {
		newW0[idNew] = NORMAL_01 * f;
		idNew++;
	}
	idNew = p2 * newInputSize;
	for (int k = 0; k < newInputSize; k++) {
		newW0[idNew] = NORMAL_01 * f;
		idNew++;
	}

	W0.reset(newW0);

	inputSize++;
	nLinesW0 += 2;

#endif

#ifdef DNN_MEMORY
	float f = powf((float)inputSize, -.5f);
	int newSize = (inputSize + 1) * sizes[1];
	float* newW = new float[newSize];
	
	int idOld = 0, idNew = 0;
	for (int j = 0; j < sizes[1]; j++) {
		for (int k = 0; k < inputSize; k++) {

			newW[idNew] = W0s[0][idOld];

			idNew++;
			idOld++;
		}
		newW[idNew] = NORMAL_01 * f;
		idNew++;
	}

	W0s[0].reset(newW);

	sizes[0]++;
	inputSize++;

#ifdef GUIDED_MUTATIONS
	setAccumulatorSize();
#endif

#endif

	return true;
}

bool MemoryNode_G::incrementOutputSize(){
	if (outputSize == MAX_MEMORY_OUTPUT_SIZE) { return false; }


#ifdef QKV_MEMORY
	outputSize++;
	setBeta();
	link.insertLineRange(link.nLines, 1);
#endif

#ifdef SRWM
	float f = powf((float)inputSize, -.5f);

	// insert a line in W0 at position outputSize.
	int newSize = inputSize * (nLinesW0 + 1);
	float* newW0 = new float[newSize];
	int idOld = 0, idNew = 0;
	for (int j = 0; j < nLinesW0 + 1; j++) {
		if (j == outputSize)
		{
			idNew += inputSize;
			continue;
		}
		for (int k = 0; k < inputSize; k++) {

			newW0[idNew] = W0[idOld];

			idNew++;
			idOld++;
		}
	}
	idNew = outputSize * inputSize;
	for (int j = 0; j < inputSize; j++) {
		newW0[idNew] = NORMAL_01 * f;
		idNew++;
	}
	W0.reset(newW0);
	nLinesW0++;
	outputSize++;
#endif
	
#ifdef DNN_MEMORY
	float f = powf((float)sizes[nLayers-1], -.5f);

	int newSize = sizes[nLayers - 1] * (sizes[nLayers]+1);
	float* newW = new float[newSize];

	int idOld = 0, idNew = 0;
	for (int j = 0; j < sizes[nLayers]; j++) {
		for (int k = 0; k < sizes[nLayers-1]; k++) {

			newW[idNew] = W0s[nLayers-1][idOld];

			idNew++;
			idOld++;
		}
	}
	for (int i = 0; i < sizes[nLayers - 1]; i++) {
		newW[idNew] = NORMAL_01 * f;
		idNew++;
	}

	W0s[nLayers-1].reset(newW);


	float* newB = new float[outputSize + 1];
	std::copy(B0s[nLayers - 1].get(), B0s[nLayers - 1].get() + outputSize, newB);
	newB[outputSize] = NORMAL_01;
	B0s[nLayers - 1].reset(newB);


	sizes[nLayers]++;
	outputSize++;

#ifdef GUIDED_MUTATIONS
	setAccumulatorSize();
#endif

#endif

	return true;
}

bool MemoryNode_G::decrementInputSize(int id){
	if (inputSize == 1) { return false; }


#ifdef QKV_MEMORY
	// decrement Q's number of columns
	int newSize = (inputSize - 1) * kernelDimension;
	float* newQ = new float[newSize];
	int idOld = 0, idNew = 0;
	for (int j = 0; j < kernelDimension; j++) {
		for (int k = 0; k < inputSize; k++) {
			if (k == id) {
				idOld++;
				continue;
			}
			newQ[idNew] = Q[idOld];

			idNew++;
			idOld++;
		}
	}
	Q.reset(newQ);


	inputSize--;
	setBeta();
	link.removeColumnRange(id, 1);
#endif


#ifdef SRWM
	// Remove a column from w0 and 2 lines.

	// first, the id-th column.
	int newSize = (inputSize - 1) * nLinesW0;
	float* newW0 = new float[newSize];
	int idOld = 0, idNew = 0;
	for (int j = 0; j < nLinesW0; j++) {
		for (int k = 0; k < inputSize; k++) {
			if (k == id) {
				idOld++;
				continue;
			}
			newW0[idNew] = W0[idOld];

			idNew++;
			idOld++;
		}
	}
	W0.reset(newW0);

	// Then, the lines at positions p1 and p2.
	int p1 = outputSize + id;
	int p2 = p1 + inputSize;

	newSize = (inputSize - 1) * (nLinesW0-2);
	newW0 = new float[newSize];
	idOld = 0;
	idNew = 0;
	for (int j = 0; j < nLinesW0; j++) {
		if (j == p1 || j == p2) {
			idOld += inputSize - 1;
			continue;
		}
		for (int k = 0; k < inputSize-1; k++) {
			
			newW0[idNew] = W0[idOld];

			idNew++;
			idOld++;
		}
	}
	W0.reset(newW0);

	inputSize--;
	nLinesW0 -= 2;
#endif

#ifdef DNN_MEMORY

	int newSize = sizes[1] * (inputSize - 1);
	float* newW = new float[newSize];

	int idOld = 0, idNew = 0;
	for (int j = 0; j < sizes[1]; j++) {
		for (int k = 0; k < inputSize; k++) {
			if (k == id) {
				idOld++;
				continue;
			}

			newW[idNew] = W0s[0][idOld];

			idNew++;
			idOld++;
		}
	}

	W0s[0].reset(newW);

	sizes[0]--;
	inputSize--;

#ifdef GUIDED_MUTATIONS
	setAccumulatorSize();
#endif

#endif

	return true;
}

bool MemoryNode_G::decrementOutputSize(int id){
	if (outputSize == 1) { return false; }


#ifdef QKV_MEMORY
	outputSize--;
	setBeta();
	link.removeLineRange(id, 1);
#endif

#ifdef SRWM
	// Remove the id-th line of w0.
	int newSize = inputSize * (nLinesW0 - 1);
	float* newW0 = new float[newSize];
	int idOld = 0, idNew = 0;
	for (int j = 0; j < nLinesW0; j++) {
		if (j == id) {
			idOld += inputSize;
			continue;
		}
		for (int k = 0; k < inputSize; k++) {

			newW0[idNew] = W0[idOld];

			idNew++;
			idOld++;
		}
	}
	W0.reset(newW0);

	outputSize--;
	nLinesW0 -= 1;
#endif

#ifdef DNN_MEMORY

	int newSize = sizes[nLayers - 1] * (sizes[nLayers] + 1);
	float* newW = new float[newSize];

	int idOld = 0, idNew = 0;
	for (int j = 0; j < sizes[nLayers]; j++) {
		if (j == id) {
			idOld += sizes[nLayers-1];
			continue;
		}
		for (int k = 0; k < sizes[nLayers - 1]; k++) {

			newW[idNew] = W0s[nLayers-1][idOld];

			idNew++;
			idOld++;
		}
	}

	W0s[nLayers - 1].reset(newW);

	float* newB = new float[outputSize - 1];
	idNew = 0;
	for (int i = 0; i < outputSize; i++) {
		if (i == id) {
			continue;
		}
		newB[idNew] = B0s[nLayers - 1][i];
		idNew++;
	}
	B0s[nLayers - 1].reset(newB);

	sizes[nLayers]--;
	outputSize--;

#ifdef GUIDED_MUTATIONS
	setAccumulatorSize();
#endif

#endif
	
	return true;
}


// Kernel dimension mutations
#ifdef QKV_MEMORY
bool MemoryNode_G::incrementKernelDimension() {
	if (kernelDimension == MAX_KERNEL_DIMENSION) { return false; }

	// increment Q's number of lines
	int newSize = inputSize * (kernelDimension + 1);
	float* newQ = new float[newSize];
	int idOld = 0, idNew = 0;
	for (int j = 0; j < kernelDimension; j++) {
		for (int k = 0; k < inputSize; k++) {

			newQ[idNew] = Q[idOld];

			idNew++;
			idOld++;
		}
	}
	for (int j = 0; j < inputSize; j++) {
		newQ[idNew] = NORMAL_01 * .2f;
		idNew++;
	}
	Q.reset(newQ);

	kernelDimension++;
	setBeta();
	return true;
}

bool MemoryNode_G::decrementKernelDimension(int id) {
	if (kernelDimension == 1) { return false; }

	// decrement Q's number of lines
	int newSize = inputSize * (kernelDimension - 1);
	float* newQ = new float[newSize];
	int idOld = 0, idNew = 0;
	for (int j = 0; j < kernelDimension; j++) {
		if (j == id) {
			idOld += inputSize;
			continue;
		}
		for (int k = 0; k < inputSize; k++) {

			newQ[idNew] = Q[idOld];

			idNew++;
			idOld++;
		}
	}
	Q.reset(newQ);

	kernelDimension--;
	setBeta();
	return true;
}


#endif
