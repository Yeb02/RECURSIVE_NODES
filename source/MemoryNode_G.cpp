#pragma once

#include "MemoryNode_G.h"
#include "Random.h"

MemoryNode_G::MemoryNode_G(MemoryNode_G* n) {
	position = n->position;
	inputSize = n->inputSize;
	outputSize = n->outputSize;
	mutationalDistance = n->mutationalDistance;
	closestNode = n->closestNode;
	kernelDimension = n->kernelDimension;
	phenotypicMultiplicity = n->phenotypicMultiplicity;
	K = n->K;
	beta = n->beta;

	link = n->link; // deep copy assignement

	int s = outputSize * kernelDimension;
	keyM = std::make_unique<float[]>(s);
	std::copy(n->keyM.get(), n->keyM.get() + s, keyM.get());

	s = inputSize * kernelDimension;
	queryM = std::make_unique<float[]>(s);
	std::copy(n->queryM.get(), n->queryM.get() + s, queryM.get());

	s = outputSize * inputSize;
	tQxK = std::make_unique<float[]>(s);
	std::copy(n->tQxK.get(), n->tQxK.get() + s, tQxK.get());

#ifdef GUIDED_MUTATIONS
	nAccumulations = n->nAccumulations;
#endif
}

MemoryNode_G::MemoryNode_G(int inputSize, int outputSize, int kernelDimension) :
	inputSize(inputSize), outputSize(outputSize), kernelDimension(kernelDimension)
{
	closestNode = NULL;
	mutationalDistance = 0;
	phenotypicMultiplicity = 0;
	position = -1;
	K = .2f;

	link = GenotypeConnexion(INPUT_NODE, OUTPUT, -1, -1, outputSize, inputSize, GenotypeConnexion::RANDOM);

	int s = outputSize * kernelDimension;
	keyM = std::make_unique<float[]>(s);
	for (int i = 0; i < s; i++) {
		keyM[i] = NORMAL_01 * .2f;
	}

	s = inputSize * kernelDimension;
	queryM = std::make_unique<float[]>(s);
	for (int i = 0; i < s; i++) {
		queryM[i] = NORMAL_01 * .2f;
	}

	s = outputSize * inputSize;
	tQxK = std::make_unique<float[]>(s);

#ifdef GUIDED_MUTATIONS
	nAccumulations = 0;
#endif
}

MemoryNode_G::MemoryNode_G(MemoryNode_G&& n) noexcept {
	position = n.position;
	inputSize = n.inputSize;
	outputSize = n.outputSize;
	mutationalDistance = n.mutationalDistance;
	closestNode = n.closestNode;
	kernelDimension = n.kernelDimension;
	K = n.K;
	beta = n.beta;
	phenotypicMultiplicity = n.phenotypicMultiplicity;
	
	link = std::move(n.link);
	keyM = std::move(n.keyM);
	queryM = std::move(n.queryM);
	tQxK = std::move(n.tQxK);

#ifdef GUIDED_MUTATIONS
	nAccumulations = n.nAccumulations;
#endif
}

void MemoryNode_G::compute_tQxK() {
	// It would be better to store tK and tQ...
	// Also more efficient to only reallocate tQxK when sizes change, but thats tedious and im lazy.
	tQxK.release();
	tQxK = std::make_unique<float[]>(inputSize * outputSize);
	for (int i = 0; i < inputSize; i++) {
		for (int j = 0; j < outputSize; j++) {
			tQxK[i * outputSize + j] = 0.0f;
			for (int k = 0; k < kernelDimension; k++) {
				tQxK[i * outputSize + j] += queryM[i + k * inputSize] * keyM[j + k * outputSize]; 
			}
		}
	}
}

void MemoryNode_G::mutateFloats() {
	constexpr float p = .4f;
	float invFactor = 0.0f;

#ifdef GUIDED_MUTATIONS
	invFactor = nAccumulations == 0 ? 0.0f : 1.0f / nAccumulations;
	nAccumulations = 0;
#endif

	link.mutateFloats(p, invFactor);
	
	int s = outputSize * kernelDimension;
	SET_BINOMIAL(s, p);
	int nMutations = BINOMIAL;
	for (int i = 0; i < nMutations; i++) {
		keyM[INT_0X(s)] += NORMAL_01 * .1f;
	}

	s = inputSize * kernelDimension;
	SET_BINOMIAL(s, p);
	nMutations = BINOMIAL;
	for (int i = 0; i < nMutations; i++) {
		queryM[INT_0X(s)] += NORMAL_01 * .1f;
	}
}


bool MemoryNode_G::incrementInputSize() {
	if (inputSize == MAX_MEMORY_INPUT_SIZE) { return false; }


	// increment query matrix's number of columns
	int newSize = (inputSize+1) * kernelDimension;
	float* newQueryM = new float[newSize];
	int idOld = 0, idNew = 0;
	for (int j = 0; j < kernelDimension; j++) {
		for (int k = 0; k < inputSize; k++) {

			newQueryM[idNew] = queryM[idOld];

			idNew++;
			idOld++;
		}
		newQueryM[idNew] = NORMAL_01 * .2f;
		idNew++;
	}
	queryM.reset(newQueryM);


	inputSize++;
	link.incrementOriginOutputSize();
	return true;
}

bool MemoryNode_G::incrementOutputSize(){
	if (outputSize == MAX_MEMORY_OUTPUT_SIZE) { return false; }

	// increment key matrix's number of columns
	int newSize = (outputSize + 1) * kernelDimension;
	float* newKeyM = new float[newSize];
	int idOld = 0, idNew = 0;
	for (int j = 0; j < kernelDimension; j++) {
		for (int k = 0; k < outputSize; k++) {

			newKeyM[idNew] = keyM[idOld];

			idNew++;
			idOld++;
		}
		newKeyM[idNew] = NORMAL_01 * .2f;
		idNew++;
	}
	keyM.reset(newKeyM);


	outputSize++;
	link.incrementDestinationInputSize();
	return true;
}

bool MemoryNode_G::decrementInputSize(int id){
	if (inputSize == 1) { return false; }

	// decrement query matrix's number of columns
	int newSize = (inputSize - 1) * kernelDimension;
	float* newQueryM = new float[newSize];
	int idOld = 0, idNew = 0;
	for (int j = 0; j < kernelDimension; j++) {
		for (int k = 0; k < inputSize; k++) {
			if (k == id) {
				idOld++;
				continue;
			}
			newQueryM[idNew] = queryM[idOld];

			idNew++;
			idOld++;
		}
	}
	queryM.reset(newQueryM);


	inputSize--;
	link.decrementOriginOutputSize(id);
	return true;
}

bool MemoryNode_G::decrementOutputSize(int id){
	if (outputSize == 1) { return false; }


	// decrement key matrix's number of columns
	int newSize = (outputSize - 1) * kernelDimension;
	float* newKeyM = new float[newSize];
	int idOld = 0, idNew = 0;
	for (int j = 0; j < kernelDimension; j++) {
		for (int k = 0; k < outputSize; k++) {
			if (k == id) {
				idOld++;
				continue;
			}
			newKeyM[idNew] = keyM[idOld];

			idNew++;
			idOld++;
		}
		newKeyM[idNew] = NORMAL_01 * .2f;
		idNew++;
	}
	keyM.reset(newKeyM);


	outputSize--;
	link.decrementDestinationInputSize(id);
	return true;
}


bool MemoryNode_G::incrementKernelDimension() {
	if (kernelDimension == MAX_KERNEL_DIMENSION) { return false; }

	// increment query matrix's number of lines
	int newSize = inputSize * (kernelDimension + 1);
	float* newQueryM = new float[newSize];
	int idOld = 0, idNew = 0;
	for (int j = 0; j < kernelDimension; j++) {
		for (int k = 0; k < inputSize; k++) {

			newQueryM[idNew] = queryM[idOld];

			idNew++;
			idOld++;
		}
	}
	for (int j = 0; j < inputSize; j++) {
		newQueryM[idNew] = NORMAL_01 * .2f;
		idNew++;
	}
	queryM.reset(newQueryM);

	// increment key matrix's number of lines
	newSize = outputSize * (kernelDimension + 1);
	float* newKeyM = new float[newSize];
	idOld = 0;
	idNew = 0;
	for (int j = 0; j < kernelDimension; j++) {
		for (int k = 0; k < outputSize; k++) {

			newKeyM[idNew] = keyM[idOld];

			idNew++;
			idOld++;
		}
	}
	for (int j = 0; j < outputSize; j++) {
		newKeyM[idNew] = NORMAL_01 * .2f;
		idNew++;
	}
	keyM.reset(newKeyM);


	kernelDimension++;
	return true;
}

bool MemoryNode_G::decrementKernelDimension(int id){
	if (kernelDimension == 1) { return false; }

	// decrement query matrix's number of lines
	int newSize = inputSize * (kernelDimension - 1);
	float* newQueryM = new float[newSize];
	int idOld = 0, idNew = 0;
	for (int j = 0; j < kernelDimension; j++) {
		if (j == id) { 
			idOld += inputSize;
			continue;
		}
		for (int k = 0; k < inputSize; k++) {

			newQueryM[idNew] = queryM[idOld];

			idNew++;
			idOld++;
		}
	}
	queryM.reset(newQueryM);

	// decrement key matrix's number of lines
	newSize = outputSize * (kernelDimension - 1);
	float* newKeyM = new float[newSize];
	idOld = 0;
	idNew = 0;
	for (int j = 0; j < kernelDimension; j++) {
		if (j == id) {
			idOld += outputSize;
			continue;
		}
		for (int k = 0; k < outputSize; k++) {

			newKeyM[idNew] = keyM[idOld];

			idNew++;
			idOld++;
		}
	}
	keyM.reset(newKeyM);


	kernelDimension--;
	return true;
}

