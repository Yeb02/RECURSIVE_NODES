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
	beta = n->beta;
	decay = n->decay;

	link = n->link; // deep copy assignement using overloaded = operator of genotypeConnexion

	int s = inputSize * kernelDimension;
	Q = std::make_unique<float[]>(s);
	std::copy(n->Q.get(), n->Q.get() + s, Q.get());

}

MemoryNode_G::MemoryNode_G(int inputSize, int outputSize, int kernelDimension) :
	inputSize(inputSize), outputSize(outputSize), kernelDimension(kernelDimension)
{
	closestNode = NULL;
	mutationalDistance = 0;
	phenotypicMultiplicity = 0;
	position = -1;
	setBeta();

	link = InternalConnexion_G(outputSize, inputSize, InternalConnexion_G::RANDOM);

	decay = UNIFORM_01 * .3f;
	
	int s = inputSize * kernelDimension;
	Q = std::make_unique<float[]>(s);
	for (int i = 0; i < s; i++) {
		Q[i] = NORMAL_01 * .2f;
	}

}

MemoryNode_G::MemoryNode_G(MemoryNode_G&& n) noexcept {
	position = n.position;
	inputSize = n.inputSize;
	outputSize = n.outputSize;
	mutationalDistance = n.mutationalDistance;
	closestNode = n.closestNode;
	kernelDimension = n.kernelDimension;
	beta = n.beta;
	phenotypicMultiplicity = n.phenotypicMultiplicity;
	
	link = std::move(n.link);
	Q = std::move(n.Q);
}

void MemoryNode_G::mutateFloats() {
	constexpr float p = .2f;

	link.mutateFloats(p);
	
	if (UNIFORM_01 > .05f) [[likely]] {
		decay += decay * (1 - decay) * (UNIFORM_01 - .5f);
	}
	else [[unlikely]] {
		decay = decay * .6f + UNIFORM_01 * .4f;
	}

	int s = inputSize * kernelDimension;
	SET_BINOMIAL(s, p);
	int nMutations = BINOMIAL;
	for (int i = 0; i < nMutations; i++) {
		Q[INT_0X(s)] += NORMAL_01 * .3f;
	}
}


bool MemoryNode_G::incrementInputSize() {
	if (inputSize == MAX_MEMORY_INPUT_SIZE) { return false; }


	// increment Q's number of columns
	int newSize = (inputSize+1) * kernelDimension;
	float* newQ = new float[newSize];
	int idOld = 0, idNew = 0;
	for (int j = 0; j < kernelDimension; j++) {
		for (int k = 0; k < inputSize; k++) {

			newQ[idNew] = Q[idOld];

			idNew++;
			idOld++;
		}
		newQ[idNew] = NORMAL_01 * .2f;
		idNew++;
	}
	Q.reset(newQ);


	inputSize++;
	setBeta();
	link.insertColumnRange(link.nColumns, 1);
	return true;
}

bool MemoryNode_G::incrementOutputSize(){
	if (outputSize == MAX_MEMORY_OUTPUT_SIZE) { return false; }

	outputSize++;
	setBeta();
	link.insertLineRange(link.nLines, 1);
	return true;
}

bool MemoryNode_G::decrementInputSize(int id){
	if (inputSize == 1) { return false; }

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
	return true;
}

bool MemoryNode_G::decrementOutputSize(int id){
	if (outputSize == 1) { return false; }

	outputSize--;
	setBeta();
	link.removeLineRange(id, 1);
	return true;
}


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

bool MemoryNode_G::decrementKernelDimension(int id){
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

