#pragma once

#include "MemoryNode_G.h"
#include "Random.h"

MemoryNode_G::MemoryNode_G(MemoryNode_G* n) {
	position = n->position;
	inputSize = n->inputSize;
	outputSize = n->outputSize;
	mutationalDistance = n->mutationalDistance;
	memoryNodeID = n->memoryNodeID;
	closestNode = n->closestNode;
	kernelDimension = n->kernelDimension;
	phenotypicMultiplicity = n->phenotypicMultiplicity;
	timeSinceLastUse = n->timeSinceLastUse;

	beta = n->beta;
	decay = n->decay;
	storage_decay = n->storage_decay;
#ifdef STDP
	STDP_storage_decay = n->STDP_storage_decay;
#endif

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
	timeSinceLastUse = 0;
	position = -1;
	memoryNodeID = -1;
	setBeta();

	link = InternalConnexion_G(outputSize, inputSize, InternalConnexion_G::RANDOM);

	storage_decay = NORMAL_01 * .2f + DECAY_PARAMETERS_STORAGE_BIAS;
#ifdef STDP
	STDP_storage_decay = NORMAL_01 * .2f + DECAY_PARAMETERS_STORAGE_BIAS;
#endif
	
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
	phenotypicMultiplicity = n.phenotypicMultiplicity;
	timeSinceLastUse = n.timeSinceLastUse;
	
	beta = n.beta;
	decay = n.decay;
	storage_decay = n.storage_decay;
#ifdef STDP
	STDP_storage_decay = n.STDP_storage_decay;
#endif
	
	link = std::move(n.link);
	Q = std::move(n.Q);
}


MemoryNode_G::MemoryNode_G(std::ifstream& is)
{
	READ_4B(inputSize, is);
	READ_4B(outputSize, is);

	READ_4B(mutationalDistance, is);
	READ_4B(timeSinceLastUse, is);
	READ_4B(memoryNodeID, is);

	READ_4B(kernelDimension, is);
	READ_4B(storage_decay, is);


#ifdef STDP
	READ_4B(STDP_storage_decay, is);
#else
	float _unused;
	READ_4B(_unused, is);
#endif

	setBeta();

	link = InternalConnexion_G(is);

	int s = inputSize * kernelDimension;
	Q = std::make_unique<float[]>(s);
	is.read(reinterpret_cast<char*>(Q.get()), s * sizeof(float));
}


void MemoryNode_G::save(std::ofstream& os)
{
	WRITE_4B(inputSize, os);
	WRITE_4B(outputSize, os);

	WRITE_4B(mutationalDistance, os);
	WRITE_4B(timeSinceLastUse, os);
	WRITE_4B(memoryNodeID, os);
	
	WRITE_4B(kernelDimension, os);
	WRITE_4B(storage_decay, os);

#ifdef STDP
	WRITE_4B(STDP_storage_decay, os);
#else
	float unused = 0.0f;
	WRITE_4B(unused, os);
#endif
	
	link.save(os);

	int s = inputSize * kernelDimension;
	os.write(reinterpret_cast<const char*>(Q.get()), s * sizeof(float));
}

void MemoryNode_G::mutateFloats(float adjustedFMutationP) {
	float p = adjustedFMutationP * log2f((float)phenotypicMultiplicity + 1.0f) / (float)phenotypicMultiplicity;

	link.mutateFloats(p);
	
	if (UNIFORM_01 < p) {
		storage_decay *= .9f + NORMAL_01 * .1f;
		storage_decay += NORMAL_01 * .1f;
	}
#ifdef STDP
	if (UNIFORM_01 < p) {
		STDP_storage_decay *= .9f + NORMAL_01 * .1f;
		STDP_storage_decay += NORMAL_01 * .1f;
	}
#endif
	

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

