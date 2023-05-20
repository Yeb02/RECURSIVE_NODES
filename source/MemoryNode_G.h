#pragma once

#include <memory>
#include "InternalConnexion_G.h"

#define MAX_KERNEL_DIMENSION 100
#define MAX_MEMORY_INPUT_SIZE  10          
#define MAX_MEMORY_OUTPUT_SIZE  10         
#define MODULATION_VECTOR_SIZE 5           // Short-term, long-term, ksi1, ksi2, ksi3

struct MemoryNode_G {
	int inputSize, outputSize;

	// util for Network. The position in the simpleGenome array.
	int position;

	// The dimension of the scalar product between keys and queries.
	int kernelDimension;

	// Util for Network. How many times this node appears in the phenotype.
	int phenotypicMultiplicity;

	MemoryNode_G* closestNode;

	// unique to each node of the network, used to match them when mating
	int memoryNodeID;

	int mutationalDistance;

	// the connexion linking input and output
	InternalConnexion_G link;

	// y = softmax(tX * tQ * K * M) * V * M, here we have the following simplifications / optimisations:
	// *   K = Q, and K*M is stored, not M.
	// *   V * M is replaced by a set of memorized output vectors.
	std::unique_ptr<float[]> Q;

	// controls the exponential average decay speed of canddidate memory, the higher the faster.
	float decay, storage_decay;

	// = 1 / sqrt(kernelDim * InSize)
	float beta;

	inline void setBeta() { beta = 1.0f / sqrtf((float) (inputSize * kernelDimension)); }

	MemoryNode_G(MemoryNode_G* n);
	MemoryNode_G(int inputSize, int outputSize, int kernelDimension);
	MemoryNode_G(MemoryNode_G&& n) noexcept;

	// Should never be called !
	MemoryNode_G(MemoryNode_G& n) {
		__debugbreak();
	}

	~MemoryNode_G() {};

	void mutateFloats();

	bool incrementInputSize();
	bool incrementOutputSize();
	bool decrementInputSize(int id);
	bool decrementOutputSize(int id);

	bool incrementKernelDimension();
	bool decrementKernelDimension(int id);
};