#pragma once

#include <memory>
#include "GenotypeConnexion.h"

#define MAX_KERNEL_DIMENSION 100
#define MAX_MEMORY_INPUT_SIZE  10          // Does not apply to the top node
#define MAX_MEMORY_OUTPUT_SIZE  10         // Does not apply to the top node
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

	int mutationalDistance;

	// the connexion linking input and output
	GenotypeConnexion link;

	// The key and query matrices
	std::unique_ptr<float[]> keyM, queryM;

	// = transpose(keyM) * queryM
	std::unique_ptr<float[]> tQxK;


	// Relative importance of output correction / angle to input of the potential memory vector. << 1
	float K;

	// Maybe = 1 / sqrt(kernelDim * InSize * OutSize)
	float beta;

#ifdef GUIDED_MUTATIONS
	int nAccumulations;
#endif

	MemoryNode_G(MemoryNode_G* n);
	MemoryNode_G(int inputSize, int outputSize, int kernelDimension);
	MemoryNode_G(MemoryNode_G&& n) noexcept;

	// Should never be called !
	MemoryNode_G(MemoryNode_G& n) {
		__debugbreak();
	}

	~MemoryNode_G() {};

	// Prepares inference time optimization, by precomputing what can be. Must be called between [any change to
	// keyM or queryM ] and [forward()].
	void precomputeUtils() {
		compute_tQxK();
		beta = 1.0f / sqrtf((float) (inputSize * outputSize * kernelDimension));
	}

	void compute_tQxK();

	void mutateFloats();

	bool incrementInputSize();
	bool incrementOutputSize();
	bool decrementInputSize(int id);
	bool decrementOutputSize(int id);

	bool incrementKernelDimension();
	bool decrementKernelDimension(int id);
};