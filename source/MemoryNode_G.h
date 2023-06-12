#pragma once

#include <memory>
#include <fstream>



#include "InternalConnexion_G.h"


struct MemoryNode_G {
	int inputSize, outputSize;

	// util for Network. The position in the simpleGenome array.
	int position;

	MemoryNode_G* closestNode;

	// unique to each node of the network, used to match them when mating
	int memoryNodeID;

	// Number of mutations its parent network has undergone since this node was created.
	int mutationalDistance;

	// How many times this node appears in the phenotype.
	int phenotypicMultiplicity;

	// Number of mutations its parent network has undergone since last time this node had a phenotypic
	// multiplicity > 0 (so timeSinceLastUse = 0 for nodes that are phenotypically active.)
	int timeSinceLastUse;


#ifdef QKV_MEMORY
	// controls the exponential average decay speed of candidate memory, the higher the faster.
	float decay, storage_decay;

	// The dimension of the scalar product between keys and queries.
	int kernelDimension;

	// the connexion linking input and output
	InternalConnexion_G link;

	// y = softmax(tX * tQ * K * M) * V * M, here we have the following simplifications / optimisations:
	// *   K = Q, and K*M is stored, not M.
	// *   V * M is replaced by a set of memorized output vectors.
	std::unique_ptr<float[]> Q;

	// = 1 / sqrt(kernelDim * InSize) ?
	float beta;

	// to be called after creation and mutations
	inline void setBeta() { beta = 1.0f / sqrtf((float)(inputSize * kernelDimension)); }

	bool incrementKernelDimension();
	bool decrementKernelDimension(int id);
#endif
	
#ifdef SRWM
	float SRWM_decay[4];
	float SRWM_storage_decay[4];

	// = 2 * inputSize + outputSize + 4.  Made an attribute to avoid recomputing it all the time.
	int nLinesW0;

	std::unique_ptr<float[]> W0;
#endif

#ifdef DNN_MEMORY
	// TODO generalize size increase and decrease to any layer. Do not forget accumulator 
	// in case of GUIDED_MUTATIONS.
	
	// Rather n weight matrices.. = 1 if there are no hidden layers.
	int nLayers;

	std::vector<std::unique_ptr<float[]>> W0s;

	std::vector<std::unique_ptr<float[]>> B0s;

	// [in1, out1=in2, out2=in3, ...] so length nLayers + 1
	std::vector<int> sizes;
	
	// treated as a decay parameter in its mutations/creation.
	float learningRate, learningRate_Storage;

#ifdef GUIDED_MUTATIONS
	// concatenated weight matrices then bias vectors of the network.
	//order: w0,b0,w1,b1, ...
	std::unique_ptr<float[]> accumulator;

	int accumulatorSize;

	// requires sizes[] be up to date
	void setAccumulatorSize() 
	{
		int s = 0;
		for (int i = 0; i < nLayers; i++) {
			s += (sizes[i] + 1) * sizes[i + 1];
		}
		accumulatorSize = s;
		accumulator.reset(new float[s]);
	}
#endif
#endif

	MemoryNode_G(MemoryNode_G* n);
	MemoryNode_G(int inputSize, int outputSize);
	MemoryNode_G(MemoryNode_G&& n) noexcept;

	// Should never be called !
	MemoryNode_G(MemoryNode_G& n);

	~MemoryNode_G() {};

	MemoryNode_G(std::ifstream& is);
	void save(std::ofstream& os);

	void transform01Parameters();

#ifdef GUIDED_MUTATIONS
	void zeroAccumulator();


#endif

	int getNParameters();

	void mutateFloats(float adjustedFMutationP);

	bool incrementInputSize();
	bool incrementOutputSize();
	bool decrementInputSize(int id);
	bool decrementOutputSize(int id);

};