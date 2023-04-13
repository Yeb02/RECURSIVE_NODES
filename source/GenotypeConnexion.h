#pragma once

#include <memory>

#include "Random.h"

#include "config.h"


// NODE_TYPE < 0 are virtual nodes, that do not have instances for efficiency.
const enum NODE_TYPE { MODULATION = -3, OUTPUT = -2, INPUT_NODE = -1, COMPLEX = 0, MEMORY = 1, SIMPLE = 3 };


struct GenotypeConnexion {
	// IDENTITY is valid only if nLines == nColumns. If not, the ZERO case is used.
	const enum initType { ZERO, RANDOM };

	// Unused if originType is either INPUT_NODE or MODULATION. Must be = -1 in this case.
	int originID;

	// Unused if destinationType is either MODULATION or OUTPUT. Must be = -1 in this case.
	int destinationID;

	NODE_TYPE originType, destinationType;

	// Corresponds to the dimension of the input of the destination node. Redundant but eliminates indirections.
	int nLines;
	// Corresponds to the dimension of the output of the origin node. Redundant but eliminates indirections.
	int nColumns;

	std::unique_ptr<float[]> A;
	std::unique_ptr<float[]> B;
	std::unique_ptr<float[]> C;
	std::unique_ptr<float[]> D;
	std::unique_ptr<float[]> eta;
	std::unique_ptr<float[]> alpha;
	std::unique_ptr<float[]> w;
#ifdef CONTINUOUS_LEARNING
	std::unique_ptr<float[]> gamma;
#endif

#ifdef GUIDED_MUTATIONS
	std::unique_ptr<float[]> accumulator;

	void zeroAccumulator() {
		for (int i = 0; i < nLines * nColumns; i++) {
			accumulator[i] = 0.0f;
		}
	}
#endif

	GenotypeConnexion() { 
		//std::cerr << " SHOULD NEVER BE CALLED !" << std::endl; 
		{
			originID = -1;
			destinationID = -1;
			originType = OUTPUT;
			destinationType = INPUT_NODE;
			nColumns = -1;
			nLines = -1;
		}
	};

	// Be careful, do not accidentaly confuse the order of one of the argument pairs.
	GenotypeConnexion(NODE_TYPE oType, NODE_TYPE dType, int oID, int dID, int dInSize, int oOutSize, initType init);

	// Required because a genotype node has a vector of connexions, not of pointers to connexions. 
	// This means that on vector reallocation the move constructor is called. But if it is not specified, it does not 
	// exist because there is a specified destructor. Therefore the constructor and the destructor are called
	// instead, which causes unwanted freeing of memory.
	// Moreover, if it is not marked noexcept std::vector will still use copy+destructor instead in some cases
	// https://stackoverflow.com/questions/9249781/are-move-constructors-required-to-be-noexcept
	GenotypeConnexion(GenotypeConnexion&& gc) noexcept;

	GenotypeConnexion(const GenotypeConnexion& gc);

	GenotypeConnexion operator=(const GenotypeConnexion& gc);

	~GenotypeConnexion() {};

	// invFactor used only when GUIDED_MUTATION is defined
	void mutateFloats(float p, float invFactor=0.0f);

	// nLines++
	void incrementDestinationInputSize();
	// nColumns++
	void incrementOriginOutputSize();
	// nLines--
	void decrementDestinationInputSize(int id);
	// nColumns--
	void decrementOriginOutputSize(int id);
};
