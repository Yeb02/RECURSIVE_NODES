#pragma once

#include <vector>
#include <memory>
#include <cmath>

#include "Random.h"


//#define RISI_NAJARRO_2020
#define USING_NEUROMODULATION
//#define DYNAMIC_MUTATION_P

#define MAX_CHILDREN_PER_BLOCK  10
#define MAX_BLOCK_INPUT_SIZE  10          // Does not apply to the top one, which is the network itself
#define MAX_BLOCK_OUTPUT_SIZE  10         // Does not apply to the top one, which is the network itself
// In the genotype, denotes the fact that a child is connected
// to the input 
#define INPUT_ID -1     

inline float ReLU(float x) { return x > 0 ? x : 0; }

struct GenotypeConnexion {
	// IDENTITY is valid only if nLines == nColumns. If not, the ZERO case is used.
	const enum initType { ZERO, IDENTITY, RANDOM };

	int originID, destinationID;

#ifdef DYNAMIC_MUTATION_P
	float pMutations[6]; // A, B, C, alpha, w, eta
#endif

	// Corresponds to the dimension of the input of the destination neuron. Redundant but eliminates indirections.
	int nLines;
	// Corresponds to the dimension of the output of the origin neuron. Redundant but eliminates indirections.
	int nColumns;

	std::unique_ptr<float[]> A;
	std::unique_ptr<float[]> B;
	std::unique_ptr<float[]> C;
	std::unique_ptr<float[]> eta;

#if defined RISI_NAJARRO_2020
	std::unique_ptr<float[]> D;

#elif defined USING_NEUROMODULATION
	std::unique_ptr<float[]> alpha;
	std::unique_ptr<float[]> w;
#endif 


	GenotypeConnexion() {}

	GenotypeConnexion(int oID, int dID, int nLines, int nColumns, initType init);

	// Required because a genotype node has a vector of connexions, not of pointers to connexions. 
	// This means that on vector reallocation the move constructor is called. But if it is not specified, it does not 
	// exist because there is a specified destructor. Therefore the constructor and the destructor are called
	// instead, which causes unwanted freeing of memory.
	// Moreover, if it is not marked noexcept std::vector will use copy+destructor instead in some cases, WTF ????
	// https://stackoverflow.com/questions/9249781/are-move-constructors-required-to-be-noexcept
	GenotypeConnexion(GenotypeConnexion&& gc) noexcept;

	GenotypeConnexion(const GenotypeConnexion& gc);

	GenotypeConnexion operator=(const GenotypeConnexion& gc);

	~GenotypeConnexion() {};
};

struct GenotypeNode {
	bool isSimpleNeuron;
	float (*f)(float); // NULL if Node is a bloc. Else pointer to tanH, cos, ReLU
	int inputSize, outputSize; // >= 1
	std::vector<float> inBias, outBias;   // length outputSize. Is added the the presynaptic activations of the tanh of the output node

	// Order matters in this vector.
	// Contains pointers to the genotypes of the children
	std::vector<GenotypeNode*> children;

	// Vector of structs containing pointers to the fixed connexion matrices linking children
	std::vector<GenotypeConnexion> childrenConnexions;

#ifdef USING_NEUROMODULATION
	// neuromodulatorySignal = tanh(neuromodulationBias + SUM(w*out))
	std::vector<float> wNeuromodulation;
	// neuromodulatorySignal = tanh(neuromodulationBias + SUM(w*out))
	float neuromodulationBias;
#endif

	// Depth of the children tree. =0 for simple neurons
	int depth;

	// The position in the genome vector. Must be genome.size() - 1 for the top node.
	int position;

	// Point towards the node it was cloned from or boxed from on the tree of life. 
	GenotypeNode* closestNode;

	// How far it went from the closestNode with mutations (does not account 
	int mutationalDistance;

	// Utils for forward:

	// = sum(child->inputSize for child in children)
	int concatenatedChildrenInputLength;
	// Has len(children) + 1 elements, and contains :
	// 0, children[0]->inputsize, children[0]->inputsize+children[1]->inputsize, ....
	std::vector<int> concatenatedChildrenInputBeacons;

	// Empty definition because many attributes are set by the network owning this. Same reason for no copy constructor.
	GenotypeNode() {};

	~GenotypeNode() {};

	// Used for the top node, as its input bias should not mutate.
	void zeroInBias() {for (int i = 0; i < inputSize; i++) inBias[i] = 0;}

	// Populates the concatenatedChildrenInputBeacons vector and sets concatenatedChildrenInputLength
	void computeBeacons();

	// genomeState is an array of the size of the genome, which has 1s where the node's depth is known and 0s elsewhere
	void updateDepth(std::vector<int>& genomeState);

	// Mutate real-valued floating point parameters
	void mutateFloats();

	// Try to connect two random children nodes. Is less likely to succed as the connexion density rises.
	void connect();

	// Disconnect two children nodes picked randomely. 
	void disconnect();

	// Add the specified child to the node. The child's depth is <= to this node's depth (which will be increased by 1 if it was ==)
	// It is initially connected to 2 random nodes, with a preference for the input and the output.
	void addChild(GenotypeNode* child);

	// Removes the rID child in the children list.
	void removeChild(int rID);

	// Resizes the matrices of every connexion between the children and the input, adding a column.
	void incrementInputSize();
	// Resizes the connexion matrices linking the children together.
	void onChildInputSizeIncremented(GenotypeNode* modifiedType);
	// Util for incrementInputSize and onChildInputSizeIncremented
	void incrementDestinationInputSize(int i);

	// Resizes the matrices of every connexion between the children and the output, adding a line.
	void incrementOutputSize();
	// Resizes the connexion matrices linking the children together.
	void onChildOutputSizeIncremented(GenotypeNode* modifiedType);
	// Util for incrementOutputSize and onChildOutputSizeIncremented
	void incrementOriginOutputSize(int i);

	// Resizes the matrices of every connexion between the children and the input, deleting the id-th column.
	void decrementInputSize(int id);
	// Resizes the connexion matrices linking the children together.
	void onChildInputSizeDecremented(GenotypeNode* modifiedType, int id);
	// Util for decrementInputSize and onChildInputSizeDecremented
	void decrementDestinationInputSize(int i, int id);

	// Resizes the matrices of every connexion between the children and the output, deleting the id-th line.
	void decrementOutputSize(int id);
	// Resizes the connexion matrices linking the children together.
	void onChildOutputSizeDecremented(GenotypeNode* modifiedType, int id);
	// Util for decrementOutputSize and onChildOutputSizeDecremented
	void decrementOriginOutputSize(int i, int id);
};