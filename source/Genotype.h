#pragma once

#include <vector>
#include <memory>
#include <cmath>
#include <iostream>

#include "Random.h"
#include "Config.h"

// Constants:
#define MAX_CHILDREN_PER_BLOCK  10
#define MAX_BLOCK_INPUT_SIZE  10          // Does not apply to the top one, which is the network itself
#define MAX_BLOCK_OUTPUT_SIZE  10         // Does not apply to the top one, which is the network itself
#define INPUT_ID -1			// In a genotype connexion, means the origin node is the parent's input.
#define MODULATION_ID -2    // In a genotypex connexion, means the destination node is the parent's modulation. 

// Utils:
inline int binarySearch(std::vector<float>& proba, float value) {
	int inf = 0;
	int sup = (int)proba.size() - 1;

	if (proba[inf] > value) {
		return inf;
	}

	int mid;
	int max_iter = 15;
	while (sup - inf >= 1 && max_iter--) {
		mid = (sup + inf) / 2;
		if (proba[mid] < value && value <= proba[mid + 1]) {
			return mid + 1;
		}
		else if (proba[mid] < value) {
			inf = mid;
		}
		else {
			sup = mid;
		}
	}
	return 0; // not necessarily a failure, since floating point approximation prevents the sum from reaching 1.
	//throw "Binary search failure !";
}



struct GenotypeConnexion {
	// IDENTITY is valid only if nLines == nColumns. If not, the ZERO case is used.
	const enum initType { ZERO, RANDOM };

	int originID, destinationID;


	// Corresponds to the dimension of the input of the destination neuron. Redundant but eliminates indirections.
	int nLines;
	// Corresponds to the dimension of the output of the origin neuron. Redundant but eliminates indirections.
	int nColumns;

	std::unique_ptr<float[]> A;
	std::unique_ptr<float[]> B;
	std::unique_ptr<float[]> C;
	std::unique_ptr<float[]> D;
	std::unique_ptr<float[]> eta;
	std::unique_ptr<float[]> alpha;
	std::unique_ptr<float[]> w;
#ifdef CONTINUOUS_LEARNING
	// proche de 0 !
	std::unique_ptr<float[]> gamma;
#endif

#ifdef GUIDED_MUTATIONS
	std::unique_ptr<float[]> accumulator;
#endif


	GenotypeConnexion() { std::cerr << " SHOULD NEVER BE CALLED !" << std::endl; };

	GenotypeConnexion(int oID, int dID, int nLines, int nColumns, initType init);

	// Required because a genotype node has a vector of connexions, not of pointers to connexions. 
	// This means that on vector reallocation the move constructor is called. But if it is not specified, it does not 
	// exist because there is a specified destructor. Therefore the constructor and the destructor are called
	// instead, which causes unwanted freeing of memory.
	// Moreover, if it is not marked noexcept std::vector will still use copy+destructor instead in some cases, WTF ????
	// https://stackoverflow.com/questions/9249781/are-move-constructors-required-to-be-noexcept
	GenotypeConnexion(GenotypeConnexion&& gc) noexcept;

	GenotypeConnexion(const GenotypeConnexion& gc);

	GenotypeConnexion operator=(const GenotypeConnexion& gc);

	~GenotypeConnexion() {};
};


struct GenotypeNode {
	// DERIVATOR outputs the difference between input at this step and input at the previous step.
	// I dont really know what to expect when it comes to applying hebbian rules to it... At least 
	// it is (in a way) linear.
	const enum NODE_TYPE { COMPLEX = 0, TANH = 1, DERIVATOR = 2 };

	NODE_TYPE nodeType; // COMPLEX, or one of the sub-types of simple.
	int inputSize, outputSize; // >= 1

	
	// Contains pointers to the genotypes of the children.
	std::vector<GenotypeNode*> children;

	// Vector of structs holding pointers to the fixed connexion matrices linking children
	std::vector<GenotypeConnexion> childrenConnexions;

	// neuromodulation bias.
	float biasM[2];

	// Depth of the children tree. =0 for simple neurons, at least 1 otherwise, even if there are no children.
	int depth;

	// The position in the genome vector. Must be genome.size() for the top node.
	int position;

	// Point towards the node it was cloned from or boxed from on the tree of life. 
	GenotypeNode* closestNode;

	// How many iterations of floating point values mutations it has undergone since it was created.
	int mutationalDistance;

	// = sum(child->inputSize for child in children)
	int sumChildrenInputSizes;

	// How many times this node appears in the phenotype.
	int phenotypicMultiplicity;

	// Has len(children) + 1 elements, and contains :
	// 0, children[0]->inputsize, children[0]->inputsize+children[1]->inputsize, ....
	std::vector<int> concatenatedChildrenInputBeacons;

	// size sumChildrenInputSizes + outputSize. Also includes this node's output bias.
	std::vector<float> childrenInBias;

#ifdef GUIDED_MUTATIONS
	//  = [How many instances of this node the network has] * [how many times wLifetime was accumulated].
	int nAccumulations;
#endif
	// Empty definition because many attributes are set by the network owning this. Same reason for no copy constructor.
	GenotypeNode() 
	{
		phenotypicMultiplicity = 0;
#ifdef GUIDED_MUTATIONS
		nAccumulations = 0;
#endif
	};

	~GenotypeNode() {};

	// WARNING ! Creates a deep copy of this node and its connexions. However, node pointers 
	// must be updated manually ! See Network(Network * n) 's implementation for an example.
	void copyParameters(GenotypeNode* n);

	// Populates the concatenatedChildrenInputBeacons vector and sets sumChildrenInputSizes
	void computeBeacons();

	// genomeState is an array of the size of the genome, which has 1s where the node's depth is known and 0s elsewhere
	void updateDepth(std::vector<int>& genomeState);

	// Used to compute the size of the array containing dynamic inputs  of the phenotype.
	void computeInArraySize(std::vector<int>& genomeState);

	// Used to compute the size of the array containing dynamic outputs of the phenotype.
	void computeOutArraySize(std::vector<int>& genomeState);


#ifdef SATURATION_PENALIZING
	// Used to compute the size of the array containing the average saturations of the phenotype.
	void computeSaturationArraySize(std::vector<int>& genomeState);
#endif 

	// Compute how many non linearities (tanh, relu, ..) the phenotype (yes, phenotype) has.
	void getNnonLinearities(std::vector<int>& genomeState);

	bool hasChild(std::vector<int>& checked, GenotypeNode* potentialChild);

	// Mutate real-valued floating point parameters
	void mutateFloats();

	// Try to connect two random children nodes. Is less likely to succed as the connexion density rises.
	void addConnexion();

	// Disconnect two children nodes picked randomely. 
	void removeConnexion();

	// Add the specified child to the node. The child's depth is <= to this node's depth (which will be increased by 1 if it was ==)
	// It is initially connected to 2 random nodes, with a preference for the input and the output.
	void addChild(GenotypeNode* child);

	// Removes the rID child in the children list.
	void removeChild(int rID);

	// Also resizes the matrices of every connexion between the children and the input, adding a column. Returns a 
	// boolean indicating whether the operation was allowed or not. If not, nothing happened. 
	bool incrementInputSize();
	// Resizes the connexion matrices linking the children together.
	void onChildInputSizeIncremented(GenotypeNode* modifiedType);
	// Util for incrementInputSize and onChildInputSizeIncremented
	void incrementDestinationInputSize(int i);

	// Also resizes the matrices of every connexion between the children and the output, adding a line. Returns a 
	// boolean indicating whether the operation was allowed or not. If not, nothing happened. 
	bool incrementOutputSize();
	// Resizes the connexion matrices linking the children together.
	void onChildOutputSizeIncremented(GenotypeNode* modifiedType);
	// Util for incrementOutputSize and onChildOutputSizeIncremented
	void incrementOriginOutputSize(int i);

	// Also resizes the matrices of every connexion between the children and the input, deleting the id-th column. 
	// Returns a boolean indicating whether the operation was allowed or not. If not, nothing happened. 
	bool decrementInputSize(int id);
	// Resizes the connexion matrices linking the children together.
	void onChildInputSizeDecremented(GenotypeNode* modifiedType, int id);
	// Util for decrementInputSize and onChildInputSizeDecremented
	void decrementDestinationInputSize(int i, int id);

	// Also resizes the matrices of every connexion between the children and the output, deleting the id-th line. 
	// Returns a boolean indicating whether the operation was allowed or not. If not, nothing happened. 
	bool decrementOutputSize(int id);
	// Resizes the connexion matrices linking the children together.
	void onChildOutputSizeDecremented(GenotypeNode* modifiedType, int id);
	// Util for decrementOutputSize and onChildOutputSizeDecremented
	void decrementOriginOutputSize(int i, int id);
};

