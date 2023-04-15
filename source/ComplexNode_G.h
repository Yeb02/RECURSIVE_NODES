#pragma once

#include <vector>
#include <memory>
#include <cmath>
#include <iostream>
#include <tuple>

#include "Random.h"
#include "Config.h"

#include "SimpleNode_G.h"
#include "MemoryNode_G.h"
#include "GenotypeConnexion.h"

// Constants:
#define MAX_SIMPLE_CHILDREN_PER_COMPLEX  20
#define MAX_COMPLEX_CHILDREN_PER_COMPLEX  10
#define MAX_MEMORY_CHILDREN_PER_COMPLEX  5
#define MAX_COMPLEX_INPUT_NODE_SIZE  10          // Does not apply to the top node
#define MAX_COMPLEX_OUTPUT_SIZE  10         // Does not apply to the top node

// Util:
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


struct ComplexNode_G {
	
	static const NODE_TYPE nodeType = NODE_TYPE::COMPLEX;    

	int inputSize, outputSize; // >= 1

	
	// Contains pointers to the children. A pointer can appear multiple times.
	std::vector<ComplexNode_G*> complexChildren;
	std::vector<SimpleNode_G*> simpleChildren;
	std::vector<MemoryNode_G*> memoryChildren;

	// Vector of structs holding pointers to the fixed connexion matrices linking children
	std::vector<GenotypeConnexion> internalConnexions;

	// neuromodulation bias.
	float modulationBias[MODULATION_VECTOR_SIZE];

	// Depth of the children tree. =0 for simple neurons, at least 1 otherwise, even if there are no children.
	int depth;

	// The position in the genome vector. Must be genome.size() for the top node.
	int position;

	// Point towards the node it was cloned from or boxed from on the tree of life. 
	ComplexNode_G* closestNode;

	// How many iterations of floating point values mutations it has undergone since it was created.
	int mutationalDistance;

	// How many times this node appears in the phenotype.
	int phenotypicMultiplicity;

	// Updated by a call to computeInternalBiasSize. Happens only at the end of structural mutations and a node creation,
	// since this value is used only during forward and floating point mutations. 
	// Equals : outputSize + nSimpleNeurons + sum(complexChild.inputSize) + sum(memoryChild.inputSize) 
	int internalBiasSize;

	// size internalBiasSize. In the following order: output -> simple -> complex -> memory
	std::vector<float> internalBias;

#ifdef GUIDED_MUTATIONS
	//  = [How many instances of this node the network has] * [how many times wLifetime was accumulated].
	int nAccumulations;
#endif

	// Does not do much, because most attributes are set by the network owning this.
	ComplexNode_G() 
	{
		phenotypicMultiplicity = 0;
#ifdef GUIDED_MUTATIONS
		nAccumulations = 0;
#endif
		mutationalDistance = 0;
		closestNode = NULL;

		for (int i = 0; i < MODULATION_VECTOR_SIZE; i++) {
			modulationBias[i] = NORMAL_01 * .2f;
		}

		// The following initializations must be done outside.
		{
			depth = -1;
			inputSize = -1;
			outputSize = -1;
			position = -1;
			internalBiasSize = -1;
		}
	};

	// WARNING ! "this" node is now a deep copy of n, but the pointers towards the children 
	// must be updated manually if "this" and n do not belong to the same Network !
	// (typically in Network(Network * n))
	ComplexNode_G(ComplexNode_G* n);

	~ComplexNode_G() {};

	// Sets internalBiasSize
	void computeInternalBiasSize();

	// genomeState is an array of the size of the genome, which has 1s where the node's depth is known and 0s elsewhere
	void updateDepth(std::vector<int>& genomeState);

	// Used to compute the size of the 4 arrays containing the activations of the phenotype.
	void computeActivationArraySize(std::vector<int>& genomeState);



#ifdef SATURATION_PENALIZING
	// Used to compute the size of the array containing the average saturations of the phenotype.
	void computeSaturationArraySize(std::vector<int>& genomeState);
#endif 

	// Compute how many non linearities (tanh, relu, ..) the phenotype (yes, phenotype) has.
	void getNnonLinearities(std::vector<int>& genomeState);

	bool hasChild(std::vector<int>& checked, ComplexNode_G* potentialChild);

	// must be called when a child is deleted. Handles re-indicing. 
	void updateConnexionsOnChildDeleted(NODE_TYPE childType, int childID);

	// Mutate real-valued floating point parameters
	void mutateFloats();

	// Util that returns a randomly picked node that can serve as a connexion's origin. There is 
	// a configurable bias towards linking to input, and modulation. More generally, the probability of
	// linking to a node is proportional to its output's size.
	std::tuple<NODE_TYPE, int, int> pickRandomOriginNode();

	// Util that returns a randomly picked node that can serve as a connexion's destination. There is 
	// a configurable bias towards linking to output, and modulation. More generally, the probability of
	// linking to a node is proportional to its input's size.
	std::tuple<NODE_TYPE, int, int> pickRandomDestinationNode();

	// Try to connect two random children nodes. Is less likely to succed as the connexion density rises.
	void addConnexion();

	// Disconnect two children nodes picked randomely. 
	void removeConnexion();

	// Add the specified child to the node. Must be followed by a call to Network::updateDepth() and Network::sortGenome().
	// The child's depth is <= to this node's depth.
	// The child is initially connected to 2 random nodes, with a preference for the INPUT_NODE and the output.
	void addComplexChild(ComplexNode_G* child);

	void addSimpleChild(SimpleNode_G* child);

	void addMemoryChild(MemoryNode_G* child);


	// Removes the rID th simple child in the children list.
	void removeSimpleChild(int rID);
	// Removes the rID th complex child in the children list.
	void removeComplexChild(int rID);
	// Removes the rID th memory child in the children list.
	void removeMemoryChild(int rID);

	// Also resizes the matrices of every connexion between the children and the INPUT_NODE, adding a column. Returns a 
	// boolean indicating whether the operation was allowed or not. If not, nothing happened. 
	bool incrementInputSize();
	// Resizes the connexion matrices linking the children together.
	void onChildInputSizeIncremented(int modifiedPosition, NODE_TYPE modifiedType);

	// Also resizes the matrices of every connexion between the children and the output, adding a line. Returns a 
	// boolean indicating whether the operation was allowed or not. If not, nothing happened. 
	bool incrementOutputSize();
	// Resizes the connexion matrices linking the children together.
	void onChildOutputSizeIncremented(int modifiedPosition, NODE_TYPE modifiedType);


	// Also resizes the matrices of every connexion between the children and the INPUT_NODE, deleting the id-th column. 
	// Returns a boolean indicating whether the operation was allowed or not. If not, nothing happened. 
	bool decrementInputSize(int id);
	// Resizes the connexion matrices linking the children together.
	void onChildInputSizeDecremented(int modifiedPosition, NODE_TYPE modifiedType, int id);


	// Also resizes the matrices of every connexion between the children and the output, deleting the id-th line. 
	// Returns a boolean indicating whether the operation was allowed or not. If not, nothing happened. 
	bool decrementOutputSize(int id);
	// Resizes the connexion matrices linking the children together.
	void onChildOutputSizeDecremented(int modifiedPosition, NODE_TYPE modifiedType, int id);
};

