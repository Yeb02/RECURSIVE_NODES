#pragma once

#include <vector>
#include <memory>
#include <cmath>
#include <iostream>
#include <tuple>

#include "Random.h"
#include "Config.h"

#include "MemoryNode_G.h"
#include "InternalConnexion_G.h"

// Constants:
#define MAX_COMPLEX_CHILDREN_PER_COMPLEX  10
#define MAX_MEMORY_CHILDREN_PER_COMPLEX  5
#define MAX_COMPLEX_INPUT_NODE_SIZE  10          // Does not apply to the top node
#define MAX_COMPLEX_OUTPUT_SIZE  10         // Does not apply to the top node


// TODO : implement DERIVATOR, that outputs the difference between INPUT_NODE at this step and INPUT_NODE at the previous step.
// CENTERED_SINE(x) = tanhf(x) * expf(-x*x) * 1/.375261
// I dont really know what to expect from SINE and CENTERED_SINE when it comes to applying 
// hebbian updates... It does not make much sense. But I plan to add cases where activations
// do not use hebbian rules.

#define N_ACTIVATIONS  2 // only using TANH and GAUSSIAN for now
const enum ACTIVATION { TANH = 0, GAUSSIAN = 1, SINE = 2, CENTERED_SINE = 3 };


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
	
	int inputSize, outputSize; // >= 1

	
	// Contains pointers to the children. A pointer can appear multiple times.
	std::vector<ComplexNode_G*> complexChildren;
	std::vector<MemoryNode_G*> memoryChildren;

	// Struct containing the constant, evolved, matrix of parameters linking internal nodes.
	// The name specifies the type of node that takes the result of the matrix operations as inputs.
	// nLines = sum(node.inputSize) for node of the type corresponding to the name
	// nColumns = this.inputSize + MODULATION_VECTOR_SIZE + sum(complexChild.inputSize) + sum(memoryChild.inputSize)
	InternalConnexion_G toComplex, toMemory, toModulation, toOutput;

	
	float modulationBias[MODULATION_VECTOR_SIZE];

	ACTIVATION modulationActivations[MODULATION_VECTOR_SIZE];

	// Depth of the children tree. =0 for simple neurons, at least 1 otherwise, even if there are no children.
	int depth;

	// The position in the genome vector. Must be genome.size() for the top node.
	int position;

	// Points towards the node this was cloned from or boxed from on the tree of life. 
	ComplexNode_G* closestNode;

	// How many iterations of floating point values mutations it has undergone since it was created.
	int mutationalDistance;

	// How many times this node appears in the phenotype.
	int phenotypicMultiplicity;

	// Updated by a call to computeBiasSizes. 
	int complexBiasSize, memoryBiasSize;

	std::vector<float> complexBias, memoryBias, outputBias;

	// arrays indicating the activation function to use on each presynaptic input.
	std::vector<ACTIVATION> complexActivations, memoryActivations, outputActivations;

	// Precomputed for efficiency, = outputSize + MODULATION_VECTOR_SIZE + sum(complexChildren.inputSize)
	int memoryPreSynOffset;
	void computeMemoryPreSynOffset() {
		int s = outputSize + MODULATION_VECTOR_SIZE;
		for (int i = 0; i < complexChildren.size(); i++) {
			s += complexChildren[i]->inputSize;
		}
		memoryPreSynOffset = s;
	}

	// Does not do much, because most attributes are set by the network owning this.
	ComplexNode_G(int inputSize, int outputSize);

	// WARNING ! "this" node is now a deep copy of n, but the pointers towards the children 
	// must be updated manually if "this" and n do not belong to the same Network !
	// (typically in Network(Network * n))
	ComplexNode_G(ComplexNode_G* n);

	~ComplexNode_G() {};

	void createInternalConnexions();

	// Sets complexBiasSize, memoryBiasSize
	void computeBiasSizes();

	// genomeState is an array of the size of the genome, which has 1s where the node's depth is known and 0s elsewhere
	void updateDepth(std::vector<int>& genomeState);

	// Compute the size of the array containing the pre synaptic activations of the phenotype.
	void computePreSynActArraySize(std::vector<int>& genomeState);

	// Compute the size of the 2 arrays containing the post synaptic activations of the phenotype,
	// previousPostSynAct and currentPostSynAct.
	void computePostSynActArraySize(std::vector<int>& genomeState);



#ifdef SATURATION_PENALIZING
	// Used to compute the size of the array containing the average saturations of the phenotype.
	void computeSaturationArraySize(std::vector<int>& genomeState);
#endif 

	// Mutate real-valued floating point parameters.
	void mutateFloats();

	// Mutate the non linearities entering the modulation, complex and memory children, and output.
	void mutateActivations();

	// Add the specified child to the node. After the call to this function, depths, genome order, and 
	// phenotypic multiplicities must be manually updated.
	void addComplexChild(ComplexNode_G* child);

	// Add the specified child to the node. After the call to this function, 
	// phenotypic multiplicities must be manually updated.
	void addMemoryChild(MemoryNode_G* child);



	// Removes the rID th complex child. Handles connexion matrices resizing.
	void removeComplexChild(int rID);

	// Removes the rID th memory child. Handles connexion matrices resizing.
	void removeMemoryChild(int rID);

	

	// Returns a boolean indicating whether the operation was allowed or not. If false, nothing happened. 
	// If true, everything was handled by this function and there is nothing to do outside.
	bool incrementInputSize();

	// Returns a boolean indicating whether the operation was allowed or not. If false, nothing happened. 
	// If true, everything was handled by this function and there is nothing to do outside.
	bool incrementOutputSize();

	// Returns a boolean indicating whether the operation was allowed or not. If false, nothing happened. 
	// If true, everything was handled by this function and there is nothing to do outside.
	bool decrementInputSize(int id);

	// Returns a boolean indicating whether the operation was allowed or not. If false, nothing happened. 
	// If true, everything was handled by this function and there is nothing to do outside.
	bool decrementOutputSize(int id);



	// Resizes the connexion matrices when a potential child has had its input size incremented.
	// bool complexNode is true if the potential child is a complex node, false if it is a memory node.
	void onChildInputSizeIncremented(void* potentialChild, bool complexNode);


	// Resizes the connexion matrices when a potential child has had its output size incremented.
	// bool complexNode is true if the potential child is a complex node, false if it is a memory node.
	void onChildOutputSizeIncremented(void* potentialChild, bool complexNode);


	// Resizes the connexion matrices when a potential child has had its input size decremented.
	// bool complexNode is true if the potential child is a complex node, false if it is a memory node.
	// id is the index of the deleted input in the potential child's interface.
	void onChildInputSizeDecremented(void* potentialChild, bool complexNode, int id);


	// Resizes the connexion matrices when a potential child has had its output size decremented.
	// bool complexNode is true if the potential child is a complex node, false if it is a memory node.
	// id is the index of the deleted output in the potential child's interface.
	void onChildOutputSizeDecremented(void* potentialChild, bool complexNode, int id);
};

