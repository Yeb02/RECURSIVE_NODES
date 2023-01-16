#pragma once
#include <string>
#include <vector>
#include <boost/archive/text_iarchive.hpp>

struct GenotypeConnexion {   // responsible of its pointers

	int originID, destinationID; // redundant. But still better for efficiency.

	// Corresponds to the dimension of the input of the destination neuron
	int nLines;
	// Corresponds to the dimension of the output of the origin neuron
	int nColumns;

	float** alpha;
	float** eta;
	float** w;
	float** A;
	float** B;
	float** C;

	~GenotypeConnexion() 
	{
		delete[] alpha;
		delete[] eta;
		delete[] w;
		delete[] A;
		delete[] B;
		delete[] C;
	}
};

struct GenotypeNode {
	// If one of these is true, the forward method is not called,
	// the appropriate function pointer is applied instead, on the pre-synaptic activity.
	float (*f)(float); // NULL if Node is a bloc. Else pointer to tanH, cos, ReLU
	int inputSize, outputSize; // outputSize is at least 2, since the last node is dedicated to neuromodulation
	std::vector<float> bias; // length inputSize. 
	// Order matters in these vectors !!!

	// The last element is the output node, which has no children nor activation function
	// Instead of calling its forward method, its parent replaces its own output
	// with the output node's input
	std::vector<GenotypeNode*> children; 

	// Vector of structs containing pointers to the fixed connexion matrices linking children
	std::vector<GenotypeConnexion*> childrenConnexions;


	// Utils for forward. The beacons array has len(children) elements, and contains :
	// 0, len(children[0]->inputsize), len(children[0]->inputsize)+len(children[1]->inputsize), ....
	int concatenatedChildrenInputLength;
	std::vector<int> concatenatedChildrenInputBeacons;

};

struct PhenotypeConnexion {   // responsible of its pointers
public:
	float** H;
	float** E;

	~PhenotypeConnexion()
	{
		delete[] H;
		delete[] E; 
	}
};

struct PhenotypeNode {
	// In case the node is a simple neuron and not a bloc.
	bool isNeuron;
	GenotypeNode* type;
	float neuromodulatorySignal; //initialized at 1 at the beginning of a trial

	// Pointers to its children. Responsible for their lifetime !
	std::vector<PhenotypeNode*> children;

	// Vector of structs containing pointers to the dynamic connexion matrices linking children
	std::vector<PhenotypeConnexion> childrenConnexions;

	// For plasticity based updates, currentOutput must be reset to all 0s at the start of each trial
	std::vector<float> previousOutput, currentOutput; 


	void forward(float* input);
};


class Network {

public:
	Network();
	std::vector<float> step(std::vector<float> obs);
	void save(std::string path);

private:
	std::vector<GenotypeNode> genome;
	PhenotypeNode network;

};