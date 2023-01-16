#pragma once
#include <string>
#include <vector>
//#include <boost/archive/text_iarchive.hpp>

// responsible of its pointers lifetime
struct GenotypeConnexion {  

	int originID, destinationID; // redundant. But still better for efficiency.

	// Corresponds to the dimension of the input of the destination neuron
	int nLines;
	// Corresponds to the dimension of the output of the origin neuron
	int nColumns;

	float* alpha;
	float* eta;
	float* w;
	float* A;
	float* B;
	float* C;

	GenotypeConnexion(int oID, int dID, int nLines, int nColumns) :
		originID(oID), destinationID(dID), nLines(nLines), nColumns(nColumns)
	{
		alpha = new float[nLines*nColumns];
		eta = new float[nLines*nColumns];
		w = new float[nLines*nColumns];
		A = new float[nLines*nColumns];
		B = new float[nLines*nColumns];
		C = new float[nLines*nColumns];

		for (int i = 0; i < nLines * nColumns; i++) {
			alpha[i] = 1;
			eta[i] = .8;
			w[i] = 1;
			A[i] = 1;
			B[i] = 1;
			C[i] = 1;
		}
		
	}

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
	bool isSimpleNeuron;
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


	// Utils for forward. 
	// = sum(child->inputSize for child in children)
	int concatenatedChildrenInputLength;
	// Has len(children) elements, and contains :
	// 0, children[0]->inputsize, children[0]->inputsize+children[1]->inputsize, ....
	std::vector<int> concatenatedChildrenInputBeacons;

	GenotypeNode() {};
};

struct PhenotypeConnexion {   // responsible of its pointers
public:
	float* H;
	float* E;

	~PhenotypeConnexion()
	{
		delete[] H;
		delete[] E; 
	}
};

struct PhenotypeNode {
	GenotypeNode* type;
	float neuromodulatorySignal; //initialized at 1 at the beginning of a trial

	// Pointers to its children. Responsible for their lifetime !
	std::vector<PhenotypeNode*> children;

	// Vector of structs containing pointers to the dynamic connexion matrices linking children
	std::vector<PhenotypeConnexion> childrenConnexions;

	// For plasticity based updates, currentOutput must be reset to all 0s at the start of each trial
	std::vector<float> previousOutput, currentOutput; 

	PhenotypeNode() {};
	void forward(float* input);
};


class Network {

public:
	Network(int inputSize, int outputSize);
	std::vector<float> step(std::vector<float> obs);
	void save(std::string path);

private:
	int inputSize, outputSize;
	std::vector<GenotypeNode> genome;
	PhenotypeNode network;

};