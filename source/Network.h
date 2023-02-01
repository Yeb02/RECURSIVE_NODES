#pragma once
#include <string>
#include <vector>
#include <iostream>
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

	GenotypeConnexion() {
		return; // TODO erase
	}
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
			eta[i] = .8f;
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
	int inputSize, outputSize; // >= 1
	std::vector<float> bias;   // length outputSize. 

	// Order matters in this vector.
	// Contains pointers to the genotypes of the children
	std::vector<GenotypeNode*> children; 

	// Vector of structs containing pointers to the fixed connexion matrices linking children
	std::vector<GenotypeConnexion> childrenConnexions;

	// neuromodulatorySignal = tanh(neuromodulationBias + SUM(w*out))
	std::vector<float> wNeuromodulation;
	// neuromodulatorySignal = tanh(neuromodulationBias + SUM(w*out))
	float neuromodulationBias;

	// Utils for forward:
	 
	// = sum(child->inputSize for child in children)
	int concatenatedChildrenInputLength;
	// Has len(children) + 1 elements, and contains :
	// 0, children[0]->inputsize, children[0]->inputsize+children[1]->inputsize, ....
	std::vector<int> concatenatedChildrenInputBeacons;

	GenotypeNode() {};
	~GenotypeNode() {};
	
	// Mutate real-valued floating point parameters
	void mutateFloats();

	// Try to connect two random children nodes. Is less likely to succed as the connexion density rises.
	void connect();

	// Disconnect two children nodes picked randomely. 
	void disconnect();

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

// responsible of its pointers lifetime
struct PhenotypeConnexion {   // responsible of its pointers
public:
	float* H;
	float* E;

	//PhenotypeConnexion() {};

	PhenotypeConnexion(int s) 
	{
		H = new float[s];
		E = new float[s];
		zero(s);
	}

	void zero(int s) {
		for (int i = 0; i < s; i++) {
			H[i] = 0;
			E[i] = 0;
		}
	}

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

	// For plasticity based updates, "previous" must be reset to all 0s at the start of each trial
	std::vector<float> previousOutput, currentOutput, previousInput;

	PhenotypeNode(GenotypeNode* type) : type(type)
	{
		neuromodulatorySignal = 1.0f;
		previousInput.resize(type->inputSize);
		previousOutput.resize(type->outputSize);
		currentOutput.resize(type->outputSize);

		// create children recursively 
		children.resize(type->children.size());
		for (int i = 0; i < type->children.size(); i++) {
			children[i] = new PhenotypeNode(type->children[i]);;
		}

		// create connexions structs
		// childrenConnexions.resize(0);
		childrenConnexions.reserve(type->childrenConnexions.size());
		for (int i = 0; i < type->childrenConnexions.size(); i++) {
			childrenConnexions.emplace_back(
				type->childrenConnexions[i].nLines *
				type->childrenConnexions[i].nColumns
			);
		}
	};

	~PhenotypeNode() {
		for (PhenotypeNode* child : children) {
			delete child;
		}
	}

	void reset() {
		neuromodulatorySignal = 1.0f;
		for (int i = 0; i < previousInput.size(); i++) {
			previousInput[i] = 0;
		}
		for (int i = 0; i < previousOutput.size(); i++) {
			previousInput[i] = 0;
		}

		for (int i = 0; i < type->children.size() - 1; i++) {
			children[i]->reset();
		}
		for (int i = 0; i < type->childrenConnexions.size(); i++) {
			childrenConnexions[i].zero(
				type->childrenConnexions[i].nLines *
				type->childrenConnexions[i].nColumns
			);
		}
	}

	void forward(float* input);
};


class Network {

public:
	Network(int inputSize, int outputSize);
	~Network();
	std::vector<float> step(float* obs);
	void save(std::string path);
	void mutate();

private:
	int nSimpleNeurons;
	int inputSize, outputSize;
	std::vector<GenotypeNode> genome;
	PhenotypeNode* topNodeP; 
};