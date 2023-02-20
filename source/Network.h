#pragma once

#define RISI_NAJARRO_2020

#include <string>
#include <vector>
#include <iostream>
#include <memory>

#include "Random.h"

//#include <boost/archive/text_iarchive.hpp>

// responsible of its pointers lifetime
struct GenotypeConnexion {  
	int originID, destinationID;

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
	
	

	GenotypeConnexion() {
		return;
	}

	GenotypeConnexion(int oID, int dID, int nLines, int nColumns, bool zeroInit = false);

	// Required because a genotype node has a vector of connexions, not of pointers to connexions. 
	// This means that on vector reallocation the move constructor is called. But if it is not specified, it does not 
	// exist because there is a specified destructor. Therefore the constructor and the destructor are called
	// instead, which causes unwanted freeing of memory.
	// Moreover, if it is not marked noexcept std::vector will use copy+destructor instead in some cases, WTF ????
	// https://stackoverflow.com/questions/9249781/are-move-constructors-required-to-be-noexcept
	GenotypeConnexion(GenotypeConnexion&& gc) noexcept {
		originID = gc.originID;
		destinationID = gc.destinationID;
		nLines = gc.nLines;
		nColumns = gc.nColumns;

		// move() removes ownership from the original pointer. Its use here is kind of an hacky workaround 
		// that vector reallocation calls move constructor AND destructor. So the pointee would be destroyed otherwise.
		// https://stackoverflow.com/questions/41864544/stdvector-calls-contained-objects-destructor-when-reallocating-no-matter-what
		A = std::move(gc.A);
		B = std::move(gc.B);
		C = std::move(gc.C);
		eta = std::move(gc.eta);

#if defined RISI_NAJARRO_2020
		D = std::move(gc.D);
#elif defined USING_NEUROMODULATION
		w = std::move(gc.w);
		alpha = std::move(gc.alpha);
#endif 
	}

	GenotypeConnexion(const GenotypeConnexion& gc) {
		destinationID = gc.destinationID;
		originID = gc.originID;
		nLines = gc.nLines;
		nColumns = gc.nColumns;

		int s = nLines * nColumns;
		eta = std::make_unique<float[]>(s);
		A = std::make_unique<float[]>(s);
		B = std::make_unique<float[]>(s);
		C = std::make_unique<float[]>(s);

		memcpy(eta.get(), gc.eta.get(), sizeof(float) * s);
		memcpy(A.get(), gc.A.get(), sizeof(float) * s);
		memcpy(B.get(), gc.B.get(), sizeof(float) * s);
		memcpy(C.get(), gc.C.get(), sizeof(float) * s);

#if defined RISI_NAJARRO_2020
		D = std::make_unique<float[]>(s);
		memcpy(D.get(), gc.D.get(), sizeof(float) * s);

#elif defined USING_NEUROMODULATION
		alpha = std::make_unique<float[]>(s);
		w = std::make_unique<float[]>(s);
		memcpy(alpha.get(), gc.alpha.get(), sizeof(float) * s);
		memcpy(w.get(), gc.w.get(), sizeof(float) * s);
#endif 
	}

	GenotypeConnexion operator=(const GenotypeConnexion& gc) {
		destinationID = gc.destinationID;
		originID = gc.originID;
		nLines = gc.nLines;
		nColumns = gc.nColumns;

		int s = nLines * nColumns;
		eta = std::make_unique<float[]>(s);
		A = std::make_unique<float[]>(s);
		B = std::make_unique<float[]>(s);
		C = std::make_unique<float[]>(s);

		memcpy(eta.get(), gc.eta.get(), sizeof(float) * s);
		memcpy(A.get(), gc.A.get(), sizeof(float) * s);
		memcpy(B.get(), gc.B.get(), sizeof(float) * s);
		memcpy(C.get(), gc.C.get(), sizeof(float) * s);

#if defined RISI_NAJARRO_2020
		D = std::make_unique<float[]>(s);
		memcpy(D.get(), gc.D.get(), sizeof(float) * s);

#elif defined USING_NEUROMODULATION
		alpha = std::make_unique<float[]>(s);
		w = std::make_unique<float[]>(s);
		memcpy(alpha.get(), gc.alpha.get(), sizeof(float) * s);
		memcpy(w.get(), gc.w.get(), sizeof(float) * s);
#endif 
		return *this;
	}

	~GenotypeConnexion() {};
};

struct GenotypeNode {
	bool isSimpleNeuron;
	float (*f)(float); // NULL if Node is a bloc. Else pointer to tanH, cos, ReLU
	int inputSize, outputSize; // >= 1
	std::vector<float> bias;   // length outputSize. Is added the the presynaptic activations of the tanh of the output node

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
	
	// Populates the concatenatedChildrenInputBeacons vector and sets concatenatedChildrenInputLength
	void computeBeacons();

	void updateDepth();

	// Mutate real-valued floating point parameters
	void mutateFloats();

	// Try to connect two random children nodes. Is less likely to succed as the connexion density rises.
	void connect();

	// Disconnect two children nodes picked randomely. 
	void disconnect();

	// Add the specified child to the node. The child's depth is <= to this node's depth (which will be increased by 1 if it was ==)
	// It is initially connected to 2 random nodes, with a preference for the input and the output.
	void addChild(GenotypeNode* child);

	// Removes a child at random.
	void removeChild();

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
	std::unique_ptr<float[]> H;

#ifdef USING_NEUROMODULATION
	std::unique_ptr<float[]> E;
#endif 

	// Should not be called !
	PhenotypeConnexion(const PhenotypeConnexion&) {
		int i = 0;
	}

	PhenotypeConnexion(int s) 
	{
		H = std::make_unique<float[]>(s);

#ifdef USING_NEUROMODULATION
		E = std::make_unique<float[]>(s);
#endif 
		zero(s);
	}

	void zero(int s) {
		for (int i = 0; i < s; i++) {
			H[i] = 0.0f;
#ifdef USING_NEUROMODULATION
			E[i] = 0.0f;
#endif 
		}
	}
	
	void randomH(int s) {
		for (int i = 0; i < s; i++) {
			H[i] = NORMAL_01*.1f;
		}
	}

	~PhenotypeConnexion() {}
};

struct PhenotypeNode {
	GenotypeNode* type;

#ifdef USING_NEUROMODULATION
	float neuromodulatorySignal; //initialized at 1 at the beginning of a trial
#endif 

	// Pointers to its children. Responsible for their lifetime !
	std::vector<std::unique_ptr<PhenotypeNode>> children;

	// Vector of structs containing pointers to the dynamic connexion matrices linking children
	std::vector<PhenotypeConnexion> childrenConnexions;

	// For plasticity based updates, "previous" must be reset to all 0s at the start of each trial
	std::vector<float> previousOutput, currentOutput, previousInput;

	PhenotypeNode(GenotypeNode* type) : type(type)
	{
#ifdef USING_NEUROMODULATION
		neuromodulatorySignal = 1.0f;
#endif 
		previousInput.resize(type->inputSize);
		previousOutput.resize(type->outputSize);
		currentOutput.resize(type->outputSize);

		// create children recursively 
		children.resize(type->children.size());
		for (int i = 0; i < type->children.size(); i++) { 
			children[i] = std::make_unique<PhenotypeNode>(new PhenotypeNode(type->children[i])); // TODO check call numbers to destructor
		}

		// create connexions structs
		childrenConnexions.reserve(type->childrenConnexions.size());
		for (int i = 0; i < type->childrenConnexions.size(); i++) {
			childrenConnexions.emplace_back(
				type->childrenConnexions[i].nLines *
				type->childrenConnexions[i].nColumns
			);
		}
	};

	~PhenotypeNode() {}

	void zero() {
		for (int i = 0; i < children.size(); i++) {
			if (!children[i]->type->isSimpleNeuron) children[i]->zero();
		}
		for (int i = 0; i < childrenConnexions.size(); i++) {
			int s = type->childrenConnexions[i].nLines * type->childrenConnexions[i].nColumns;
			childrenConnexions[i].zero(s);
		}
	}

	void randomH() {
		for (int i = 0; i < children.size(); i++) {
			if (!children[i]->type->isSimpleNeuron) children[i]->randomH();
		}
		for (int i = 0; i < childrenConnexions.size(); i++) {
			int s = type->childrenConnexions[i].nLines * type->childrenConnexions[i].nColumns;
			childrenConnexions[i].randomH(s);
		}
	}

	void reset() {
#ifdef USING_NEUROMODULATION
		neuromodulatorySignal = 1.0f;
#endif 
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
	// Does NOT create the phenotype tree ! No "topNodeP = new PhenotypeNode(&genome[genome.size()-1]);"
	Network(Network* n);
	~Network() {};

	std::vector<float> getOutput();
	void step(std::vector<float> obs);
	void save(std::string path);
	void mutate();

	// sets to 0 the dynamic elements of the phenotype
	void intertrialReset();

	// a positive float, increasing with the networks number of parameters.
	float getSizeRegularizationLoss();
	// a positive float, increasing with the amplitude of the network's parameters. 
	float getAmplitudeRegularizationLoss();

private:
	int nSimpleNeurons;
	int inputSize, outputSize;
	std::vector<std::unique_ptr<GenotypeNode>> genome;
	std::unique_ptr<PhenotypeNode> topNodeP;
};