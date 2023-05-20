#include "ComplexNode_G.h"

ComplexNode_G::ComplexNode_G(int inputSize, int outputSize) :
	inputSize(inputSize), outputSize(outputSize),
	toComplex(0, 0, InternalConnexion_G::ZERO),
	toMemory(0, 0, InternalConnexion_G::ZERO),
	toModulation(0, 0, InternalConnexion_G::ZERO),
	toOutput(0, 0, InternalConnexion_G::ZERO)
{
	mutationalDistance = 0;
	closestNode = NULL;

	outputBias.resize(outputSize);
	outputActivations.resize(outputSize);
	for (int i = 0; i < outputSize; i++) {
		outputBias[i] = NORMAL_01 * .2f;
		outputActivations[i] = static_cast<ACTIVATION>(INT_0X(N_ACTIVATIONS));
	}

	for (int i = 0; i < MODULATION_VECTOR_SIZE; i++) {
		modulationBias[i] = NORMAL_01 * .2f;
		modulationActivations[i] = static_cast<ACTIVATION>(INT_0X(N_ACTIVATIONS));
	}

	complexBiasSize = 0;
	memoryBiasSize = 0;

	// The following initializations MUST be done outside.
	{
		depth = -1;
		position = -1;
		phenotypicMultiplicity = -1;
		memoryPreSynOffset = -1;
		complexNodeID = -1;
	}
};

ComplexNode_G::ComplexNode_G(ComplexNode_G* n) {

	inputSize = n->inputSize;
	outputSize = n->outputSize;
	complexNodeID = n->complexNodeID;

	toComplex = n->toComplex;
	toMemory = n->toMemory;
	toModulation = n->toModulation;
	toOutput = n->toOutput;


	outputBias.assign(n->outputBias.begin(), n->outputBias.end());
	complexBias.assign(n->complexBias.begin(), n->complexBias.end());
	memoryBias.assign(n->memoryBias.begin(), n->memoryBias.end());

	outputActivations.assign(n->outputActivations.begin(), n->outputActivations.end());
	complexActivations.assign(n->complexActivations.begin(), n->complexActivations.end());
	memoryActivations.assign(n->memoryActivations.begin(), n->memoryActivations.end());

	for (int i = 0; i < MODULATION_VECTOR_SIZE; i++) {
		modulationBias[i] = n->modulationBias[i];
	}

	complexBiasSize = n->complexBiasSize;
	memoryBiasSize = n->memoryBiasSize;
	depth = n->depth;
	position = n->position;
	mutationalDistance = n->mutationalDistance;
	phenotypicMultiplicity = n->phenotypicMultiplicity;
	memoryPreSynOffset = n->memoryPreSynOffset;

	// The following enclosed section is useless if n is not part of the same network as "this", 
	// and it must be repeated where this function was called.
	{

		complexChildren.reserve((int)((float)n->complexChildren.size() * 1.5f));
		for (int j = 0; j < n->complexChildren.size(); j++) {
			complexChildren.emplace_back(n->complexChildren[j]);
		}
		memoryChildren.reserve((int)((float)n->memoryChildren.size() * 1.5f));
		for (int j = 0; j < n->memoryChildren.size(); j++) {
			memoryChildren.emplace_back(n->memoryChildren[j]);
		}
		closestNode = n->closestNode;
	}
}


void ComplexNode_G::createInternalConnexions() {

	int nColumns = inputSize + MODULATION_VECTOR_SIZE;
	for (int i = 0; i < complexChildren.size(); i++) {
		nColumns += complexChildren[i]->outputSize;
	}
	for (int i = 0; i < memoryChildren.size(); i++) {
		nColumns += memoryChildren[i]->outputSize;
	}

	int nLines;

	nLines = 0;
	for (int i = 0; i < complexChildren.size(); i++) {
		nLines += complexChildren[i]->inputSize;
	}
	toComplex = InternalConnexion_G(nLines, nColumns, InternalConnexion_G::RANDOM);

	nLines = 0;
	for (int i = 0; i < memoryChildren.size(); i++) {
		nLines += memoryChildren[i]->inputSize;
	}
	toMemory = InternalConnexion_G(nLines, nColumns, InternalConnexion_G::RANDOM);

	nLines = outputSize;
	toOutput = InternalConnexion_G(nLines, nColumns, InternalConnexion_G::RANDOM);

	nLines = MODULATION_VECTOR_SIZE;
	toModulation = InternalConnexion_G(nLines, nColumns, InternalConnexion_G::RANDOM);
}

void ComplexNode_G::computeBiasSizes() {
	int _s = 0;
	for (int i = 0; i < complexChildren.size(); i++) {
		_s += complexChildren[i]->inputSize;
	}
	complexBiasSize = _s;

	_s = 0;
	for (int i = 0; i < memoryChildren.size(); i++) {
		_s += memoryChildren[i]->inputSize;
	}
	memoryBiasSize = _s;
}

void ComplexNode_G::mutateFloats() {
	constexpr float p = .4f;

	toComplex.mutateFloats(p);
	toMemory.mutateFloats(p);
	toModulation.mutateFloats(p);
	toOutput.mutateFloats(p);

	
	auto mutateBiasArray =  [] (float* v, int size)
	{
		SET_BINOMIAL(size, p);
		int _nMutations = BINOMIAL;
		for (int i = 0; i < _nMutations; i++) {
			int id = INT_0X(size);
			v[id] *= .8f + NORMAL_01 * .2f; // .8 < 1 to drive the weight towards 0.
			v[id] += NORMAL_01 * .2f;
		}
	};

	mutateBiasArray(modulationBias, MODULATION_VECTOR_SIZE);
	mutateBiasArray(outputBias.data(), outputSize);
	mutateBiasArray(complexBias.data(), complexBiasSize);
	mutateBiasArray(memoryBias.data(), memoryBiasSize);
		
}

void ComplexNode_G::mutateActivations() {
	constexpr float p = .05f;

	auto mutateActivationsArray = [](ACTIVATION* v, int size)
	{
		SET_BINOMIAL(size, p);
		int _nMutations = BINOMIAL;
		for (int i = 0; i < _nMutations; i++) {
			int id = INT_0X(size);
			v[id] = static_cast<ACTIVATION>(INT_0X(N_ACTIVATIONS));
		}
	};

	mutateActivationsArray(modulationActivations, MODULATION_VECTOR_SIZE);
	mutateActivationsArray(outputActivations.data(), outputSize);
	mutateActivationsArray(complexActivations.data(), complexBiasSize);
	mutateActivationsArray(memoryActivations.data(), memoryBiasSize);

}


void ComplexNode_G::addComplexChild(ComplexNode_G* child) {

	// insert random bias and activations range
	{
		int id = 0;
		for (int i = 0; i < complexChildren.size(); i++) {
			id += complexChildren[i]->inputSize;
		}

		complexBias.insert(complexBias.begin() + id, child->inputSize, 0.0f);
		complexActivations.insert(complexActivations.begin() + id, child->inputSize, TANH);

		for (int i = id; i < id + child->inputSize; i++) {
			complexBias[i] = NORMAL_01 * .2f;
			complexActivations[i] = static_cast<ACTIVATION>(INT_0X(N_ACTIVATIONS));
		}
	}
	
	// insert random weight range
	{
		toComplex.insertLineRange(toComplex.nLines, child->inputSize);

		int pos = inputSize + MODULATION_VECTOR_SIZE;
		for (int i = 0; i < complexChildren.size(); i++) {
			pos += complexChildren[i]->outputSize;
		}
		toComplex.insertColumnRange(pos, child->outputSize);
		toMemory.insertColumnRange(pos, child->outputSize);
		toModulation.insertColumnRange(pos, child->outputSize);
		toOutput.insertColumnRange(pos, child->outputSize);
	}

	complexChildren.emplace_back(child);
}
void ComplexNode_G::addMemoryChild(MemoryNode_G* child) {

	// insert random bias and activation range
	{
		int id = 0;
		for (int i = 0; i < memoryChildren.size(); i++) {
			id += memoryChildren[i]->inputSize;
		}
		memoryBias.insert(memoryBias.begin() + id, child->inputSize, 0.0f);
		memoryActivations.insert(memoryActivations.begin() + id, child->inputSize, TANH);
		for (int i = id; i < id + child->inputSize; i++) {
			memoryBias[i] = NORMAL_01 * .2f;
			memoryActivations[i] = static_cast<ACTIVATION>(INT_0X(N_ACTIVATIONS));
		}
	}

	// insert random weight range
	{
		toMemory.insertLineRange(toMemory.nLines, child->inputSize);

		int pos = inputSize + MODULATION_VECTOR_SIZE;
		for (int i = 0; i < complexChildren.size(); i++) {
			pos += complexChildren[i]->outputSize;
		}
		for (int i = 0; i < memoryChildren.size(); i++) {
			pos += memoryChildren[i]->outputSize;
		}
		toComplex.insertColumnRange(pos, child->outputSize);
		toMemory.insertColumnRange(pos, child->outputSize);
		toModulation.insertColumnRange(pos, child->outputSize);
		toOutput.insertColumnRange(pos, child->outputSize);
	}

	memoryChildren.emplace_back(child);
}


void ComplexNode_G::removeComplexChild(int rID) {

	// erase the corresponding bias and activations range
	{
		int id = 0;
		for (int i = 0; i < rID; i++) {
			id += complexChildren[i]->inputSize;
		}
		complexBias.erase(complexBias.begin() + id, complexBias.begin() + id + complexChildren[rID]->inputSize);
		complexActivations.erase(complexActivations.begin() + id, complexActivations.begin() + id + complexChildren[rID]->inputSize);
	}

	// Erase the corresponding weight ranges
	{
		int pos = 0;
		for (int i = 0; i < rID; i++) {
			pos += complexChildren[i]->inputSize;
		}
		toComplex.removeLineRange(pos, complexChildren[rID]->inputSize);

		pos = inputSize + MODULATION_VECTOR_SIZE;
		for (int i = 0; i < rID; i++) {
			pos += complexChildren[i]->outputSize;
		}
		toComplex.removeColumnRange(pos, complexChildren[rID]->outputSize);
		toMemory.removeColumnRange(pos, complexChildren[rID]->outputSize);
		toModulation.removeColumnRange(pos, complexChildren[rID]->outputSize);
		toOutput.removeColumnRange(pos, complexChildren[rID]->outputSize);
	}


	complexChildren.erase(complexChildren.begin() + rID);
}
void ComplexNode_G::removeMemoryChild(int rID) {

	// erase the corresponding bias range
	{
		int id = 0;
		for (int i = 0; i < rID; i++) {
			id += memoryChildren[i]->inputSize;
		}
		memoryBias.erase(memoryBias.begin() + id, memoryBias.begin() + id + memoryChildren[rID]->inputSize);
		memoryActivations.erase(memoryActivations.begin() + id, memoryActivations.begin() + id + memoryChildren[rID]->inputSize);
	}

	// Erase the corresponding weight ranges
	{
		int pos = 0;
		for (int i = 0; i < rID; i++) {
			pos += memoryChildren[i]->inputSize;
		}
		toMemory.removeLineRange(pos, memoryChildren[rID]->inputSize);

		pos = inputSize + MODULATION_VECTOR_SIZE;
		for (int i = 0; i < complexChildren.size(); i++) {
			pos += complexChildren[i]->outputSize;
		}
		for (int i = 0; i < rID; i++) {
			pos += memoryChildren[i]->outputSize;
		}
		toComplex.removeColumnRange(pos, memoryChildren[rID]->outputSize);
		toMemory.removeColumnRange(pos, memoryChildren[rID]->outputSize);
		toModulation.removeColumnRange(pos, memoryChildren[rID]->outputSize);
		toOutput.removeColumnRange(pos, memoryChildren[rID]->outputSize);
	}

	memoryChildren.erase(memoryChildren.begin() + rID);
}


bool ComplexNode_G::incrementInputSize() {
	if (inputSize >= MAX_COMPLEX_INPUT_NODE_SIZE) return false;

	toComplex.insertColumnRange(inputSize, 1);
	toMemory.insertColumnRange(inputSize, 1);
	toModulation.insertColumnRange(inputSize, 1);
	toOutput.insertColumnRange(inputSize, 1);

	inputSize++;
	return true;
}
void ComplexNode_G::onChildInputSizeIncremented(void* potentialChild, bool complexNode) {
	
	int id = 0;

	if (complexNode) {

		ComplexNode_G* potentialChildC = static_cast<ComplexNode_G*>(potentialChild);

		for (int i = 0; i < complexChildren.size(); i++) {
			id += complexChildren[i]->inputSize;
			if (complexChildren[i] == potentialChildC) {
				toComplex.insertLineRange(id - 1,1); // id-1 because the child's inputSize has already been updated.
				complexBias.insert(complexBias.begin() + (id - 1), NORMAL_01 * .2f);
				complexActivations.insert(complexActivations.begin() + (id - 1), static_cast<ACTIVATION>(INT_0X(N_ACTIVATIONS)));
			}
		}

	}
	else {

		MemoryNode_G* potentialChildM = static_cast<MemoryNode_G*>(potentialChild);

		for (int i = 0; i < memoryChildren.size(); i++) {
			id += memoryChildren[i]->inputSize;
			if (memoryChildren[i] == potentialChildM) {
				toMemory.insertLineRange(id - 1,1);// id-1 because the child's inputSize has already been updated.
				memoryBias.insert(memoryBias.begin() + (id - 1), NORMAL_01 * .2f);
				memoryActivations.insert(memoryActivations.begin() + (id - 1), static_cast<ACTIVATION>(INT_0X(N_ACTIVATIONS)));
			}
		}

	}
}

bool ComplexNode_G::incrementOutputSize() {
	
	if (outputSize >= MAX_COMPLEX_OUTPUT_SIZE)
	{ 
		return false;
	}
	
	toOutput.insertLineRange(outputSize,1);

	outputBias.insert(outputBias.begin() + outputSize, NORMAL_01*.2f);
	outputActivations.insert(outputActivations.begin() + outputSize, static_cast<ACTIVATION>(INT_0X(N_ACTIVATIONS)));
	outputSize++;
	return true;
}
void ComplexNode_G::onChildOutputSizeIncremented(void* potentialChild, bool complexNode) {

	int column = inputSize + MODULATION_VECTOR_SIZE;

	if (complexNode) {

		ComplexNode_G* potentialChildC = static_cast<ComplexNode_G*>(potentialChild);

		for (int i = 0; i < complexChildren.size(); i++) {
			column += complexChildren[i]->outputSize;
			if (complexChildren[i] == potentialChildC) {
				toComplex.insertColumnRange(column-1, 1); // column-1 because the child's outputSize has already been updated.
				toMemory.insertColumnRange(column-1, 1);
				toModulation.insertColumnRange(column-1, 1);
				toOutput.insertColumnRange(column-1, 1);
			}
		}

	}
	else {

		MemoryNode_G* potentialChildM = static_cast<MemoryNode_G*>(potentialChild);

		for (int i = 0; i < complexChildren.size(); i++) {
			column += complexChildren[i]->outputSize;
		}

		for (int i = 0; i < memoryChildren.size(); i++) {
			column += memoryChildren[i]->outputSize;
			if (memoryChildren[i] == potentialChildM) {
				toComplex.insertColumnRange(column-1, 1); // column-1 because the child's outputSize has already been updated.
				toMemory.insertColumnRange(column-1, 1);
				toModulation.insertColumnRange(column-1, 1);
				toOutput.insertColumnRange(column-1, 1);
			}
		}

	}
}

bool ComplexNode_G::decrementInputSize(int id) {
	if (inputSize <= 1) {
		return false;
	}

	toComplex.removeColumnRange(id, 1);
	toMemory.removeColumnRange(id, 1);
	toModulation.removeColumnRange(id, 1);
	toOutput.removeColumnRange(id, 1);

	inputSize--;
	return true;
}
void ComplexNode_G::onChildInputSizeDecremented(void* potentialChild, bool complexNode, int id) {
	int aID = 0;

	if (complexNode) {

		ComplexNode_G* potentialChildC = static_cast<ComplexNode_G*>(potentialChild);

		for (int i = 0; i < complexChildren.size(); i++) {
			
			
			if (complexChildren[i] == potentialChildC) {
				toComplex.removeLineRange(aID +id, 1);
				complexBias.erase(complexBias.begin() + aID + id);
				complexActivations.erase(complexActivations.begin() + aID + id);
			}
			aID += complexChildren[i]->inputSize;
		}

	}
	else {

		MemoryNode_G* potentialChildM = static_cast<MemoryNode_G*>(potentialChild);

		for (int i = 0; i < memoryChildren.size(); i++) {
			
			
			if (memoryChildren[i] == potentialChildM) {
				toMemory.removeLineRange(aID + id, 1);
				memoryBias.erase(memoryBias.begin() + aID + id);
				memoryActivations.erase(memoryActivations.begin() + aID + id);
			}
			aID += memoryChildren[i]->inputSize;
		}

	}
	
}

bool ComplexNode_G::decrementOutputSize(int id) {
	if (outputSize <= 1) return false;

	toOutput.removeLineRange(id, 1);

	outputBias.erase(outputBias.begin() + id);
	outputActivations.erase(outputActivations.begin() + id);
	outputSize--;
	return true;
}
void ComplexNode_G::onChildOutputSizeDecremented(void* potentialChild, bool complexNode, int id) {

	int column = inputSize + MODULATION_VECTOR_SIZE;

	if (complexNode) {

		ComplexNode_G* potentialChildC = static_cast<ComplexNode_G*>(potentialChild);

		for (int i = 0; i < complexChildren.size(); i++) {
			
			if (complexChildren[i] == potentialChildC) {
				toComplex.removeColumnRange(column+id, 1);
				toMemory.removeColumnRange(column+id, 1);
				toModulation.removeColumnRange(column+id, 1);
				toOutput.removeColumnRange(column+id, 1);
			}
			column += complexChildren[i]->outputSize;
		}

	}
	else {

		MemoryNode_G* potentialChildM = static_cast<MemoryNode_G*>(potentialChild);

		for (int i = 0; i < complexChildren.size(); i++) {
			column += complexChildren[i]->outputSize;
		}

		for (int i = 0; i < memoryChildren.size(); i++) {
			
			if (memoryChildren[i] == potentialChildM) {
				toComplex.removeColumnRange(column+id, 1);
				toMemory.removeColumnRange(column+id, 1);
				toModulation.removeColumnRange(column+id, 1);
				toOutput.removeColumnRange(column+id, 1);
			}
			column += memoryChildren[i]->outputSize;
		}

	}
}


void ComplexNode_G::updateDepth(std::vector<int>& genomeState) {
	int dmax = 0;
	for (int i = 0; i < complexChildren.size(); i++) {
		if (genomeState[complexChildren[i]->position] == 0) complexChildren[i]->updateDepth(genomeState);
		if (complexChildren[i]->depth > dmax) dmax = complexChildren[i]->depth;
	}
	depth = dmax + 1;
	genomeState[position] = 1;
}

void ComplexNode_G::computePreSynActArraySize(std::vector<int>& genomeState) {
	int s = outputSize + MODULATION_VECTOR_SIZE;
	for (int i = 0; i < complexChildren.size(); i++) {
		s += complexChildren[i]->inputSize;
		if (genomeState[complexChildren[i]->position] == 0) {
			complexChildren[i]->computePreSynActArraySize(genomeState);
		}
		s += genomeState[complexChildren[i]->position];
	}
	for (int i = 0; i < memoryChildren.size(); i++) {
		s += memoryChildren[i]->inputSize; // this node's phenotype only uses the memory child's preSyn input
		s += memoryChildren[i]->outputSize; // while the child itself only uses its preSyn output.
	}
	genomeState[position] = s;
}

void ComplexNode_G::computePostSynActArraySize(std::vector<int>& genomeState) {
	int s = inputSize + MODULATION_VECTOR_SIZE;
	for (int i = 0; i < complexChildren.size(); i++) {
		s += complexChildren[i]->outputSize;
		if (genomeState[complexChildren[i]->position] == 0) {
			complexChildren[i]->computePostSynActArraySize(genomeState);
		}
		s += genomeState[complexChildren[i]->position];
	}
	for (int i = 0; i < memoryChildren.size(); i++) {
		s += memoryChildren[i]->outputSize; // this node's phenotype only uses the memory child's postSyn output
		s += memoryChildren[i]->outputSize + memoryChildren[i]->outputSize; // while the child itself uses its postSyn input and output.
	}
	genomeState[position] = s;
}

#ifdef SATURATION_PENALIZING
// Used to compute the size of the array containing the average saturations of the phenotype.
void ComplexNode_G::computeSaturationArraySize(std::vector<int>& genomeState) {
	int s = inputSize + outputSize + MODULATION_VECTOR_SIZE;
	for (int i = 0; i < complexChildren.size(); i++) {
		if (genomeState[complexChildren[i]->position] == 0) {
			complexChildren[i]->computeSaturationArraySize(genomeState);
		}
		s += genomeState[complexChildren[i]->position];
	}
	for (int i = 0; i < memoryChildren.size(); i++) {
		s += memoryChildren[i]->inputSize;
	}
	genomeState[position] = s;
}
#endif 


