#include "ComplexNode_G.h"

ComplexNode_G::ComplexNode_G(int inputSize, int outputSize) :
	inputSize(inputSize), outputSize(outputSize),
	toComplex(0, 0, InternalConnexion_G::ZERO),
	toMemory(0, 0, InternalConnexion_G::ZERO),
	toModulation(0, 0, InternalConnexion_G::ZERO),
	toOutput(0, 0, InternalConnexion_G::ZERO)
{
	mutationalDistance = 0;
	timeSinceLastUse = 0;
	closestNode = NULL;

	

	// The following initializations MUST be done outside.
	{
		depth = -1;
		position = -1;
		phenotypicMultiplicity = -1;
		complexNodeID = -1;
	}

	// The following initializations are useless, as the parameters are set somewhere else.
	// Here for completeness, warning suppression and ensuring (sould be the case already)
	// that the behaviour is the same in debug and release mode.
	{

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


	depth = n->depth;
	position = n->position;
	mutationalDistance = n->mutationalDistance;
	phenotypicMultiplicity = n->phenotypicMultiplicity;
	timeSinceLastUse = n->timeSinceLastUse;

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


ComplexNode_G::ComplexNode_G(std::ifstream& is) {
	// These must be set by the network in its Network(std::ifstream& is) constructor.
	{
		depth = -1;
		position = -1;
		phenotypicMultiplicity = -1;
	}

	READ_4B(inputSize, is);
	READ_4B(outputSize, is);

	int _s;
	READ_4B(_s, is);
	complexChildren.resize(_s);
	READ_4B(_s, is);
	memoryChildren.resize(_s);

	READ_4B(complexNodeID, is);
	READ_4B(mutationalDistance, is);
	READ_4B(timeSinceLastUse, is);


	toComplex = InternalConnexion_G(is);
	toMemory = InternalConnexion_G(is);
	toModulation = InternalConnexion_G(is);
	toOutput = InternalConnexion_G(is);

}

void ComplexNode_G::save(std::ofstream& os) {
	WRITE_4B(inputSize, os);
	WRITE_4B(outputSize, os);

	int _s;
	_s = (int)complexChildren.size();
	WRITE_4B(_s, os);
	_s = (int)memoryChildren.size();
	WRITE_4B(_s, os);


	WRITE_4B(complexNodeID, os);
	WRITE_4B(mutationalDistance, os);
	WRITE_4B(timeSinceLastUse, os);


	toComplex.save(os);
	toMemory.save(os);
	toModulation.save(os);
	toOutput.save(os);
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


void ComplexNode_G::mutateFloats(float adjustedFMutationP) {
	float p = adjustedFMutationP * log2f((float)phenotypicMultiplicity + 1.0f) / (float)phenotypicMultiplicity;

	toComplex.mutateFloats(p);
	toMemory.mutateFloats(p);
	toModulation.mutateFloats(p);
	toOutput.mutateFloats(p);	
}



void ComplexNode_G::addComplexChild(ComplexNode_G* child) {
	toComplex.insertLineRange(toComplex.nLines, child->inputSize);

	int pos = inputSize + MODULATION_VECTOR_SIZE;
	for (int i = 0; i < complexChildren.size(); i++) {
		pos += complexChildren[i]->outputSize;
	}
	toComplex.insertColumnRange(pos, child->outputSize);
	toMemory.insertColumnRange(pos, child->outputSize);
	toModulation.insertColumnRange(pos, child->outputSize);
	toOutput.insertColumnRange(pos, child->outputSize);
	
	complexChildren.emplace_back(child);
}
void ComplexNode_G::addMemoryChild(MemoryNode_G* child) {

	
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
	

	memoryChildren.emplace_back(child);
}


void ComplexNode_G::removeComplexChild(int rID) {

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
	
	complexChildren.erase(complexChildren.begin() + rID);
}
void ComplexNode_G::removeMemoryChild(int rID) {
	
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
			}
		}

	}
	else {

		MemoryNode_G* potentialChildM = static_cast<MemoryNode_G*>(potentialChild);

		for (int i = 0; i < memoryChildren.size(); i++) {
			id += memoryChildren[i]->inputSize;
			if (memoryChildren[i] == potentialChildM) {
				toMemory.insertLineRange(id - 1,1);// id-1 because the child's inputSize has already been updated.
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
			}
			aID += complexChildren[i]->inputSize;
		}

	}
	else {

		MemoryNode_G* potentialChildM = static_cast<MemoryNode_G*>(potentialChild);

		for (int i = 0; i < memoryChildren.size(); i++) {
			
			
			if (memoryChildren[i] == potentialChildM) {
				toMemory.removeLineRange(aID + id, 1);
			}
			aID += memoryChildren[i]->inputSize;
		}

	}
	
}

bool ComplexNode_G::decrementOutputSize(int id) {
	if (outputSize <= 1) return false;

	toOutput.removeLineRange(id, 1);

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
		s += memoryChildren[i]->inputSize; 
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
		s += memoryChildren[i]->outputSize;
	}
	genomeState[position] = s;
}

#ifdef SATURATION_PENALIZING
// Used to compute the size of the array containing the average saturations of the phenotype.
void ComplexNode_G::computeSaturationArraySize(std::vector<int>& genomeState) {
	int s = MODULATION_VECTOR_SIZE;
	for (int i = 0; i < complexChildren.size(); i++) {
		s += complexChildren[i]->inputSize;
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



