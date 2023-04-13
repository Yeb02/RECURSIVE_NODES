#include "ComplexNode_G.h"


void ComplexNode_G::computeInternalBiasSize() {
	int _s = outputSize;
	_s += (int) simpleChildren.size();
	for (int i = 0; i < complexChildren.size(); i++) {
		_s += complexChildren[i]->inputSize;
	}
	for (int i = 0; i < memoryChildren.size(); i++) {
		_s += memoryChildren[i]->inputSize;
	}
	internalBiasSize = _s;
}

void ComplexNode_G::mutateFloats() {
	constexpr float p = .4f;
	float invFactor = 0.0f;

#ifdef GUIDED_MUTATIONS
	invFactor = nAccumulations == 0 ? 0.0f : 1.0f / nAccumulations;
	nAccumulations = 0;
#endif

	for (int i = 0; i < internalConnexions.size(); i++) {
		internalConnexions[i].mutateFloats(p, invFactor);
	}

	// Ordinary bias mutations
	SET_BINOMIAL(internalBiasSize, p);
	int _nMutations = BINOMIAL;
	for (int i = 0; i < _nMutations; i++) {
		int id = INT_0X(internalBiasSize);
		internalBias[id] *= .8f + NORMAL_01 * .2f;
		internalBias[id] += NORMAL_01 * .2f;
	}

	// Modulation bias mutations
	SET_BINOMIAL(MODULATION_VECTOR_SIZE, p);
	_nMutations = BINOMIAL;
	for (int i = 0; i < _nMutations; i++) {
		int id = INT_0X(MODULATION_VECTOR_SIZE);
		modulationBias[id] *= .8f + NORMAL_01 *.2f;
		modulationBias[id] += NORMAL_01 * .2f;
	}
	
}

std::tuple<NODE_TYPE, int, int> ComplexNode_G::pickRandomOriginNode() {
	// dictates how much more likely the INPUT_NODE is to be selected as the origin node than any other node.
	constexpr int _inputBias = 2;

	// dictates how much more likely the modulation is to be selected as the origin node than any other node.
	constexpr int _modulationOutBias = 1;


	int originN = inputSize * _inputBias + MODULATION_VECTOR_SIZE * _modulationOutBias + (int)simpleChildren.size();
	for (int i = 0; i < complexChildren.size(); i++) {
		originN += complexChildren[i]->outputSize;
	}
	for (int i = 0; i < memoryChildren.size(); i++) {
		originN += memoryChildren[i]->outputSize;
	}

	NODE_TYPE oType;
	int oID, oOutSize;
	int origin_n = INT_0X(originN);

	if ((origin_n -= inputSize * _inputBias) < 0) {
		oType = INPUT_NODE;
		oOutSize = inputSize;
		oID = -1;
	}
	else if ((origin_n -= MODULATION_VECTOR_SIZE * _modulationOutBias) < 0) {
		oType = MODULATION;
		oOutSize = MODULATION_VECTOR_SIZE;
		oID = -1;
	}
	else if ((origin_n -= (int)simpleChildren.size()) < 0) {
		oType = SIMPLE;
		oOutSize = 1;
		oID = origin_n + (int)simpleChildren.size();
	}
	else {
		for (int i = 0; i < complexChildren.size(); i++) {
			if ((origin_n -= complexChildren[i]->outputSize) < 0) {
				oType = COMPLEX;
				oOutSize = complexChildren[i]->outputSize;
				oID = i;
				break;
			}
		}
		if (origin_n >= 0) {
			for (int i = 0; i < memoryChildren.size(); i++) {
				if ((origin_n -= memoryChildren[i]->outputSize) < 0) {
					oType = MEMORY;
					oOutSize = memoryChildren[i]->outputSize;
					oID = i;
					break;
				}
			}
		}
	}

	return { oType, oID, oOutSize };
}

std::tuple<NODE_TYPE, int, int> ComplexNode_G::pickRandomDestinationNode() {

	// dictates how much more likely the output is to be selected as the destination node than any other node.
	constexpr int _outputBias = 2;

	// dictates how much more likely the modulation is to be selected as the destination node than any other node.
	constexpr int _modulationInBias = 1;

	int destinationN = outputSize * _outputBias + MODULATION_VECTOR_SIZE * _modulationInBias + (int)simpleChildren.size();
	for (int i = 0; i < complexChildren.size(); i++) {
		destinationN += complexChildren[i]->inputSize;
	}
	for (int i = 0; i < memoryChildren.size(); i++) {
		destinationN += memoryChildren[i]->inputSize;
	}

	NODE_TYPE dType;
	int dID, dInSize;
	int destination_n = INT_0X(destinationN);

	if ((destination_n -= outputSize * _outputBias) < 0) {
		dType = OUTPUT;
		dInSize = outputSize;
		dID = -1;
	}
	else if ((destination_n -= MODULATION_VECTOR_SIZE * _modulationInBias) < 0) {
		dType = MODULATION;
		dInSize = MODULATION_VECTOR_SIZE;
		dID = -1;
	}
	else if ((destination_n -= (int)simpleChildren.size()) < 0) {
		dType = SIMPLE;
		dInSize = 1;
		dID = destination_n + (int)simpleChildren.size();
	}
	else {
		for (int i = 0; i < complexChildren.size(); i++) {
			if ((destination_n -= complexChildren[i]->outputSize) < 0) {
				dType = COMPLEX;
				dInSize = complexChildren[i]->outputSize;
				dID = i;
				break;
			}
		}
		if (destination_n >= 0) {
			for (int i = 0; i < memoryChildren.size(); i++) {
				if ((destination_n -= memoryChildren[i]->outputSize) < 0) {
					dType = MEMORY;
					dInSize = memoryChildren[i]->outputSize;
					dID = i;
					break;
				}
			}
		}
	}

	return { dType, dID, dInSize };
}

// Note that a child node can be connected to itself. Also, several connexion can link the same two children. (can be toggled)
void ComplexNode_G::addConnexion() {
	

	constexpr int maxAttempts = 3;

	for (int i = 0; i < maxAttempts; i++) {

		// not really optimized, but unimportant.
		auto [dType, dID, dInSize]  = pickRandomDestinationNode();
		auto [oType, oID, oOutSize] = pickRandomOriginNode();
		
		int nSameConnexion = 0;
		for (int j = 0; j < internalConnexions.size(); j++) {
			if (internalConnexions[j].originType == oType && 
				internalConnexions[j].destinationType == dType &&
				internalConnexions[j].originID == oID && 
				internalConnexions[j].destinationID == dID) 
			{
				nSameConnexion++;
			}
		}

		if (true) { // If true, only one connexion allowed between two nodes
			if (nSameConnexion>0) continue;
		}
		else {  
			// the probability of a connexion being created diminishes exponentially as the number of preexisting connexions 
			// between the same 2 nodes increases.
			constexpr float failureProbability = .7f;
			if (UNIFORM_01 < 1.0f - powf(failureProbability, (float)nSameConnexion)) {
				continue;
			}
		}
		
		// ZERO initialization to minimize disturbance of the network
		internalConnexions.emplace_back(oType, dType, oID, dID, dInSize, oOutSize, GenotypeConnexion::ZERO);
		break;
	}
}
void ComplexNode_G::removeConnexion() {
	if (internalConnexions.size() <= 1) return; 
	int id = INT_0X(internalConnexions.size());
	//if (origin = INPUT_NODE && internalConnexions[id].destinationID == MODULATION_ID) return; // not allowed to disconnect INPUT_NODE->neuromodulation.
	internalConnexions.erase(internalConnexions.begin() + id);
}


void ComplexNode_G::addComplexChild(ComplexNode_G* child) {

	int newChildID = (int)complexChildren.size();
	
	auto [oType, oID, oOutSize] = pickRandomOriginNode();
	internalConnexions.emplace_back(oType, COMPLEX, oID, newChildID,  child->inputSize, oOutSize,GenotypeConnexion::RANDOM);

	auto [dType, dID, dInSize] = pickRandomDestinationNode();
	internalConnexions.emplace_back(COMPLEX, dType, newChildID, dID, dInSize, child->outputSize, GenotypeConnexion::ZERO);

	int biasID = outputSize + (int)simpleChildren.size();
	for (int i = 0; i < complexChildren.size(); i++) {
		biasID += complexChildren[i]->inputSize;
	}
	internalBias.insert(internalBias.begin() + biasID, child->inputSize, 0.0f);
	for (int i = biasID; i < biasID + child->inputSize; i++) {
		internalBias[i] = NORMAL_01 * .2f;
	}

	complexChildren.emplace_back(child);
}
void ComplexNode_G::addSimpleChild(SimpleNode_G* child) {

	int newChildID = (int)simpleChildren.size();

	auto [oType, oID, oOutSize] = pickRandomOriginNode();
	internalConnexions.emplace_back(oType, SIMPLE, oID, newChildID, 1, oOutSize, GenotypeConnexion::RANDOM);

	auto [dType, dID, dInSize] = pickRandomDestinationNode();
	internalConnexions.emplace_back(SIMPLE, dType, newChildID, dID, dInSize, 1, GenotypeConnexion::ZERO);

	int biasID = outputSize + (int)simpleChildren.size();

	internalBias.insert(internalBias.begin() + biasID, NORMAL_01 * .2f);

	simpleChildren.emplace_back(child);
}
void ComplexNode_G::addMemoryChild(MemoryNode_G* child) {

	int newChildID = (int)memoryChildren.size();

	auto [oType, oID, oOutSize] = pickRandomOriginNode();
	internalConnexions.emplace_back(oType, MEMORY, oID, newChildID, child->inputSize, oOutSize, GenotypeConnexion::RANDOM);

	auto [dType, dID, dInSize] = pickRandomDestinationNode();
	internalConnexions.emplace_back(MEMORY, dType, newChildID, dID, dInSize, child->outputSize, GenotypeConnexion::ZERO);

	int biasID = outputSize + (int)simpleChildren.size();
	for (int i = 0; i < complexChildren.size(); i++) {
		biasID += complexChildren[i]->inputSize;
	}
	for (int i = 0; i < memoryChildren.size(); i++) {
		biasID += memoryChildren[i]->inputSize;
	}
	internalBias.insert(internalBias.begin() + biasID, child->inputSize, 0.0f);
	for (int i = biasID; i < biasID + child->inputSize; i++) {
		internalBias[i] = NORMAL_01 * .2f;
	}

	memoryChildren.emplace_back(child);
}


void ComplexNode_G::removeComplexChild(int rID) {

	// Erase connexions that lead to the removed child. 
	std::vector<GenotypeConnexion> keptConnexions;
	keptConnexions.reserve(internalConnexions.size());
	for (int i = 0; i < internalConnexions.size(); i++){
		if ((internalConnexions[i].destinationType != COMPLEX || internalConnexions[i].destinationID != rID) &&
			(internalConnexions[i].originType != COMPLEX      || internalConnexions[i].originID != rID))
		{
			keptConnexions.emplace_back(internalConnexions[i]);
		}
	}
	internalConnexions = std::move(keptConnexions);

	// Update the IDs in the connexions that are kept
	for (int i = 0; i < internalConnexions.size(); i++) {
		if (internalConnexions[i].destinationType == COMPLEX && internalConnexions[i].destinationID > rID) 
		{
			internalConnexions[i].destinationID--;
		}
		if (internalConnexions[i].originType == COMPLEX && internalConnexions[i].originID > rID)
		{
			internalConnexions[i].originID--;
		}
	}

	// Update the bias array
	int biasID = outputSize + (int) simpleChildren.size();
	for (int i = 0; i < rID; i++) {
		biasID += complexChildren[i]->inputSize;
	}
	internalBias.erase(internalBias.begin() + biasID, internalBias.begin() + biasID + complexChildren[rID]->inputSize);

	complexChildren.erase(complexChildren.begin() + rID);
}
void ComplexNode_G::removeSimpleChild(int rID) {

	// Erase connexions that lead to the removed child. 
	std::vector<GenotypeConnexion> keptConnexions;
	keptConnexions.reserve(internalConnexions.size());
	for (int i = 0; i < internalConnexions.size(); i++) {
		if ((internalConnexions[i].destinationType != SIMPLE || internalConnexions[i].destinationID != rID) &&
			(internalConnexions[i].originType != SIMPLE || internalConnexions[i].originID != rID))
		{
			keptConnexions.emplace_back(internalConnexions[i]);
		}
	}
	internalConnexions = std::move(keptConnexions);

	// Update the IDs in the connexions that are kept
	for (int i = 0; i < internalConnexions.size(); i++) {
		if (internalConnexions[i].destinationType == SIMPLE && internalConnexions[i].destinationID > rID)
		{
			internalConnexions[i].destinationID--;
		}
		if (internalConnexions[i].originType == SIMPLE && internalConnexions[i].originID > rID)
		{
			internalConnexions[i].originID--;
		}
	}

	// Update the bias array
	internalBias.erase(internalBias.begin() + outputSize + rID);

	simpleChildren.erase(simpleChildren.begin() + rID);
}
void ComplexNode_G::removeMemoryChild(int rID) {

	// Erase connexions that lead to the removed child. 
	std::vector<GenotypeConnexion> keptConnexions;
	keptConnexions.reserve(internalConnexions.size());
	for (int i = 0; i < internalConnexions.size(); i++) {
		if ((internalConnexions[i].destinationType != MEMORY || internalConnexions[i].destinationID != rID) &&
			(internalConnexions[i].originType != MEMORY || internalConnexions[i].originID != rID))
		{
			keptConnexions.emplace_back(internalConnexions[i]);
		}
	}
	internalConnexions = std::move(keptConnexions);

	// Update the IDs in the connexions that are kept
	for (int i = 0; i < internalConnexions.size(); i++) {
		if (internalConnexions[i].destinationType == MEMORY && internalConnexions[i].destinationID > rID)
		{
			internalConnexions[i].destinationID--;
		}
		if (internalConnexions[i].originType == MEMORY && internalConnexions[i].originID > rID)
		{
			internalConnexions[i].originID--;
		}
	}

	// Update the bias array
	int biasID = outputSize + (int)simpleChildren.size();
	for (int i = 0; i < complexChildren.size(); i++) {
		biasID += complexChildren[i]->inputSize;
	}
	for (int i = 0; i < rID; i++) {
		biasID += memoryChildren[i]->inputSize;
	}
	internalBias.erase(internalBias.begin() + biasID, internalBias.begin() + biasID + memoryChildren[rID]->inputSize);

	memoryChildren.erase(memoryChildren.begin() + rID);
}


bool ComplexNode_G::incrementInputSize() {
	if (inputSize >= MAX_COMPLEX_INPUT_NODE_SIZE) return false;

	for (int i = 0; i < internalConnexions.size(); i++) {
		if (internalConnexions[i].originType == INPUT_NODE) {
			internalConnexions[i].incrementOriginOutputSize();
		}
	}
	inputSize++;
	return true;
}
void ComplexNode_G::onChildInputSizeIncremented(int modifiedPosition, NODE_TYPE modifiedType) {

	// check if the modified node is among the children

	for (int i = 0; i < internalConnexions.size(); i++) {
		NODE_TYPE destinationType = internalConnexions[i].destinationType;
		if (destinationType == modifiedType) {
			int destinationPosition = modifiedType == COMPLEX ?
				complexChildren[internalConnexions[i].destinationID]->position :
				memoryChildren[internalConnexions[i].destinationID]->position;
			if (destinationPosition == modifiedPosition) {
				internalConnexions[i].incrementDestinationInputSize();
			}
		}
	}

	// insert a random bias where need be.
	int nInsertions = 0;
	int biasID = outputSize + (int)simpleChildren.size();
	if (modifiedType == COMPLEX) {
		for (int i = 0; i < complexChildren.size(); i++) {
			if (modifiedType == COMPLEX && complexChildren[i]->position == modifiedPosition) {
				internalBias.insert(internalBias.begin() + biasID + nInsertions, NORMAL_01 * .2f);
				nInsertions++;
			}
			biasID += complexChildren[i]->inputSize;
		}
	}
	else {
		for (int i = 0; i < complexChildren.size(); i++) {
			biasID += complexChildren[i]->inputSize;
		}
		for (int i = 0; i < memoryChildren.size(); i++) {
			if (memoryChildren[i]->position == modifiedPosition) {
				internalBias.insert(internalBias.begin() + biasID + nInsertions, NORMAL_01 * .2f);
				nInsertions++;
			}
			biasID += memoryChildren[i]->inputSize;
		}
	}
}

bool ComplexNode_G::incrementOutputSize() {
	
	if (outputSize >= MAX_COMPLEX_OUTPUT_SIZE)
	{ 
		return false;
	}
	
	for (int i = 0; i < internalConnexions.size(); i++) {
		if (internalConnexions[i].destinationType == OUTPUT) {
			internalConnexions[i].incrementDestinationInputSize();
		}
	}
	internalBias.insert(internalBias.begin() + outputSize, NORMAL_01*.2f);
	outputSize++;
	return true;
}
void ComplexNode_G::onChildOutputSizeIncremented(int modifiedPosition, NODE_TYPE modifiedType) {

	// check if the modified node is among the children

	for (int i = 0; i < internalConnexions.size(); i++) {
		NODE_TYPE originType = internalConnexions[i].originType;
		if (originType == modifiedType) {
			int originPosition = modifiedType == COMPLEX ?
				complexChildren[internalConnexions[i].originID]->position :
				memoryChildren[internalConnexions[i].originID]->position;
			if (originPosition == modifiedPosition) {
				internalConnexions[i].incrementOriginOutputSize();
			}
		}
	}
}

bool ComplexNode_G::decrementInputSize(int id) {
	if (inputSize <= 1) {
		return false;
	}

	for (int i = 0; i < internalConnexions.size(); i++) {
		if (internalConnexions[i].originType == INPUT_NODE) {
			internalConnexions[i].decrementOriginOutputSize(id);
		}
	}
	inputSize--;
	return true;
}
void ComplexNode_G::onChildInputSizeDecremented(int modifiedPosition, NODE_TYPE modifiedType, int id) {

	// id is the indice of the neuron that was deleted in the child's INPUT_NODE.

	// check if the modified node is among the children

	for (int i = 0; i < internalConnexions.size(); i++) {
		NODE_TYPE destinationType = internalConnexions[i].destinationType;
		if (destinationType == modifiedType) {
			int destinationPosition = modifiedType == COMPLEX ?
				complexChildren[internalConnexions[i].destinationID]->position :
				memoryChildren[internalConnexions[i].destinationID]->position;
			if (destinationPosition == modifiedPosition) {
				internalConnexions[i].decrementDestinationInputSize(id);
			}
		}
	
	}

	// insert a random bias where need be.
	int nErasures = 0;
	int biasID = outputSize + (int)simpleChildren.size();
	if (modifiedType == COMPLEX) {
		for (int i = 0; i < complexChildren.size(); i++) {
			if (modifiedType == COMPLEX && complexChildren[i]->position == modifiedPosition) {
				internalBias.erase(internalBias.begin() + biasID + id - nErasures);
				nErasures++;
			}
			biasID += complexChildren[i]->inputSize;
		}
	}
	else {
		for (int i = 0; i < complexChildren.size(); i++) {
			biasID += complexChildren[i]->inputSize;
		}
		for (int i = 0; i < memoryChildren.size(); i++) {
			if (memoryChildren[i]->position == modifiedPosition) {
				internalBias.erase(internalBias.begin() + biasID + id - nErasures);
				nErasures++;
			}
			biasID += memoryChildren[i]->inputSize;
		}
	}
}

bool ComplexNode_G::decrementOutputSize(int id) {
	if (outputSize <= 1) return false;

	for (int i = 0; i < internalConnexions.size(); i++) {
		if (internalConnexions[i].destinationID == OUTPUT) {
			internalConnexions[i].decrementDestinationInputSize(id);
		}
	}
	internalBias.erase(internalBias.begin() + id);
	outputSize--;
	return true;
}
void ComplexNode_G::onChildOutputSizeDecremented(int modifiedPosition, NODE_TYPE modifiedType, int id) {

	// check if the modified node is among the children

	for (int i = 0; i < internalConnexions.size(); i++) {
		NODE_TYPE originType = internalConnexions[i].originType;
		if (originType == modifiedType) {
			int originPosition = modifiedType == COMPLEX ?
				complexChildren[internalConnexions[i].originID]->position :
				memoryChildren[internalConnexions[i].originID]->position;
			if (originPosition == modifiedPosition) {
				internalConnexions[i].decrementOriginOutputSize(id);
			}
		}
	}
}


void ComplexNode_G::getNnonLinearities(std::vector<int>& genomeState) {
	constexpr int modulationMultiplier = 0; // must be set to the same value in Phenotype::forward. TODO cleaner.
	int n = inputSize + outputSize + (int) simpleChildren.size() + 2 * modulationMultiplier;
	for (int i = 0; i < complexChildren.size(); i++) {
		if (genomeState[complexChildren[i]->position] == 0) complexChildren[i]->getNnonLinearities(genomeState);
		n += genomeState[complexChildren[i]->position];
	}
	for (int i = 0; i < memoryChildren.size(); i++) {
		n += memoryChildren[i]->inputSize + memoryChildren[i]->outputSize;
	}
	genomeState[position] = n;
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

void ComplexNode_G::computeActivationArraySize(std::vector<int>& genomeState) {
	int s = inputSize + outputSize + (int)simpleChildren.size();
	for (int i = 0; i < complexChildren.size(); i++) {
		if (genomeState[complexChildren[i]->position] == 0) {
			complexChildren[i]->computeActivationArraySize(genomeState);
		}
		s += genomeState[complexChildren[i]->position];
	}
	for (int i = 0; i < memoryChildren.size(); i++) {
		s += memoryChildren[i]->inputSize + memoryChildren[i]->outputSize;
	}
	genomeState[position] = s;
}


#ifdef SATURATION_PENALIZING
// Used to compute the size of the array containing the average saturations of the phenotype.
void ComplexNode_G::computeSaturationArraySize(std::vector<int>& genomeState) {
	int s = INPUT_NODESize + 2 + outputSize;
	for (int i = 0; i < children.size(); i++) {
		if (genomeState[children[i]->position] == 0) {
			children[i]->computeSaturationArraySize(genomeState);
		}
		s += genomeState[children[i]->position];
	}
	genomeState[position] = s;
}
#endif 

ComplexNode_G::ComplexNode_G(ComplexNode_G* n) {

	inputSize = n->inputSize;
	outputSize = n->outputSize;
	internalBias.assign(n->internalBias.begin(), n->internalBias.end());
	for (int i = 0; i < MODULATION_VECTOR_SIZE; i++) {
		modulationBias[i] = n->modulationBias[i];
	}
	internalBiasSize = n->internalBiasSize;
	depth = n->depth;
	position = n->position;
	mutationalDistance = n->mutationalDistance;


	internalConnexions.reserve((int)((float)n->internalConnexions.size() * 1.5f));
	for (int j = 0; j < n->internalConnexions.size(); j++) {
		internalConnexions.emplace_back(n->internalConnexions[j]);
	}

#ifdef GUIDED_MUTATIONS
	nAccumulations = n->nAccumulations;
#endif
	phenotypicMultiplicity = n->phenotypicMultiplicity;

	// The following enclosed section is useless if n is not part of the same network as "this", 
	// and it must be repeated where this function was called.
	{
		simpleChildren.reserve((int)((float)n->simpleChildren.size() * 1.5f));
		for (int j = 0; j < n->simpleChildren.size(); j++) {
			simpleChildren.emplace_back(n->simpleChildren[j]);
		}
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

bool ComplexNode_G::hasChild(std::vector<int>& checked, ComplexNode_G* potentialChild) {
	if (depth <= potentialChild->depth) return false;

	for (int i = 0; i < (int)complexChildren.size(); i++) {
		if (checked[complexChildren[i]->position] == 1) continue;
		if (complexChildren[i] == potentialChild) return true;
		if (complexChildren[i]->hasChild(checked,potentialChild)) return true;
		checked[complexChildren[i]->position] = 1;
	}
	return false;
}