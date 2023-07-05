#pragma once

#include <iostream>
#include <algorithm> // std::sort
#include <mutex>     // threading
#include <condition_variable> // threading

#include "Population.h"
#include "Random.h"


// Threading utils
std::mutex m;
std::condition_variable startProcessing;
std::condition_variable doneProcessing;
std::condition_variable hasTerminated;
bool bStartProcessing = false;
int nTerminated = 0;
int nDoneProcessing = 0;

int PhylogeneticNode::maxListSize = 0;

// src is unchanged.
void normalizeArray(float* src, float* dst, int size) {
	float avg = 0.0f;
	for (int i = 0; i < size; i++) {
		avg += src[i];
	}
	avg /= (float)size;
	float variance = 0.0f;
	for (int i = 0; i < size; i++) {
		dst[i] = src[i] - avg;
		variance += dst[i] * dst[i];
	}
	if (variance < .001f) return;
	float InvStddev = 1.0f / sqrtf(variance / (float) size);
	for (int i = 0; i < size; i++) {
		dst[i] *= InvStddev;
	}
}

// src is unchanged. Results in [-1, 1]
void rankArray(float* src, float* dst, int size) {
	std::vector<int> positions(size);
	for (int i = 0; i < size; i++) {
		positions[i] = i;
	}
	// sort position by ascending value.
	std::sort(positions.begin(), positions.end(), [src](int a, int b) -> bool
		{
			return src[a] < src[b];
		}
	);
	float invSize = 1.0f / (float)size;
	for (int i = 0; i < size; i++) {
		// linear in [-1,1], -1 for the worst specimen, 1 for the best
		float positionValue = (float)(2 * i - size) * invSize;
		// arbitrary, to make it a bit more selective. 
		positionValue = 1.953f * powf(positionValue * .8f, 3.0f);

		dst[positions[i]] = positionValue;
	}
	return;
}

Population::Population(int IN_SIZE, int OUT_SIZE, int nSpecimens, bool fromDLL) :
	nSpecimens(nSpecimens), fromDLL(fromDLL),
	N_THREADS(0), threadIteration(0), mustTerminate(false)
{
	threads.reserve(0);
	globalTrials.reserve(0);
	rawScores.reserve(0);
	batchTransformedScores.reserve(0);
	networks.resize(nSpecimens);
	fitnesses.resize(nSpecimens);
	for (int i = 0; i < nSpecimens; i++) {
		networks[i] = new Network(IN_SIZE, OUT_SIZE);
	}
	
	PopulationEvolutionParameters defaultParams;
	setEvolutionParameters(defaultParams);

	phylogeneticTree = new PhylogeneticNode[MAX_MATING_DEPTH * nSpecimens];
	for (int i = 0; i < nSpecimens; i++) {
		phylogeneticTree[i].children.resize(0);
		phylogeneticTree[i].networkIndice = i;
		phylogeneticTree[i].parent = nullptr;
	}

	fittestSpecimen = 0;
	evolutionStep = 1; // starting at 1 is important for the phylogeneticTree.
	nTrialsAtThisStep = -1;
}

void Population::stopThreads() {
	if (threads.size() == 0) return;

	{
		std::lock_guard<std::mutex> lg(m);
		mustTerminate = true;
		nTerminated = (int) threads.size();
	}
	startProcessing.notify_all();

	// wait on workers.
	{
		std::unique_lock<std::mutex> lg(m);
		hasTerminated.wait(lg, [] {return nTerminated == 0; });
	}

	for (int t = 0; t < N_THREADS; t++) threads[t].join();

	N_THREADS = 0;

}

void Population::startThreads(int N_THREADS) {
	stopThreads();

	mustTerminate = false;
	this->N_THREADS = N_THREADS;
	threads.resize(0); // to kill all previously existing threads.

	if (N_THREADS < 2) return;

	threads.reserve(N_THREADS);
	threadIteration = -1;
	int subArraySize = nSpecimens / N_THREADS;
	int i0;
	for (int t = 0; t < N_THREADS; t++) {  
		i0 = t * subArraySize;
		threads.emplace_back(&Population::threadLoop, this, i0, subArraySize);
	}
}

Population::~Population() {
	for (const Network* n : networks) {
		delete n;
	}
	delete[] phylogeneticTree;
}

void Population::threadLoop(const int i0, const int subArraySize) {
	std::vector<std::unique_ptr<Trial>> localTrials;

	int currentThreadIteration = 0;
	while (true) {
		std::unique_lock<std::mutex> ul(m);
		startProcessing.wait(ul, [&currentThreadIteration,this] {return (currentThreadIteration == threadIteration) || mustTerminate; });
		if (mustTerminate) {
			nTerminated--;
			if (nTerminated == 0) {
				ul.unlock();
				hasTerminated.notify_one();
			}
			break;
		}
		ul.unlock();

		currentThreadIteration++;

		// Copy init. Read only, so no mutex required.
		if (localTrials.size() != globalTrials.size()) {
			localTrials.resize(0);
			for (int i = 0; i < globalTrials.size(); i++) {
				localTrials.emplace_back(globalTrials[i]->clone());
			}
		}
		else {
			for (int i = 0; i < globalTrials.size(); i++) {
				localTrials[i]->copy(globalTrials[i]);
				localTrials[i]->reset(true);
			}
		}

		for (int i = i0; i < i0 + subArraySize; i++) {

			networks[i]->mutate(); 
			networks[i]->createPhenotype();
		}
		

		for (int i = 0; i < localTrials.size(); i++) {

			evaluate(i0, subArraySize, localTrials[i].get(), rawScores.data() + i * nSpecimens);

			ul.lock();
			nDoneProcessing--;
			if (nDoneProcessing == 0) {
				ul.unlock();
				doneProcessing.notify_one();
			} else { ul.unlock(); }

			// WAIT FOR NON THREADED OPERATION ON THE SCORE ARRAY.

			ul.lock();
			startProcessing.wait(ul, [&currentThreadIteration, this] {return (currentThreadIteration == threadIteration) || mustTerminate; });
			ul.unlock();
			currentThreadIteration++;

			for (int j = i0; j < i0 + subArraySize; j++) {
				networks[j]->postTrialUpdate(batchTransformedScores[i * nSpecimens + j], i);
			}
		}
		
		ul.lock();
		nDoneProcessing--;
		if (nDoneProcessing == 0) {
			ul.unlock();
			doneProcessing.notify_one();
		}
	}
}

// TODO this order forces all phenotypes to be loaded in RAM during the same period, only one per thread is
// a priority
void Population::evaluate(const int i0, const int subArraySize, Trial* trial, float* rawScores) {
	for (int i = i0; i < i0 + subArraySize; i++) {
		networks[i]->preTrialReset();
		trial->reset(useSameTrialInit); 
		while (!trial->isTrialOver) {
			networks[i]->step(trial->observations);
			trial->step(networks[i]->getOutput());
		}
		rawScores[i] = trial->score;
	}
}

void Population::step(std::vector<std::unique_ptr<Trial>>& trials, int nTrialsEvaluated) {
	// utils
	nTrialsAtThisStep = (int)trials.size();
	if (nSpecimens * nTrialsAtThisStep != rawScores.size()) {
		rawScores.resize(nSpecimens * nTrialsAtThisStep);
		batchTransformedScores.resize(nSpecimens * nTrialsAtThisStep);
	}
	// The indice of the first trial that will be used for fitness
	int i0 = nTrialsAtThisStep - nTrialsEvaluated;


	// Mutate, then evaluate the specimens on trials. Threaded operation if specified.
	if (N_THREADS > 1) {

		// acquire pointers to this step's trials.
		globalTrials.resize(nTrialsAtThisStep);
		for (int j = 0; j <nTrialsAtThisStep; j++) {
			globalTrials[j] = trials[j].get();
		}

		for (int i = 0; i <nTrialsAtThisStep + 1; i++) {
			
			// send msg to workers to start processing
			{
				std::lock_guard<std::mutex> lg(m);
				nDoneProcessing = N_THREADS;
				threadIteration++;
			}
			startProcessing.notify_all();

			// wait on workers.
			{
				std::unique_lock<std::mutex> lg(m);
				doneProcessing.wait(lg, [] {return nDoneProcessing == 0; });
			}

			// Non threaded operations on the raw scores array:
			if (i <nTrialsAtThisStep) {
				switch (scoreBatchTransformation) {
				case NORMALIZE:
					normalizeArray(rawScores.data() + i * nSpecimens, batchTransformedScores.data() + i * nSpecimens, nSpecimens);
					break;
				case RANK:
					rankArray(rawScores.data() + i * nSpecimens, batchTransformedScores.data() + i * nSpecimens, nSpecimens);
					break;
				case NONE:
					std::copy(rawScores.data() + i * nSpecimens,
						rawScores.data() + (i + 1) * nSpecimens,
						batchTransformedScores.data() + i * nSpecimens);
					break;
				}	
			}
		}
	}
	else {
		for (int i = 0; i < nSpecimens; i++) {
			networks[i]->mutate();
			networks[i]->createPhenotype();
		}

		for (int i = 0; i < nTrialsAtThisStep; i++) {
			evaluate(0, nSpecimens, trials[i].get(), rawScores.data() + i * nSpecimens);

			switch (scoreBatchTransformation) {
			case NORMALIZE:
				normalizeArray(rawScores.data() + i * nSpecimens, batchTransformedScores.data() + i * nSpecimens, nSpecimens);
				break;
			case RANK:
				rankArray(rawScores.data() + i * nSpecimens, batchTransformedScores.data() + i * nSpecimens, nSpecimens);
				break;
			case NONE:
				std::copy(rawScores.data() + i * nSpecimens,
					rawScores.data() + (i + 1) * nSpecimens,
					batchTransformedScores.data() + i * nSpecimens);
				break;
			}

			for (int j = 0; j < nSpecimens; j++) {
				networks[j]->postTrialUpdate(batchTransformedScores[j], i);
			}
		}
	}


	// logging scores. monitoring only, can be disabled.
	if (true) {
	
		std::vector<float> avgScoresPerTrial(nTrialsEvaluated);
		float maxScore = -1000000.0f;
		int maxScoreID = 0;
		float score, avgFactor = 1.0f / (float)nTrialsEvaluated;
		for (int i = 0; i < nSpecimens; i++) {
			score = 0;
			for (int j = i0; j <nTrialsAtThisStep; j++) {
				score += rawScores[i + j * nSpecimens];
				avgScoresPerTrial[j - i0] += rawScores[i + j * nSpecimens];
			}
			score *= avgFactor;
			if (score > maxScore) {
				maxScore = score;
				maxScoreID = i;
			}
		}
		for (int j = 0; j < nTrialsEvaluated; j++) avgScoresPerTrial[j] /= (float)nSpecimens;
		float avgavgf = 0.0f;
		for (float f : avgScoresPerTrial) avgavgf += f;
		avgavgf /= nTrialsEvaluated;
		std::cout << "At generation " << evolutionStep
		<< ", max score = " << maxScore
		<< ", average score per specimen per trial = " << avgavgf << ".\n";
		//std::cout << maxScore << ", ";
		trials[0]->reset();
		Network* n = networks[maxScoreID];
		n->createPhenotype(); // should be already created
		n->preTrialReset();
		while (!trials[0]->isTrialOver) {
			n->step(trials[0]->observations);
			trials[0]->step(n->getOutput());
		}
		std::cout << "Best specimen's score on new trial = " << trials[0]->score << std::endl;
		
	}

	// Competition score adjustment. Assuming that either all networks have a valid parentData, or none has.
	// Uses batchTransformed scores. Happens here and not in computeFitnesses because is specific for each trial.
	if (competitionFactor != 0.0f && networks[0]->parentData.isAvailable) { 
		for (int i = 0; i < nSpecimens; i++) {
			for (int j = i0; j <nTrialsAtThisStep; j++) {
				batchTransformedScores[i + j * nSpecimens] += competitionFactor *
						(batchTransformedScores[i + j * nSpecimens] - networks[i]->parentData.scores[j]);
			}
		}
	}


	// Compute the final score per specimen.
	std::vector<float> avgScorePerSpecimen(nSpecimens);
#ifdef SPECIALIZATION_INCENTIVE
	float invP = 1.0f / SPECIALIZATION_INCENTIVE;
	for (int i = 0; i < nSpecimens; i++) {
		float s = 0.0f;
		for (int j = i0; j <nTrialsAtThisStep; j++) {
			s += powf(std::max( batchTransformedScores[i + j * nSpecimens], 0.0f) , SPECIALIZATION_INCENTIVE);
		}
		avgScorePerSpecimen[i] = powf(s, invP);
	}
#else
	for (int i = 0; i < nSpecimens; i++) {
		for (int j = i0; j <nTrialsAtThisStep; j++) {
			avgScorePerSpecimen[i] += batchTransformedScores[i + j * nSpecimens];
		}
	}
#endif

	// Adds regularization, saturation penalization, ...
	computeFitnesses(avgScorePerSpecimen);

	createOffsprings();
}

void Population::computeFitnesses(std::vector<float>& avgScorePerSpecimen) {

	// compute and rank the regularization term
	std::vector<float> regularizationScore(nSpecimens);
	for (int i = 0; i < nSpecimens; i++) {
		regularizationScore[i] = networks[i]->getRegularizationLoss();
	}
	//normalizeArray(regularizationScore.data(), regularizationScore.data(), nSpecimens);
	rankArray(regularizationScore.data(), regularizationScore.data(), nSpecimens);


#ifdef SATURATION_PENALIZING
	// compute and normalize the saturation term
	std::vector<float> saturationScore(nSpecimens);
	for (int i = 0; i < nSpecimens; i++) {
		saturationScore[i] = networks[i]->getSaturationPenalization();
	}
	//normalizeArray(saturationScore.data(), saturationScore.data(), nSpecimens);
	rankArray(saturationScore.data(), saturationScore.data(), nSpecimens);
#endif


	
	if (rankingFitness) {
		rankArray(avgScorePerSpecimen.data(), avgScorePerSpecimen.data(), nSpecimens);
	}
	/*else {
		normalizeArray(avgScorePerSpecimen.data(), avgScorePerSpecimen.data(), nSpecimens);
	}*/


	// Then, the fitness is simply a weighted sum of score, regularization, and saturation if enabled.
	for (int i = 0; i < nSpecimens; i++) {
		fitnesses[i] = avgScorePerSpecimen[i] - regularizationFactor * regularizationScore[i];
#ifdef SATURATION_PENALIZING
		fitnesses[i] -= saturationFactor * saturationScore[i];
#endif
	}
}

Network* Population::createChild(PhylogeneticNode* primaryParent) {

	// Reminder: when this function is called, nParents > 1.

	std::vector<int> parents;
	parents.push_back(primaryParent->networkIndice);
	
	std::vector<float> rawWeights;
	rawWeights.resize(nParents);

	// fill parents, and weights are initialized with a function of the phylogenetic distance.
	{
		PhylogeneticNode* previousRoot = primaryParent;
		PhylogeneticNode* root = primaryParent->parent;
		bool listFilled = false;

		auto distanceValue = [] (float d) 
		{
			return powf(1.0f+d, -0.6f);
		};

		for (int depth = 1; depth < MAX_MATING_DEPTH; depth++) {

			if (depth <= CONSANGUINITY_DISTANCE) 
			{
				previousRoot = root;
				root = root->parent;
				continue;
			}

			float w = distanceValue((float) depth);
			std::fill(rawWeights.begin() + (int)parents.size(), rawWeights.end(), w);

			int nC = (int)root->children.size();

			// we start at a random indice in the root children, so that if the list is filled before the end of this
			// loop iteration it is not always the same networks that are selected.
			int i0 = INT_0X(nC); 

			for (int i = 0; i < nC; i++) {
				int id = (i + i0) % nC;
				if (root->children[id] != previousRoot) {
					if (!root->children[id]->addToList(parents, depth-1)) {
						listFilled = true;
						break;
					}
				}
			}

			// parents.size() == nParents can happen if there were exactly as many parents 
			// as there was room left in the array. This does not set listFilled = true.
			if (listFilled || parents.size() == nParents) break; 

			previousRoot = root;
			root = root->parent;
		}
	}

	// weights are an increasing function of the relative fitness.
	{
		rawWeights[0] = 1.0f;
		float f0;
		if (networks[primaryParent->networkIndice]->parentData.isAvailable && USE_PARENT_FITNESS_AS_ZERO) { // TODO experimental
			f0 = networks[primaryParent->networkIndice]->parentData.f;
		}
		else {
			f0 = fitnesses[primaryParent->networkIndice];
		}
		for (int i = 1; i < parents.size(); i++) {
			rawWeights[i] *= (fitnesses[parents[i]] - f0); // TODO better
		}
	}

	std::vector<Network*> parentNetworks;
	parentNetworks.resize(parents.size());
	for (int i = 0; i < parents.size(); i++) {
		parentNetworks[i] = networks[parents[i]];
	}
	return Network::combine(parentNetworks, rawWeights);
}

void Population::createOffsprings() {
	uint64_t start = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();

	float* phase1Probabilities = new float[nSpecimens];
	float* phase2Probabilities = new float[nSpecimens];
	Network** tempNetworks = new Network* [nSpecimens];

	float fMax = -10000.0f;
	for (int i = 0; i < nSpecimens; i++) {
		if (fitnesses[i] > fMax) {
			fMax = fitnesses[i];
			fittestSpecimen = i;
		}
	}

	float normalizationFactor = 1.0f / (fMax - selectionPressure.first);
	for (int i = 0; i < nSpecimens; i++) {
		phase1Probabilities[i] = (fitnesses[i] - selectionPressure.first)*normalizationFactor;
		if (fitnesses[i] < selectionPressure.second) phase2Probabilities[i] = 0.0f;
		else phase2Probabilities[i] = fitnesses[i] - selectionPressure.second;
	}

	

	int parentPhylogeneticTreeID = ((evolutionStep-1) % MAX_MATING_DEPTH) * nSpecimens;
	int phylogeneticTreeID = (evolutionStep % MAX_MATING_DEPTH) * nSpecimens;
	int nReconductedSpecimens = 0;

	// lambda just not to write this twice.
	auto doEverything = [&](int parentID, int childID) {

		PhylogeneticNode* childNode = &phylogeneticTree[phylogeneticTreeID + childID];
		childNode->children.resize(0);
		childNode->parent = &phylogeneticTree[parentPhylogeneticTreeID + parentID];
		childNode->networkIndice = childID;
		childNode->parent->children.push_back(childNode);

		if (evolutionStep < MAX_MATING_DEPTH || nParents == 1) {
			tempNetworks[childID] = new Network(networks[parentID]);
		}
		else {
			tempNetworks[childID] = createChild(childNode->parent);
		}

		if (!fromDLL) {
			tempNetworks[childID]->parentData.isAvailable = true;
			tempNetworks[childID]->parentData.scoreSize = nTrialsAtThisStep;
			tempNetworks[childID]->parentData.scores = new float[nTrialsAtThisStep];
			for (int i = 0; i < nTrialsAtThisStep; i++) {
				tempNetworks[childID]->parentData.scores[i] = batchTransformedScores[i * nSpecimens + parentID];
			}
			tempNetworks[childID]->parentData.f = fitnesses[parentID];
		}

		return;
	};

	
	for (int i = 0; i < nSpecimens; i++) {
		if (UNIFORM_01 < phase1Probabilities[i]) {
			if (i == fittestSpecimen) fittestSpecimen = nReconductedSpecimens; // happens once and only once
			doEverything(i, nReconductedSpecimens);
			nReconductedSpecimens++;
		}
	}
	std::cout << "reconducted fraction : " << (float)nReconductedSpecimens / (float)nSpecimens << std::endl;

	// Compute probabilities for roulette wheel selection.
	float invProbaSum = 0.0f;
	for (int i = 0; i < nSpecimens; i++) {
		invProbaSum += phase2Probabilities[i];
	}
	invProbaSum = 1.0f / invProbaSum;

	phase2Probabilities[0] = phase2Probabilities[0] * invProbaSum;
	for (int i = 1; i < nSpecimens; i++) {
		phase2Probabilities[i] = phase2Probabilities[i - 1] + phase2Probabilities[i] * invProbaSum;
	}

	int parentID;
	for (int i = nReconductedSpecimens; i < nSpecimens; i++) {
		parentID = binarySearch(phase2Probabilities, UNIFORM_01, nSpecimens);
		doEverything(parentID, i);
	}


	// Clean up and update
	for (int i = 0; i < nSpecimens; i++) {
		delete networks[i];
	}

	for (int i = 0; i < nSpecimens; i++) {
		networks[i] = tempNetworks[i];
	}

	// d is used because for the MAX_MATING_DEPTH-1 first steps we cant traverse the deeper layers
	// of the tree. 2 is the stable value.
	int d = 2 + std::max(0, MAX_MATING_DEPTH - evolutionStep - 1);
	for (int i = parentPhylogeneticTreeID; i < parentPhylogeneticTreeID + nSpecimens; i++) {
		if (phylogeneticTree[i].children.size() == 0) {
			phylogeneticTree[i].erase(d);
		}
	}

	delete[] tempNetworks;
	delete[] phase1Probabilities;
	delete[] phase2Probabilities;
	
	evolutionStep++;

	uint64_t stop = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
	std::cout << "Offspring creation took " << stop - start << " ms." << std::endl;
}
