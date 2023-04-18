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


// Util:
void normalizeArray(float* v, int size) {
	float avg = 0.0f;
	for (int i = 0; i < size; i++) {
		avg += v[i];
	}
	avg /= (float)size;
	float variance = 0.0f;
	for (int i = 0; i < size; i++) {
		v[i] -= avg;
		variance += v[i] * v[i];
	}
	if (variance < .001f) return;
	float InvStddev = 1.0f / sqrtf(variance / (float) size);
	for (int i = 0; i < size; i++) {
		v[i] *= InvStddev;
	}
}

void convertRawToRank(float* v, int size) {
	std::vector<int> positions(size);
	for (int i = 0; i < size; i++) {
		positions[i] = i;
	}
	// sort position by ascending fitnesses.
	std::sort(positions.begin(), positions.end(), [v](int a, int b) -> bool
		{
			return v[a] < v[b];
		}
	);
	float invSize = 1.0f / (float)size;
	for (int i = 0; i < size; i++) {
		// linear in [-1,1], -1 for the worst specimen, 1 for the best
		float positionValue = (float)(2 * i - size) * invSize;
		// arbitrary, to make it a bit more selective. 
		positionValue = 4.0f * powf(positionValue * .8f, 3.0f);

		v[positions[i]] = positionValue;
	}
	return;
}

Population::Population(int IN_SIZE, int OUT_SIZE, int nSpecimens) :
	nSpecimens(nSpecimens), N_THREADS(0), pScores(nullptr), threadIteration(0), mustTerminate(false)
{
	threads.reserve(0);
	globalTrials.reserve(0);
	networks.resize(nSpecimens);
	fitnesses.resize(nSpecimens);
	for (int i = 0; i < nSpecimens; i++) {
		networks[i] = new Network(IN_SIZE, OUT_SIZE);
	}
	
	PopulationEvolutionParameters defaultParams;
	setEvolutionParameters(defaultParams);

	fittestSpecimen = 0;
	evolutionStep = 0;
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
			float* localScorePtr = pScores + i * nSpecimens;

			for (int j = i0; j < i0 + subArraySize; j++) {
				networks[j]->preTrialReset();
			}
			evaluate(i0, subArraySize, localTrials[i].get(), localScorePtr);


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
				networks[j]->postTrialUpdate(localScorePtr[j], i);
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

void Population::evaluate(const int i0, const int subArraySize, Trial* trial, float* scores) {
	for (int i = i0; i < i0 + subArraySize; i++) {
		networks[i]->preTrialReset();
		trial->reset(useSameTrialInit); // "true" to reduce (eliminate !) fitness function stochasticity
		while (!trial->isTrialOver) {
			networks[i]->step(trial->observations);
			trial->step(networks[i]->getOutput());
		}
		scores[i] = trial->score;		
	}
}

void Population::step(std::vector<std::unique_ptr<Trial>>& trials, int nTrialsEvaluated) {
	// utils
	int tSize = (int)trials.size();
	int i0 = tSize - nTrialsEvaluated;

	// Mutate, then evaluate the specimens on trials. Threaded operation if specified.
	std::vector<float> scores(tSize * nSpecimens);
	if (N_THREADS > 1) {

		// acquire pointers to this step's trials.
		globalTrials.resize(tSize);
		for (int j = 0; j < tSize; j++) {
			globalTrials[j] = trials[j].get();
		}

		pScores = scores.data();
		for (int i = 0; i < tSize + 1; i++) {
			
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

			// Non threaded operations on score array:
			if (i < tSize) {
				if (rankingFitness) {
					convertRawToRank(pScores + i * nSpecimens, nSpecimens);
				}
				else {
					normalizeArray(pScores + i * nSpecimens, nSpecimens);
				}
			}
		}
	}
	else {
		for (int i = 0; i < nSpecimens; i++) {
			networks[i]->mutate();
			networks[i]->createPhenotype();
		}

		for (int i = 0; i < tSize; i++) {
			pScores = &scores[i * nSpecimens];
			evaluate(0, nSpecimens, trials[i].get(), pScores);

			if (rankingFitness) {
				convertRawToRank(pScores, nSpecimens);
			}
			else {
				normalizeArray(pScores, nSpecimens);
			}

			for (int j = 0; j < nSpecimens; j++) {
				networks[j]->postTrialUpdate(pScores[j], i);
			}
		}
		pScores = scores.data();
	}


	// logging scores. monitoring only, can be disabled.
	if (true) {
		if (rankingFitness) {
			float maxScore = -1000000.0f, score;
			int maxScoreID = -1;
			for (int i = 0; i < nSpecimens; i++) {
				score = 0;
				for (int j = i0; j < tSize; j++) {
					score += scores[i + j * nSpecimens];
				}
				if (score > maxScore) {
					maxScore = score;
					maxScoreID = i;
				}
			}
			trials[0]->reset();
			Network* n = networks[maxScoreID];
			n->createPhenotype();
			n->preTrialReset();
			while (!trials[0]->isTrialOver) {
				n->step(trials[0]->observations);
				trials[0]->step(n->getOutput());
			}
			std::cout << "At generation " << evolutionStep
				<< ", best specimen's score on new trial = " << trials[0]->score << std::endl;
		}
		else {
			std::vector<float> avgScoresPerTrial(nTrialsEvaluated);
			float maxScore = -1000000.0f;
			float score, avgFactor = 1.0f / (float)nTrialsEvaluated;
			for (int i = 0; i < nSpecimens; i++) {
				score = 0;
				for (int j = i0; j < tSize; j++) {
					score += scores[i + j * nSpecimens];
					avgScoresPerTrial[j - i0] += scores[i + j * nSpecimens];
				}
				score *= avgFactor;
				if (score > maxScore) {
					maxScore = score;
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
		}
	}


	// Result-based niching, if nichingNorm > 0. Otherwise, simple average.
	std::vector<float> avgScorePerSpecimen(nSpecimens);
	if (nichingNorm > 0.0f && nTrialsEvaluated > 1) {
		float invNichingNorm = 1.0f/ nichingNorm;
		for (int i = 0; i < nSpecimens; i++) {
			float s = 0.0f;
			if (networks[i]->parentData.isAvailable && competitionFactor > 0.0f) {
				for (int j = i0; j < tSize; j++) {
					float _score = scores[i + j * nSpecimens] + competitionFactor *
						powf(
							std::clamp(networks[i]->parentData.scores[j] - scores[i + j * nSpecimens], -1.0f, 1.0f),
							3.0f
						);

					s += powf(_score, nichingNorm);
				}
			}
			else {
				for (int j = i0; j < tSize; j++) {
					s += powf(scores[i + j * nSpecimens], nichingNorm);
				}
			}
			
			// Dividing s by nTrialsEvaluated so that changing nTrialsEvaluated does not
			// influence too much other hyperparameters ?
			//avgScorePerSpecimen[i] = powf(s/(float) nTrialsEvaluated, invNichingNorm);
			avgScorePerSpecimen[i] = powf(s, invNichingNorm); 
		}
	} 
	else {
		for (int i = 0; i < nSpecimens; i++) {
			float s = 0.0f;
			if (networks[i]->parentData.isAvailable && competitionFactor > 0.0f) {
				for (int j = i0; j < tSize; j++) {
					s += scores[i + j * nSpecimens] + competitionFactor *
						powf(
							std::clamp(networks[i]->parentData.scores[j] - scores[i + j * nSpecimens], -1.0f, 1.0f),
							3.0f
						);
				}
			}
			else {
				for (int j = i0; j < tSize; j++) {
					s += scores[i + j * nSpecimens];
				}
			}
			avgScorePerSpecimen[i] = s;
		}
	}

	computeFitnesses(avgScorePerSpecimen);

	createOffsprings((int)trials.size());
	
	evolutionStep++;
}

void Population::computeFitnesses(std::vector<float> &avgScorePerSpecimen) {

	// compute and normalize the regularization term
	std::vector<float> regularizationScore(nSpecimens);
	for (int i = 0; i < nSpecimens; i++) {
		regularizationScore[i] = networks[i]->getRegularizationLoss();
	}
	normalizeArray(regularizationScore.data(), nSpecimens);


#ifdef SATURATION_PENALIZING
	// compute and normalize the saturation term
	std::vector<float> saturationScore(nSpecimens);
	for (int i = 0; i < nSpecimens; i++) {
		saturationScore[i] = networks[i]->getSaturationPenalization();
	}
	normalizeArray(saturationScore.data(), nSpecimens);
#endif


	// These happen twice if there is only one trial. Redundant 
	if (rankingFitness) {
		convertRawToRank(avgScorePerSpecimen.data(), nSpecimens); 
	}
	else {
		normalizeArray(avgScorePerSpecimen.data(), nSpecimens);
	}



	// Then, the fitness is simply a weighted sum of the 2 intermediate measures, score and regularization.
	float fMax = -10000.0f;
	for (int i = 0; i < nSpecimens; i++) {
		fitnesses[i] = avgScorePerSpecimen[i] - regularizationFactor * regularizationScore[i];
#ifdef SATURATION_PENALIZING
		fitnesses[i] -= saturationFactor * saturationScore[i];
#endif

		if (fitnesses[i] > fMax) {
			fMax = fitnesses[i];
			fittestSpecimen = i;
		}
	}
}

void Population::createOffsprings(int nTrials) {
	std::vector<float> probabilities(nSpecimens);
	std::vector<Network*> tempNetworks(nSpecimens);

	// already happens in computeFitnesses...
	float fMax = -10000.0f;
	for (int i = 0; i < nSpecimens; i++) {
		if (fitnesses[i] > fMax) {
			fMax = fitnesses[i];
		}
	}

	float normalizationFactor = 1.0f / (fMax - selectionPressure.first);
	for (int i = 0; i < nSpecimens; i++) {
		probabilities[i] = (fitnesses[i] - selectionPressure.first)*normalizationFactor;
		if (fitnesses[i] < selectionPressure.second) fitnesses[i] = 0.0f;
		else fitnesses[i] -= selectionPressure.second;
	}

	auto setParentData = [this, nTrials](int parentID, Network* offspring) {
		if (pScores != nullptr) { // should be true if and only if running the exe and not the dll, for now. TODO
			offspring->parentData.isAvailable = true;
			offspring->parentData.scoreSize = nTrials;
			offspring->parentData.scores = new float[nTrials];
			for (int i = 0; i < nTrials; i++) {
				offspring->parentData.scores[i] = pScores[i * nSpecimens + parentID];
			}
		}
		return;
	};

	int nReconductedSpecimens = 0;
	for (int i = 0; i < nSpecimens; i++) {
		if (UNIFORM_01 < probabilities[i]) {
			tempNetworks[nReconductedSpecimens] = new Network(networks[i]);
			setParentData(i, tempNetworks[nReconductedSpecimens]);
			nReconductedSpecimens++;
		}
	}

	// Compute probabilities for roulette wheel selection.
	float fitnessSum = 0.0f;
	for (int i = 0; i < nSpecimens; i++) {
		fitnessSum += fitnesses[i];
	}
	
	probabilities[0] = fitnesses[0] / fitnessSum;
	for (int i = 1; i < nSpecimens; i++) {
		probabilities[i] = probabilities[i - 1] + fitnesses[i] / fitnessSum;
	}

	int parentID;
	for (int i = nReconductedSpecimens; i < nSpecimens; i++) {
		parentID = binarySearch(probabilities, UNIFORM_01);
		tempNetworks[i] = new Network(networks[parentID]);
		setParentData(parentID, tempNetworks[i]);

		if (parentID == fittestSpecimen) {
			fittestSpecimen = i; 
		}
	}


	// Clean up and update
	for (int i = 0; i < nSpecimens; i++) {
		delete networks[i];
	}

	for (int i = 0; i < nSpecimens; i++) {
		networks[i] = tempNetworks[i];
	}
}
