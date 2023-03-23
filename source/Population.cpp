#pragma once

#include <iostream>
#include <mutex>
#include <condition_variable>

#include "Population.h"
#include "Random.h"


#ifdef KMEANS
// KMeans implementation: https://github.com/genbattle/dkm.git
#include "dkm.h"
#endif

std::mutex m;
std::condition_variable startProcessing;
std::condition_variable doneProcessing;
std::condition_variable hasTerminated;
bool bStartProcessing = false;
int nTerminated = 0;
int nDoneProcessing = 0;


// Utils:
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
	float InvStddev = 1.0f / sqrtf(variance / (float) size);
	if (abs(InvStddev) < .001f) return;
	for (int i = 0; i < size; i++) {
		v[i] *= InvStddev;
	}
}
int binarySearch(std::vector<float>& proba, float value) {
	int inf = 0;
	int sup = (int) proba.size() - 1;

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


Population::Population(int IN_SIZE, int OUT_SIZE, int N_SPECIMENS) :
	N_SPECIMENS(N_SPECIMENS), N_THREADS(0), pScores(nullptr), threadIteration(0), mustTerminate(false)
{
	threads.reserve(0);
	globalTrials.reserve(0);
	networks.resize(N_SPECIMENS);
	fitnesses.resize(N_SPECIMENS);
	for (int i = 0; i < N_SPECIMENS; i++) {
		networks[i] = new Network(IN_SIZE, OUT_SIZE);
	}
	fittestSpecimen = 0;
	regularizationFactor = .1f;
	selectionPressure = .0f;
	nichingNorm = 1.0f;
	useSameTrialInit = false;

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

}

void Population::startThreads(int N_THREADS) {
	stopThreads();

	mustTerminate = false;
	this->N_THREADS = N_THREADS;
	threads.resize(0); // to kill all previously existing threads.

	if (N_THREADS < 2) return;

	threads.reserve(N_THREADS);
	threadIteration = -1;
	int subArraySize = N_SPECIMENS / N_THREADS;
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
			float* localScorePtr = pScores + i * N_SPECIMENS;

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
				networks[j]->postTrialUpdate(localScorePtr[j]);
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

	// Mutate, then evaluate the specimens on trials:

	bool normalizedScoreGradients = false; // experimental, default=false. Only used with CONTINUOUS_LEARNING && GUIDED_MUTATIONS
	std::vector<float> scores(tSize * N_SPECIMENS);
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
			if (i < tSize && normalizedScoreGradients) {
				normalizeArray(pScores + i * N_SPECIMENS, N_SPECIMENS);
			}
		}
	}
	else {
		for (int i = 0; i < N_SPECIMENS; i++) {
			networks[i]->mutate();
			networks[i]->createPhenotype();
		}

		for (int i = 0; i < tSize; i++) {
			pScores = &scores[i * N_SPECIMENS];
			evaluate(0, N_SPECIMENS, trials[i].get(), pScores);

			if (normalizedScoreGradients) {
				normalizeArray(pScores, N_SPECIMENS);
			}

			for (int i = 0; i < N_SPECIMENS; i++) {
				networks[i]->postTrialUpdate(pScores[i]);
			}
		}
	}


	// LOGGING SCORES. MONITORING ONLY, CAN BE DISABLED.
	if (true) {
		if (normalizedScoreGradients) {
			float maxScore = -1000000.0f, score;
			int maxScoreID = -1;
			for (int i = 0; i < N_SPECIMENS; i++) {
				score = 0;
				for (int j = i0; j < tSize; j++) {
					score += scores[i + j * N_SPECIMENS];
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
			for (int i = 0; i < N_SPECIMENS; i++) {
				score = 0;
				for (int j = i0; j < tSize; j++) {
					score += scores[i + j * N_SPECIMENS];
					avgScoresPerTrial[j - i0] += scores[i + j * N_SPECIMENS];
				}
				score *= avgFactor;
				if (score > maxScore) {
					maxScore = score;
				}
			}
			for (int j = 0; j < nTrialsEvaluated; j++) avgScoresPerTrial[j] /= (float)N_SPECIMENS;
			float avgavgf = 0.0f;
			for (float f : avgScoresPerTrial) avgavgf += f;
			avgavgf /= nTrialsEvaluated;
			std::cout << "At generation " << evolutionStep
			<< ", max score = " << maxScore
			<< ", avg avg score = " << avgavgf << ".\n";
		//std::cout << maxScore << ", ";
		}
	}


	// Result-based niching, if nichingNorm > 0. Otherwise, simple sum and normalization.
	std::vector<float> avgScorePerSpecimen(N_SPECIMENS);
	if (nichingNorm > 0.0f) {
		std::vector<float> sums, mins;
		sums.resize(nTrialsEvaluated);
		mins.resize(nTrialsEvaluated); // 0 init is fine cause there WILL be values < 0.
		for (int j = i0; j < tSize; j++) {
			for (int i = 0; i < N_SPECIMENS; i++) {
				sums[j - i0] += scores[i + j * N_SPECIMENS];
				if (scores[i + j * N_SPECIMENS] < mins[j - i0]) {
					mins[j - i0] = scores[i + j * N_SPECIMENS];
				}
			}
		}

		for (int i = 0; i < nTrialsEvaluated; i++) {
			sums[i] -= mins[i] * (float)N_SPECIMENS;
			sums[i] = 1.0f / sums[i];
		}

		float invNichingNorm = 1.0f/ nichingNorm;
		for (int i = 0; i < N_SPECIMENS; i++) {
			float s = 0.0f;
			for (int j = i0; j < tSize; j++) {
				s += powf((scores[i + j * N_SPECIMENS] - mins[j-i0]) * sums[j-i0], nichingNorm);
			}
			// Dividing s by nTrialsEvaluated so that changing nTrialsEvaluated does not
			// influence too much other hyperparameters.
			avgScorePerSpecimen[i] = powf(s/(float) nTrialsEvaluated, invNichingNorm);
		}
	} 
	else {
		for (int j = i0; j < tSize; j++) {
			for (int i = 0; i < N_SPECIMENS; i++) {
				avgScorePerSpecimen[i] += scores[i + j * N_SPECIMENS];
			}
		}
	}

	normalizeArray(avgScorePerSpecimen.data(), N_SPECIMENS); // TODO, check if necessary.


	computeFitnesses(avgScorePerSpecimen);

	createOffsprings();
	
	evolutionStep++;
}

void Population::computeFitnesses(std::vector<float> avgScorePerSpecimen) {

	// compute and normalize the regularization term
	std::vector<float> regularizationScore(N_SPECIMENS);
	for (int i = 0; i < N_SPECIMENS; i++) {
		regularizationScore[i] = networks[i]->getRegularizationLoss();
	}
	normalizeArray(regularizationScore.data(), N_SPECIMENS);


#ifdef SATURATION_PENALIZING
	// compute and normalize the saturation term
	std::vector<float> saturationScore(N_SPECIMENS);
	for (int i = 0; i < N_SPECIMENS; i++) {
		saturationScore[i] = networks[i]->getSaturationPenalization();
	}
	normalizeArray(saturationScore.data(), N_SPECIMENS);
#endif


	// Then, the fitness is simply a weighted sum of the 2 intermediate measures, score and regularization.
	float fMax = -10000.0f;
	for (int i = 0; i < N_SPECIMENS; i++) {
		fitnesses[i] = avgScorePerSpecimen[i] - regularizationFactor * regularizationScore[i];
#ifdef SATURATION_PENALIZING
		fitnesses[i] += -.05f * saturationScore[i];
#endif

		if (fitnesses[i] > fMax) {
			fMax = fitnesses[i];
			fittestSpecimen = i;
		}
		if (fitnesses[i] < selectionPressure) fitnesses[i] = 0.0f;
		else fitnesses[i] -= selectionPressure;
	}

	if (fMax == selectionPressure) {
		std::cerr <<
		"WARNING : selectionPressure TOO HIGH, ALL SPECIMENS REJECTED. < 1 RECOMMENDED FOR STABILITY, < 0 TO BE SURE."
		<< std::endl;

		fitnesses[fittestSpecimen] = 1.0f;
	}

}

void Population::createOffsprings() {
	float fitnessSum = 0.0f;
	for (int i = 0; i < N_SPECIMENS; i++) {
		fitnessSum += fitnesses[i];
	}

	std::vector<float> probabilities(N_SPECIMENS);
	probabilities[0] = fitnesses[0] / fitnessSum;
	for (int i = 1; i < N_SPECIMENS; i++) {
		probabilities[i] = probabilities[i - 1] + fitnesses[i] / fitnessSum;
	}

	bool updated = false;  // we keep track of one of the fittest specimen's clones.
	int parentID;
	std::vector<Network*> tempNetworks(N_SPECIMENS);
	for (int i = 0; i < N_SPECIMENS; i++) {
		parentID = binarySearch(probabilities, UNIFORM_01);
		tempNetworks[i] = new Network(networks[parentID]);
		if (parentID == fittestSpecimen && !updated) {
			fittestSpecimen = i; 
			updated = true;
		}
	}

	for (int i = 0; i < N_SPECIMENS; i++) {
		delete networks[i];
		networks[i] = tempNetworks[i];
	}
}
