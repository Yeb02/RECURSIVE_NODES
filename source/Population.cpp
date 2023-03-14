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
void normalizeVector(std::vector<float>& v) {
	float avg = 0.0f;
	for (int i = 0; i < v.size(); i++) {
		avg += v[i];
	}
	avg /= (float)v.size();
	float variance = 0.0f;
	for (int i = 0; i < v.size(); i++) {
		v[i] -= avg;
		variance += v[i] * v[i];
	}
	float InvStddev = 1.0f / sqrtf(variance / (float)v.size());
	if (abs(InvStddev) < .001f) return;
	for (int i = 0; i < v.size(); i++) {
		v[i] *= InvStddev;
	}
}
inline float L2Dist(float* a, float* b, int size) {
	float sum = 0.0f;
	float d;
	for (int i = 0; i < size; i++) {
		d = *(a + i) - *(b + i);
		sum += d * d;
	}
	return sqrtf(sum);
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
	N_SPECIMENS(N_SPECIMENS), N_THREADS(0), pScores(nullptr), iteration(0), mustTerminate(false)
{
	threads.reserve(0);
	globalTrials.reserve(0);
	networks.resize(N_SPECIMENS);
	fitnesses.resize(N_SPECIMENS);
	for (int i = 0; i < N_SPECIMENS; i++) {
		networks[i] = new Network(IN_SIZE, OUT_SIZE);
	}
	fittestSpecimen = 0;
	regularizationFactor = .03f;
	f0 = .2f;
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
	iteration = -1;
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

	int currentIteration = 0;
	while (true) {
		std::unique_lock<std::mutex> ul(m);
		startProcessing.wait(ul, [&currentIteration,this] {return (currentIteration == iteration) || mustTerminate; });
		if (mustTerminate) {
			nTerminated--;
			if (nTerminated == 0) {
				ul.unlock();
				hasTerminated.notify_one();
			}
			break;
		}
		ul.unlock();

		currentIteration++;

		// Copy init. Read only, so no mutex required.
		if (localTrials.size() != globalTrials.size()) {
			for (int i = 0; i < globalTrials.size(); i++) {
				localTrials.emplace_back(globalTrials[i]->clone());
			}
		}
		else {
			for (int i = 0; i < globalTrials.size(); i++) {
				localTrials[i]->copy(globalTrials[i]);
			}
		}

		// Heavy lifting.
		evaluate(i0, subArraySize, localTrials);
		
		ul.lock();
		nDoneProcessing--;
		if (nDoneProcessing == 0) {
			ul.unlock();
			doneProcessing.notify_one();
		}
	}
}

//networks[i]->mutate();   TODO UNCOMMENT WHEN THREAD SAFE RNG IS IMPLEMENTED
void Population::evaluate(const int i0, const int subArraySize, std::vector<std::unique_ptr<Trial>>& localTrials) {
	for (int i = i0; i < i0 + subArraySize; i++) {
		//networks[i]->mutate();   TODO UNCOMMENT WHEN THREAD SAFE RNG IS IMPLEMENTED
		networks[i]->intertrialReset(); // remove
		for (int j = 0; j < localTrials.size(); j++) {
			localTrials[j]->reset(true);
			//networks[i]->intertrialReset(); // add
			while (!localTrials[j]->isTrialOver) {
				networks[i]->step(localTrials[j]->observations);
				localTrials[j]->step(networks[i]->getOutput());
			}
			pScores[i * localTrials.size() + j] = localTrials[j]->score;
		}
	}
}

void Population::step(std::vector<std::unique_ptr<Trial>>& trials) {

	for (int i = 0; i < N_SPECIMENS; i++) networks[i]->mutate();  // TODO COMMENT WHEN THREAD SAFE RNG IS IMPLEMENTED
	
	// evaluate the specimens on trials
	std::vector<float> scores(trials.size() * N_SPECIMENS);
	pScores = scores.data();
	if (N_THREADS > 1) {
		globalTrials.resize(trials.size());
		for (int j = 0; j < trials.size(); j++) {
			trials[j]->reset();
			globalTrials[j] = trials[j].get();
		}

		// send msg to workers to start processing
		{
			std::lock_guard<std::mutex> lg(m);
			nDoneProcessing = N_THREADS;
			iteration++;
		}
		startProcessing.notify_all();

		// wait on workers.
		{
			std::unique_lock<std::mutex> lg(m);
			doneProcessing.wait(lg, [] {return nDoneProcessing == 0; });
		}
	}
	else {
		evaluate(0, N_SPECIMENS, trials);
	}

	// operation on scores
	std::vector<float> avgScorePerSpecimen(N_SPECIMENS);
	{
		std::vector<float> avgScoresPerTrial(trials.size());

		float maxScore = 0.0f;
		int maxScoreID = 0;
		float score, avgFactor=1.0f/(float) trials.size();
		for (int i = 0; i < N_SPECIMENS; i++) {
			score = 0;
			for (int j = 0; j < trials.size(); j++) { 
			//for (int j = trials.size() - 1; j < trials.size(); j++) { // all trials but the last are used for life-long learning
				score += scores[i * trials.size() + j];
				avgScoresPerTrial[j] += scores[i * trials.size() + j];
			}
			score *= avgFactor;
			avgScorePerSpecimen[i] = score;
			if (score > maxScore) {
				maxScore = score;
				maxScoreID = i;
			}
		}

		fittestSpecimen = maxScoreID;

		// LOGS, TODO REMOVE IN THE FINAL BUILD
		for (int j = 0; j < trials.size(); j++) avgScoresPerTrial[j] /= (float)N_SPECIMENS;
		float avgavgf = 0.0f;
		for (float f : avgScoresPerTrial) avgavgf += f;
		avgavgf /= trials.size();
		std::cout << "At iteration " << iteration 
			<< ", max score = " << avgScorePerSpecimen[fittestSpecimen]
			<< ", avg avg score = " << avgavgf << ".\n";


		normalizeVector(avgScorePerSpecimen);
	}

	computeFitnesses(avgScorePerSpecimen);

	createOffsprings();
	
}

void Population::computeFitnesses(std::vector<float> avgScorePerSpecimen) {

	// compute and normalize the regularization term
	std::vector<float> regularizationScore(N_SPECIMENS);
	for (int i = 0; i < N_SPECIMENS; i++) {
		regularizationScore[i] = networks[i]->getRegularizationLoss();
	}
	normalizeVector(regularizationScore);

	// Then, the fitness is simply a weighted sum of the 3 measures. 
	for (int i = 0; i < N_SPECIMENS; i++) {
		fitnesses[i] = avgScorePerSpecimen[i] - regularizationFactor * regularizationScore[i];
	}

}

void Population::createOffsprings() {
	float fitnessSum = 0.0f, fitnessMin=100000.0f;
	for (int i = 0; i < N_SPECIMENS; i++) {
		fitnessSum += fitnesses[i];
		if (fitnesses[i] < fitnessMin) fitnessMin = fitnesses[i];
	}
	fitnessMin -= f0;
	fitnessSum -= fitnessMin * (float)N_SPECIMENS;

	std::vector<float> probabilities(N_SPECIMENS);
	probabilities[0] = (fitnesses[0] - fitnessMin) / fitnessSum;
	for (int i = 1; i < N_SPECIMENS; i++) {
		probabilities[i] = probabilities[i - 1] + (fitnesses[i] - fitnessMin) / fitnessSum;
	}

	bool updated = false;
	int parentID;
	std::vector<Network*> tempNetworks(N_SPECIMENS);
	for (int i = 0; i < N_SPECIMENS; i++) {
		parentID = binarySearch(probabilities, UNIFORM_01);
		tempNetworks[i] = new Network(networks[parentID]);
		if (parentID == fittestSpecimen && !updated) {
			fittestSpecimen = i; updated = true;
		}
	}

	for (int i = 0; i < N_SPECIMENS; i++) {
		delete networks[i];
		networks[i] = tempNetworks[i];
	}
}
