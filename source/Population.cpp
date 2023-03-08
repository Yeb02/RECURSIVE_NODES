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
	for (int i = 0; i < N_SPECIMENS; i++) {
		networks[i] = new Network(IN_SIZE, OUT_SIZE);
	}
	fittestSpecimen = 0;
}

void Population::stopThreads() {
	if (threads.size() == 0) return;

	{
		std::lock_guard<std::mutex> lg(m);
		mustTerminate = true;
		nTerminated = threads.size();
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
		threads.emplace_back(&Population::mutateNevaluateThreaded, this, i0, subArraySize);
	}
}

Population::~Population() {
	for (const Network* n : networks) {
		delete n;
	}
}

void Population::mutateNevaluateThreaded(const int i0, const int subArraySize) {
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
		

		for (int i = i0; i < i0 + subArraySize; i++) {
			networks[i]->mutate();
			for (int j = 0; j < localTrials.size(); j++) {
				localTrials[j]->reset(true);
				networks[i]->intertrialReset();
				while (!localTrials[j]->isTrialOver) {
					networks[i]->step(localTrials[j]->observations);
					localTrials[j]->step(networks[i]->getOutput());
				}
				pScores[i * localTrials.size() + j] = localTrials[j]->score;
			}
		}

		ul.lock();
		nDoneProcessing--;
		if (nDoneProcessing == 0) {
			ul.unlock();
			doneProcessing.notify_one();
		}
		// Not thread safe :
		//std::cout << "iteration " << currentIteration-1 << " from thread " << std::this_thread::get_id()  << " over." << std::endl;

	}
}

void Population::step(std::vector<Trial*> trials) {

	std::vector<float> scores(trials.size() * N_SPECIMENS);

	// mutate and evaluate the specimens on trials
	if (N_THREADS > 1) {
		pScores = scores.data();
		globalTrials.resize(trials.size());
		for (int j = 0; j < trials.size(); j++) {
			trials[j]->reset();
			globalTrials[j] = trials[j];
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

		// mutate
		for (int i = 0; i < N_SPECIMENS; i++) networks[i]->mutate();
		
		// evaluate on trials to get the normalized vector of scores.
		for (int i = 0; i < N_SPECIMENS; i++) {
			for (int j = 0; j < trials.size(); j++) {
				trials[j]->reset(true);
				networks[i]->intertrialReset();
				while (!trials[j]->isTrialOver) {
					networks[i]->step(trials[j]->observations);
					trials[j]->step(networks[i]->getOutput());
				}
				scores[i * trials.size() + j] = trials[j]->score;
			}
		}
	}

	// normalize scores
	{
		std::vector<float> avgScores(trials.size());

		float maxScore = 0.0f;
		int maxScoreID = 0;
		float score;
		for (int i = 0; i < N_SPECIMENS; i++) {
			score = 0;
			for (int j = 0; j < trials.size(); j++) {
				score += scores[i * trials.size() + j];
				avgScores[j] += scores[i * trials.size() + j];
			}
			if (score > maxScore) {
				maxScore = score;
				maxScoreID = i;
			}
		}
		fittestSpecimen = maxScoreID;
		for (int j = 0; j < trials.size(); j++) avgScores[j] /= (float)N_SPECIMENS;

		/*for (float f : avgScores) std::cout << f << " ";
		std::cout << std::endl;*/
		float avgavgf = 0.0f;
		for (float f : avgScores) avgavgf += f;
		avgavgf /= trials.size();
		std::cout << "At iteration " << iteration 
			<< ", max score = " << (int)(maxScore / (float)trials.size() )
			<< ", avg avg score = " << (int)avgavgf << ".\n";


		// Normalize scores
		float stddev;
		for (int j = 0; j < trials.size(); j++) {
			avgScores[j] /= (float) N_SPECIMENS;
			stddev = 0;
			float diff;
			for (int i = 0; i < N_SPECIMENS; i++) {
				diff = scores[i * trials.size() + j] - avgScores[j];
				stddev += diff * diff;
			}
			stddev = sqrtf(stddev/ (float) N_SPECIMENS);
			//assert(abs(stddev) > 0.00001f);

			if (abs(stddev) > 0.001f) {
				for (int i = 0; i < N_SPECIMENS; i++) {
					scores[i * trials.size() + j] = (scores[i * trials.size() + j] - avgScores[j]) / stddev;
				}
			} else {
				for (int i = 0; i < N_SPECIMENS; i++) {
					scores[i * trials.size() + j] = 0.0f;
				}
			}
		}
	}


#ifdef KMEANS
	// apply K-Means algorithm    TODO change dkm.h to use arrays, and pass K and max_iter.
	int k = sqrt(trials.size());
	dkm::clustering_parameters c(k);
	auto [centroids, ids] = dkm::kmeans_lloyd(scores, c);
#else  
	// estimate local density by random sampling.  
	std::vector<float> distances(N_SPECIMENS);
	{
		int nSamples = (int) sqrt(N_SPECIMENS);
		for (int i = 0; i < N_SPECIMENS; i++) {
			distances[i] = 0;
			int rID;
			for (int j = 0; j < nSamples; j++) {
				rID = (int)(UNIFORM_01 * (float)N_SPECIMENS);
				distances[i] += L2Dist(&scores[i], &scores[rID], (int)trials.size());
			}
			// no /nSamples here, because distances are normalized over all specimens.
		}
	}

#endif 
	

	// compute raw fitnesses 
	constexpr float scoreFactor = 1.0f, distanceFactor = .0f, regularizationFactor = .05f;
	std::vector<float> fitnesses(N_SPECIMENS);
	float fitnessSum, fitnessMin;
	{
		// compute and normalize the regularization term and the clustering term:

		std::vector<float> regularization(N_SPECIMENS);

#ifdef KMEANS
		// TODO remove this line and get it directly from the KMeans algorithm :
		std::vector<float> distances(N_SPECIMENS);
#endif 

		float sumr = 0, sumd = 0;
		for (int i = 0; i < N_SPECIMENS; i++) {
			regularization[i] = networks[i]->getRegularizationLoss();
			sumr += regularization[i];
			sumd += distances[i];
		}
		float avgr = sumr / (float) N_SPECIMENS;
		float avgd = sumd / (float) N_SPECIMENS;
		float stddevr = 0.0f, stddevd = 0.0f;
		for (int i = 0; i < N_SPECIMENS; i++) {
			stddevr += (regularization[i] - avgr) * (regularization[i] - avgr);
			stddevd += (distances[i] - avgd) * (distances[i] - avgd);
		}
		stddevr = sqrtf(stddevr / (float)N_SPECIMENS);
		stddevd = sqrtf(stddevd / (float)N_SPECIMENS);
		

		if (abs(stddevd) > 0.00001f) {
			for (int i = 0; i < N_SPECIMENS; i++) {
				distances[i] =(distances[i] - avgd) / stddevd;
			}
		} else {
			for (int i = 0; i < N_SPECIMENS; i++) {
				distances[i] = 0.0f;
			}
		}

		if (abs(stddevr) > 0.00001f) {
			for (int i = 0; i < N_SPECIMENS; i++) {
				regularization[i] = (regularization[i] - avgr) / stddevr;
			}
		}
		else {
			for (int i = 0; i < N_SPECIMENS; i++) {
				regularization[i] = 0.0f;
			}
		}

		// Then, the fitness is simply a weighted sum of the 3 measures. 
		fitnessSum = 0, fitnessMin = 100000.0f;
		for (int i = 0; i < N_SPECIMENS; i++) {
			float score = 0.0f;
			for (int j = 0; j < trials.size(); j++) score += scores[i*trials.size() + j];
			score /= (float) trials.size();

			fitnesses[i] = score * scoreFactor - regularizationFactor * regularization[i] + distanceFactor * distances[i];
			fitnessSum += fitnesses[i];
			if (fitnesses[i] < fitnessMin) fitnessMin = fitnesses[i];
		}
	}

	
	// The higher f0, the lower the selection pressure. When f0 = 0, the least fit individual has a probability of 0
	// to have children.
	//constexpr float f0 = 0.0f;
	//float f0 = 1.0f + UNIFORM_01 * 1.0f;
	float f0 = .2f + .2f*sinf((float)iteration/3.0f);
	// create offsprings 
	{
		//float f0 = .1f / (float) N_SPECIMENS; 
		fitnessMin -= f0;
		fitnessSum -= fitnessMin * (float)N_SPECIMENS;
		std::vector<float> probabilities(N_SPECIMENS);
		probabilities[0] = (fitnesses[0] - fitnessMin) / fitnessSum;
		for (int i = 1; i < N_SPECIMENS; i++) {
			probabilities[i] =  probabilities[i-1] + (fitnesses[i] - fitnessMin) / fitnessSum;
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
}
