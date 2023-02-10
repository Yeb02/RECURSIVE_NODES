#pragma once

#include "Population.h"

#include "Random.h"

Population::Population(int IN_SIZE, int OUT_SIZE, int N_SPECIMENS) :
	N_SPECIMENS(N_SPECIMENS)
{
	for (int i = 0; i < N_SPECIMENS; i++) {
		networks[i] = new Network(IN_SIZE, OUT_SIZE);
	}
}

Population::~Population() {
	for (const Network* n : networks) {
		delete n;
	}
}

void Population::step(std::vector<Trial*> trials) {

	std::vector<float> scores;
	scores.resize(N_SPECIMENS * trials.size());

	// evaluate on trials
	for (int i = 0; i < N_SPECIMENS; i++) {
		for (int j = 0; j < trials.size(); j++) {
			trials[j]->reset();
			networks[i]->intertrialReset();
			while (!trials[j]->isTrialOver) {
				networks[i]->step(trials[j]->observations);
			}
			scores[i] += trials[j]->score;
		}
	}

	// pick parents

	// breed

	// mutate
}