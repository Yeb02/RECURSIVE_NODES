#pragma once

#include <random>
#include <functional> // std::bind

#include "Population.h"

std::default_random_engine generator;
std::uniform_real_distribution<float> Udistribution(0, 1);
std::normal_distribution<float> Ndistribution(0.0, 1.0);
auto uniform01 = std::bind(Udistribution, generator);
auto normal01 = std::bind(Ndistribution, generator);

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
	scores.resize(N_SPECIMENS);

	// evaluate on trials
	for (int i = 0; i < N_SPECIMENS; i++) {
		for (int j = 0; j < trials.size(); j++) {
			trials[j]->reset();
			networks[i]->intertrialReset();
			while (!trials[j]->isTrialOver) {
				networks[i]->step(trials[j]->observations);
			}
		}
	}

	// pick parents

	// breed

	// mutate
}