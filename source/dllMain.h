#pragma once

#include "Population.h"

#define NOMINMAX
#include <windows.h>

#define RECURSIVE_NODES_API __declspec(dllexport)

// optional:
#define DRAWING
#ifdef DRAWING
#include "Drawer.h"
#endif


extern "C" {

	BOOL WINAPI DllMain(HINSTANCE hinstDLL, DWORD fdwReason, LPVOID lpvReserved);

	RECURSIVE_NODES_API Population* create_population(int IN_SIZE, int OUT_SIZE, int N_SPECIMENS)
	{
		return new Population(IN_SIZE, OUT_SIZE, N_SPECIMENS);
	}
	RECURSIVE_NODES_API void destroy_population(Population* population)
	{
		delete population;
	}
	RECURSIVE_NODES_API void compute_fitnesses(Population* population, float* arrayPtr) {
		std::vector<float> avgScorePerSpecimen(arrayPtr, arrayPtr+population->get_N_SPECIMENS());
		population->computeFitnesses(avgScorePerSpecimen);
	}
	RECURSIVE_NODES_API void create_offsprings(Population* population) {
		population->createOffsprings();
	}
	RECURSIVE_NODES_API void mutate_population(Population* population) {
		population->mutatePopulation();
	}
	RECURSIVE_NODES_API Network* get_network_handle(Population* population, int i) {
		return population->getSpecimenPointer(i);
	}
	RECURSIVE_NODES_API Network* get_fittest_network_handle(Population* population) {
		return population->getSpecimenPointer(population->fittestSpecimen);
	}
	RECURSIVE_NODES_API void set_evolution_parameters(Population* population, float f0, float regularizationFactor) {
		population->setEvolutionParameters(f0, regularizationFactor);
	}
	
	RECURSIVE_NODES_API void prepare_network(Network* n) {
		n->intertrialReset();
	}
	RECURSIVE_NODES_API float* get_actions(Network* n, float* observations) {
		std::vector<float> observationsV(observations, observations + n->inputSize);
		n->step(observationsV);
		std::vector<float> actionsV = n->getOutput();
		float* actions = new float[actionsV.size()];
		memcpy(actions, actionsV.data(), actionsV.size());
		return actions;
	}

#ifdef DRAWING
	RECURSIVE_NODES_API Drawer* initialize_drawer(int w, int h) {
		return new Drawer(w, h);
	}
	RECURSIVE_NODES_API void draw_network(Drawer* drawer, Network* n) {
		drawer->draw(n);
	}
#endif
}