#pragma once

#include "Population.h"

#define NOMINMAX
#include <windows.h>

#define RECURSIVE_NODES_API __declspec(dllexport)

#ifdef DRAWING
#include "Drawer.h"
#endif 


extern "C" {

	BOOL WINAPI DllMain(HINSTANCE hinstDLL, DWORD fdwReason, LPVOID lpvReserved);

	RECURSIVE_NODES_API Population* create_population(int IN_SIZE, int OUT_SIZE, int N_SPECIMENS)
	{
		Population* p = new Population(IN_SIZE, OUT_SIZE, N_SPECIMENS);
		//void* p2 = reinterpret_cast<void*>(p);
		return p;
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
		n->createPhenotype();
		n->preTrialReset();
	}
	RECURSIVE_NODES_API void end_trial(Network* n) {
		n->postTrialUpdate(1.0f);
	}
	RECURSIVE_NODES_API void get_actions(Network* n, float* observations, float* actions) {
		std::vector<float> observationsV(observations, observations + n->inputSize);
		n->step(observationsV);
		float* actionsTemp = n->getOutput();
		std::copy(actionsTemp, actionsTemp+n->outputSize, actions);
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