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

	RECURSIVE_NODES_API Population* create_population(int IN_SIZE, int OUT_SIZE, int nSpecimens)
	{
		Population* p = new Population(IN_SIZE, OUT_SIZE, nSpecimens, true);
		//void* p2 = reinterpret_cast<void*>(p);
		return p;
	}
	RECURSIVE_NODES_API void destroy_population(Population* population)
	{
		delete population;
	}
	RECURSIVE_NODES_API void compute_fitnesses(Population* population, float* arrayPtr) {
		std::vector<float> avgScorePerSpecimen(arrayPtr, arrayPtr+population->get_nSpecimens());
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
	RECURSIVE_NODES_API void set_evolution_parameters(Population* population, float selectionPressure, float regularizationFactor, float nichingNorm) {
		PopulationEvolutionParameters params;
		params.selectionPressure.second = selectionPressure;
		params.regularizationFactor = regularizationFactor;
		population->setEvolutionParameters(params);
	}
	
	RECURSIVE_NODES_API void prepare_network(Network* n) {
		n->createPhenotype();
		n->preTrialReset();
	}
	RECURSIVE_NODES_API void end_trial(Network* n) {
		n->postTrialUpdate(1.0f, -1);
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
	RECURSIVE_NODES_API void draw_network(Drawer* drawer, Network* n, int step) {
		drawer->draw(n, step);
	}
#endif
}