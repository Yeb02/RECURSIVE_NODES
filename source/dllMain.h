#pragma once

#include "Population.h"

#define NOMINMAX
#include <windows.h>

#define RECURSIVE_NODES_API __declspec(dllexport)

////////////////////////////////////
///// USER COMPILATION CHOICES /////
////////////////////////////////////

// Comment or uncomment the preprocessor directives to compile versions of the code
// Or use the -D flag.

// Draws a specimen at each step, using SFML. Requires the appropriate DLLs 
// alongside the generated executable, details in readme.md .
#define DRAWING 

// Should be on if there is just 1 trial, or no trials at all. Could be on even if there are multiple trials, 
// but it disables the intertrial update of wLifetime. Not recommended in his case.
// Enable and disable it in Genotype.h, it does not do anything here !
//#define CONTINUOUS_LEARNING

////////////////////////////////////
////////////////////////////////////

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
		n->intertrialReset();
	}
	RECURSIVE_NODES_API void get_actions(Network* n, float* observations, float* actions) {
		std::vector<float> observationsV(observations, observations + n->inputSize);
		n->step(observationsV);
		std::vector<float> actionsV = n->getOutput();
		memcpy(actions, actionsV.data(), actionsV.size());
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