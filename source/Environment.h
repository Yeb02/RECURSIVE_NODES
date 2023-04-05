#pragma once

#include "Network.h"

struct GridAgent {
	Network* n;
	std::vector<int> position;
	int age;
	float energy;
};

// A grid-structured hypercube, roamed by a host of agents. Each agent sees the grid slots directly adjacent to
// it (so 2 * dimension), and the slot it is in.
class Environment {

public :
	Environment(int s, int d);
	void reset();
	void step();

private:
	int s, d;
	int paintDimension;
	std::unique_ptr<float*> grid;

	// A two way linked list would be a more intuitive representation, but is less efficient.
	std::vector<GridAgent*> agents;

};