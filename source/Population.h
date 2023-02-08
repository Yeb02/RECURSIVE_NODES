#pragma once

#include "Network.h"
#include "Trial.h"

class Population {

public:
	Population(int IN_SIZE, int OUT_SIZE, int N_SPECIMENS);
	~Population();
	void step(std::vector<Trial*>);

private:
	int N_SPECIMENS;
	std::vector<Network*> networks;
	void mutate();
	void evaluate();
};