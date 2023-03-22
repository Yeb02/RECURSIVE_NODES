#pragma once

#include "Network.h"

struct GridAgent {
	Network n;
	int x, y;
};

class Grid {

public :
	Grid(int, int, int);
	void reset();
	void step();

private:
	int w, h;
	int paintDimension;
	std::unique_ptr<float*> canvas;

};