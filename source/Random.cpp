#include "Random.h"

std::default_random_engine generator;
std::uniform_real_distribution<float> Udistribution(0.0f, 1.0f);
std::normal_distribution<float> Ndistribution(0.0, 1.0);