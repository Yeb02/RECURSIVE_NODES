#pragma once

#include <random>

extern std::default_random_engine generator;
extern std::uniform_real_distribution<float> Udistribution;
extern std::normal_distribution<float> Ndistribution;

#define UNIFORM_01 Udistribution(generator)
#define NORMAL_01 Ndistribution(generator)
