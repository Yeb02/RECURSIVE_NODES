#pragma once

#include <random>

extern std::default_random_engine generator;
extern std::uniform_real_distribution<float> Udistribution;
extern std::normal_distribution<float> Ndistribution;
//extern std::binomial_distribution<int> Bdistribution;

#define UNIFORM_01 Udistribution(generator)
#define NORMAL_01 Ndistribution(generator)
//#define BINOMIAL Bdistribution(generator)
//#define SET_BINOMIAL(n, p) Bdistribution.param(std::binomial_distribution<int>::param_type(n, p))
