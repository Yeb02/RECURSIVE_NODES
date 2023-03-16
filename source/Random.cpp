#include "Random.h"

std::default_random_engine generator;
std::uniform_real_distribution<float> Udistribution(0.0f, 1.0f);
std::uniform_int_distribution<uint32_t> UIdistribution(0, UINT32_MAX);
std::normal_distribution<float> Ndistribution(0.0, 1.0);
std::binomial_distribution<int> Bdistribution(1, .0f);