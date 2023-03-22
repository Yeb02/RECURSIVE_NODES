#include "Random.h"

thread_local std::default_random_engine generator;
thread_local std::uniform_real_distribution<float> Udistribution(0.0f, 1.0f);
thread_local std::uniform_int_distribution<uint32_t> UIdistribution(0, UINT32_MAX);
thread_local std::normal_distribution<float> Ndistribution(0.0, 1.0);
thread_local std::binomial_distribution<int> Bdistribution(1, .0f);