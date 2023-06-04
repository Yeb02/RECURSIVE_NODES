#pragma once

#include <random>
#include <chrono>

extern thread_local std::chrono::system_clock::rep seed;
extern thread_local std::default_random_engine generator;
extern thread_local std::uniform_real_distribution<float> Udistribution;
extern thread_local std::uniform_int_distribution<uint32_t> UIdistribution;
extern thread_local std::normal_distribution<float> Ndistribution;
extern thread_local std::binomial_distribution<int> Bdistribution;

#define UNIFORM_01 Udistribution(generator)
#define NORMAL_01 Ndistribution(generator)
#define BINOMIAL Bdistribution(generator)
#define SET_BINOMIAL(n, p) Bdistribution.param(std::binomial_distribution<int>::param_type(n, p))


// yeah .... see Notes of https://en.cppreference.com/w/cpp/numeric/random/uniform_real_distribution
//#define INT_0X(x) (int)(Udistribution(generator) * (float)(x)-.000000000001f) 
// This is blazing fast:

// Random integer in [0, x-1]
#define INT_0X(x) (int)(((uint64_t)UIdistribution(generator) * (x)) >> 32)
