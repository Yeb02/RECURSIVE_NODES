#include "Random.h"



// I am unsure of the correctness of this process...
thread_local std::chrono::system_clock::rep seed = std::chrono::system_clock::now().time_since_epoch().count();
thread_local std::default_random_engine generator((unsigned int)seed);

thread_local std::uniform_real_distribution<float> Udistribution(0.0, 1.0);
thread_local std::uniform_int_distribution<uint32_t> UIdistribution(0, UINT32_MAX);
thread_local std::normal_distribution<float> Ndistribution(0.0, 1.0);
thread_local std::binomial_distribution<int> Bdistribution(3.0, .1); // initializing p at 0 causes div by 0.