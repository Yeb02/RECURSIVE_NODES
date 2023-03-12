#pragma once

#include <vector>
#include "Random.h"
// The base class which any trial should inherit from. Each derived class must implement
// a copy constructor that takes a pointer to an instance of the derived class, and return
// a trial that has the same initialisation.
// The score attribute must be a positive measure of the success of the run.
class Trial {

public:
	// given the actions of the network, proceeds one step forward in the trial
	virtual void step(const std::vector<float>& actions) = 0;

	// To be called at the end of the trial, AFTER fetching the score !
	// When sameSeed is true, the random values are kept between runs.
	virtual void reset(bool sameSeed = false) = 0;

	// copies the constant and per-run parameters of t. Must cast to derived class:
	// DerivedTrial* t = dynamic_cast<DerivedTrial*>(t0);
	virtual void copy(Trial* t0) = 0;

	// returns a pointer to a new instance OF THE DERIVED CLASS, cast to a pointer of the base class.
	virtual Trial* clone() = 0;

	std::vector<float> observations;

	// the required network dimensions
	int netInSize, netOutSize;

	float score;

	virtual ~Trial() = default; // otherwise derived destructors will not be called.

	bool isTrialOver;

protected:

	// the current elapsed steps in the trial. To be set to 0 in reset.
	int currentNStep;
};



/* v1 and v2 are binary vectors (-1 or 1), randomly initialized. The trial is split in 3 phases. In the first one,
the observation is the vector v1. In the second, it is v2. In the third, the observation is a vector of 0s.
During the last phase, the expected output of the network is the termwise XOR of v1 and v2. Only the sign of
the network's output is used. There are 2^(2*vSize) different trials possible. */
class XorTrial : public Trial {

public:
	// Required network sizes: input = vectorSize, output = vectorSize.
	XorTrial(int vectorSize);
	void step(const std::vector<float>& actions) override;
	void reset(bool sameSeed = false) override;
	void copy(Trial* t) override;
	Trial* clone() override;

private:
	int vSize;
	std::vector<bool> v1, v2, v1_xor_v2;
};

// Classic CartPole, adapted from the python version of 
// https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py
class CartPoleTrial : public Trial {

public:
	// Required network sizes: input = vectorSize, output = vectorSize.
	CartPoleTrial();
	void step(const std::vector<float>& actions) override;
	void reset(bool sameSeed = false) override;
	void copy(Trial* t) override;
	Trial* clone() override;

	// or 30000... Gym's baseline is either 200 or 500, which is quite short with tau=0.02.
	static const int STEP_LIMIT = 1000; 

private:
	float x, xDot, theta, thetaDot;
	float x0, xDot0, theta0, thetaDot0;
};

