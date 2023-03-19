#pragma once

#include <vector>
#include "Random.h"
// The base virtual class which any trial should inherit from. 
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

	// Handle for updates coming from the outer loop (main.cpp 's loop)
	virtual void outerLoopUpdate(void* data) = 0;

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
the network's output is used. There are 2^(2*vSize) different trials possible. 
Note that be it with or without CONTINUOUS_LEARNING, lifelong learning on this task is completely useless.
Therefore, it can be seen as a robustness test, a benchmark on catastrophic forgetting (even though
what is to be "remembered" was not learned during the lifetime, but a genetic trait).
*/
class XorTrial : public Trial {

public:
	// Required network sizes: input = vectorSize, output = vectorSize.
	XorTrial(int vectorSize, int delay);
	void step(const std::vector<float>& actions) override;
	void reset(bool sameSeed = false) override;
	void copy(Trial* t) override;
	Trial* clone() override;
	void outerLoopUpdate(void* data) override {};

private:
	int vSize;
	int delay;
	std::vector<bool> v1, v2, v1_xor_v2;
};


// Classic CartPole, adapted from the python version of 
// https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py
class CartPoleTrial : public Trial {

public:
	CartPoleTrial(bool continuousControl);
	void step(const std::vector<float>& actions) override;
	void reset(bool sameSeed = false) override;
	void copy(Trial* t) override;
	Trial* clone() override;
	void outerLoopUpdate(void* data) override {};

	// or 30000... Gym's baseline is either 200 or 500, which is quite short with tau=0.02.
	static const int STEP_LIMIT = 1000; 

private:
	bool continuousControl;
	float x, xDot, theta, thetaDot;
	float x0, xDot0, theta0, thetaDot0;
};



// as per Soltoggio et al. (2008)
class TMazeTrial : public Trial{
public:
	TMazeTrial(bool switchesSide);
	void step(const std::vector<float>& actions) override;
	void reset(bool sameSeed = false) override;
	void copy(Trial* t) override;
	Trial* clone() override;
	void outerLoopUpdate(void* data) override {
		switchesSide = static_cast<int*>(data)[0];
	};

	static const int corridorLength = 5;

private:
	// to be set in main.cpp's loop at each step.
	bool switchesSide;
	void subTrialReset();

	bool wentLeft;
	int nSubTrials;
};


// The goal of this trial is to evaluate lifelong learning capabilities. It does so by creating a polynome
// of a given degree with random coefficients, and compares the output of the network with the polynome's evaluation 
// at any time t (t in [-1,1], = 2*nCurrentSteps/nSteps  - 1). Importantly, all trials at a given time step must have
// the same parameters (they are exactly the same). The only thing changing between evaluations is the network itself 
// through dynamic learning. Setting common parameters is done by calling outerLoopUpdate in main's loop, as intended. 
// The network's input is the degree+1 parameters and t (so degree + 2 input size). Polynom's evaluation are clipped in [-1,1]
class PolynomeTrial : public Trial {

public:
	PolynomeTrial(int nSteps, int degree);
	void step(const std::vector<float>& actions) override;
	void reset(bool sameSeed = false) override;
	void copy(Trial* t) override;
	Trial* clone() override;

	// WARNING DATA MUST BE THE SAME SIZE AS PARAMETERS ! Copies data into its own parameters. 
	void outerLoopUpdate(void* data)  {
		float* baseParameters = static_cast<float*>(data);
		setParameters(baseParameters);
	};

private:
	void setParameters(float* p) {
		for (int i = 0; i < degree + 1; i++) {
			parameters[i] = p[i];
		}
		computeValues();
	}

	int nSteps, degree;
	std::vector<float> parameters;

	// For efficiency, the polynom's value are computed once, at parameter's setting.
	std::vector<float> evaluatedValues;
	void computeValues();
};