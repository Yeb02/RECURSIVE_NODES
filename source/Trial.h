#pragma once

#include <vector>
#include "Random.h"
// The base virtual class which any trial should inherit from. 
// The score attribute must be a positive measure of the success of the run.
class Trial {

public:
	// Given the actions of the network, proceeds one step forward in the trial.
	virtual void step(const float* actions) = 0;

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
	void step(const float* actions) override;
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
	void step(const float* actions) override;
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
	void step(const float* actions) override;
	void reset(bool sameSeed = false) override;
	void copy(Trial* t) override;
	Trial* clone() override;
	void outerLoopUpdate(void* data) override {
		switchesSide = *static_cast<bool*>(data);
	};

	static const int corridorLength = 5;
	static const int nInferencesBetweenEnvSteps = 3;

private:
	// the network performs nInferencesBetweenEnvSteps inferences between each step of the environment,
	// to give enough time for information to flow through.
	int inferenceStep;

	// to be set in main.cpp's loop at each step.
	bool switchesSide;

	void subTrialReset();

	bool wentLeft;
	int nSubTrials;
};


// Observation are: [cartX, cosTheta1 ,sinTheta1, cosTheta2, ... sinThetaNLinks)], where cosines and sines  
// are with respect to the global axis. The zero radians is the trigonometric standard, horizontal right.
// The network is not fed the speed of the arms. It model will have to infer it by itself. For this purpose,
// the "DERIVATOR" simple neuron was added. 
class NLinksPendulumTrial : public Trial {

public:
	NLinksPendulumTrial(bool continuousControl, int nJoins);
	void step(const float* actions) override;
	void reset(bool sameSeed = false) override;
	void copy(Trial* t) override;
	Trial* clone() override;
	void outerLoopUpdate(void* data) override {};

	static const int STEP_LIMIT = 1000;

private:
	bool continuousControl;
	int nLinks;

	// initial values
	float x0;
	std::unique_ptr<float[]> thetas0;

	// Redundant but useful
	std::unique_ptr<float[]> thetas;

	// Cartesian state
	std::unique_ptr<float[]> xs, vxs, ys, vys;

	// Positions at previous step.
	std::unique_ptr<float[]> pxs, pys;
};


// The trial is split in two phases. In the first one, the network is presented with a set of 
// motif-response pairs. In the second phase, the network is presented with motifs from the first
// step, but the responses are set to 0. The task of the network is to output the associated response 
// for each motif. The parameter binary of the constructor determines whether the motif-response pair 
// should be binary {-1,1}, or continuous [-1,1]. 
class MemoryTrial : public Trial {

public:
	MemoryTrial(int nMotifs, int motifSize, int responseSize, bool binary=true);
	void step(const float* actions) override;
	void reset(bool sameSeed = false) override;
	void copy(Trial* t) override;
	Trial* clone() override;
	void outerLoopUpdate(void* data) override {};


private:
	int nMotifs, motifSize, responseSize;

	bool binary;

	// matrix of size nMotifs * (motifSize+responseSize)
	std::unique_ptr<float[]> motifResponsePairs;
};