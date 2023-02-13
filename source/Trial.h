#pragma once

#include <vector>

// The base class which any trial should inherit from.
// The score must be a positive measure of the success of the run.
class Trial {

public:

	// given the actions of the network, proceeds one step forward in the trial
	virtual void step(std::vector<float> actions) = 0;

	// To be called at the end of the trial, AFTER fetching the score !
	// When sameSeed is true, the random values are kept between runs.
	virtual void reset(bool sameSeed = false) = 0;

	// the required network dimensions
	int netInSize, netOutSize;

	// the current elapsed steps in the trial. To be set to 0 in reset.
	int currentNStep;

	// the trial should end in less than STEP_LIMIT steps
	static const int STEP_LIMIT = 200;

	std::vector<float> observations;

	float score;

	bool isTrialOver;
};



/* The observation phase is split in 2 parts, in each of which the observation is a binary vector.
During the evaluation phase, the expected output is their XOR.*/
class XorTrial : public Trial {

public:
	// Required network sizes: input = vectorSize, output = vectorSize.
	XorTrial(int vectorSize);
	void step(std::vector<float> actions) override;
	void reset(bool sameSeed = false) override;

private:
	int vSize;
	std::vector<bool> v1, v2, v1_xor_v2;
};
