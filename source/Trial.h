#pragma once

#include <vector>

// The base class which any trial should inherit from.
class Trial {

public:

	// given the actions of the network, proceeds one step forward in the trial
	virtual void step(std::vector<float> actions) = 0;

	// to be called at the end of the trial, AFTER fetching the score !
	virtual void reset() = 0;

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


// The observation phase is split in 2 parts, in each of which the observation is a binary vector.
// During the evaluation phase, the expected output is their XOR, where . Thus inSize = 2*outSize.
class XorTrial : public Trial {

public:
	XorTrial(int vectorSize);
	void step(std::vector<float> actions) override;
	void reset() override;

private:
	int vSize;
	std::vector<bool> v1, v2, v1_xor_v2;
};
