#pragma once

#include "Trial.h"

#include "Random.h"


XorTrial::XorTrial(int vSize) :
	vSize(vSize)
{
	netInSize = 2 * vSize;
	netOutSize = vSize;

	v1.resize(vSize);
	v2.resize(vSize);
	v1_xor_v2.resize(vSize);
	observations.resize(2 * vSize);

	reset();
}

void XorTrial::reset() {
	currentNStep = 0;
	score = 0.0f;
	isTrialOver = false;

	for (int i = 0; i < vSize; i++) {
		v1[i] = UNIFORM_01 < .5;
		v2[i] = UNIFORM_01 < .5;
		v1_xor_v2[i] = v1[i] ^ v2[i];
		observations[i] = v1[i] ? 1.0f : -1.0f;
	}
}

void XorTrial::step(std::vector<float> actions) {
	constexpr int endV1Phase = 20;
	constexpr int endV2Phase = 40;
	constexpr int startResponsePhase = 50;
	constexpr int endResponsePhase = 60;

	
	if (currentNStep == endV1Phase) {
		for (int i = 0; i < vSize; i++)  observations[i] = v2[i] ? 1.0f : -1.0f;
	}

	if (currentNStep == endV2Phase) {
		for (int i = 0; i < vSize; i++)  observations[i] = 0.0f;
	}

	if (currentNStep < endResponsePhase && currentNStep >= startResponsePhase) {
		int z = 0;
		for (int i = 0; i < vSize; i++)  
			score += (float) (actions[i] > 0) == v1_xor_v2[i]; 
	}

	if (currentNStep >= endResponsePhase) {
		isTrialOver = true;

		// score normalization
		if (currentNStep == endResponsePhase) score /= (endResponsePhase - startResponsePhase) * vSize;
	}

	currentNStep++;
}
