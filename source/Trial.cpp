#pragma once

#include "Trial.h"


XorTrial::XorTrial(int vSize) :
	vSize(vSize)
{
	netInSize = vSize;
	netOutSize = vSize;

	v1.resize(vSize);
	v2.resize(vSize);
	v1_xor_v2.resize(vSize);
	observations.resize(vSize);

	reset();
}

void XorTrial::reset(bool sameSeed) {
	currentNStep = 0;
	score = 0.0f;
	isTrialOver = false;

	if (!sameSeed) {
		for (int i = 0; i < vSize; i++) {
			v1[i] = UNIFORM_01 < .5;
			v2[i] = UNIFORM_01 < .5;
			v1_xor_v2[i] = v1[i] ^ v2[i];
			observations[i] = v1[i] ? 1.0f : -1.0f;
		}
	}
}

void XorTrial::step(const std::vector<float>& actions) {
	constexpr int endV1Phase = 10;
	constexpr int endV2Phase = 20;
	constexpr int startResponsePhase = 25;
	constexpr int endResponsePhase = 35;

	
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

		// score normalization, not necessary
		if (currentNStep == endResponsePhase) score /= (endResponsePhase - startResponsePhase) * vSize;
	}

	currentNStep++;
}

void XorTrial::copy(Trial* t0) {
	XorTrial* t = dynamic_cast<XorTrial*>(t0);
	vSize = t->vSize;
	netInSize = vSize;
	netOutSize = vSize;
	v1.resize(vSize);
	v2.resize(vSize);
	v1_xor_v2.resize(vSize);
	observations.resize(vSize);
	for (int i = 0; i < vSize; i++) {
		v1[i] = t->v1[i];
		v2[i] = t->v2[i];
		v1_xor_v2[i] = t->v1_xor_v2[i];
		observations[i] = t->observations[i];
	}
	reset(true);
}

Trial* XorTrial::clone() {
	XorTrial* t = new XorTrial(vSize);
	for (int i = 0; i < vSize; i++) {
		t->v1[i] = v1[i];
		t->v2[i] = v2[i];
		t->v1_xor_v2[i] = v1_xor_v2[i];
		t->observations[i] = observations[i];
	}
	return (Trial*)t;
}