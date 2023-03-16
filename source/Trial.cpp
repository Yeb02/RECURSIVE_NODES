#pragma once

#include <cmath> 

#include "Trial.h"


XorTrial::XorTrial(int vSize, int delay) :
	vSize(vSize), delay(delay)
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
	//constexpr int endV1Phase = 5;
	//constexpr int endV2Phase = 10;
	//constexpr int startResponsePhase = 10;
	//constexpr int endResponsePhase = 15;

	
	if (currentNStep == delay) {
		for (int i = 0; i < vSize; i++)  observations[i] = v2[i] ? 1.0f : -1.0f;
	}

	if (currentNStep == delay*2) {
		for (int i = 0; i < vSize; i++)  observations[i] = 0.0f;
	}

	if (currentNStep < delay*3 && currentNStep >= delay*2) {
		for (int i = 0; i < vSize; i++)  
			score += (float) (actions[i] > 0) == v1_xor_v2[i]; 
	}

	if (currentNStep >= delay*3) {
		isTrialOver = true;

		// score normalization, not necessary
		if (currentNStep == delay * 3) score /= (float) (delay * vSize);
	}

	currentNStep++;
}

void XorTrial::copy(Trial* t0) {
	XorTrial* t = dynamic_cast<XorTrial*>(t0);
	vSize = t->vSize;
	delay = t->delay;
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
	XorTrial* t = new XorTrial(vSize, delay);
	for (int i = 0; i < vSize; i++) {
		t->v1[i] = v1[i];
		t->v2[i] = v2[i];
		t->v1_xor_v2[i] = v1_xor_v2[i];
		t->observations[i] = observations[i];
	}
	return (Trial*)t;
}



CartPoleTrial::CartPoleTrial() {
	observations.resize(4);
	netInSize = 4;
	netOutSize = 1;

	reset();
}

void CartPoleTrial::reset(bool sameSeed) {
	score = 0.0f;
	isTrialOver = false;
	currentNStep = 0;

	if (!sameSeed) {
		// gym initializes all 4 in [-0.05, 0.05], so *.1f if we follow their setup. 
		x0 = (UNIFORM_01 - .5f) * .1f; 
		xDot0 = (UNIFORM_01 - .5f) * .1f;
		theta0 = (UNIFORM_01 - .5f) * .1f;
		thetaDot0 = (UNIFORM_01 - .5f) * .1f;
	}

	x = x0;
	xDot = xDot0;
	theta = theta0;
	thetaDot = thetaDot0;

	observations[0] = x;
	observations[1] = xDot;
	observations[2] = theta;
	observations[3] = thetaDot;
}

void CartPoleTrial::step(const std::vector<float>& actions) {
	constexpr float tau = .02f ; // .02f baseline
	constexpr float gravity = 9.8f; //9.8f baseline
	constexpr float masscart = 1.0f;
	constexpr float masspole = 0.1f;
	constexpr float total_mass = masspole + masscart;
	constexpr float length = 0.5f;
	constexpr float polemass_length = masspole * length;
	constexpr float force_mag = 10.0f;

	if (abs(theta) > .21f || abs(x) > 2.5f || currentNStep >= STEP_LIMIT) isTrialOver = true; 
	// A final reward slightly improves robustness, but surprisingly increasing the steps limit is the way to improve
	// average performance (drastically !). This is probably because of the dynamic nature of connexions in this model. 
	// TODO investigate !
	/*if (currentNStep == STEP_LIMIT) { 
		score *= 1.5f;
		currentNStep++;
	}*/
	if (isTrialOver) return;

	currentNStep++;
	score += 1.0f;

	// To give time to the initial observation to propagate to the network. Avoiding a cold start,
	// in case the initial observation requires quick actions. This hinders outer and inner learning though.
	if (currentNStep < 5) return; 


	//if (currentNStep % 2 == 0) return;  Giving the network twice the "time to think" worsens performances. 
	// Maybe because of the unexpected jump in the physics every other step. Halving the time step (doubling the total
	// number of steps) has not shown a statistically significant improvement either. More torough tests are needed.

	float force;
	// Gym uses force = sign(actions[0]). It makes convergence much faster (if at all), but has not as good average
	// performance as a continuous control (i.e. force = actions[0]) when the problem is "easy", or "nearly solved".
	force = actions[0] > 0 ? 1.0f : -1.0f; 
	//force = actions[0];  

	// update as per https://coneural.org/florian/papers/05_cart_pole.pdf

	float cosTheta = cosf(theta), sinTheta = sinf(theta);

	// I re-ordered the terms in order to minimize the number of division operation.
	float temp = (force + polemass_length * thetaDot * thetaDot * sinTheta);
	float thetaacc = (gravity * sinTheta * total_mass - cosTheta * temp) /
		(length * (1.33333f * total_mass - masspole * cosTheta * cosTheta));
	float xacc = (temp - polemass_length * thetaacc * cosTheta) / total_mass;

	// Explicit euler. Could use implicit.
	x = x + tau * xDot;
	xDot = xDot + tau * xacc;
	theta = theta + tau * thetaDot;
	thetaDot = thetaDot + tau * thetaacc;

	observations[0] = x;
	observations[1] = xDot;
	observations[2] = theta;
	observations[3] = thetaDot;

}

void CartPoleTrial::copy(Trial* t0) {
	CartPoleTrial* t = dynamic_cast<CartPoleTrial*>(t0);
	x0 = t->x0;
	xDot0 = t->xDot0;
	theta0 = t->theta0;
	thetaDot0 = t->thetaDot0;
	reset(true);
}

Trial* CartPoleTrial::clone() {
	CartPoleTrial* t = new CartPoleTrial();
	t->x0 = x0;
	t->xDot0 = xDot0;
	t->theta0 = theta0;
	t->thetaDot0 = thetaDot0;
	t->reset(true);
	return (Trial*)t;
}