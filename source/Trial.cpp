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

void XorTrial::step(const float* actions) {
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
			score += (float) ((actions[i] > 0) == v1_xor_v2[i]); 
	}

	if (currentNStep >= delay*3) {
		isTrialOver = true;

		// score normalization, not necessary
		if (currentNStep == delay * 3) {
			//score /= (float)(delay * vSize);
			score = (score >= (vSize * delay - .0001f)) ? 1.0f : 0.0f;
			
		}
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



CartPoleTrial::CartPoleTrial(bool continuousControl) :
	continuousControl(continuousControl)
{
	netInSize = 4;
	netOutSize = 1;
	observations.resize(netInSize);

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

void CartPoleTrial::step(const float* actions) {
	constexpr float tau = .02f ; // .02f baseline
	constexpr float gravity = 9.8f; //9.8f baseline
	constexpr float masscart = 1.0f;
	constexpr float masspole = 0.1f;
	constexpr float total_mass = masspole + masscart;
	constexpr float length = 0.5f;
	constexpr float polemass_length = masspole * length;
	constexpr float force_mag = 10.0f;

	if (abs(theta) > .21f || abs(x) > 2.5f || currentNStep >= STEP_LIMIT) isTrialOver = true; 
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
	// Gym uses force = sign(actions[0]). It makes convergence much faster, but has not as good 
	// performance as a continuous control with force = actions[0].
	//force = actions[0] > 0 ? 1.0f : -1.0f; 
	if (continuousControl) force = actions[0];
	else force = actions[0] > 0 ? 1.0f : -1.0f;


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
	continuousControl = t->continuousControl;
	reset(true);
}

Trial* CartPoleTrial::clone() {
	CartPoleTrial* t = new CartPoleTrial(continuousControl);
	t->x0 = x0;
	t->xDot0 = xDot0;
	t->theta0 = theta0;
	t->thetaDot0 = thetaDot0;
	t->reset(true);
	return (Trial*)t;
}



TMazeTrial::TMazeTrial(bool switchesSides) :
	switchesSide(switchesSide)
{
	netInSize = 3;
	netOutSize = 1;
	observations.resize(netInSize);

	reset();
}

void TMazeTrial::reset(bool sameSeed) {
	nSubTrials = -1;
	score = 0.0f;
	isTrialOver = false;
	subTrialReset();
}

void TMazeTrial::subTrialReset() {
	wentLeft = false;
	currentNStep = 0;
	nSubTrials++;
	observations[0] = -1.0f;
	observations[1] = -1.0f;
	observations[2] = -1.0f;
}

void TMazeTrial::step(const float* actions) {
	constexpr int straight1 = corridorLength + 1;
	constexpr int turn1 = straight1 + 1;
	constexpr int straight2 = turn1 + corridorLength;
	constexpr int ME = straight2 + 1;
	constexpr int straight3 = ME + corridorLength;
	constexpr int turn2 = straight3 + corridorLength;
	constexpr int straight4 = turn2 + corridorLength;

	float a = actions[0]; //readability
	
	if (currentNStep < straight1) {
		if (abs(a) > .3f) {
			score -= .4f;
			subTrialReset();
		}
		else if (currentNStep == straight1-1) observations[0] = 1.0f;

	} else if (currentNStep < turn1) {
		if (abs(a) < .3f) {
			score -= .4f;
			subTrialReset();
		}
		else {
			if (a < 0.0f) wentLeft = true;
			observations[0] = -1.0f;
		}
	} else if (currentNStep < straight2) {
		if (abs(a) > .3f) {
			score -= .4f;
			subTrialReset();
		}
		else if (currentNStep == straight2 - 1) {
			observations[1] = 1.0f;
			float reward = (wentLeft ^ (switchesSide && (nSubTrials >= 10))) ? 1.0f : .2f;
			observations[2] = reward;
			score += reward;
		}
	} 
	else if (currentNStep < ME) {
		observations[1] = -1.0f;
		if (abs(a) > .3f) {
			score -= .4f;
			subTrialReset();
		}
	}
	else if (currentNStep < straight3) {
		if (abs(a) > .3f) {
			score -= .4f;
			subTrialReset();
		}
		else if (currentNStep == straight3 - 1) {
			observations[0] = 1.0f;
		}
	}
	else if (currentNStep < turn2) {
		if (abs(a) < .3f) {
			score -= .4f;
			subTrialReset();
		}
		else {
			if ((a>0) ^ wentLeft) {
				score -= .3f;
				subTrialReset();
			}
			else {
				observations[0] = -1.0f;
			}
		}
	} else if(currentNStep < straight3) {
		if (abs(a) > .3f) {
			score -= .4f;
			subTrialReset();
		}
	} else {
		nSubTrials++;
		subTrialReset();
	}

	currentNStep++;
	if (nSubTrials >= 20) {
		isTrialOver = true;
	}
}

void TMazeTrial::copy(Trial* t0) {
	TMazeTrial* t = dynamic_cast<TMazeTrial*>(t0);
	switchesSide = t->switchesSide;
}

Trial* TMazeTrial::clone() {
	TMazeTrial* t = new TMazeTrial(switchesSide);
	return (Trial*)t;
}




NLinksPendulumTrial::NLinksPendulumTrial(bool continuousControl, int nLinks) :
	continuousControl(continuousControl), nLinks(nLinks)
{
	netInSize = 2 + nLinks * 3;
	netOutSize = 1;

	observations.resize(netInSize);

	thetas = std::make_unique<float[]>(nLinks);
	thetas0 = std::make_unique<float[]>(nLinks);

	xs  = std::make_unique<float[]>(nLinks+1);
	vxs = std::make_unique<float[]>(nLinks+1);
	ys  = std::make_unique<float[]>(nLinks+1);
	vys = std::make_unique<float[]>(nLinks+1);
	pxs = std::make_unique<float[]>(nLinks + 1);
	pys = std::make_unique<float[]>(nLinks + 1);

	reset();
}

void NLinksPendulumTrial::reset(bool sameSeed) {
	score = 0.0f;
	isTrialOver = false;
	currentNStep = 0;

	if (!sameSeed) {
		x0 = UNIFORM_01 - .5f;
		thetas0[0] = -1.57079f + (UNIFORM_01 - .5f);
		for (int i = 1; i < nLinks; i++) {
			thetas0[i] = (UNIFORM_01 - .5f);
		}
		
	}

	xs[0] = x0;
	ys[0] = 0.0f;

	// pxs and pys need not be initialized.

	observations[0] = xs[0];

	float _x = xs[0], _y = ys[0], l = 1.0f;
	for (int i = 0; i < nLinks; i++) {
		thetas[i] = thetas0[i];

		_x += l * cosf(thetas[i]);
		_y += l * cosf(thetas[i]);
		xs[i + 1] = _x;
		ys[i + 1] = _y;
		observations[2*i+1] = cosf(thetas[i]);
		observations[2*i+2] = sinf(thetas[i]);
		l *= .7f;
	}
}

void NLinksPendulumTrial::step(const float* actions) {
	constexpr float gravity = 9.8f; 
	constexpr float mCart = 1.0f;
	constexpr float force_mag = 10.0f;

	constexpr float tau = .01f;
	constexpr int nSubSteps = 5;

	// DO NOT CHANGE THESE 5, OR IF YOU DO MAKE SURE THEY ARE ALSO CHANGED IN reset() (spaghetti mama mia)
	constexpr float mPole = .2f;
	constexpr float mPoleDecay = .7f; // = m of pole i+1 / m of pole i
	constexpr float lPole = 1.0f;
	constexpr float lPoleDecay = .7f; // = l of pole i+1 / l of pole i. Change it in reset() too !!!
	constexpr float xRange = 3.0f;  // serves in reset() for x0 initialisation.


	if (abs(xs[0]) > xRange || currentNStep >= STEP_LIMIT) isTrialOver = true;
	if (isTrialOver) return;


	float force;
	if (continuousControl) force = actions[0];
	else force = actions[0] > 0 ? 1.0f : -1.0f;


	// physics update:
	for (int s = 0; s < nSubSteps; s++) {

		xs[0] += tau * force * force_mag / mCart;
		pxs[0] = xs[0];
		pys[0] = ys[0];
		for (int i = 1; i < nLinks + 1; i++) {
			ys[i] -= tau * gravity;
			pxs[i] = xs[i];
			pys[i] = ys[i];
			xs[i] += vxs[i] * tau;
			ys[i] += vys[i] * tau;
		}

		float m = mPole, l = lPole;
		for (int i = 1; i < nLinks+1; i++) {
			float dx = xs[i] - xs[i-1];
			float dy = ys[i] - ys[i-1];
			float d = sqrtf(dx * dx + dy * dy);
			float w0 = i==1 ? 0.0f : 1.0f / (m/mPoleDecay);
			float w1 = m;
			float corr = (l - d) / (d *(w0 + w1));
			xs[i - 1] -= w0 * corr * dx;
			ys[i - 1] -= w0 * corr * dy;
			xs[i] += w1 * corr * dx;
			ys[i] += w1 * corr * dy;
			m *= mPoleDecay;
			l *= lPoleDecay;
		}
		for (int i = 1; i < nLinks+1; i++) {
			vxs[i] = (xs[i] - pxs[i]) / tau;
			vys[i] = (ys[i] - pys[i]) / tau;
		}
	}

	// thetas and observations update
	observations[0] = xs[0];
	float l = lPole;
	for (int i = 0; i < nLinks; i++) {
		float theta_i = acosf((xs[i + 1] - xs[i]) / l);
		theta_i = ys[i + 1] > ys[i] ? theta_i : -theta_i;  // be careful of the discontinuity (mod 2pi). 

		observations[2 * i + 1] = (xs[i + 1] - xs[i]) / l;             // cos theta
		observations[2 * i + 2] = (ys[i + 1] - ys[i]) / l;             // sin theta

		thetas[i] = theta_i;
		l *= lPoleDecay;
	}

	currentNStep++;
	score += ys[0];
	return;
}

void NLinksPendulumTrial::copy(Trial* t0) {
	NLinksPendulumTrial* t = dynamic_cast<NLinksPendulumTrial*>(t0);
	nLinks = t->nLinks;
	continuousControl = t->continuousControl;
	netInSize = t->netInSize;

	observations.resize(netInSize);

	thetas.reset(new float[nLinks]);
	thetas0.reset(new float[nLinks]);
	
	xs.reset(new float[nLinks + 1]);
	vxs.reset(new float[nLinks + 1]);
	ys.reset(new float[nLinks + 1]);
	vys.reset(new float[nLinks + 1]);
	pxs.reset(new float[nLinks + 1]);
	pys.reset(new float[nLinks + 1]);

	x0 = t->x0;
	for (int i = 0; i < nLinks; i++) {
		thetas0[i] = t->thetas0[i];
	}
	
	reset(true);
}

Trial* NLinksPendulumTrial::clone() {
	NLinksPendulumTrial* t = new NLinksPendulumTrial(continuousControl, nLinks);
	
	t->x0 = x0;

	for (int i = 0; i < nLinks; i++) {
		t->thetas0[i] = thetas0[i];
	}

	t->reset(true);

	return (Trial*)t;
}

