#pragma once

#include "RocketSim.h"



RocketSimTrial::RocketSimTrial(Arena* _arena, Car* _car)
{
	netInSize = 24; 
	netOutSize = 8;
	observations.resize(netInSize);

	if (_arena == nullptr) {
		this->arena.reset(Arena::Create(GameMode::SOCCAR));
		this->car = this->arena->AddCar(Team::BLUE);
		std::cout << "created arena" << std::endl;
	}
	else {
		this->arena.reset(_arena);
		this->car = _car;
	}

	reset(false);
}

void RocketSimTrial::reset(bool sameSeed) {
	score = 0.0f;
	isTrialOver = false;
	currentNStep = 0;

	if (!sameSeed) {
		// data on objects at rest found there:
		// https://github.com/RLBot/RLBot/wiki/Useful-Game-Values
		// angles are YPR, YR in -pi,pi and P in -pi/2,pi/2
		initialCarState.pos = { 6000.0f * (UNIFORM_01 - .5f), 9800.0f * (UNIFORM_01 - .5f), 17.0f + UNIFORM_01 * 200.0f}; // zGrounded = 17
		initialCarState.vel = { 400.0f * (UNIFORM_01-.5f), 400.0f * (UNIFORM_01 - .5f), 200 * (UNIFORM_01 - .2f) };
		Angle carAng = Angle(M_PI * 2.0f * (UNIFORM_01 - .5f), M_PI * .4f * (UNIFORM_01 - .5f), M_PI * .4f * (UNIFORM_01 - .5f));
		initialCarState.rotMat = carAng.ToRotMat();

		initialCarState.boost = 100.0f * UNIFORM_01;

		initialBallState.pos = { 6000.0f * (UNIFORM_01 - .5f), 9800.0f * (UNIFORM_01 - .5f), (UNIFORM_01 - .5f) * 1000.0f + 92.75f}; // zGrounded = 92.75f
		//initialBallState.pos = { 6000.0f * (UNIFORM_01 - .5f), 9800.0f * (UNIFORM_01 - .5f), 92.75f + UNIFORM_01  * 1900.0f};
		initialBallState.vel = { 300.0f * (UNIFORM_01 - .5f), 300.0f * (UNIFORM_01 - .5f), 300.0f * (UNIFORM_01 - .5f) };
		//initialBallState.vel = { .0f, .0f, .0f };
	}

	car->SetState(initialCarState);
	arena->ball->SetState(initialBallState);

	setObservations();
}

void RocketSimTrial::setObservations() {
	constexpr float invFieldX = 1.0f / 4096.0f;
	constexpr float invFieldY = 1.0f / 5120.0f;
	constexpr float invHalfCeilingZ = 1.0f / 2044.0f;
	constexpr float invMaxCarVel = 1.0f / 2300.0f;
	constexpr float invMaxBallVel = 1.0f / 3000.0f;  // technically 6000, but it rarely reaches such speeds.
	constexpr float invMaxCarAngVel = 1.0f / 5.5f;  
	constexpr float invHalfMaxDist = 2.0f / 13272.f;  

	CarState currentCarState = arena->GetCars()[0]->GetState();

	Vec carForward = car->GetForwardDir();
	
	Vec& carPos = currentCarState.pos;
	Vec& carVel = currentCarState.vel;
	Angle carAng;
	carAng = Angle::FromRotMat(currentCarState.rotMat);
	Vec& carAngVel = currentCarState.angVel;

	BallState currentBallState = arena->ball->GetState();
	Vec& ballPos = currentBallState.pos;
	Vec& ballVel = currentBallState.vel;

	Vec car2Ball = ballPos - carPos;
	float carDist2Ball = car2Ball.Length();

	float alpha = atan2f(car2Ball.y, car2Ball.x);
	float beta = atan2f(carForward.y, carForward.x);
	float carYaw2Ball = alpha - beta;
	if (carYaw2Ball > M_PI) {
		carYaw2Ball -= 2.0f * M_PI;
	}
	else if (carYaw2Ball < -M_PI) {
		carYaw2Ball += 2.0f * M_PI;
	}

	float aleph = atan2f(car2Ball.z, sqrtf(car2Ball.x * car2Ball.x + car2Ball.y * car2Ball.y));
	float beth = atan2f(carForward.z, sqrtf(carForward.x * carForward.x + carForward.y * carForward.y));
	float carPitch2Ball = aleph - beth;
	if (carPitch2Ball > M_PI_2) {
		carPitch2Ball -= M_PI;
	}
	else if (carPitch2Ball < -M_PI_2) {
		carPitch2Ball += M_PI;
	}
	

	int i = 0;
	observations[i++] = carPos[0]*invFieldX;
	observations[i++] = carPos[1]*invFieldY;
	observations[i++] = carPos[2]*invHalfCeilingZ-1.0f;

	observations[i++] = carVel[0] * invMaxCarVel;
	observations[i++] = carVel[1] * invMaxCarVel;
	observations[i++] = carVel[2] * invMaxCarVel;

	observations[i++] = carAng.yaw;
	observations[i++] = carAng.pitch;
	observations[i++] = carAng.roll;

	observations[i++] = carAngVel[0] * invMaxCarAngVel;
	observations[i++] = carAngVel[1] * invMaxCarAngVel;
	observations[i++] = carAngVel[2] * invMaxCarAngVel;

	observations[i++] = ballPos[0] * invFieldX;
	observations[i++] = ballPos[1] * invFieldY;
	observations[i++] = ballPos[2] * invHalfCeilingZ - 1.0f;

	observations[i++] = ballVel[0] * invMaxBallVel;
	observations[i++] = ballVel[1] * invMaxBallVel;
	observations[i++] = ballVel[2] * invMaxBallVel;


	observations[i++] = currentCarState.boost * .02f - 1.0f;
	observations[i++] = carDist2Ball * invHalfMaxDist - 1.0f;
	observations[i++] = carYaw2Ball *.3f; // raw in -pi, pi
	observations[i++] = carPitch2Ball * .6f; // raw in -pi/2, pi/2

	observations[i++] = (currentCarState.isOnGround * 2.0f) - 1.0f;
	observations[i++] = (currentCarState.hasJumped * 2.0f) - 1.0f;
}

void RocketSimTrial::step(const float* actions) {
	constexpr int tickStride = 6; // 120 ticks per second in the game. (However the client recieves only 60 per second in the real game)
	constexpr float amplitude = 1.2f; // could be much higher. Never below 1.


	if (currentNStep * tickStride >= TICK_LIMIT) {
		score = 1.0f - score / ((float)currentNStep * 20000.0f); // 0 < score < 1.0f if this is how it ends. 20000.0f is above max dist.
		isTrialOver = true;
		return;
	}

	// floats are clamped in [-1, 1] in the simulation's code, so I do not bother here
	int i = 0;  
	car->controls.throttle = actions[i++] * amplitude;
	car->controls.steer = actions[i++] * amplitude;
	car->controls.pitch = actions[i++] * amplitude;
	car->controls.yaw = actions[i++] * amplitude;
	car->controls.roll = actions[i++] * amplitude;
	car->controls.boost = actions[i++] > 0;
	car->controls.jump = actions[i++] > 0;
	car->controls.handbrake = actions[i++] > 0;

	arena->Step(tickStride);

	setObservations();

	score += car->GetState().pos.Dist(arena->ball->GetState().pos);

	if (arena->tickCount - car->GetState().lastHitBallTick < tickStride) // ball was hit at this step
	{
		score = 1.0f + 2.0f * (1.0f - (float)(currentNStep * tickStride) / (float)TICK_LIMIT);
		isTrialOver = true;
	}
	currentNStep++;
}


void RocketSimTrial::copy(Trial* t0) {
	RocketSimTrial* t = dynamic_cast<RocketSimTrial*>(t0);
	
	initialBallState = t->initialBallState;
	initialCarState = t->initialCarState;

	reset(true);
}


Trial* RocketSimTrial::clone() {
	RocketSimTrial* t = new RocketSimTrial();
	
	t->initialCarState = initialCarState;
	t->initialBallState = initialBallState;
	
	t->reset(true);
	return (Trial*)t;
}

