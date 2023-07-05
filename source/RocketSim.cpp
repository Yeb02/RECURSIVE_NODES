#pragma once

#include "RocketSim.h"



RocketSimTrial::RocketSimTrial(Arena* _arena, Car* _car) :
	velocityOctree(2, 3), positionOctree(3,3)
{
	netInSize = (3+4+1) + (positionOctree.depth * 8 + 3 + velocityOctree.depth * 8 + 3) * 2;
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

	boostR = .001f;
	jumpR = .001f;
	throttleR = .001f;
	
	reset(false);
}

void RocketSimTrial::reset(bool sameSeed) {
	score = 0.0f;

	boostS = 0.0f;
	jumpS = 0.0f;
	throttleS = 0.0f;

	isTrialOver = false;
	currentNStep = 0;

	if (!sameSeed) {
		// data on objects at rest found there:
		// https://github.com/RLBot/RLBot/wiki/Useful-Game-Values
		// angles are YPR, YR in -pi,pi and P in -pi/2,pi/2
		initialCarState.pos = { 6000.0f * (UNIFORM_01 - .5f), 9800.0f * (UNIFORM_01 - .5f), 17.0f }; // zGrounded = 17
		initialCarState.vel = { 400.0f * (UNIFORM_01 - .5f), 400.0f * (UNIFORM_01 - .5f), 0.0f };
		Angle carAng = Angle(M_PI * 2.0f * (UNIFORM_01 - .5f), 0.0f, 0.0f);
		initialCarState.rotMat = carAng.ToRotMat();

		initialCarState.boost = 100.0f * UNIFORM_01;

		initialBallState.pos = { 6000.0f * (UNIFORM_01 - .5f), 9800.0f * (UNIFORM_01 - .5f), (UNIFORM_01 - .5f) * 1000.0f + 92.75f }; // zGrounded = 92.75f
		//initialBallState.pos = { 6000.0f * (UNIFORM_01 - .5f), 9800.0f * (UNIFORM_01 - .5f), 92.75f + UNIFORM_01  * 1900.0f};
		initialBallState.vel = { 300.0f * (UNIFORM_01 - .5f), 300.0f * (UNIFORM_01 - .5f), 300.0f * (UNIFORM_01 - .5f) };
		//initialBallState.vel = { .0f, .0f, .0f };
	}

	car->SetState(initialCarState);
	arena->ball->SetState(initialBallState);

	d0 = car->GetState().pos.Dist(arena->ball->GetState().pos);

	setObservations();
}

void RocketSimTrial::compare2Game()
{
	initialCarState.pos = { 1000.0f, -2000.0f, 650.0f }; 
	initialCarState.vel = { -400.0f, 600.0f, 500.0f };
	initialCarState.angVel = { 1.0f, -2.0f, 5.0f };
	Angle carAng = Angle(2.f, -1.f, -2.8f);
	initialCarState.rotMat = carAng.ToRotMat();
	initialCarState.boost = 87.0f;
	initialBallState.pos = { 100.0f, -200.0f, 300.0f }; 
	initialBallState.vel = { -500.0f, 20.0f, -1000.0f };


	car->SetState(initialCarState);
	arena->ball->SetState(initialBallState);
	setObservations();

	arena->Step(6);
	setObservations();

	arena->Step(6);
	setObservations();
}

void RocketSimTrial::setObservations() {
	constexpr float invFieldX = 1.0f / 4096.0f;
	constexpr float invFieldY = 1.0f / (5120.0f+880.0f);
	constexpr float invHalfCeilingZ = 1.0f / 1022.0f; 
	constexpr float invMaxCarVel = 1.0f / 2300.0f;
	constexpr float invMaxBallVel = 1.0f / 4600.0f; // technically 6000 but rarely reached
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
	
	float octreeInput[3];
	int i = 0;

	octreeInput[0] = carPos[0]*invFieldX;
	octreeInput[1] = carPos[1]*invFieldY;
	octreeInput[2] = carPos[2]*invHalfCeilingZ-1.0f;
	i += positionOctree.encode(octreeInput, &observations[i]);

	octreeInput[0] = carVel[0] * invMaxCarVel;
	octreeInput[1] = carVel[1] * invMaxCarVel;
	octreeInput[2] = carVel[2] * invMaxCarVel;
	i += velocityOctree.encode(octreeInput, &observations[i]);

	observations[i++] = carAng.yaw;
	observations[i++] = carAng.pitch;
	observations[i++] = carAng.roll;

	//observations[i++] = carAngVel[0] * invMaxCarAngVel; 
	//observations[i++] = carAngVel[1] * invMaxCarAngVel; 
	//observations[i++] = carAngVel[2] * invMaxCarAngVel; 

	octreeInput[0] = ballPos[0] * invFieldX;
	octreeInput[1] = ballPos[1] * invFieldY;
	octreeInput[2] = ballPos[2] * invHalfCeilingZ - 1.0f;
	i += positionOctree.encode(octreeInput, &observations[i]);

	octreeInput[0] = ballVel[0] * invMaxBallVel;
	octreeInput[1] = ballVel[1] * invMaxBallVel;
	octreeInput[2] = ballVel[2] * invMaxBallVel;
	i += velocityOctree.encode(octreeInput, &observations[i]);

	observations[i++] = currentCarState.boost * .02f - 1.0f;
	observations[i++] = carDist2Ball * invHalfMaxDist - 1.0f;
	observations[i++] = carYaw2Ball *.3f; // raw in -pi, pi
	observations[i++] = carPitch2Ball * .6f; // raw in -pi/2, pi/2

	observations[i++] = (currentCarState.isOnGround * 2.0f) - 1.0f;
	//observations[i++] = (currentCarState.hasJumped * 2.0f) - 1.0f;
}

void RocketSimTrial::step(const float* actions) {
	// 120 ticks per second in the game. (But the client recieves only 60 ticks
	// per second from the server). Inferences per second = 120 / tickStride .
	constexpr int tickStride = 12; 
	constexpr float amplitude = 1.2f; // could be much higher. Never below 1.


	if (currentNStep * tickStride >= TICK_LIMIT) {
	
		//score = car->GetState().vel.Length() / 2300.0f - score/ ((float)currentNStep * d0);
		//score = 1.0f + (jumpS > 0)*jumpR - (score * inv_d0 - throttleS * throttleR - boostR * boostS)/ (float)currentNStep; 
		score = 1.0f - score / ((float)currentNStep * d0);

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

	int delta = INT_0X(3) - 1; // random int in {-1,0,1} to reproduce the game's fluctuations.
	arena->Step(tickStride+delta);

	setObservations();

	score += car->GetState().pos.Dist(arena->ball->GetState().pos);

	//throttleS += actions[0];
	//boostS += actions[5]>0;
	//jumpS += actions[6]>0;


	if (arena->tickCount - car->GetState().lastHitBallTick < tickStride) // ball was hit at this step
	{
		//score = 1.0f + (1.0f - (float)(currentNStep * tickStride) / (float)TICK_LIMIT) + car->GetState().vel.Length() / 2300;
		//score = 1.0f + (1.0f - (float)(currentNStep * tickStride) / (float)TICK_LIMIT) + car->GetState().vel.Length() / 2300;
		score = 1.0f + 2.0f * (1.0f - (float)(currentNStep * tickStride) / (float)TICK_LIMIT);
		isTrialOver = true;
	}
	currentNStep++;
}


void RocketSimTrial::copy(Trial* t0) {
	RocketSimTrial* t = dynamic_cast<RocketSimTrial*>(t0);
	
	initialBallState = t->initialBallState;
	initialCarState = t->initialCarState;

	boostR = t->boostR;
	jumpR = t->jumpR;
	throttleR = t->throttleR;

	reset(true);
}


Trial* RocketSimTrial::clone() {
	RocketSimTrial* t = new RocketSimTrial();
	
	t->initialCarState = initialCarState;
	t->initialBallState = initialBallState;
	
	t->boostR = boostR;
	t->jumpR = jumpR;
	t->throttleR = throttleR;

	t->reset(true);
	return (Trial*)t;
}

