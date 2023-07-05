#pragma once

#include "Trial.h"

#include "RocketSim/src/Sim/Arena/Arena.h"
#include "RocketSim/src/RocketSim.h"

/* Data I collected on how the sim behaves compared to the game.

ROCKET LEAGUE :

SetState:  CarAng(Rotator(pitch, yaw, roll)). Yeah, it is $#@ % !reversed !
GetState : CarAng(Rotator(yaw, pitch, roll)), pitch in - pi / 2, pi / 2.

SetState : CarAngVel(-rollvel, -pitchvel, +yawvel)  
Get state yields the same as was set.

ROCKET SIM:
SetState:  CarAng(Rotator(yaw, pitch, roll)), pitch in - pi / 2, pi / 2
GetState : CarAng(Rotator(yaw, pitch, roll)). 

SetState : CarAngVel(-rollvel, -pitchvel, +yawvel)  
Get state yields the same as was set.
*/


/*
Observations of size 24, laid out like this:
carPos  carVel  carAng  carAngVel  ballPos ballVel (vec3s of floats)
boost, carDist2Ball, carYaw2Ball, carPitch2Ball							(float, in [-1, 1])
isOnGround hasJumped   (bools, in {-1, 1})

Actions of size 8, laid out like this:
throttle, steer, pitch, yaw, roll   (floats, in [-1, 1])
boost, jump, handbrakes				(bools)
*/
class RocketSimTrial : public Trial {

public:
	RocketSimTrial(Arena* arena = nullptr, Car* car = nullptr);
	void step(const float* actions) override;
	void reset(bool sameSeed = false) override;
	void copy(Trial* t) override;
	Trial* clone() override;
	void outerLoopUpdate(void* data) override 
	{ 
		float* fdata = static_cast<float*>(data);
		jumpR = fdata[0];
		boostR = fdata[1];
		throttleR = fdata[2];
	};

	void compare2Game();

	static const int TICK_LIMIT = 4*120; // 4 seconds.
	
	float jumpR, boostR, throttleR;

private:
	void setObservations();

	OctreeEncoder velocityOctree;
	OctreeEncoder positionOctree;

	std::unique_ptr<Arena> arena;
	Car* car; // managed by arena. Used as a shortcut for perfs.
	CarState initialCarState;
	BallState initialBallState;

	// Used for reward computations. The initial distance between the car and the ball.
	float d0;

	float jumpS, boostS, throttleS;
};