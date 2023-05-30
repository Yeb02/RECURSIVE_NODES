#pragma once

#include "Trial.h"

#include "C:/Users/alpha/Bureau/RLRL/RocketSim/src/Sim/Arena/Arena.h"
#include "C:/Users/alpha/Bureau/RLRL/RocketSim/src/RocketSim.h"


//class RocketSimTrialData : public TrialData {
//	RocketSimTrialData(RocketSimTrialData& t);
//	RocketSimTrialData();
//	void init(TrialData& t) override;
//};

/*
Observations of size 25, laid out like this:
carPos  carVel  carAng  carAngVel  ballPos ballVel (vec3s of floats)
boost, carDist2Ball, carYaw2Ball, carPitch2Ball							(float, in [-1, 1])
isOnGround isJumping hasJump   (bools, in {-1, 1})

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
	void outerLoopUpdate(void* data) override {};

	static const int TICK_LIMIT = 1000;

private:
	void setObservations();

	std::unique_ptr<Arena> arena;
	Car* car; // managed by arena. Used as a shortcut for perfs.
	CarState initialCarState;
	BallState initialBallState;
};