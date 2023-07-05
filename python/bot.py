from rlbot.agents.base_agent import BaseAgent, SimpleControllerState
from rlbot.messages.flat.QuickChatSelection import QuickChatSelection
from rlbot.utils.structures.game_data_struct import GameTickPacket
from rlbot.utils.game_state_util import GameState, BallState, CarState, Physics, Vector3, Rotator, GameInfoState

from util.vec import Vec3

from numpy import arctan2, sqrt, pi, clip

from RECURSIVE_NODES import *
import ctypes


"""
Data I collected on how the sim behaves compared to the game:

ROCKET LEAGUE :

SetState:  CarAng(Rotator(pitch, yaw, roll)), pitch in - pi / 2 i / 2
GetState : CarAng(Rotator(yaw, pitch, roll)). 

SetState : CarAngVel(-rollvel, -pitchvel, +yawvel)  
Get state yields the same as was set.

ROCKET SIM:
SetState:  CarAng(Rotator(yaw, pitch, roll)), p in - pi / 2 i / 2
GetState : CarAng(Rotator(yaw, pitch, roll)). 

SetState : CarAngVel(-rollvel, -pitchvel, +yawvel)  
Get state yields the same as was set.
"""
class ReNo(BaseAgent):

    def __init__(self, name, team, index):
        super().__init__(name, team, index)

        self.network = load_existing_network(b"src\\topNet_1686828256060_751.renon")
        prepare_network(self.network)
        self.observations = (ctypes.c_float * get_observations_size(self.network))()                          
        self.actions = (ctypes.c_float * get_actions_size(self.network))()  
        self.physicsFrame = 0  
 

    def get_output(self, packet: GameTickPacket) -> SimpleControllerState:
        #as of now, reno is active one in 12 physics ticks (i.e. 10 times a second). Im setting 5 to allow for fluctuations.
        if not packet.game_info.is_round_active:
            self.physicsFrame = packet.game_info.frame_num
            return SimpleControllerState()
        
        print("delta ticks : " + str(packet.game_info.frame_num - self.physicsFrame))
        self.physicsFrame = packet.game_info.frame_num
        

        my_car = packet.game_cars[self.index]
        car_location = Vec3(my_car.physics.location)
        car_velocity = Vec3(my_car.physics.velocity)
        car_angle = Vec3(my_car.physics.rotation.yaw, my_car.physics.rotation.pitch, my_car.physics.rotation.roll)  # x=Yaw, y=Pitch, z=Roll
        car_angular_velocity = Vec3(my_car.physics.angular_velocity)
        ball_location = Vec3(packet.game_ball.physics.location)
        ball_velocity = Vec3(packet.game_ball.physics.velocity)

        invFieldX = 1.0 / 4096.0
        invFieldY = 1.0 / (5120.0 + 880)
        invHalfCeilingZ = 1.0 / 1022
        invMaxCarVel = 1.0 / 2300.0
        invMaxBallVel = 1.0 / 4600.0  
        invMaxCarAngVel = 1.0/ 5.5 
        invHalfMaxDist = 2. / 13272

	
	
        car2Ball = ball_location - car_location
        carDist2Ball = car2Ball.length()

        alpha = arctan2(car2Ball.y, car2Ball.x)
        beta = car_angle.x 
        carYaw2Ball = alpha - beta
        if carYaw2Ball > pi :
            carYaw2Ball -= 2.0 * pi
        elif carYaw2Ball < -pi:
            carYaw2Ball += 2.0 * pi
        

        aleph = arctan2(car2Ball.z, sqrt(car2Ball.x * car2Ball.x + car2Ball.y * car2Ball.y))
        beth = car_angle.y 
        carPitch2Ball = aleph - beth
        if carPitch2Ball > pi*.5:
            carPitch2Ball -= pi
        elif carPitch2Ball < -pi*.5:
            carPitch2Ball += pi
        
	

        i = 0
        self.observations[i] = car_location.x*invFieldX
        self.observations[i+1] = car_location.y*invFieldY
        self.observations[i+2] = car_location.z*invHalfCeilingZ-1.0
        i += 3

        self.observations[i] = car_velocity.x * invMaxCarVel
        self.observations[i+1] = car_velocity.y * invMaxCarVel
        self.observations[i+2] = car_velocity.z * invMaxCarVel
        i += 3

        self.observations[i] = car_angle.x 
        self.observations[i+1] = car_angle.y 
        self.observations[i+2] = car_angle.z
        i+=3

        # self.observations[i] = car_angular_velocity.x * invMaxCarAngVel 
        # self.observations[i+1] = car_angular_velocity.y * invMaxCarAngVel 
        # self.observations[i+2] = car_angular_velocity.z * invMaxCarAngVel
        # i+=3

        self.observations[i] = ball_location.x * invFieldX
        self.observations[i+1] = ball_location.y * invFieldY
        self.observations[i+2] = ball_location.z * invHalfCeilingZ - 1.0
        i+=3

        self.observations[i] = ball_velocity.x * invMaxBallVel
        self.observations[i+1] = ball_velocity.y * invMaxBallVel
        self.observations[i+2] = ball_velocity.z * invMaxBallVel
        i+=3

        self.observations[i] = my_car.boost * .02 - 1.0
        self.observations[i+1] = carDist2Ball * invHalfMaxDist - 1.0
        self.observations[i+2] = carYaw2Ball *.3 # raw in -pi, pi
        self.observations[i+3] = carPitch2Ball * .6 # raw in -pi/2, pi/2
        i+=4

        self.observations[i] = (my_car.has_wheel_contact * 2.0) - 1.0
        # self.observations[i+1] = (my_car.jumped * 2.0) - 1.0
    

        get_actions(self.network, self.observations, self.actions)

        amplitude = 1.2
        controls = SimpleControllerState()
        controls.throttle = clip(self.actions[0] * amplitude, -1, 1)
        controls.steer = clip(self.actions[1] * amplitude, -1, 1)
        controls.pitch = clip(self.actions[2] * amplitude, -1, 1)
        controls.yaw = clip(self.actions[3] * amplitude, -1, 1)
        controls.roll = clip(self.actions[4] * amplitude, -1, 1)
        controls.boost = self.actions[5] > 0
        controls.jump = self.actions[6] > 0
        controls.handbrake = self.actions[7] > 0

        return controls

# myBot = ReNo("ReNo", 0, 0)
# get_actions(myBot.network, myBot.observations, myBot.actions)
# print(myBot.actions[0])