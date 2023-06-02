from rlbot.agents.base_agent import BaseAgent, SimpleControllerState
from rlbot.messages.flat.QuickChatSelection import QuickChatSelection
from rlbot.utils.structures.game_data_struct import GameTickPacket

from util.vec import Vec3

from numpy import arctan2, sqrt, pi, clip

from RECURSIVE_NODES import *
import ctypes

class ReNo(BaseAgent):

    def __init__(self, name, team, index):
        super().__init__(name, team, index)

        self.network = load_existing_network(b"src\\topNet_1685528330066_1001.renon")
        prepare_network(self.network)
        self.observations = (ctypes.c_float * 24)()                          
        self.actions = (ctypes.c_float * 8)()    
 

    def get_output(self, packet: GameTickPacket) -> SimpleControllerState:
        """
        This function will be called by the framework many times per second. This is where you can
        see the motion of the ball, etc. and return controls to drive your car.
        """

        # Gather some information about our car and the ball
        my_car = packet.game_cars[self.index]
        car_location = Vec3(my_car.physics.location)
        car_velocity = Vec3(my_car.physics.velocity)
        car_angle = Vec3(my_car.physics.rotation.yaw, my_car.physics.rotation.pitch, my_car.physics.rotation.roll)  # x=Yaw, y=Pitch, z=Roll
        car_angular_velocity = Vec3(my_car.physics.angular_velocity)
        ball_location = Vec3(packet.game_ball.physics.location)
        ball_velocity = Vec3(packet.game_ball.physics.velocity)

        invFieldX = 1.0 / 4096.0
        invFieldY = 1.0 / 5120.0
        invHalfCeilingZ = 1.0 / 2044.0
        invMaxCarVel = 1.0 / 2300.0
        invMaxBallVel = 1.0 / 3000.0  
        invMaxCarAngVel = 1.0/ 5.5 
        invHalfMaxDist = 2. / 13272

	
	
        car2Ball = ball_location - car_location
        carDist2Ball = car2Ball.length()

        alpha = arctan2(car2Ball.y, car2Ball.x)
        beta = car_angle.x #yaw
        carYaw2Ball = alpha - beta
        if carYaw2Ball > pi :
            carYaw2Ball -= 2.0 * pi
        elif carYaw2Ball < -pi:
            carYaw2Ball += 2.0 * pi
        

        aleph = arctan2(car2Ball.z, sqrt(car2Ball.x * car2Ball.x + car2Ball.y * car2Ball.y))
        beth = car_angle.y #pitch
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

        self.observations[i] = car_angle.x # yaw
        self.observations[i+1] = car_angle.y #pitch
        self.observations[i+2] = car_angle.z
        i+=3

        self.observations[i] = car_angular_velocity.y * invMaxCarAngVel #yaw
        self.observations[i+1] = car_angular_velocity.x * invMaxCarAngVel #pitch
        self.observations[i+2] = car_angular_velocity.z * invMaxCarAngVel
        i+=3

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
        self.observations[i+1] = (my_car.jumped * 2.0) - 1.0
    
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