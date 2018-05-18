import os
import sys
import time
import numpy as np

python_path = os.path.abspath('AirSim/PythonClient')
sys.path.append(python_path)
from AirSimClient import *

class CarConnector:
    def __init__(self):
        self.MAX_SPEED    = 10.0
        self.MIN_SPEED    = 0.1

        self.index        = 0
        self.trajectory   = 0

        self.max_actions = 64

        self.modes = ['acc', 'deacc', 'still', 'brake']
        self.mode  = 'still'

        self.client = CarClient()
        self.client.confirmConnection()
        self.client.enableApiControl(True)

        self.car_controls          = CarControls()
        self.car_controls.brake    = 0
        self.car_controls.throttle = 0
        self.car_controls.steering = 0

    def disconnect(self):
        self.client.enableApiControl(False)

    def get_position(self):
        state = self.client.getCarState()
        pos   = state.kinematics_true.position
        return pos

    def get_orientation(self):
        state = self.client.getCarState()
        ort   = state.kinematics_true.orientation
        return self.client.getPitchRollYaw(ort)

    def get_position_and_orientation(self):
        state = self.client.getCarState()
        pos   = state.kinematics_true.position
        ort   = state.kinematics_true.orientation
        return pos, self.client.toEulerianAngle(ort)

    def reset(self):
        self.car_controls.brake    = 1
        self.client.setCarControls(self.car_controls)
        time.sleep(1)

        self.index                 = 0
        self.car_controls.brake    = 0
        self.car_controls.throttle = 0
        self.car_controls.steering = 0

        self.client.reset()
        self.client.enableApiControl(True)
        pos, ort = self.get_position_and_orientation()
        return pos, ort

    def drive(self):
        self.get_controls()
        self.client.setCarControls(self.car_controls)

    def get_controls(self):
        # if self.index % self.max_actions == 0:
        #     self.mode = np.random.choice(self.modes, 1)

        if self.mode=="still":
            self.car_controls.brake    = 0
            self.car_controls.throttle = 0
            self.car_controls.steering = 0

        if self.mode=="acc":
            if self.car_controls.is_manual_gear:
                self.car_controls.is_manual_gear = False
                self.car_controls.manual_gear    = 0
            self.car_controls.brake    = 0
            self.car_controls.throttle = 0.0625
            self.car_controls.steering = 0

        if self.mode=="deacc":
            self.car_controls.throttle       = -0.1
            self.car_controls.is_manual_gear = True
            self.car_controls.manual_gear    = -1

        if self.mode=="brake":
            if self.car_controls.is_manual_gear:
                self.car_controls.is_manual_gear = False
                self.car_controls.manual_gear    = 0
            self.car_controls.brake    = 0.125
            self.car_controls.throttle = 0
            self.car_controls.steering = 0

        if self.mode=="turn":
            if self.car_controls.is_manual_gear:
                self.car_controls.is_manual_gear = False
                self.car_controls.manual_gear    = 0
            self.car_controls.brake    = 0
            self.car_controls.throttle = 1
            self.car_controls.steering = np.random.sample()*2*0.125 - 0.125

        elif self.mode=="acc-deacc":
            max_actions = self.max_actions
            if self.index % max_actions == 0:
                if self.car_controls.is_manual_gear:
                    self.car_controls.is_manual_gear = False
                    self.car_controls.manual_gear    = 0
                self.index                       = 0
                self.car_controls.throttle       = 0.125
            elif self.index == (max_actions*7/8):
                if self.car_controls.is_manual_gear:
                    self.car_controls.is_manual_gear = False
                    self.car_controls.manual_gear    = 0
                self.car_controls.throttle       = 0
            elif self.index == (max_actions*4/8):
                self.car_controls.throttle       = -0.125
                self.car_controls.is_manual_gear = True
                self.car_controls.manual_gear    = -1
            elif self.index == (max_actions*2/8):
                if self.car_controls.is_manual_gear:
                    self.car_controls.is_manual_gear = False
                    self.car_controls.manual_gear    = 0
                self.car_controls.throttle       = 0

        elif self.mode=="random":
            if self.index % 4 == 0:
                x = np.random.randint(1, high=4)
                if x==1:
                    if self.client.getCarState().speed >= self.MAX_SPEED-5.0:
                        print "[DRIVER]: BRAKE"
                        self.car_controls.brake    = 1
                        self.car_controls.throttle = 0
                    else:
                        print "[DRIVER]: THROTLE"
                        self.car_controls.brake    = 0
                        self.car_controls.throttle = 0.125
                elif x==2:
                    print "[DRIVER]: CONST."
                    self.car_controls.brake    = 0
                    self.car_controls.throttle = 0
                else:
                    if self.client.getCarState().speed >= self.MAX_SPEED/2.0:
                        print "[DRIVER]: BRAKE"
                        self.car_controls.brake    = 1
                        self.car_controls.throttle = 0
                    else:
                        print "[DRIVER]: THROTLE"
                        self.car_controls.brake    = 0
                        self.car_controls.throttle = 0.125
                self.car_controls.steering = np.random.sample()*2*0.125 - 0.125
        self.index += 1
