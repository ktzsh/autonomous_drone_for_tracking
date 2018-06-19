import os
import sys
import time
import json
import numpy as np

python_path = os.path.abspath('AirSim/PythonClient')
sys.path.append(python_path)
from AirSimClient import *

class CarConnector:
    def __init__(self, config):
        self.max_speed    = config['MAX_SPEED']
        self.min_speed    = config['MIN_SPEED']
        self.max_actions  = config['MAX_ACTIONS_PER_SEQ']
        self.modes        = config['SELECTED_MODES']
        self.throttle     = config['THROTTLE']
        self.max_steering = config['STEERING']

        self.index        = 0
        self.trajectory   = 0

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
        if self.index % self.max_actions == 0:
            self.mode = np.random.choice(self.modes, 1)

        if self.mode=="still":
            self.car_controls.brake    = 0
            self.car_controls.throttle = 0
            self.car_controls.steering = 0

        if self.mode=="acc":
            if self.car_controls.is_manual_gear:
                self.car_controls.is_manual_gear = False
                self.car_controls.manual_gear    = 0
            self.car_controls.brake    = 0
            self.car_controls.throttle = self.throttle
            self.car_controls.steering = 0

        if self.mode=="deacc":
            self.car_controls.throttle       = -1*self.throttle
            self.car_controls.is_manual_gear = True
            self.car_controls.manual_gear    = -1

        if self.mode=="brake":
            if self.car_controls.is_manual_gear:
                self.car_controls.is_manual_gear = False
                self.car_controls.manual_gear    = 0
            self.car_controls.brake    = 1
            self.car_controls.throttle = 0
            self.car_controls.steering = 0

        if self.mode=="turn":
            if self.car_controls.is_manual_gear:
                self.car_controls.is_manual_gear = False
                self.car_controls.manual_gear    = 0
            self.car_controls.brake    = 0
            self.car_controls.throttle = self.throttle
            self.car_controls.steering = (np.random.sample()*2 - 1)*self.max_steering

        elif self.mode=="random":
            if self.index % 4 == 0:
                x = np.random.randint(1, high=4)
                if x==1:
                    if self.client.getCarState().speed >= self.max_speed-5.0:
                        self.car_controls.brake    = 1
                        self.car_controls.throttle = 0
                    else:
                        self.car_controls.brake    = 0
                        self.car_controls.throttle = self.throttle
                elif x==2:
                    self.car_controls.brake    = 0
                    self.car_controls.throttle = 0
                else:
                    if self.client.getCarState().speed >= self.max_speed/2.0:
                        self.car_controls.brake    = 1
                        self.car_controls.throttle = 0
                    else:
                        self.car_controls.brake    = 0
                        self.car_controls.throttle = self.throttle
                self.car_controls.steering = (np.random.sample()*2 - 1.0)*self.throttle
        self.index += 1
