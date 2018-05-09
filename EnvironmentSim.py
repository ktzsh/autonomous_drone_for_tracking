import os
import sys
import cv2
import time
import math
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
import xml.etree.ElementTree as ET

python_path = os.path.abspath('TF_ObjectDetection')
sys.path.append(python_path)
from object_detection.utils import visualization_utils as vis_util

from Detector import Detector
from MultiRotorConnector import MultiRotorConnector
from CarConnector import CarConnector

class State():
    DELTA_X    = 0.0
    DELTA_Y    = 0.0
    DELTA_Z    = 0.0
    pass

class EnvironmentSim:
    def __init__(self, image_shape=(720, 1280)):
        self.current_episode  = 0
        self.current_timestep = 0

        self.im_width  = image_shape[1]
        self.im_height = image_shape[0]

        self.max_dist   = 80.0
        self.max_dist_z = 10.0

        self.MIN_ALTITUDE = -0.5
        self.MAX_ALTITUDE = -40.0

        self.current_frame    = None
        self.current_output   = None

        self._detector      = Detector()
        self._uav_connector = MultiRotorConnector()
        self._car_connector = CarConnector()

    def state_to_array(self, state):
        out = np.zeros((3,), dtype='float32')
        out[0] = float(state.DELTA_X)/float(self.max_dist)
        out[1] = float(state.DELTA_Y)/float(self.max_dist)
        out[2] = float(state.DELTA_Z)/float(self.max_dist_z)
        return out

    def reset(self):
        self.current_episode  += 1
        self.current_timestep += 1

        car_pos, car_ort = self._car_connector.reset()
        self._uav_connector.reset()
        self._uav_connector.move_by_angle(car_ort, self._uav_connector.INIT_Z)
        self._car_connector.drive()

        car_pos = self._car_connector.get_position()
        uav_pos = self._uav_connector.get_position()

        _state = State()
        _state.DELTA_X = car_pos.x_val - uav_pos.x_val
        _state.DELTA_Y = car_pos.y_val - uav_pos.y_val
        _state.DELTA_Z = uav_pos.z_val - self._uav_connector.INIT_Z
        print "\n-----Parameters-----"
        print "Delta     X:", _state.DELTA_X
        print "Delta     Y:", _state.DELTA_Y
        print "Delta     Z:", _state.DELTA_Z

        self.current_state  = _state

        return self.state_to_array(_state)


    def step(self, action):
        # TRACKER
        done   = 0
        reward = 0.0

        uav_vel = self._uav_connector.get_velocity()
        print "\n-----Parameters-----"
        print "Delta X    :", self.current_state.DELTA_X
        print "Delta Y    :", self.current_state.DELTA_Y
        print "Delta Z    :", self.current_state.DELTA_Z
        print "UAV Vel    :", (uav_vel.x_val, uav_vel.y_val, uav_vel.z_val)

        self._car_connector.drive()
        self._uav_connector.move_by_velocity(action)

        car_pos = self._car_connector.get_position()
        uav_pos = self._uav_connector.get_position()
        print "Car Pos    :", car_pos.x_val, car_pos.y_val, car_pos.z_val
        print "UAV Pos    :", uav_pos.x_val, uav_pos.y_val, uav_pos.z_val

        dist_x = car_pos.x_val - uav_pos.x_val
        dist_y = car_pos.y_val - uav_pos.y_val
        dist_xy = np.linalg.norm([dist_x, dist_y])
        reward_xy = (1.0 - dist_xy/self.max_dist) * 1.0

        dist_z = abs(uav_pos.z_val - self._uav_connector.INIT_Z)
        reward_z  = dist_z/self.max_dist_z
        if reward_z > 1.0:
            reward_z = -1.0 * math.exp(reward_z)
        else:
            reward_z = -1.0 * reward_z

        reward    = (reward_xy + reward_z)
        print "Distance XY:", dist_xy, \
            "\nDistance Z :", dist_z
        print "Reward XY  :", reward_xy, \
            "\nReward Z   :", reward_z

        print "Reward (+T):", reward
        reward += 0.15

        # cv2.imshow('Simulation', frame)
        # cv2.waitKey(5)

        # NEXT frame
        _state = State()

        if (uav_pos.z_val > self.MIN_ALTITUDE) or \
           (uav_pos.z_val < self.MAX_ALTITUDE) or \
           reward_xy<0.0 or \
           car_pos.z_val<=-2.0:
            reward = -10
            done   = 1
        else:
            _state.DELTA_X = car_pos.x_val - uav_pos.x_val
            _state.DELTA_Y = car_pos.y_val - uav_pos.y_val
            _state.DELTA_Z = uav_pos.z_val - self._uav_connector.INIT_Z
            self.current_state = _state

        print "Reward     :", reward
        print "Action RL  :", action, "+m/s"
        print "Done       :", done

        self.current_timestep += 1

        return self.state_to_array(_state), reward, done
