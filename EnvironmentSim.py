import os
import sys
import cv2
import time
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
    pass

class EnvironmentSim:
    def __init__(self, image_shape=(720, 1280), max_dist=30.0, max_guided_eps=1000):
        self.current_episode = 0
        self.max_guided_eps  = max_guided_eps

        self.im_width  = image_shape[1]
        self.im_height = image_shape[0]

        self.max_dist   = max_dist

        self.current_frame    = None
        self.current_output   = None

        self._detector      = Detector()
        self._uav_connector = MultiRotorConnector()
        self._car_connector = CarConnector()

    def state_to_array(self, state):
        out = np.zeros((2,), dtype='float32')
        out[0] = float(state.DELTA_X)/float(self.max_dist)
        out[1] = float(state.DELTA_Y)/float(self.max_dist)
        return out

    def reset(self):
        self.current_episode += 1

        car_pos, car_ort = self._car_connector.reset()
        self._uav_connector.reset()
        self._uav_connector.move_by_angle(car_ort, self._uav_connector.INIT_Z)
        self._car_connector.drive()

        car_pos = self._car_connector.get_position()
        uav_pos = self._uav_connector.get_position()

        _state = State()
        _state.DELTA_X = car_pos.x_val - uav_pos.x_val
        _state.DELTA_Y = car_pos.y_val - uav_pos.y_val
        print "\n-----Parameters-----"
        print "Delta    X:", _state.DELTA_X
        print "Delta    Y:", _state.DELTA_Y

        self.current_state  = _state

        return self.state_to_array(_state)


    def step(self, action):
        # TRACKER
        done   = 0
        reward = 0.0

        uav_vel = self._uav_connector.get_velocity()
        print "\n-----Parameters-----"
        print "Delta    X:", self.current_state.DELTA_X
        print "Delta    Y:", self.current_state.DELTA_Y
        print "UAV Vel   :", (uav_vel.x_val, uav_vel.y_val, uav_vel.z_val)

        self._car_connector.drive()
        self._uav_connector.move_by_velocityz(action)

        car_pos = self._car_connector.get_position()
        uav_pos = self._uav_connector.get_position()
        print "Car Pos    :", car_pos.x_val, car_pos.y_val, car_pos.z_val
        print "UAV Pos    :", uav_pos.x_val, uav_pos.y_val, uav_pos.z_val

        dist_x = float(car_pos.x_val - uav_pos.x_val)
        dist_y = float(car_pos.y_val - uav_pos.y_val)
        dist = np.linalg.norm([dist_x, dist_y])/self.max_dist
        reward = (1-2.0*dist)
        print "Distance   :", dist, \
            "\nDist Reward:", (1-dist)
        print "Action RL  :", action, "+m/s"

        # cv2.imshow('Simulation', frame)
        # cv2.waitKey(5)

        # NEXT frame
        _state = State()

        if reward<=-1.0 or car_pos.z_val<=-2.0 or (self._car_connector.index % self._car_connector.max_actions == 0):
            done   = 1
        else:
            _state.DELTA_X = car_pos.x_val - uav_pos.x_val
            _state.DELTA_Y = car_pos.y_val - uav_pos.y_val
            self.current_state = _state

        print "Reward     :", reward
        print "Done       :", done

        return self.state_to_array(_state), reward, done
