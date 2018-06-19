import os
import gym
import cv2
import math
import yaml
import random
import numpy as np
from gym import wrappers

from connectors.MultiRotorConnector import MultiRotorConnector
from connectors.CarConnector import CarConnector

class State():
    DELTA_X    = 0.0
    DELTA_Y    = 0.0
    DELTA_Z    = 0.0

class Environment(object):
    def __init__(self, name):
        with open('cfg/' + name + '.yml', 'rb') as stream:
            self.config = yaml.load(stream)

        self.quiet       = self.config['QUIET']
        self.repeat      = self.config['REPEAT_ACTION']
        self.init_x      = self.config['ENVIRONMENT']['MULTIROTOR']['INIT_X']
        self.init_y      = self.config['ENVIRONMENT']['MULTIROTOR']['INIT_Y']
        self.init_z      = self.config['ENVIRONMENT']['MULTIROTOR']['INIT_Z']
        self.max_dist_xy = self.config['ENVIRONMENT']['MAX_DIST_XY']
        self.max_dist_z  = self.config['ENVIRONMENT']['MAX_DIST_Z']
        self.scale_xy    = self.config['ENVIRONMENT']['REWARD']['SCALE_XY']
        self.scale_z     = self.config['ENVIRONMENT']['REWARD']['SCALE_Z']

        self.scaling_factor    = self.config['ENVIRONMENT']['ACTION_SCALING_FACTOR']
        self.action_space      = np.arange(self.config['ENVIRONMENT']['NUM_ACTIONS'])
        self.observation_dims  = tuple(self.config['ENVIRONMENT']['STATES_SHAPE'])

        self.current_episode  = 0
        self.current_timestep = 0

        self._uav_connector = MultiRotorConnector(self.init_x, self.init_y, self.init_z)
        self._car_connector = CarConnector(self.config['ENVIRONMENT']['CAR'])

    def observation_shape(self):
        return self.observation_dims

    def nb_actions(self):
        return len(self.action_space)

    def interpret_action(self, action):
        if action == 0:
            quad_offset = (0, 0, 0)
        elif action == 1:
            quad_offset = (self.scaling_factor, 0, 0)
        elif action == 2:
            quad_offset = (0, self.scaling_factor, 0)
        elif action == 3:
            quad_offset = (0, 0 , self.scaling_factor)
        elif action == 4:
            quad_offset = (-self.scaling_factor, 0, 0)
        elif action == 5:
            quad_offset = (0, -self.scaling_factor, 0)
        elif action == 6:
            quad_offset = (0, 0, -self.scaling_factor)

        return quad_offset

    def state_to_array(self, state):
        out = np.zeros((3,), dtype='float32')
        out[0] = float(state.DELTA_X)/float(self.max_dist_xy)
        out[1] = float(state.DELTA_Y)/float(self.max_dist_xy)
        out[2] = float(state.DELTA_Z)/float(self.max_dist_z)
        return out

    def reset(self):
        self.current_episode  += 1
        self.current_timestep += 1

        car_pos, car_ort = self._car_connector.reset()
        self._uav_connector.reset()
        # self._uav_connector.move_by_angle(car_ort, self._uav_connector.INIT_Z)
        self._car_connector.drive()

        car_pos = self._car_connector.get_position()
        uav_pos = self._uav_connector.get_position()

        _state = State()
        _state.DELTA_X = car_pos.x_val - uav_pos.x_val
        _state.DELTA_Y = car_pos.y_val - uav_pos.y_val
        _state.DELTA_Z = uav_pos.z_val - self.init_z

        if not self.quiet:
            print "\n-----Parameters-----"
            print "Delta     X:", _state.DELTA_X
            print "Delta     Y:", _state.DELTA_Y
            print "Delta     Z:", _state.DELTA_Z

        self.current_state  = _state
        return self.state_to_array(_state)

    def step(self, action):
        done   = 0
        reward = 0.0
        action = self.interpret_action(action)

        uav_vel = self._uav_connector.get_velocity()
        if not self.quiet:
            print "\n-----Parameters-----"
            print "Delta X    :", self.current_state.DELTA_X
            print "Delta Y    :", self.current_state.DELTA_Y
            print "Delta Z    :", self.current_state.DELTA_Z
            print "UAV Vel    :", (uav_vel.x_val, uav_vel.y_val, uav_vel.z_val)

        for i in range(random.choice(self.repeat)):
            self._car_connector.drive()
            self._uav_connector.move_by_velocity(action)

            car_pos = self._car_connector.get_position()
            uav_pos = self._uav_connector.get_position()

            if not self.quiet:
                print "Car Pos    :", car_pos.x_val, car_pos.y_val, car_pos.z_val
                print "UAV Pos    :", uav_pos.x_val, uav_pos.y_val, uav_pos.z_val

            dist_x    = car_pos.x_val - uav_pos.x_val
            dist_y    = car_pos.y_val - uav_pos.y_val
            dist_xy   = np.linalg.norm([dist_x, dist_y])
            reward_xy = max(0.0, 1.0 - dist_xy/self.max_dist_xy)

            dist_z    = abs(uav_pos.z_val - self.init_z)
            reward_z  = -1.0 * (dist_z/self.max_dist_z)
            reward_z  = max(-1.0, (reward_z ** 2))

            reward   += (self.scale_xy * reward_xy + self.scale_z * reward_z)

            if not self.quiet:
                print "Reward XY  :", reward_xy, \
                    "\nReward Z   :", reward_z

            # NEXT frame
            _state = State()
            _state.DELTA_X = car_pos.x_val - uav_pos.x_val
            _state.DELTA_Y = car_pos.y_val - uav_pos.y_val
            _state.DELTA_Z = uav_pos.z_val - self.init_z
            self.current_state = _state

            if dist_z > self.max_dist_z or dist_xy > self.max_dist_xy:
                reward = -10.0
                done   = 1
                return self.current_state, reward, done

        reward += 0.1

        if not self.quiet:
            print "Reward     :", reward
            print "Action RL  :", action, "+m/s"
            print "Done       :", done

        self.current_timestep += 1
        return self.state_to_array(_state), reward, done
