import os
import sys
import time
import numpy as np

python_path = os.path.abspath('AirSim/PythonClient')
sys.path.append(python_path)
from AirSimClient import *

class MultiRotorConnector:
    client = None

    INIT_X = -.55265
    INIT_Y = -31.9786
    INIT_Z = -19.0225

    def __init__(self):
        self.client = MultirotorClient()
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)

        self.client.takeoff()
        self.client.moveToPosition(self.INIT_X, self.INIT_Y, self.INIT_Z, 5)
        self.client.moveByVelocity(1, -0.67, -0.8, 5)
        time.sleep(0.5)

    # The camera ID 0 to 4 corresponds to center front, left front, right front, center downward, center rear respectively.
    def get_frame(self, camera_id=3):
        response = self.client.simGetImages([ImageRequest(camera_id, AirSimImageType.Scene, True, False)])[0]
        img_1d = np.array(response.image_data_float, dtype=np.float) # get numpy array
        img_rgba = img_1d.reshape(response.height, response.width, 4) # reshape array to 4 channel image array H X W X 4
        img_rgba = np.flipud(img_rgba) # original image is fliped vertically

        # DEBUG
        self.client.write_png(os.path.normpath('frame.png'), img_rgba)

        # TODO: convert img to RGB, check flip and decide reshape size
        img = img_rgba
        return img

    def get_velocity(self):
        return self.client.getVelocity()

    def get_position(self):
        return self.client.getPosition()

    def get_collision_info(self):
        return self.client.getCollisionInfo()

    def move_by_velocity(self, offset, duration=5):
        quad_vel = self.client.getVelocity()
        self.client.moveByVelocity( quad_vel.x_val + offset[0],
                                    quad_vel.y_val + offset[1],
                                    quad_vel.z_val + offset[2],
                                    duration
                                  )
        time.sleep(0.5)
