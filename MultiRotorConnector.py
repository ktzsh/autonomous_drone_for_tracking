import os
import sys
import time
import numpy as np
from PIL import Image

python_path = os.path.abspath('AirSim/PythonClient')
sys.path.append(python_path)
from AirSimClient import *

class MultiRotorConnector:
    client = None

    INIT_X = -0.3
    INIT_Y = -0.3
    INIT_Z = -15

    def __init__(self):
        self.client = MultirotorClient()
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)

        self.client.takeoff()
        self.client.moveToPosition(self.INIT_X, self.INIT_Y, self.INIT_Z, 5)
        # self.client.moveByVelocity(1, -0.67, -0.8, 5)
        time.sleep(0.5)

    # The camera ID 0 to 4 corresponds to center front, left front, right front, center downward, center rear respectively.
    def get_frame(self, camera_id=3):
        response = self.client.simGetImages([ImageRequest(camera_id, AirSimImageType.Scene, False, False)])[0]
        img1d = np.fromstring(response.image_data_uint8, dtype=np.uint8) # get numpy array
        img_rgba = img1d.reshape(response.height, response.width, 4) # reshape array to 4 channel image array H X W X 4
        img_rgba = np.flipud(img_rgba) # original image is fliped vertically

        # TODO - Implement conversion of uncmpressed RGBA to RGB without writing to disk
        self.client.write_png(os.path.normpath('frame.png'), img_rgba)
        img_rgb = np.asarray(Image.open('frame.png').convert('RGB'), dtype=np.uint8)
        return img_rgb

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
