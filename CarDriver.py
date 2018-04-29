import os
import sys
import time
import numpy as np

python_path = os.path.abspath('AirSim/PythonClient')
sys.path.append(python_path)
from AirSimClient import *

class CarDriver:
    def __init__(self):
        self.MAX_SPEED    = 50
        self.MIN_SPEED    = 0
        
        self.trajectory   = 0
        self.trajectories = ['acc', 'deacc', 'still', 'rand-turn', 'circle']  
        
        self.max_actions = 100
              
        self.client = CarClient()
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        self.car_controls = CarControls()

    def disconnect(self):
        self._connector.enableApiControl(False)

    def drive(self):
        trajectories()
        self.client.setCarControls(self.car_controls)
        time.sleep(0.5)
    
    def drive(self):
        self.car_controls.brake = 0
        self.car_controls.throttle = 1
        self.car_controls.steering = 0
        # if self.trajectory=0:
        steer = (np.random.sample(1))*2*0.1 - 0.1
        self.car_controls.steering = steer
        self.trajectory = (self.trajectory + 1) % len(self.trajectories)
