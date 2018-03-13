import os
import sys
import time
import numpy as np

python_path = os.path.abspath('AirSim/PythonClient')
sys.path.append(python_path)
from AirSimClient import *

class CarConnector:
    def __init__(self):
        self._connector = CarClient()
        self._connector.confirmConnection()
        self.api_control = False

    def connect(self):
        self._connector.enableApiControl(True)
        self.api_control = True

    def disconnect(self):
        self._connector.enableApiControl(False)
        self.api_control = False

    def get_position(self):
        if not self.api_control:
            self.connect()
        state = self._connector.getCarState()
        print state
        return state

    def reset():
        self._connector.reset()
        state = self.get_position()
        self.disconnect()
        return state
