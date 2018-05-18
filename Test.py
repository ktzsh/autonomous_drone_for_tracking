import os
import sys
import time
python_path = os.path.abspath('AirSim/PythonClient')
sys.path.append(python_path)
from AirSimClient import *

def test_drone():
    client = MultirotorClient()
    client.confirmConnection()
    client.enableApiControl(True)
    client.armDisarm(True)
    client.takeoff()
    client.moveToPosition(0, 0, -15, 10)
    i = 0
    while True:
        offset = [0.25, 0.25, 0.0]
        quad_vel = client.getVelocity()
        quad_pos = client.getPosition()
        print i
        print quad_vel
        print quad_pos, "\n\n"
        # client.moveByVelocityZ( quad_vel.x_val + offset[0],
        #                         quad_vel.y_val + offset[1],
        #                        -15, 2)
        client.moveByVelocity( quad_vel.x_val + offset[0],
                                    quad_vel.y_val + offset[1],
                                    quad_vel.z_val + offset[2],
                                    5)
        time.sleep(0.5)
        i += 1

def test_car_reset():
    from CarConnector import CarConnector
    connector = CarConnector()
    state = connector.reset()

def test_detection():
    from Environment import Environment
    env = Environment()
    while True:
        current_state = env.reset()
        time.sleep(1)

def test_get_images_at_positions():
    from MultiRotorConnector import MultiRotorConnector
    connector = MultiRotorConnector()

    _ = connector.get_frame(path='dummy.png')
    position = connector.get_position()
    x_val = position.x_val
    y_val = position.y_val
    z_val = position.z_val
    print "Initial Positions:", x_val, y_val, z_val

    count = 0
    for i,z in enumerate([-9, -6, -3, 0, 3, 6, 9]):
        for j,x in enumerate([0, 5, -5]):
            for k,y in enumerate([0, 5, -5]):
                connector.move_to_position([x_val+x, y_val+y, z_val+z])
                time.sleep(1)
                path = 'TF_ObjectDetection/data/orig_data/' + str(count) + '.png'
                count += 1
                _ = connector.get_frame(path=path)
                print "\tTest Case:", x_val+x, y_val+y, z_val+z, path

if __name__=='__main__':
    test_drone()
