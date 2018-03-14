import time

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
        for j,x in enumerate([-3, 0, 3]):
            for k,y in enumerate([-3, 0, 3]):
                connector.move_to_position([x_val+x, y_val+y, z_val+z])
                time.sleep(1)
                path = 'TF_ObjectDetection/data/orig_data/' + str(count) + '.png'
                count += 1
                _ = connector.get_frame(path=path)
                print "\tTest Case:", x_val+x, y_val+y, z_val+z, path

if __name__=='__main__':

    test_get_images_at_positions()
