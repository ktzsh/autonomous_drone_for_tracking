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

def test_images_at_altitude():
    from MultiRotorConnector import MultiRotorConnector
    connector = MultiRotorConnector()
    time.sleep(3)

    _ = connector.get_frame(path='dummy.png')
    position = connector.get_position()
    x_val = position.x_val
    y_val = position.y_val
    z_val = position.z_val
    print "Initial Positions:", x_val, y_val, z_val

    for i,z in enumerate([z_val-10, z_val-5, z_val, z_val+5, z_val+10]):
        connector.move_to_position([x_val, y_val, z])
        time.sleep(3)
        path = str(i+1) + '.png'
        _ = connector.get_frame(path=path)
        print "\tTest Case:", x_val, y_val, z, path

if __name__=='__main__':

    test_car_reset()
