from Detector import Detector
from MultiRotorConnector import MultiRotorConnector

class State():
    VEL_X    = None
    VEL_Y    = None
    VEL_Z    = None
    POS_X    = None
    POS_Y    = None
    WIDTH    = None
    HEIGHT   = None
    ALTITUDE = None
    pass

class Environment:
    def __init__(self):
        self._connector = MultiRotorConnector()
        self._detector  = Detector()

    def update(self, state):
        velocity = self._connector.get_velocity()
        state.VEL_X = velocity.x_val
        state.VEL_Y = velocity.y_val
        state.VEL_Z = velocity.z_val

        position = self._connector.get_position()
        state.ALTITUDE = position.z_val

        frame        = self._connector.get_frame()
        output       = self._detector.detect(frame)
        state.POS_X  = output[0]
        state.POS_Y  = output[1]
        state.WIDTH  = output[2]
        state.HEIGHT = output[3]

        print "----State Values----"
        print "Velocity X:", state.VEL_X
        print "Velocity Y:", state.VEL_Y
        print "Velocity Z:", state.VEL_Z
        print "Altitude  :", state.ALTITUDE
        print "Position X:", state.POS_X
        print "Position Y:", state.POS_Y
        print "Width     :", state.WIDTH
        print "Height    :", state.HEIGHT

    def reset(self):
        _state = State()
        self.update(_state)
        return _state

    def step(self, action, duration=5):
        _state = State()
        self._connector.move_by_velocity(action, duration=duration)
        time.sleep(1)
        self.update(_state)
        return _state
