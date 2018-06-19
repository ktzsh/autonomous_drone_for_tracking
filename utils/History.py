import numpy as np

class History(object):
    """
    Accumulator keeping track of the N previous frames to be used by the agent
    for evaluation
    """
    def __init__(self, shape):
        self._buffer = np.zeros(shape, dtype=np.float32)

    @property
    def value(self):
        """ Underlying buffer with N previous states stacked along first axis
        """
        return self._buffer

    def append(self, state):
        """ Append state to the history
        """
        self._buffer[:-1] = self._buffer[1:]
        self._buffer[-1]  = state

    def reset(self):
        """ Reset the memory. Underlying buffer set all indexes to 0
        """
        self._buffer.fill(0)
