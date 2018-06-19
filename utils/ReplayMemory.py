import numpy as np
import pickle

class ReplayMemory(object):
    """
    ReplayMemory keeps track of the environment dynamic.
    We store all the transitions (s(t), action, s(t+1), reward, done).
    The replay memory allows us to efficiently sample minibatches from it, and generate the correct state representation
    (w.r.t the number of previous frames needed).
    """
    def __init__(self, size, sample_shape, history_length=4, restore=None):
        self._pos            = 0
        self._count          = 0
        self._max_size       = size
        self._history_length = max(1, history_length)
        self._state_shape    = sample_shape

        if restore is not None:
            self._states     = np.load(restore + 'states.npy')
            self._actions    = np.load(restore + 'actions.npy')
            self._rewards    = np.load(restore + 'rewards.npy')
            self._terminals  = np.load(restore + 'terminals.npy')
        else:
            self._states     = np.zeros((size,) + sample_shape, dtype=np.float32)
            self._actions    = np.zeros(size, dtype=np.uint8)
            self._rewards    = np.zeros(size, dtype=np.float32)
            self._terminals  = np.zeros(size, dtype=np.float32)

    def __len__(self):
        """ Returns the number of items currently present in the memory
        """
        return self._count

    def append(self, state, action, reward, done):
        """ Appends the specified transition to the memory.
        """
        assert state.shape == self._state_shape, \
            'Invalid state shape (required: %s, got: %s)' % (self._state_shape, state.shape)

        self._states[self._pos]    = state
        self._actions[self._pos]   = action
        self._rewards[self._pos]   = reward
        self._terminals[self._pos] = done

        self._count = max(self._count, self._pos + 1)
        self._pos   = (self._pos + 1) % self._max_size

    def sample(self, size):
        """ Generate size random integers mapping indices in the memory.
            The returned indices can be retrieved using #get_state().
            See the method #minibatch() if you want to retrieve samples directly.
        """
        # Local variable access is faster in loops
        count, pos, history_len, terminals = self._count - 1, self._pos, \
                                             self._history_length, self._terminals
        indexes = []

        while len(indexes) < size:
            index = np.random.randint(history_len, count)

            if index not in indexes:

                # if not wrapping over current pointer,
                # then check if there is terminal state wrapped inside
                if not (index >= pos > index - history_len):
                    if not terminals[(index - history_len):index].any():
                        indexes.append(index)

        return indexes

    def minibatch(self, size):
        """ Generate a minibatch with the number of samples specified by the size parameter.
        """
        indexes = self.sample(size)

        pre_states  = np.array([self.get_state(index) for index in indexes], dtype=np.float32)
        post_states = np.array([self.get_state(index + 1) for index in indexes], dtype=np.float32)
        actions     = self._actions[indexes]
        rewards     = self._rewards[indexes]
        dones       = self._terminals[indexes]

        return pre_states, actions, post_states, rewards, dones

    def get_state(self, index):
        """
        Return the specified state with the replay memory. A state consists of
        the last `history_length` perceptions.
        """
        if self._count == 0:
            raise IndexError('Empty Memory')

        index         %= self._count
        history_length = self._history_length

        # If index > history_length, take from a slice
        if index >= history_length:
            return self._states[(index - (history_length - 1)):index + 1, ...]
        else:
            indexes = np.arange(index - history_length + 1, index + 1)
            return self._states.take(indexes, mode='wrap', axis=0)

    def save(self, path):
        np.save(path + 'states.npy', self._states)
        np.save(path + 'actions.npy', self._actions)
        np.save(path + 'rewards.npy', self._rewards)
        np.save(path + 'terminals.npy', self._terminals)
