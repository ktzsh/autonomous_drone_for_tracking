import os
import yaml
import pickle
import random
import numpy as np

from utils.History import History
from utils.ReplayMemory import ReplayMemory

class BaseAgent(object):
    def __init__(self, name, input_shape, nb_actions):

        with open('cfg/' + name + '.yml', 'rb') as stream:
            self.config = yaml.load(stream)

        self.ENV_NAME               = name
        self.DOUBLE_Q               = self.config['DOUBLE_Q']
        self.QUIET                  = self.config['QUIET']

        self.MAX_EPISODES           = self.config['AGENT']['MAX_EPISODES']
        self.STATE_LENGTH           = self.config['AGENT']['STATE_LENGTH']
        self.GAMMA                  = self.config['AGENT']['GAMMA']
        self.EXPLORATION_STEPS      = self.config['AGENT']['EXPLORATION_STEPS']
        self.INITIAL_EPSILON        = self.config['AGENT']['INITIAL_EPSILON']
        self.FINAL_EPSILON          = self.config['AGENT']['FINAL_EPSILON']
        self.INITIAL_REPLAY_SIZE    = self.config['AGENT']['INITIAL_REPLAY_SIZE']
        self.MEMORY_SIZE            = self.config['AGENT']['MEMORY_SIZE']
        self.BATCH_SIZE             = self.config['AGENT']['BATCH_SIZE']
        self.TARGET_UPDATE_INTERVAL = self.config['AGENT']['TARGET_UPDATE_INTERVAL']
        self.TRAIN_INTERVAL         = self.config['AGENT']['TRAIN_INTERVAL']
        self.LEARNING_RATE          = self.config['AGENT']['LEARNING_RATE']
        self.MOMENTUM               = self.config['AGENT']['MOMENTUM']
        self.MIN_GRAD               = self.config['AGENT']['MIN_GRAD']
        self.DECAY_RATE             = self.config['AGENT']['DECAY_RATE']
        self.SAVE_INTERVAL          = self.config['AGENT']['SAVE_INTERVAL']
        self.LOAD_NETWORK           = self.config['AGENT']['LOAD_NETWORK']
        self.SAVE_NETWORK_PATH      = self.config['AGENT']['SAVE_NETWORK_PATH']
        self.SAVE_SUMMARY_PATH      = self.config['AGENT']['SAVE_SUMMARY_PATH']
        self.SAVE_TRAIN_STATE       = self.config['AGENT']['SAVE_TRAIN_STATE']
        self.SAVE_TRAIN_STATE_PATH  = self.config['AGENT']['SAVE_TRAIN_STATE_PATH']

        self.t             = 0
        self.epsilon       = self.INITIAL_EPSILON
        self.epsilon_step  = (self.INITIAL_EPSILON - self.FINAL_EPSILON) / (self.EXPLORATION_STEPS )

        self.total_reward  = 0.0
        self.total_q_max   = 0.0
        self.total_q_mean  = 0.0
        self.total_loss    = 0
        self.duration      = 0
        self.episode       = 0

        self.input_shape   = (self.STATE_LENGTH, ) + input_shape
        self.nb_actions    = nb_actions
        self._history      = History(self.input_shape)

        if not os.path.exists(self.SAVE_NETWORK_PATH + self.ENV_NAME):
            os.makedirs(self.SAVE_NETWORK_PATH + self.ENV_NAME)
        if not os.path.exists(self.SAVE_SUMMARY_PATH + self.ENV_NAME):
            os.makedirs(self.SAVE_SUMMARY_PATH + self.ENV_NAME)
        if not os.path.exists(self.SAVE_TRAIN_STATE_PATH + self.ENV_NAME):
            os.makedirs(self.SAVE_TRAIN_STATE_PATH + self.ENV_NAME)

        restore_prefix = self.SAVE_TRAIN_STATE_PATH + self.ENV_NAME + '/'
        if os.path.exists(restore_prefix + 'snapshot.lock'):
            print "Restoring Training State Snapshot from", restore_prefix
            with open(restore_prefix + 'snapshot.lock', 'rb') as f:
                params = f.read().split('\n')
                self.episode            = int(params[1])
                self.t                  = int(params[2])
                self.epsilon            = float(params[3])

            self._memory     = ReplayMemory(self.MEMORY_SIZE, self.input_shape[1:], self.STATE_LENGTH,
                                                                                    restore=restore_prefix)
            self.tb_counter  = len([log for log in os.listdir(os.path.expanduser(
                                            self.SAVE_SUMMARY_PATH + self.ENV_NAME)) if 'Experiment_' in log])
        else:
            self._memory     = ReplayMemory(self.MEMORY_SIZE, self.input_shape[1:], self.STATE_LENGTH)
            self.tb_counter  = len([log for log in os.listdir(os.path.expanduser(
                                            self.SAVE_SUMMARY_PATH + self.ENV_NAME)) if 'Experiment_' in log]) + 1
            os.makedirs(self.SAVE_SUMMARY_PATH + self.ENV_NAME + '/Experiment_' + str(self.tb_counter))

        # Save snapshot of run configuration used
        with open(self.SAVE_SUMMARY_PATH + self.ENV_NAME + '/Experiment_'
                    + str(self.tb_counter) + '/config.run.yml', 'wb') as stream:
            yaml.dump(self.config, stream, default_flow_style=False)

    def build_network(self, input_shape):
        raise NotImplementedError

    def build_training_op(self, q_network_weights):
        raise NotImplementedError

    def act(self, state):
        """ This allows the agent to select the next action to perform in regard of the current state of the environment.
        It follows the terminology used in the Nature paper.
        """
        # Append the state to the short term memory (ie. History)
        self._history.append(state)

        if self.epsilon >= random.random() or self.t < self.INITIAL_REPLAY_SIZE:
            # Choose an action randomly
            action = random.randrange(self.nb_actions)
        else:
            # Use the network to output the best action
            env_with_history = self._history.value
            action = np.argmax(self.sess.run(self.q_values, feed_dict={self.s: env_with_history.reshape((1,) + env_with_history.shape)}))

        # Anneal epsilon linearly over time
        if self.epsilon > self.FINAL_EPSILON and self.t >= self.INITIAL_REPLAY_SIZE:
            self.epsilon -= self.epsilon_step

        self.t += 1
        return action

    def observe(self, old_state, action, reward, done):
        """ This allows the agent to observe the output of doing the action it selected through act() on the old_state
        """
        # If done, reset short term memory (ie. History)
        self.total_reward += reward
        env_with_history = self._history.value
        q_vals = self.sess.run(self.q_values, feed_dict={self.s: env_with_history.reshape((1,) + env_with_history.shape)})
        self.total_q_max += np.max(q_vals)
        self.total_q_mean += np.mean(q_vals)
        self.duration += 1

        if done:
            # Write summary
            if self.t >= self.INITIAL_REPLAY_SIZE:
                stats = [self.total_reward, self.total_q_max / float(self.duration),
                            self.total_q_mean / float(self.duration), self.duration,
                            self.total_loss / (float(self.duration)), self.t]
                for i in range(len(stats)):
                    self.sess.run(self.update_ops[i], feed_dict={
                        self.summary_placeholders[i]: float(stats[i])
                    })
                summary_str = self.sess.run(self.summary_op)
                self.summary_writer.add_summary(summary_str, self.episode + 1)

            # Debug
            if self.t < self.INITIAL_REPLAY_SIZE:
                mode = 'random'
            elif self.INITIAL_REPLAY_SIZE <= self.t < self.INITIAL_REPLAY_SIZE + self.EXPLORATION_STEPS:
                mode = 'explore'
            else:
                mode = 'exploit'

            if not self.QUIET:
                print "-----EPISODE SUMMARY-----"
                print "EPISODE    :", self.episode + 1, \
                    "\nTIMESTEP   :", self.t, \
                    "\nDURATION   :", self.duration, \
                    "\nEPSILON    :", self.epsilon, \
                    "\nTOTALREWARD:", self.total_reward, \
                    "\nAVG_MAX_Q  :", self.total_q_max / float(self.duration), \
                    "\nAVG_MEAN_Q :", self.total_q_mean / float(self.duration), \
                    "\nAVG_LOSS   :", self.total_loss / float(self.duration), \
                    "\nMODE       :", mode
                print "-------------------------"

            self.total_reward = 0
            self.total_q_max  = 0
            self.total_q_mean = 0
            self.total_loss   = 0
            self.duration     = 0
            self.episode     += 1

            # Reset the short term memory
            self._history.reset()

        # Append to long term memory
        self._memory.append(old_state, action, reward, done)

        if self.SAVE_TRAIN_STATE and (self.t % self.SAVE_INTERVAL == 0 ):
            if os.path.exists(self.SAVE_TRAIN_STATE_PATH + self.ENV_NAME + '/snapshot.lock'):
                os.remove(self.SAVE_TRAIN_STATE_PATH + self.ENV_NAME + '/snapshot.lock')
            snapshot_params =  'Snapshot Parameters [Episode, Timestep, Actions Taken, Epsilon]\n' + \
                                        str(self.episode) + '\n' + \
                                        str(self.t) + '\n' + \
                                        str(self.epsilon) + '\n'
            self._memory.save(self.SAVE_TRAIN_STATE_PATH + self.ENV_NAME + '/')
            with open(self.SAVE_TRAIN_STATE_PATH + self.ENV_NAME + '/snapshot.lock', 'wb') as f:
                f.write(snapshot_params)

    def train(self):
        """ This allows the agent to train itself to better understand the environment dynamics.
        The agent will compute the expected reward for the state(t+1)
        and update the expected reward at step t according to this.

        The target expectation is computed through the Target Network, which is a more stable version
        of the Action Value Network for increasing training stability.

        The Target Network is a frozen copy of the Action Value Network updated as regular intervals.
        """
        # Train network
        if (self.t % self.TRAIN_INTERVAL) == 0:
            self.train_network()

        # Update target network
        if self.t % self.TARGET_UPDATE_INTERVAL == 0:
            self.sess.run(self.update_target_network)

        # Save network
        if self.t % self.SAVE_INTERVAL == 0:
            save_path = self.saver.save(self.sess, self.SAVE_NETWORK_PATH + self.ENV_NAME + '/chkpnt', global_step=self.t)
            if not self.QUIET: print "Successfully saved:", save_path

    def train_network(self):
        raise NotImplementedError

    def setup_summary(self):
        raise NotImplementedError

    def load_network(self):
        raise NotImplementedError

    def test(self, state):
        self.t += 1
        self._history.append(state)

        if self.t >= self.STATE_LENGTH:
            env_with_history = self._history.value
            action = np.argmax(self.sess.run(self.q_values, feed_dict={self.s: env_with_history.reshape((1,) + env_with_history.shape)}))
            return action
        else:
            return 0
