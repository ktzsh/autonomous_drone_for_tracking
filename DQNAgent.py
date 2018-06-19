import sys
import time
import yaml
import argparse
import numpy as np
from collections import deque

from Base import BaseAgent
from Environment import Environment

import tensorflow as tf
from keras.layers import Input, ConvLSTM2D, Dense, Flatten, Conv2D, Reshape
from keras.models import Model

class Agent(BaseAgent):
    def __init__(self, name, input_shape, action_space):
        # Call Base Class init method
        super(Agent, self).__init__(name, input_shape, action_space)

        tf.reset_default_graph()

        # Action Value model (used by agent to interact with the environment)
        self.s, self.q_values, self.q_network = self.build_network()
        self.q_network_weights                = self.q_network.trainable_weights
        self.q_network.summary()

        # Target model used to compute the target Q-values in training, updated
        # less frequently for increased stability.
        self.st, self.target_q_values, self.target_network = self.build_network()
        self.target_network_weights                        = self.target_network.trainable_weights

        # Define target network update operation
        self.update_target_network = [ self.target_network_weights[i].assign(self.q_network_weights[i])
                                            for i in range(len(self.target_network_weights))]

        # Define loss and gradient update operation
        self.a, self.y, self.loss, self.grads_update = self.build_training_op()

        self.sess  = tf.Session()
        self.saver = tf.train.Saver(self.q_network_weights)

        self.summary_placeholders, self.update_ops, self.summary_op = self.setup_summary()

        self.summary_writer = tf.summary.FileWriter(self.SAVE_SUMMARY_PATH + self.ENV_NAME + '/Experiment_'
                                                        + str(self.tb_counter), self.sess.graph)

        self.sess.run(tf.global_variables_initializer())

        # Load network
        if self.LOAD_NETWORK:
            self.load_network()

        # Initialize target network
        self.sess.run(self.update_target_network)


    def build_network(self):
        input_shape  = self.input_shape
        input_frames = Input(shape=input_shape, dtype='float32')
        x = Reshape(input_shape[1:-1] + (input_shape[0] * input_shape[-1],))(input_frames)

        # x = Flatten()(x)
        x = Dense(128, activation='tanh')(x)
        x = Dense(128, activation='tanh')(x)

        output = Dense(self.nb_actions, activation='linear')(x)
        model = Model(inputs=input_frames, outputs=output)

        s = tf.placeholder(tf.float32, (None,) + input_shape)
        q_values = model(s)

        return s, q_values, model

    def build_training_op(self):
        a = tf.placeholder(tf.int64, [None])
        y = tf.placeholder(tf.float32, [None])

        # Convert action to one hot vector
        a_one_hot = tf.one_hot(a, self.nb_actions, 1.0, 0.0)
        q_value   = tf.reduce_sum(tf.multiply(self.q_values, a_one_hot), reduction_indices=1)

        # Clip the error, the loss is quadratic when the error is in (-1, 1), and linear outside of that region
        error          = tf.abs(y - q_value)
        clipped_error  = tf.where(error < 1.0, 0.5 * tf.square(error), error - 0.5)
        loss           = tf.reduce_mean(clipped_error)

        optimizer      = tf.train.RMSPropOptimizer(self.LEARNING_RATE,
                                                    momentum=self.MOMENTUM,
                                                    epsilon=self.MIN_GRAD,
                                                    decay=self.DECAY_RATE)
        # optimizer      = tf.train.AdamOptimizer(self.LEARNING_RATE)
        grads_update   = optimizer.minimize(loss, var_list=self.q_network_weights)

        return a, y, loss, grads_update

    def train_network(self):
        ''' Extension to train() call - Batch generation and graph computations
        '''
        # Sample random minibatch of transition from replay memory
        state_batch, action_batch, next_state_batch, reward_batch, terminal_batch = self._memory.minibatch(self.BATCH_SIZE)

        if self.DOUBLE_Q:
            q_values_batch = self.sess.run(self.q_values, feed_dict={self.s: next_state_batch})
            max_q_value_idx = np.argmax(q_values_batch, axis=1)
            target_q_values_at_idx_batch = self.sess.run(self.target_q_values, feed_dict={self.st: next_state_batch})[:, max_q_value_idx]
            y_batch = reward_batch + (1 - terminal_batch) * self.GAMMA * target_q_values_at_idx_batch
        else:
            target_q_values_batch = self.sess.run(self.target_q_values, feed_dict={self.st: next_state_batch})
            y_batch = reward_batch + (1 - terminal_batch) * self.GAMMA * np.max(target_q_values_batch, axis=1)

        loss, _ = self.sess.run([self.loss, self.grads_update], feed_dict={
            self.s: state_batch,
            self.a: action_batch,
            self.y: y_batch
        })
        self.total_loss += loss

    def setup_summary(self):
        episode_total_reward = tf.Variable(0.)
        episode_avg_max_q    = tf.Variable(0.)
        episode_avg_mean_q   = tf.Variable(0.)
        episode_duration     = tf.Variable(0.)
        episode_avg_loss     = tf.Variable(0.)
        episode_timestep     = tf.Variable(0.)

        tf.summary.scalar(self.ENV_NAME + '/Total Reward/Episode', episode_total_reward)
        tf.summary.scalar(self.ENV_NAME + '/Average Max Q/Episode', episode_avg_max_q)
        tf.summary.scalar(self.ENV_NAME + '/Average Mean Q/Episode', episode_avg_mean_q)
        tf.summary.scalar(self.ENV_NAME + '/Duration/Episode', episode_duration)
        tf.summary.scalar(self.ENV_NAME + '/Average Loss/Episode', episode_avg_loss)
        tf.summary.scalar(self.ENV_NAME + '/Timestep/Episode', episode_timestep)

        summary_vars         = [episode_total_reward, episode_avg_max_q, episode_avg_mean_q, episode_duration, episode_avg_loss, episode_timestep]
        summary_placeholders = [tf.placeholder(tf.float32) for _ in range(len(summary_vars))]
        update_ops           = [summary_vars[i].assign(summary_placeholders[i]) for i in range(len(summary_vars))]
        summary_op           = tf.summary.merge_all()

        return summary_placeholders, update_ops, summary_op

    def load_network(self):
        checkpoint = tf.train.get_checkpoint_state(self.SAVE_NETWORK_PATH + self.ENV_NAME)
        if checkpoint and checkpoint.model_checkpoint_path:
            self.saver.restore(self.sess, checkpoint.model_checkpoint_path)
            print('Successfully loaded: ' + checkpoint.model_checkpoint_path)
        else:
            print('Training new network...')

    def run(self, env):
        scores = deque(maxlen=100)

        # Agent Observes the environment - no ops steps
        print "Warming Up..."
        current_state = env.reset()
        for i in range(self.INITIAL_REPLAY_SIZE):

            action = self.act(current_state)
            new_state, reward, done = env.step(action)
            self.observe(current_state, action, reward, done)

            current_state = new_state
            if done:
                current_state = env.reset()

        # Actual Training Begins
        print "Begin Training..."
        for i in range(self.MAX_EPISODES):
            t    = 0
            done = False
            while not done:
                action                  = self.act(current_state)
                new_state, reward, done = env.step(action)
                self.observe(current_state, action, reward, done)
                self.train()

                if not agent.QUIET:
                    print "Reward     :", reward
                    print "Action     :", action
                    print "Done       :", terminal

                current_state = new_state
                t += 1
            current_state = env.reset()

            scores.append(t)
            mean_score = np.mean(scores)

            if i % 100 == 0 and i != 0:
                print('[Episode {}] - Mean survival time over last 100 episodes was {} ticks.. Epsilon - {}'.format(str(i).zfill(6), mean_score, self.epsilon))


if __name__=="__main__":
    _name = "Simulator-v0"
    environment  = Environment(_name)
    input_shape  = environment.observation_shape()
    nb_actions   = environment.nb_actions()

    agent = Agent(_name, input_shape, nb_actions)
    agent.run(environment)
