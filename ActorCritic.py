import os
import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense


class CriticNetwork(keras.Model):
    def __init__(self, layer1_size, layer2_size, layer3_size,
                 name='critic', chkpt_dir='tmp/model_weights'):
        super().__init__()
        self.model_name = name  # in order to distinguish between target and main networks
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir,
                                            self.model_name + '_ddpg.h5')

        self.layer1 = Dense(layer1_size, activation='relu')
        self.layer2 = Dense(layer2_size, activation='relu')
        self.layer3 = Dense(layer3_size, activation='relu')
        self.q = Dense(1)

    def call(self, state, action):
        q_val1 = self.layer1(tf.concat([state, action], axis=1))  # ToDo: check how input looks like after concat
        q_val2 = self.layer2(q_val1)
        q_val3 = self.layer3(q_val2)
        q_val = self.q(q_val3)
        return q_val


class ActorNetwork(keras.Model):
    def __init__(self, layer1_size, layer2_size, action_dim,
                 act_bound=1, name='actor', chkpt_dir='tmp/model_weights'):
        super().__init__()
        self.model_name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir,
                                            self.model_name + '_ddpg.h5')

        self.action_dim = action_dim
        self.action_bound = act_bound

        self.layer1 = Dense(layer1_size, activation='relu')
        self.layer2 = Dense(layer2_size, activation='relu')
        self.mu = Dense(self.action_dim, activation='tanh')

    def call(self, state):
        mu_val1 = self.layer1(state)
        mu_val2 = self.layer2(mu_val1)
        mu_val = self.action_bound * self.mu(mu_val2)
        return mu_val
