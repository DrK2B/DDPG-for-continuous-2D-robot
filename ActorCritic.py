import os
import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense


class CriticNetwork(keras.Model):
    def __init__(self, layer_sizes, name='critic', chkpt_dir='tmp/models'):
        super().__init__()
        self.model_name = name  # in order to distinguish between target and main networks
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, self.model_name)

        self.hidden_layers = []
        for i in range(len(layer_sizes)):
            self.hidden_layers.append(Dense(layer_sizes[i], activation='relu'))
        self.q = Dense(1)

    def call(self, state, action):
        x = tf.concat([state, action], axis=-1)
        for layer in self.hidden_layers:
            x = layer(x)
        q_val = self.q(x)
        return q_val


class ActorNetwork(keras.Model):
    def __init__(self, layer_sizes, action_dim, act_bound=1, name='actor',
                 chkpt_dir='tmp/models'):
        super().__init__()
        self.model_name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, self.model_name)

        self.action_dim = action_dim
        self.action_bound = act_bound

        self.hidden_layers = []
        for i in range(len(layer_sizes)):
            self.hidden_layers.append(Dense(layer_sizes[i], activation='relu'))
        self.mu = Dense(self.action_dim, activation='tanh')

    def call(self, state):
        x = state
        for layer in self.hidden_layers:
            x = layer(x)
        mu_val = self.action_bound * self.mu(x)
        return mu_val
