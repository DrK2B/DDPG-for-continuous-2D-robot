import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.optimizers import Adam
from Memory import ReplayMemory
from ActorCritic import ActorNetwork, CriticNetwork


class DDPG_Agent:
    def __int__(self, state_dim, action_dim, lr_actor=0.001, lr_critic=0.002, env=None,
                discount_factor=0.99, mem_size=1e6, polyak=0.005,
                layer1_size=40, layer2_size=30, batch_size=64, noise=0.1):
        self.discount_factor = discount_factor
        self.polyak = polyak
        self.batch_size = batch_size
        self.action_dim = action_dim
        self.memory = ReplayMemory(mem_size, state_dim, action_dim)

        self.noise = noise      # standard deviation of zero-mean Gaussian noise

        self.env = env
        self.max_action = env.action_space.high[0]
        self.min_action = env.action_space.low[0]
        self.action_bound = np.max(np.abs([self.min_action, self.max_action]))

        self.actor = ActorNetwork(layer1_size=layer1_size, layer2_size=layer2_size, action_dim=action_dim,
                                  act_bound=self.action_bound, name='actor')
        self.target_actor = ActorNetwork(layer1_size=layer1_size, layer2_size=layer2_size, action_dim=action_dim,
                                         act_bound=self.action_bound, name='target_actor')
        self.critic = CriticNetwork(name='critic', layer1_size=layer1_size,
                                    layer2_size=layer2_size)
        self.target_critic = CriticNetwork(name='target_critic',
                                           layer1_size=layer1_size,
                                           layer2_size=layer2_size)

        self.actor.compile(optimizer=Adam(learning_rate=lr_actor))
        self.target_actor.compile(optimizer=Adam(learning_rate=lr_actor))
        self.critic.compile(optimizer=Adam(learning_rate=lr_critic))
        self.target_critic.compile(optimizer=Adam(learning_rate=lr_critic))

        self.update_network_parameters(polyak=1)

    # Performs soft update on the target networks with update rate polyak
    def update_network_parameters(self, polyak=None):
        if polyak is None:
            polyak = self.polyak

        weights = []
        targets = self.target_actor.weights
        for i, weight in enumerate(self.actor.weights):
            weights.append(weight * polyak + targets[i] * (1 - polyak))
        self.target_actor.set_weights(weights)

        weights = []
        targets = self.target_critic.weights
        for i, weight in enumerate(self.critic.weights):
            weights.append(weight * polyak + targets[i] * (1 - polyak))
        self.target_critic.set_weights(weights)

    # Stores transition in replay memory
    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def save_models(self):
        print('... saving model weights ...')
        self.actor.save_weights(self.actor.checkpoint_file)
        self.target_actor.save_weights(self.target_actor.checkpoint_file)
        self.critic.save_weights(self.critic.checkpoint_file)
        self.target_critic.save_weights(self.target_critic.checkpoint_file)

    def load_models(self):
        print('... loading model weights ...')
        self.actor.load_weights(self.actor.checkpoint_file)
        self.target_actor.load_weights(self.target_actor.checkpoint_file)
        self.critic.load_weights(self.critic.checkpoint_file)
        self.target_critic.load_weights(self.target_critic.checkpoint_file)

    def choose_action(self, state, evaluate=False, exploration_boost=False):
        # at the start of the training, actions are sampled from a uniform random
        # distribution over valid actions for a fixed number of steps (batch size?)
        if exploration_boost:
            action = tf.convert_to_tensor(self.env.action_space.sample())
        else:
            state = tf.convert_to_tensor([state], dtype=tf.float32)
            action = self.actor(state)
            if not evaluate:
                action += tf.random.normal(shape=[self.action_dim], mean=0.0,
                                           stddev=self.noise)

        action = tf.clip_by_value(action, self.min_action, self.max_action)
        return action[0]  # return 0th element of tensor, which is a np array

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return

        states, actions, rewards, new_states, dones = \
            self.memory.sample(self.batch_size)

        states = tf.convert_to_tensor(states, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.float32)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        new_states = tf.convert_to_tensor(new_states, dtype=tf.float32)

        with tf.GradientTape() as tape:  # ToDo: look up what this line does exactly
            # computing the targets (y) of mean-squared Bellman error (MSBE) function
            new_target_actions = self.target_actor(new_states)
            new_target_q_values = tf.squeeze(self.target_critic(
                new_states, new_target_actions), -1)  # might need to change axis back to value 1
            q_values = tf.squeeze(self.critic(states, actions), -1)  # might need to change axis back to value 1
            targets = rewards + self.discount_factor * (1 - dones) * new_target_q_values

            critic_loss = keras.losses.MSE(targets, q_values)

        critic_network_grad = tape.gradient(critic_loss,
                                            self.critic.trainable_variables)
        self.critic.optimizer.apply_gradients(
            zip(critic_network_grad, self.critic.trainable_variables))

        with tf.GradientTape() as tape:
            policy_actions = self.actor(states)
            # q value function has to be maximized, hence its negative has to minimized
            # loss function of actor corresponds to negative of q value function
            actor_loss = -self.critic(states, policy_actions)
            actor_loss = tf.math.reduce_mean(actor_loss)

        actor_network_grad = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor.optimizer.apply_gradients(
            zip(actor_network_grad, self.actor.trainable_variables))

        # soft update of the target networks
        self.update_network_parameters()
