import gymnasium as gym
import numpy as np
import tensorflow as tf
from Agent import ddpgAgent
from Noise import OUNoise
from utils import plot_learning_curve, save_learningCurveData_to_csv


def DDPG():
    # Hyperparameters and settings
    HPARAMS = {
        "Episodes": 2,
        "Time steps": 500,
        "Explorations": 0,                  # number of episodes with (random) exploration only
        "Critic learning rate": 0.001,
        "Actor learning rate": 0.002,
        "Discount factor": 0.99,
        "Memory size": 100000,
        "Polyak averaging": 0.005,
        "Critic layer sizes": (64, 64),     # critic networks are designed to have 2 hidden layers
        "Actor layer sizes": (64, 64),      # actor networks are designed to have 2 hidden layers
        "Batch size": 64,
        "Noise std dev.": 0.25              # std dev of zero-mean gaussian distributed noise
    }

    EPISODES = 1000
    TIME_STEPS = 500
    EXPLORATIONS = 0  # number of episodes with (random) exploration only
    LR_ACTOR = 0.001
    LR_CRITIC = 0.002
    DISCOUNT_FACTOR = 0.99
    MEM_SIZE = 100000
    POLYAK = 0.005
    CRITIC_LAYER_SIZES = (64, 64)   # critic networks are designed to have 2 hidden layers
    ACTOR_LAYER_SIZES = (64, 64)    # actor networks are designed to have 2 hidden layers
    BATCH_SIZE = 64
    NOISE = 0.25  # std dev of zero-mean gaussian distributed noise

    EVALUATE = False
    ROLLING_WINDOW_SIZE_AVG_SCORE = 100  # size of the rolling window for averaging the episode scores
    FILENAME_FIG = 'MountainCarContinuous-v0_01'

    # Create environment, agent and noise process
    env = gym.make('MountainCarContinuous-v0', render_mode='human')
    agent = ddpgAgent(env=env, lr_actor=HPARAMS["Actor learning rate"], lr_critic=HPARAMS["Critic learning rate"],
                      discount_factor=HPARAMS["Discount factor"], mem_size=HPARAMS["Memory size"],
                      polyak=HPARAMS["Polyak averaging"], critic_layer_sizes=HPARAMS["Critic layer sizes"],
                      actor_layer_sizes=HPARAMS["Actor layer sizes"], batch_size=HPARAMS["Batch size"])
    noise = OUNoise(action_space=env.action_space, max_sigma=HPARAMS["Noise std dev."])

    best_score = env.reward_range[0]  # initialize with worst reward value
    score_history = []

    # start training or evaluation
    if EVALUATE:
        # model weights cannot be directly be load into an empty new model; hence, it is necessary to initialize the
        # model parameters by learning from randomly generated state transitions
        for n in range(agent.batch_size):
            state = env.reset()[0]
            action = env.action_space.sample()
            new_state, reward, done, truncated, _ = env.step(action)
            agent.remember(state, action, reward, new_state, done)

        agent.learn()
        agent.load_models()

    # ToDo: add termination criterion: stop if there's no improvement (opt.)
    for episode in range(1, HPARAMS["Episodes"] + 1):
        state = env.reset()[0]
        noise.reset()
        score = 0
        xp_boost = True if (episode <= HPARAMS["Explorations"] and not EVALUATE) else False

        for time in range(1, HPARAMS["Time steps"] + 1):
            action = agent.choose_action(state, xp_boost)
            if not EVALUATE:
                noisy_action = noise.add_noise(action, t=time)
                new_state, reward, done, _, _ = env.step(noisy_action)
                agent.remember(state, tf.squeeze(noisy_action), reward, new_state, done)
                agent.learn()
                # print("time: %d | action: %f | reward: %f" % (time, noisy_action, reward))
            else:
                new_state, reward, done, _, _ = env.step(action)

            score += reward

            state = new_state

            if done:
                break

        score_history.append(score)
        avg_score = np.mean(score_history[-ROLLING_WINDOW_SIZE_AVG_SCORE:])
        if avg_score > best_score:
            best_score = avg_score
            if not EVALUATE:
                agent.save_models()
                print("models' parameters saved at episode: ", episode)

        print("Completed in {} steps.... episode: {}/{}, episode reward: {},"
              " average episode reward: {}".format(time, episode, HPARAMS["Episodes"], score, avg_score))

    # Close the environment
    env.close()

    if not EVALUATE:
        save_learningCurveData_to_csv(score_history, FILENAME_FIG)
        plot_learning_curve(score_history, FILENAME_FIG, ROLLING_WINDOW_SIZE_AVG_SCORE, **HPARAMS)

    print('--- Finished DDPG ---')


if __name__ == '__main__':
    DDPG()
