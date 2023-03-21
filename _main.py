import gymnasium as gym
import numpy as np
import tensorflow as tf
from Agent import ddpgAgent
from Noise import OUNoise
from utils import plot_learningCurve, save_learningCurveData_to_csv, create_unique_filename


def DDPG():
    # Hyperparameters
    HPARAMS = {
        "Episodes": 1000,
        "Time steps": 500,
        "Explorations": 0,  # number of episodes with (random) exploration only
        "Critic learning rate": 0.002,
        "Actor learning rate": 0.001,
        "Discount factor": 0.99,
        "Memory size": 100000,
        "Polyak averaging": 0.005,
        "Critic layer sizes": (8, 8),  # critic networks are designed to have 2 hidden layers
        "Actor layer sizes": (8, 8),  # actor networks are designed to have 2 hidden layers
        "Batch size": 64,
        "Noise std. dev.": 0.25  # std dev of zero-mean gaussian distributed noise
    }

    # settings
    ENV_NAME = 'MountainCarContinuous-v0'
    EVALUATE = False
    ROLLING_WINDOW_SIZE_AVG_SCORE = 100  # size of the rolling window for averaging the episode scores

    # Create environment, agent and noise process
    env = gym.make(ENV_NAME, render_mode='human')
    agent = ddpgAgent(env=env, env_name=ENV_NAME, lr_actor=HPARAMS["Actor learning rate"],
                      lr_critic=HPARAMS["Critic learning rate"], discount_factor=HPARAMS["Discount factor"],
                      mem_size=HPARAMS["Memory size"], polyak=HPARAMS["Polyak averaging"],
                      critic_layer_sizes=HPARAMS["Critic layer sizes"], actor_layer_sizes=HPARAMS["Actor layer sizes"],
                      batch_size=HPARAMS["Batch size"])
    noise = OUNoise(action_space=env.action_space, max_sigma=HPARAMS["Noise std. dev."])

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

    for episode in range(1, HPARAMS["Episodes"] + 1):
        state = env.reset()[0]
        noise.reset()
        score = 0
        xp_boost = True if (episode <= HPARAMS["Explorations"] and not EVALUATE) else False

        time_len = 0
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
            time_len = time

            if done:
                break

        score_history.append(score)
        avg_score = np.mean(score_history[-ROLLING_WINDOW_SIZE_AVG_SCORE:])
        if avg_score > best_score:
            best_score = avg_score
            if not EVALUATE:
                agent.save_models()
                print("models' parameters saved at episode: ", episode)

        plot_learningCurve(score_history, ROLLING_WINDOW_SIZE_AVG_SCORE, **HPARAMS)
        print("Completed in {} steps.... episode: {}/{}, episode reward: {},"
              " average episode reward: {}".format(time_len, episode, HPARAMS["Episodes"], score, avg_score))

    # Close the environment
    env.close()

    if not EVALUATE:
        # save data and plot learning curve
        filename = create_unique_filename(ENV_NAME)
        save_learningCurveData_to_csv(score_history, filename)
        plot_learningCurve(score_history, ROLLING_WINDOW_SIZE_AVG_SCORE, filename=filename, **HPARAMS)

    print('--- Finished DDPG ---')


if __name__ == '__main__':
    DDPG()
