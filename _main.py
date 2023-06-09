import gymnasium as gym
import numpy as np
import tensorflow as tf
from Agent import ddpgAgent
from Noise import OUNoise, GaussianNoise
from utils import plot_learningCurve, save_learningCurveData_to_csv, create_unique_filename, plot_agentTrajectories

from utils import feedForward_Actor


def DDPG():
    # Hyperparameters
    HPARAMS = {
        "Episodes": 1000,
        "Time steps": 500,
        "Explorations": 0,  # number of episodes with (random) exploration only (and no exploitation)
        "Critic learning rate": 0.002,
        "Actor learning rate": 0.001,
        "Discount factor": 0.99,
        "Memory size": 100000,
        "Polyak averaging": 0.005,
        "Critic layer sizes": (8, 8),  # number of hidden layers is variable and corresponds to tuple length
        "Actor layer sizes": (8, 8),  # number of hidden layers is variable and corresponds to tuple length
        "Batch size": 64,
        "Noise type": "OU",
        "Noise std. dev.": 0.2  # std dev of zero-mean gaussian distributed noise
    }

    # settings
    ENV_NAME = 'MountainCarContinuous-v0'
    # ENV_NAME = 'gym_examples:2DRobot-v0'
    render_mode = None  # options: None, 'human', 'rgb_array'
    EVALUATE = False
    ROLLING_WINDOW_SIZE_AVG_SCORE = 100  # size of the rolling window for averaging the episode scores

    # Create environment, agent and noise process
    env = gym.make(ENV_NAME, render_mode=render_mode)
    agent = ddpgAgent(env=env, env_name=ENV_NAME, lr_actor=HPARAMS["Actor learning rate"],
                      lr_critic=HPARAMS["Critic learning rate"], discount_factor=HPARAMS["Discount factor"],
                      mem_size=HPARAMS["Memory size"], polyak=HPARAMS["Polyak averaging"],
                      critic_layer_sizes=HPARAMS["Critic layer sizes"], actor_layer_sizes=HPARAMS["Actor layer sizes"],
                      batch_size=HPARAMS["Batch size"])
    noise = OUNoise(action_space=env.action_space, max_sigma=HPARAMS["Noise std. dev."]) \
        if HPARAMS["Noise type"] == "OU" \
        else GaussianNoise(action_space=env.action_space, sigma=HPARAMS["Noise std. dev."])

    best_score = env.reward_range[0]  # initialize with worst reward value
    score_history = []
    trajectories = []

    # start training or evaluation
    if EVALUATE:
        # model weights cannot be directly load into an empty new model; hence, it is necessary to initialize the
        # model parameters by learning from randomly generated state transitions
        for _ in range(agent.batch_size):
            state = env.reset()[0]
            action = env.action_space.sample()
            new_state, reward, done, _, _ = env.step(action)
            agent.remember(state, action, reward, new_state, done)

        agent.learn()
        agent.load_models()

        # print(agent.actor(np.array([[5, 5]])))  # for Andrew

    for episode in range(1, HPARAMS["Episodes"] + 1):
        state = env.reset()[0]
        noise.reset()
        score = 0
        xp_boost = True if (episode <= HPARAMS["Explorations"] and not EVALUATE) else False

        # initializations for trajectory plotting
        states = np.zeros((env.observation_space.shape[0], HPARAMS["Time steps"]+1), dtype=np.float32)
        states[0][0] = state[0]
        states[1][0] = state[1]

        time_len = 0
        for time in range(1, HPARAMS["Time steps"] + 1):
            action = agent.choose_action(state, xp_boost)
            if not EVALUATE:
                noisy_action = noise.add_noise(action, t=time) if HPARAMS["Noise type"] == "OU" \
                    else noise.add_noise(action)
                new_state, reward, done, _, _ = env.step(noisy_action)
                agent.remember(state, tf.squeeze(noisy_action), reward, new_state, done)
                agent.learn()
                # print("time: %d | action: %f | reward: %f" % (time, noisy_action, reward))
            else:
                new_state, reward, done, _, _ = env.step(action)

                states[0][time] = np.copy(state[0])
                states[1][time] = np.copy(state[1])

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

        if EVALUATE:
            trajectories.append(states.copy())
            # plot agent trajectory
            # plot_agentTrajectory(states, env, ENV_NAME, save=False)

        # plot_learningCurve(score_history, ROLLING_WINDOW_SIZE_AVG_SCORE, **HPARAMS)
        print("Completed in {} steps.... episode: {}/{}, episode reward: {},"
              " average episode reward: {}".format(time_len, episode, HPARAMS["Episodes"], score, avg_score))

    # Close the environment
    env.close()

    if not EVALUATE:
        # save data and plot learning curve
        filename = create_unique_filename(ENV_NAME)
        save_learningCurveData_to_csv(score_history, filename)
        plot_learningCurve(score_history, ROLLING_WINDOW_SIZE_AVG_SCORE, filename=filename, **HPARAMS)
    else:
        # plot agent trajectories of all episodes
        plot_agentTrajectories(trajectories, env, ENV_NAME, save=False)

    print('--- Finished DDPG ---')


if __name__ == '__main__':
    DDPG()

    # envi = gym.make('MountainCarContinuous-v0', render_mode=None)
    # print(feedForward_Actor([-0.1, -0.05], "tmp/models/actor_MountainCarContinuous-v0_ddpg.h5", (8, 8), envi))
