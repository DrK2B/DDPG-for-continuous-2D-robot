import gymnasium as gym
import numpy as np
from Agent import ddpgAgent
from utils import plot_learning_curve


def DDPG():
    # Hyperparameters and settings
    EVALUATE = False

    EPISODES = 500
    TIME_STEPS = 500
    EXPLORATIONS = 50   # number of episodes with (random) exploration only
    LR_ACTOR = 0.001
    LR_CRITIC = 0.002
    DISCOUNT_FACTOR = 0.99
    MEM_SIZE = 1000000
    POLYAK = 0.005
    LAYER1_SIZE = 10
    LAYER2_SIZE = 10
    BATCH_SIZE = 64
    NOISE = 0.1    # std dev of zero-mean gaussian distributed noise
    ROLLING_WINDOW_SIZE_AVG_SCORE = 100  # size of the rolling window for averaging the episode scores
    FILENAME_FIG = 'MountainCarContinuous-v0_01.png'

    # Create environment and agent
    env = gym.make('MountainCarContinuous-v0', render_mode='human')
    agent = ddpgAgent(state_dim=env.observation_space.shape[0], action_dim=env.action_space.shape[0], env=env,
                      lr_actor=LR_ACTOR, lr_critic=LR_CRITIC, discount_factor=DISCOUNT_FACTOR, mem_size=MEM_SIZE,
                      polyak=POLYAK, layer1_size=LAYER1_SIZE, layer2_size=LAYER2_SIZE,
                      batch_size=BATCH_SIZE, noise=NOISE)

    best_score = env.reward_range[0]  # initialize with worst reward value
    score_history = []

    if EVALUATE:
        # model weights cannot be directly be load into an empty new model
        # hence, it is necessary to initialize the model weights by learning from randomly generated state transitions
        for n in range(agent.batch_size):
            state = env.reset()
            action = env.action_space.sample()
            new_state, reward, done, truncated, info = env.step(action)
            agent.remember(state, action, reward, new_state, done)

        agent.learn()
        agent.load_models()

    for episode in range(1, EPISODES+1):
        state = env.reset()[0]
        score = 0
        xp_boost = True if (episode <= EXPLORATIONS and not EVALUATE) else False

        for time in range(1, TIME_STEPS+1):
            env.render()

            action = agent.choose_action(state, EVALUATE, xp_boost)
            new_state, reward, done, truncated, info = env.step(action)
            agent.remember(state, action, reward, new_state, done)
            score += reward

            if not EVALUATE:
                agent.learn()

            state = new_state

            if done:
                break

        score_history.append(score)
        avg_score = np.mean(score_history[-ROLLING_WINDOW_SIZE_AVG_SCORE:])
        if avg_score > best_score:
            best_score = avg_score
            if not EVALUATE:
                agent.save_models()
                print("models' weights saved at episode: ", episode)

        print("Completed in {} steps.... episode: {}/{}, episode reward: {},"
              " average episode reward: {}".format(time, episode, EPISODES, score, avg_score))

    if not EVALUATE:
        episode_idx = [episode + 1 for episode in range(EPISODES)]
        plot_learning_curve(episode_idx, score_history, FILENAME_FIG)

    print('Successful')


if __name__ == '__main__':
    DDPG()
