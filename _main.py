import gym
import numpy as np
from Agent import DDPG_Agent
from utils import plot_learning_curve


def DDPG():
    env = gym.make('MountainCarContinuous-v0', render_mode='human')
    agent = DDPG_Agent(state_dim=env.observation_space.shape, env=env,
                       action_dim=env.action_space.shape[0])

    figure_filename = 'MountainCarContinuous-v0_01.png'

    best_score = env.reward_range[1]    # initialize with worst reward value
    score_history = []

    EPISODES = 500
    TIME_STEPS = 250
    EVALUATE = False
    EXPLORATION_TIME = 50

    if EVALUATE:
        for n in range(agent.batch_size):
            state = env.reset()
            action = env.action_space.sample()
            new_state, reward, done, truncated, info = env.step(action)
            agent.remember((state, action, reward, new_state, done))

        agent.learn()
        agent.load_models()

    for episode in range(EPISODES):
        state = env.reset()
        done = False
        score = 0
        xp_boost = True \
            if (episode < EXPLORATION_TIME and not EVALUATE) else False

        for time in range(TIME_STEPS):
            action = agent.choose_action(state, EVALUATE, xp_boost)
            new_state, reward, done, truncated, info = env.step(action)
            agent.remember(state, action, reward, new_state, done)
            score += reward

            if not EVALUATE:
                agent.learn()

            state = new_state

        score_history.append(score)
        avg_score = np.mean(score_history[-100:])
        if avg_score > best_score:
            best_score = avg_score
            if not EVALUATE:
                agent.save_models()

        print('episode')
        print("Completed in {} steps.... episode: {}/{}, episode reward: {},"
              " average episode reward"
              .format(time, episode+1, EPISODES, score, avg_score))

    if not EVALUATE:
        episode_idx = [episode+1 for episode in range(EPISODES)]
        plot_learning_curve(episode_idx, score_history, figure_filename)


if __name__ == '__main__':
    DDPG()

