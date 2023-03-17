import gym
import numpy as np
from Agent import ddpgAgent
from utils import plot_learning_curve


def DDPG():
    env = gym.make('MountainCarContinuous-v0', render_mode='human')

    agent = ddpgAgent(env.observation_space.shape[0], env.action_space.shape[0], env=env)

    figure_filename = 'MountainCarContinuous-v0_01.png'

    best_score = env.reward_range[1]  # initialize with worst reward value
    score_history = []

    # Hyperparameters
    EPISODES = 500
    TIME_STEPS = 250
    EVALUATE = False
    EXPLORATIONS = 50   # number of episodes with (random) exploration only

    # the following hyperparameters are optional inputs to agent object
    # LR_ACTOR
    # LR_CRITIC
    # DISCOUNT_FACTOR
    # MEM_SIZE
    # POLYAK
    # BATCH_SIZE
    # NOISE aka std dev (zero-mean gaussian)
    # LAYER1_SIZE
    # LAYER2_SIZE

    if EVALUATE:
        # model weights cannot be directly be load into an empty new model
        # hence, it is necessary to initialize the model weights by learning from randomly generated state transitions
        for n in range(agent.batch_size):
            state = env.reset()
            action = env.action_space.sample()
            new_state, reward, done, truncated, info = env.step(action)
            agent.remember((state, action, reward, new_state, done))

        agent.learn()
        agent.load_models()

    for episode in range(EPISODES):
        state = env.reset()
        score = 0
        xp_boost = True \
            if (episode < EXPLORATIONS and not EVALUATE) else False

        for time in range(TIME_STEPS):
            env.render()

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
              " average episode reward".format(time, episode + 1, EPISODES, score, avg_score))

    if not EVALUATE:
        episode_idx = [episode + 1 for episode in range(EPISODES)]
        plot_learning_curve(episode_idx, score_history, figure_filename)

    print('Successful')


if __name__ == '__main__':
    DDPG()
