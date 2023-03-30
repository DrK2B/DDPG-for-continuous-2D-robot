import numpy as np
import matplotlib.pyplot as plt
import os
import csv
from datetime import datetime


def plot_learningCurve(scores, rolling_window_size=100, filename=None, **hyperparameters):
    # score averaging
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i - rolling_window_size):(i + 1)])

    # plotting
    fig, ax = plt.subplots()
    ax.plot(range(1, len(running_avg) + 1), running_avg)
    ax.set_title('Running average of previous %d episode rewards' % rolling_window_size)
    plt.xlabel('episode')
    plt.ylabel('average episode reward')
    plt.grid()

    # text box with hyperparameter settings
    infos = '\n'.join((': '.join((param[0], str(param[1]))) for param in hyperparameters.items()))
    properties = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.65, 0.05, infos, transform=ax.transAxes, fontsize=8, verticalalignment='bottom', bbox=properties)

    # Set environment variable to avoid multiple loading of a shared library
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    if filename is not None:
        # save the plot of the learning curve
        filepath = os.path.join('tmp/figures', filename + '_plot.png')
        plt.savefig(filepath)
        print("Saved learning curve plot")

    plt.show()


def plot_agentTrajectory(time_steps, states, env, env_name):
    # ToDo: Implementation
    # first step: plot trajectory of one evaluation episode
    # second step: plot several episode trajectory in the same plot

    assert env_name in ('MountainCarContinuous-v0', 'gym_examples:2DRobot-v0'), \
        "plot_agentTrajectory: The specified environment does not exist."

    # plotting
    if env_name == 'MountainCarContinuous-v0':
        # only consider first state component at plotting
        states = [states[i][0] for i in range(len(states))]
        plt.plot(time_steps, states, color='blue', label='trajectory')

        # plot target position
        target_pos = env.goal_position
        plt.plot(time_steps, [target_pos for _ in range(len(time_steps))], color='green', label='target')

        # details
        plt.xlabel('time step')
        plt.ylabel("agent's position")
        plt.title("The agent's trajectory in %s" % env_name)
        plt.legend()

        plt.show()
    else:  # 2D robot environment
        # reshape states array [[s11, s12], [s21, s22], ...,[sn1, sn2]] to [[s11, s21, ..., sn1], [s12, s22, ..., sn2]]
        states = np.reshape(states, (states.shape[1], states.shape[0]))

        # plot start area
        start_width = env.start_area.high[0] - env.start_area.low[0]
        start_height = env.start_area.high[1] - env.start_area.low[1]
        start = plt.Rectangle((env.start_area.low[0], env.start_area.low[1]), start_width, start_height,
                              label='start area', linewidth=2, edgecolor='b', facecolor='none')
        plt.gca().add_patch(start)

        # plot target area
        target_width = env.target_area.high[0] - env.target_area.low[0]
        target_height = env.target_area.high[1] - env.target_area.low[1]
        target = plt.Rectangle((env.target_area.low[0], env.target_area.low[1]), target_width, target_height,
                               label='target area', linewidth=2, edgecolor='g', facecolor='none')
        plt.gca().add_patch(target)

        # plot s2 over s1
        plt.scatter(states[0], states[1], color='blue', label='trajectory')

        # details
        plt.xlabel("agent's x position")
        plt.ylabel("agent's y position")
        plt.title("The agent's trajectory in the environment %s" % env_name)
        plt.legend()

        plt.show()


def save_learningCurveData_to_csv(scores, filename):
    filepath = os.path.join('tmp/figures', filename + '_plot_data.csv')
    with open(filepath, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['episode', 'episode reward'])
        for i in range(1, len(scores) + 1):
            writer.writerow([i, scores[i - 1]])
    print("Saved learning curve data")


def create_unique_filename(filename):
    """
    appends a time stamp (format YYYYMMDD-hhmmss) to the file name
    :param filename: original file name
    :return: new file name with time stamp
    """
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    # example: get rid of 'gym_examples:' in 'gym_examples:2DRobot-v0'
    if ':' in filename:
        filename = filename.split(":")[-1]

    filename = '_'.join((filename, timestamp))
    return filename
