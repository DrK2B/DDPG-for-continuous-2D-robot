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


def plot_agentTrajectory(states, env, env_name, save=False):
    assert env_name in ('MountainCarContinuous-v0', 'gym_examples:2DRobot-v0'), \
        "plot_agentTrajectory: The specified environment does not exist."

    # plotting
    if env_name == 'MountainCarContinuous-v0':
        time_steps = [t for t in range(501)]

        # plot environment boundaries
        lb = [env.min_position for _ in range(len(time_steps))]
        plt.plot(time_steps, lb, color='black', label='lower boundary')
        ub = [env.max_position for _ in range(len(time_steps))]
        plt.plot(time_steps, ub, color='black', label='upper boundary')

        # only consider first state component at plotting
        states = [states[0][i] for i in range(len(states[0]))]
        while states and states[-1] == 0:
            states.pop()
        plt.plot([t for t in range(len(states))], states, color='orange', label='trajectory')

        # plot target position
        target_pos = env.goal_position
        plt.plot(time_steps, [target_pos for _ in range(len(time_steps))], color='green', label='target')

        # details
        t_min, t_max = 0, 500   # number of time steps
        y_min, y_max = -1.5, 1.5
        plt.xlim([t_min - 1, t_max + 1])
        plt.ylim([y_min, y_max])

        plt.xlabel('time step')
        plt.ylabel("agent's position")
        plt.title("The agent's trajectory in the environment %s" % env_name)
        plt.legend(loc='upper left')

    else:  # 2D robot environment
        # plot environment boundaries
        bound_width = env.max_position - env.min_position
        bound_height = bound_width
        bound = plt.Rectangle((env.min_position, env.min_position), bound_width, bound_height,
                              label='boundaries', linewidth=2, edgecolor='black', facecolor='none')
        plt.gca().add_patch(bound)

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
        x_states = states[0]
        y_states = states[1]
        x_states_trimmed = np.trim_zeros(x_states)  # remove trailing zeros
        y_states_trimmed = np.trim_zeros(y_states)  # remove trailing zeros
        plt.scatter(x_states_trimmed, y_states_trimmed, color='orange', label='trajectory', marker='.')

        # plot details
        x_min, x_max = env.min_position, env.max_position
        y_min, y_max = env.min_position, env.max_position
        plt.xlim([x_min - 1, x_max + 1])
        plt.ylim([y_min - 1, y_max + 1])

        plt.xlabel("agent's x position")
        plt.ylabel("agent's y position")
        plt.title("The agent's trajectory in the environment %s" % env_name.split(':')[-1])
        plt.legend()

    if save:
        # Set environment variable to avoid multiple loading of a shared library
        os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
        # save the plot of the learning curve
        filepath = os.path.join('tmp/trajectories', create_unique_filename(env_name) + '_traj.png')
        plt.savefig(filepath)
        print("Saved trajectory")

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
