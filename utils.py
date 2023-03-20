import numpy as np
import matplotlib.pyplot as plt
import os
import csv


def plot_learning_curve(scores, filename, rolling_window_size=100, **hyperparameters):
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i - rolling_window_size):(i + 1)])

    fig, ax = plt.subplots()
    ax.plot(range(1, len(running_avg)+1), running_avg)
    ax.set_title('Running average of previous %d episode rewards' % rolling_window_size)
    plt.xlabel('episode')
    plt.ylabel('average episode reward')

    # text box with hyperparameter settings
    infos = '\n'.join((': '.join((param[0], str(param[1]))) for param in hyperparameters.items()))
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.75, 0.05, infos, transform=ax.transAxes, fontsize=8, verticalalignment='bottom', bbox=props)

    filepath = os.path.join('tmp/figures', filename + '_plot.png')
    # Set environment variable to avoid multiple loading of a shared library
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    plt.savefig(filepath)
    plt.show()
    print("Saved learning curve plot")


def save_learningCurveData_to_csv(scores, filename):
    filepath = os.path.join('tmp/figures', filename + '_plot_data.csv')
    with open(filepath, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['x', 'y'])
        for i in range(1, len(scores)+1):
            writer.writerow([i, scores[i-1]])
    print("Saved learning curve data")
