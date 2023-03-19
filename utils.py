import numpy as np
import matplotlib.pyplot as plt
import os
import csv


def plot_learning_curve(scores, filename, rolling_window_size=100, *hyperparameters):
    # ToDo: Add to title of a plot the applied values for hyperparameters
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i - rolling_window_size):(i + 1)])

    plt.plot(range(len(running_avg)), running_avg)
    plt.title('Running average of previous %d scores' % rolling_window_size)
    plt.xlabel('episode')
    plt.ylabel('average score')
    file_path = os.path.join('tmp/figures', filename+'_plot.png')
    # Set environment variable to avoid multiple loading of a shared library
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    plt.savefig(file_path)
    plt.show()
    print("saved learning curve plot")


def save_learningCurveData_to_csv(scores, filename):
    filepath = os.path.join('tmp/figures', filename + '_plot_data.csv')
    with open(filepath, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['x', 'y'])
        for i in range(1, len(scores)+1):
            writer.writerow([i, scores[i]])
    print("saved learning curve data")
