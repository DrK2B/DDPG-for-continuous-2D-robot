import numpy as np
import matplotlib.pyplot as plt
import os


def plot_learning_curve(x, scores, figure_file):
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i-100):(i+1)])

    plt.plot(x, running_avg)
    plt.title('Running average of previous 100 scores')
    plt.xlabel('episode')
    plt.ylabel('score')
    file_path = os.path.join('tmp/figures', figure_file)
    # Set environment variable to avoid multiple loading of a shared library
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    plt.savefig(file_path)
    plt.show()
    print("saved learning curve")
