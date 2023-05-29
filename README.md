# DDPG-for-continuous-2D-robot

This Python project is an application of the Deep Deterministic Policy Gradient (DDPG) to generate a Neural Network
controller for an agent in a given environment. It is only required to consider the _main.py file in order to run the
DDPG algorithm, but for more details on how the code works, it is advised to have a closer look at the other Python
files as well. 

BEFORE running the code:

To start the algorithm, it is necessary to set the hyperparameter values and configure the settings.

The settings are accessible in code lines 28 to 33 in _main.py. There, a specific environment can be chosen via the changing
the value of the variable ENV_NAME. Moreover, the rendering mode has to be set. Possible are the options 'human', which
results in a window displaying the agent in the specified environment, 'rgb_array' in order to obtain a frame 
representing RGB values of a pixel image, or None if no rendering at all is desired. Furthermore, it has to be specified
whether a training process should be started in order to generate a new Neural Network controller, or whether an already
existing Neural Network controller should be evaluated. This setting can be done by determining the boolean value of the
variable EVALUATE. If it equals False, then the algorithm will start a training process. The last setting variable 
ROLLING_WINDOW_SIZE_AVG_SCORE considers the size of the rolling window for averaging the episode rewards throughout the 
training process.

The hyperparameters are enlisted within a dictionary in the code lines 12 to 25: the number of episodes, the maximum 
number of time steps, the number of explorations which is the number of episodes with random agent behaviour in the 
beginning of the training process, the learning rates of the critic neural network and the actor neural network,
the discount factor, the memory size for the collected transition samples, the polyak averaging factor for the update of
the target neural networks, the number and size of the layers of the critic neural network and the actor neural network 
the batch size, the noise type (either Gaussian or Ornstein-Uhlenbeck) which is added onto the actions of the agent
during the training process for the sake of exploration of the environment, and the standard deviation of the noise on
the agent's actions.

DURING running the code:

Throughout the training process the neural networks of the actor, the critic, the target actor and the target critic are
saved each in a folder structure inside the folder tmp/models. They are updated as soon the running average score of the
current training episode surpasses the best running average score so far.

AFTER running the code:

After the training process finished, the learning curve of the training process is displayed and saved (with csv a file
containing the numerical values of the learning curve) in the folder tmp/figures.

In case of evaluation, only a plot showing the agent's trajectories for each episode is displayed and saved in the
folder tmp/trajectories.