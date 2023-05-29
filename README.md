# DDPG-for-continuous-2D-robot

This Python project is an application of the Deep Deterministic Policy Gradient (DDPG) to generate a Neural Network
controller for an agent in a given environment. It is only required to consider the _main.py file in order to run the
DDPG algorithm, but for more details on how the code works, it is advised to have a closer look at the other Python
files as well.

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

