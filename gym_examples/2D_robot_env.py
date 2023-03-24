import numpy as np
import pygame
from pygame import gfxdraw
import math
import gymnasium as gym
from gymnasium import spaces


class Continuous_2D_RobotEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(self, render_mode=None, size=10):
        self.agent_mass = 1  # mass of the agent
        self.agent_radius = 0.25  # radius of the robot (circle)

        self.time_step = 0.1

        # states are represented by the (2D) position of the agent
        self.state = None

        self.max_action = 1.0
        self.min_action = -self.max_action
        self.max_position = 5.0
        self.min_position = -self.max_position

        # target area = [3.0, 4.0]x[2.5, 3.5]
        self.target_area = spaces.Box(low=np.array([3.0, 2.5]),
                                      high=np.array([4.0, 3.5]), dtype=np.float32)

        self.size = size  # The size of the square environment
        self.window_size = 600  # The size of the PyGame window
        self.border_thickness = 50
        self.canvas = None

        # Observations are the agent's position in the square 2D environment.
        self.observation_space = spaces.Box(low=self.min_position,
                                            high=self.max_position,
                                            shape=(2,), dtype=np.float32)

        # The agent can perform actions with continuous value
        self.action_space = spaces.Box(low=self.min_action, high=self.max_action,
                                       shape=(2,), dtype=np.float32)

        assert render_mode is None or render_mode in self.metadata["render_modes"], \
            "Passed render mode is not valid"
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None
        self.isOpen = True

    # %%
    # Reset
    # ~~~~~
    #
    # The ``reset`` method will be called to initiate a new episode. You may
    # assume that the ``step`` method will not be called before ``reset`` has
    # been called. Moreover, ``reset`` should be called whenever a done signal
    # has been issued. Users may pass the ``seed`` keyword to ``reset`` to
    # initialize any random number generator that is used by the environment
    # to a deterministic state. It is recommended to use the random number
    # generator ``self.np_random`` that is provided by the environment’s base
    # class, ``gymnasium.Env``. If you only use this RNG, you do not need to
    # worry much about seeding, *but you need to remember to call
    # ``super().reset(seed=seed)``* to make sure that ``gymnasium.Env``
    # correctly seeds the RNG. Once this is done, we can randomly set the
    # state of our environment. In our case, we randomly choose the agent’s
    # location until it does not lie within the target area.
    #
    # The ``reset`` method should return a tuple of the initial observation
    # and some auxiliary information.

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # initialization of the agent's state in the middle of the target area
        target_low = self.target_area.low
        target_high = self.target_area.high
        self.state = np.array([np.mean([target_low[0], target_high[0]]),
                               np.mean([target_low[1], target_high[1]])])

        # Choose the agent's initial position uniformly at random and make sure it is not in the target area
        while target_low[0] <= self.state[0] <= target_high[0] \
                and target_low[1] <= self.state[1] <= target_high[1]:
            self.state = np.array([self.np_random.uniform(low=self.min_position, high=self.max_position)
                                   for _ in range(self.observation_space.shape[0])], dtype=np.float32)

        info = {}

        if self.render_mode == "human":
            self.render()

        return self.state, info

    # %%
    # Step
    # ~~~~
    #
    # The ``step`` method usually contains most of the logic of your
    # environment. It accepts an ``action``, computes the state of the
    # environment after applying that action and returns the 5-tuple
    # ``(observation, reward, terminated, truncated, info)``. Once the new state of the
    # environment has been computed, we can check whether it is a terminal
    # state, and we set ``terminated`` accordingly.

    def step(self, action):
        # time step
        dt = self.time_step

        # update state
        self.state[0] += dt * action[0]
        self.state[1] += dt * action[1]

        # make sure the agent does not leave its environment
        if self.state[0] < self.min_position:
            self.state[0] = self.min_position
        if self.state[0] > self.max_position:
            self.state[0] = self.max_position
        if self.state[1] < self.min_position:
            self.state[1] = self.min_position
        if self.state[1] > self.max_position:
            self.state[1] = self.max_position

        self.state.astype(np.float32)

        # check whether terminated
        target_low = self.target_area.low
        target_high = self.target_area.high
        terminated = bool(target_low[0] <= self.state[0] <= target_high[0]
                          and target_low[1] <= self.state[1] <= target_high[1])

        # penalise the increase in (kinetic) energy consumption
        reward = -0.5 * self.agent_mass * (math.pow(action[0], 2) + math.pow(action[1], 2))
        # grant reward if target has been reached
        if terminated:
            reward += 1000

        info = {}
        truncated = False

        if self.render_mode == "human":
            self.render()

        return self.state, reward, terminated, truncated, info

    def env2canvas(self, pos):
        """
        transforms environment (1D) coordinate into pixel coordinate on canvas
        :param pos: 1D coordinate in environment
        :return: pixel coordinates on canvas
        """
        offset = self.window_size / 2
        scale = (offset - self.border_thickness - self.agent_radius) / self.max_position

        return scale * pos + offset

    # %%
    # Rendering
    # ~~~~~~~~~
    #
    # Here, we are using PyGame for rendering. A similar approach to rendering
    # is used in many environments that are included with Gymnasium and you
    # can use it as a skeleton for your own environments:

    def render(self):
        if self.render_mode is None:
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym("{self.spec.id}", render_mode="rgb_array")'
            )
            return

        if self.window is None:  # and self.render_mode == "human":
            pygame.init()
            if self.render_mode == "human":
                pygame.display.init()
                self.window = pygame.display.set_mode(
                    (self.window_size, self.window_size)
                )
            else:  # mode == "rgb_array"
                self.window = pygame.Surface((self.window_size, self.window_size))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        # background
        self.canvas = pygame.Surface((self.window_size, self.window_size))
        self.canvas.fill((0, 0, 0))  # set background colour

        # draw the borders
        gfxdraw.vline(self.canvas, self.border_thickness, self.border_thickness,
                      self.window_size - self.border_thickness, (255, 255, 255))
        gfxdraw.vline(self.canvas, self.window_size - self.border_thickness,
                      self.border_thickness, self.window_size - self.border_thickness,
                      (255, 255, 255))
        gfxdraw.hline(self.canvas, self.border_thickness, self.window_size - self.border_thickness,
                      self.border_thickness, (255, 255, 255))
        gfxdraw.hline(self.canvas, self.border_thickness, self.window_size - self.border_thickness,
                      self.window_size - self.border_thickness, (255, 255, 255))

        # draw the (green, rectangular) target
        target_left = self.env2canvas(self.target_area.low[0])
        target_top = self.env2canvas(self.target_area.high[1])
        target_width = self.env2canvas(self.target_area.high[0]) - self.env2canvas(self.target_area.low[0])
        target_height = self.env2canvas(self.target_area.high[1]) - self.env2canvas(self.target_area.low[1])
        pygame.draw.rect(
            self.canvas,
            (0, 255, 0),
            pygame.Rect((target_left, target_top), (target_width, target_height))
        )

        # draw the (blue, circular) agent
        scale = (self.window_size/2 - self.border_thickness - self.agent_radius)/self.max_position
        pygame.draw.circle(
            self.canvas,
            (255, 0, 255),
            (self.env2canvas(self.state[0]), self.env2canvas(self.state[1])),
            scale * self.agent_radius
        )

        # change y direction of canvas from downward to upward
        self.canvas = pygame.transform.flip(self.canvas, False, True)
        # put canvas onto window
        self.window.blit(self.canvas, (0, 0))
        if self.render_mode == "human":
            # update window
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.canvas)), axes=(1, 0, 2)
            )

    # %%
    # Close
    # ~~~~~
    #
    # The ``close`` method should close any open resources that were used by
    # the environment. In many cases, you don’t actually have to bother to
    # implement this method. However, in our example ``render_mode`` may be
    # ``"human"`` and we might need to close the window that has been opened:

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
            self.isOpen = False
