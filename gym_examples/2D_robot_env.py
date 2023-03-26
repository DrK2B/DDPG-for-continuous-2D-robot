import numpy as np
import pygame
from pygame import gfxdraw
import math
import gymnasium as gym
from gymnasium import spaces


class Continuous_2D_RobotEnv_v0(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 100}

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
        self.window_size = 500  # The size of the PyGame window
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

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # initialization of the agent's state in the middle of the target area
        target_low = self.target_area.low
        target_high = self.target_area.high
        self.state = np.array([np.mean([target_low[0], target_high[0]]),
                               np.mean([target_low[1], target_high[1]])], dtype=np.float32)

        # Choose the agent's initial position uniformly at random and make sure it is not in the target area
        while target_low[0] <= self.state[0] <= target_high[0] \
                and target_low[1] <= self.state[1] <= target_high[1]:
            self.state = np.array([self.np_random.uniform(low=self.min_position, high=self.max_position)
                                   for _ in range(self.observation_space.shape[0])], dtype=np.float32)

        '''
        # Choose the agent's initial position uniformly at random in the area [-5,-2.5]²
        self.state = np.array([self.np_random.uniform(low=self.min_position, high=-2.5)
                               for _ in range(self.observation_space.shape[0])], dtype=np.float32)
        '''

        info = {}

        if self.render_mode == "human":
            self.render()

        return self.state, info

    def _calcDist2Target(self):
        target_low = self.target_area.low
        target_high = self.target_area.high

        # the center of the target area
        centerTarget = np.array([np.mean([target_low[0], target_high[0]]),
                                 np.mean([target_low[1], target_high[1]])], dtype=np.float32)

        dist = np.linalg.norm(centerTarget - self.state)
        return dist

    def step(self, action):
        # distance to target
        dist = self._calcDist2Target()

        # update state
        dt = self.time_step
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

        # new distance to target
        new_dist = self._calcDist2Target()

        # check whether terminated
        target_low = self.target_area.low
        target_high = self.target_area.high
        terminated = bool(target_low[0] <= self.state[0] <= target_high[0]
                          and target_low[1] <= self.state[1] <= target_high[1])

        # penalise "(kinetic) energy consumption"
        reward = -0.5 * self.agent_mass * (math.pow(action[0], 2) + math.pow(action[1], 2))
        # grant reward if closer to target, otherwise penalise
        reward += 1 if new_dist < dist else -1
        # grant reward if target has been reached
        if terminated:
            reward += 1000

        info = {}
        truncated = False

        if self.render_mode == "human":
            self.render()

        return self.state, reward, terminated, truncated, info

    def _env2canvas(self, pos):
        """
        transforms environment x or y coordinate into pixel coordinate on canvas
        :param pos: x or y coordinate in environment
        :return: pixel coordinate on canvas
        """
        offset = self.window_size / 2
        scale = (offset - self.border_thickness) / self.max_position
        # taking agent's radius into account such that agent won't cross the environment's borders
        scale = scale / (1 + self.agent_radius / self.max_position)

        return scale * pos + offset

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
        target_left = self._env2canvas(self.target_area.low[0])
        target_top = self._env2canvas(self.target_area.high[1])
        target_width = self._env2canvas(self.target_area.high[0]) - self._env2canvas(self.target_area.low[0])
        target_height = self._env2canvas(self.target_area.high[1]) - self._env2canvas(self.target_area.low[1])
        pygame.draw.rect(
            self.canvas,
            (0, 255, 0),
            pygame.Rect((target_left, target_top), (target_width, target_height))
        )

        # draw the (purple, circular) agent
        scale = (self.window_size / 2 - self.border_thickness - self.agent_radius) / self.max_position
        # taking agent's radius into account such that agent won't cross the environment's borders
        scale = scale / (1 + self.agent_radius / self.max_position)
        pygame.draw.circle(
            self.canvas,
            (255, 120, 0),
            (self._env2canvas(self.state[0]), self._env2canvas(self.state[1])),
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

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
            self.isOpen = False
