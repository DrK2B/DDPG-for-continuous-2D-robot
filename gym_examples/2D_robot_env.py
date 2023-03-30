import gymnasium as gym
import numpy as np
import pygame
from gymnasium import spaces
from pygame import gfxdraw


class Continuous_2D_RobotEnv_v0(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(self, render_mode=None, size=10):
        self.agent_mass = 1  # mass of the agent
        self.agent_radius = 0.25  # radius of the robot (circle)

        self.time_step = 0.1

        # states are represented by the (2D) position of the agent
        self.state = None
        self.max_position = 5.0
        self.min_position = -self.max_position
        # Observations are the agent's position in the square 2D environment
        self.observation_space = spaces.Box(low=self.min_position,
                                            high=self.max_position,
                                            shape=(2,), dtype=np.float32)

        # start area = [-5,-2.5]²
        self.start_area = spaces.Box(low=np.array([self.min_position, self.min_position]),
                                     high=np.array([-2.5, -2.5]), dtype=np.float32)

        # target area = [3.0, 4.0]x[2.5, 3.5]
        self.target_area = spaces.Box(low=np.array([3.0, 2.5]),
                                      high=np.array([4.0, 3.5]), dtype=np.float32)

        self.max_action = 1.0
        self.min_action = -self.max_action
        # The agent can perform actions with continuous value
        self.action_space = spaces.Box(low=self.min_action, high=self.max_action,
                                       shape=(2,), dtype=np.float32)

        self.size = size  # The size of the square environment
        self.window_size = 500  # The size of the PyGame window
        self.border_thickness = 50
        self.canvas = None

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

        # Choose the agent's initial position uniformly at random in the start area
        self.state = np.array([self.np_random.uniform(low=self.start_area.low[0], high=self.start_area.high[0]),
                               self.np_random.uniform(low=self.start_area.low[1], high=self.start_area.high[1])],
                              dtype=np.float32)

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

        reward = 0
        # penalise if not closer to target
        if new_dist >= dist:
            reward += -1
        # grant large reward if target has been reached
        if terminated:
            reward += 100

        info = {}
        truncated = False

        if self.render_mode == "human":
            self.render()

        return self.state, reward, terminated, truncated, info

    def _env2canvas(self, pos, offset=True):
        """
        transforms environment x or y coordinate into pixel coordinate on canvas
        :param pos: x or y coordinate in environment
        :param offset: considers offset in position if true
        :return: pixel coordinate on canvas
        """
        offset = self.window_size / 2 if offset else 0
        scale = (self.window_size - 2*self.border_thickness)/(2*self.max_position + 2*self.agent_radius)

        return scale * pos + offset

    def render(self):
        """
        visualizes gym environment if render mode is "human".
        Note that y coordinate axis of canvas is pointing downwards!!!
        :return: rgb array if render mode is "rgb_array"
        """
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

        # draw the (blue, rectangular) start area
        start_left = self._env2canvas(self.start_area.low[0])
        start_top = self._env2canvas(self.start_area.low[1])
        start_width = self._env2canvas(self.start_area.high[0]) - self._env2canvas(self.start_area.low[0])
        start_height = self._env2canvas(self.start_area.high[1]) - self._env2canvas(self.start_area.low[1])
        pygame.draw.rect(
            self.canvas,
            (0, 0, 255),
            pygame.Rect(start_left, start_top, start_width, start_height)
        )

        # draw the (green, rectangular) target area
        target_left = self._env2canvas(self.target_area.low[0])
        target_top = self._env2canvas(self.target_area.low[1])
        target_width = self._env2canvas(self.target_area.high[0]) - self._env2canvas(self.target_area.low[0])
        target_height = self._env2canvas(self.target_area.high[1]) - self._env2canvas(self.target_area.low[1])
        pygame.draw.rect(
            self.canvas,
            (0, 255, 0),
            pygame.Rect(target_left, target_top, target_width, target_height)
        )

        # draw the (orange, circular) agent
        pygame.draw.circle(
            self.canvas,
            (255, 120, 0),
            (self._env2canvas(self.state[0]), self._env2canvas(self.state[1])),
            self._env2canvas(self.agent_radius, offset=False)
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
