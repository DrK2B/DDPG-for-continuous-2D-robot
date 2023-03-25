import numpy as np


class Noise:
    def __init__(self, action_space):
        self.action_dim = action_space.shape[0]
        self.low = action_space.low
        self.high = action_space.high

    def reset(self):
        pass

    def add_noise(self, action):
        raise NotImplementedError


class GaussianNoise(Noise):
    def __init__(self, action_space, mu=0.0, sigma=0.3):
        super().__init__(action_space)
        self.mu = mu
        self.sigma = sigma

    def add_noise(self, action):
        noise = np.random.normal(self.mu, self.sigma, self.action_dim)
        return np.clip(action + noise, self.low, self.high)


class OUNoise(Noise):
    """
    This strategy implements the Ornstein-Uhlenbeck process, which adds
    time-correlated noise to the actions taken by the deterministic policy.
    The OU process satisfies the following stochastic differential equation:
    dxt = theta*(mu - xt)*dt + sigma*dWt
    where Wt denotes the Wiener process
    """

    def __init__(self, action_space, mu=0.0, theta=0.15, max_sigma=0.3, min_sigma=None, decay_period=100000):
        super().__init__(action_space)
        self.state = None
        self.mu = mu
        self.theta = theta
        self.sigma = max_sigma
        self.max_sigma = max_sigma
        if min_sigma is None:
            min_sigma = max_sigma
        self.min_sigma = min_sigma
        self.decay_period = decay_period
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu

    def evolve_state(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.action_dim)
        self.state = x + dx
        return self.state

    def add_noise(self, action, t=0):
        ou_state = self.evolve_state()
        self.sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(1.0, t / self.decay_period)
        return np.clip(action + ou_state, self.low, self.high)
