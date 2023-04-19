"""This is an 'associative bandit' problem, where the behavior of the agent should depend on the state, i.e., the day of the week.
"""

import matplotlib.pyplot as plt
import numpy as np
from abc import ABC, abstractmethod

class Env(ABC):
    def __init__(self):
        pass
    
    @abstractmethod
    def allocate(self, ffl_store: int, day_of_week: int):
        """Take an action in a specific env state and return a reward."""
        pass


class TwoStoresEnv(Env):
    def __init__(self):
        self.ptfs = np.array([
            [0.66, 0.75, 0.78, 0.62, 0.60, 0.54, 0.57], # location 1
            [0.45, 0.56, 0.66, 0.75, 0.78, 0.60, 0.52], # location 2
        ])

    def allocate(self, ffl_store, day_of_week, return_label=False):
        """Allocate an order to store `ffl_store` on day-of-week `day_of_week`.
        Return a reward indicating success or failure of the allocation.
        """
        ptf = self.ptfs[ffl_store, day_of_week]
        
        # flip a coin
        r = np.random.binomial(n=1, p=ptf) # 1 = success
        if return_label:
            return r
        return 1.0 if r == 1 else -1.0

    def plot(self, ax = None):
        if ax is None:
            _, ax = plt.subplots(figsize=(5,3))

        colors = ('red', 'blue')
        for i in range(self.ptfs.shape[0]):
            ax.plot(self.ptfs[i,:], c=colors[i], label=f"location {i}")
        ax.set(ylim=[0,1], ylabel='PTF', xlabel='day of week', xticks=range(7))
        ax.legend()
        plt.show()


class Agent():
    default_policy = lambda s: np.random.randint(0, 2) # uniform random
    
    def __init__(self, policy = default_policy):
        self._history = []
        self.policy_func = policy
        self.reward_total = 0.0
        self.gamma = 1.0

    def __call__(self, state, env = None):
        """Follow the agent's policy to select an action.
            If env is not None, then the action is passed to the env,
            a reward is obtained, and the reward is integrated into the
            agent's history.
        """
        # select an action
        action = self.policy_func(state)

        # if env passed, take action and return a reward
        reward = None
        if env is not None:
            reward = env(action)

        results = (state, action, reward or 0.0)

        # track history
        self._history.append(results)
        # total reward
        if reward is not None:
            self.reward_total += self.gamma * reward

    def reset(self) -> None:
        self._history = []