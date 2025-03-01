import gymnasium as gym
from abc import ABC, abstractmethod

class Agent(ABC):
    def __init__(
        self,
        env: gym.Env,
    ):
        self.env = env

    @abstractmethod
    def get_action(self, observation):
        pass