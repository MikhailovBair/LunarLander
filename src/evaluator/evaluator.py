import gymnasium as gym
from agent import Agent

from abc import ABC, abstractmethod

class Evaluator(ABC):
    def __init__(self,
                 agent: Agent,
                 env: gym.Env):
        self.agent = agent
        self.env = env

    @abstractmethod
    def evaluate_agent(self, num_times: int):
        pass
