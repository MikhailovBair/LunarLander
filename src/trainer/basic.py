from agent import Agent
from abc import ABC, abstractmethod

class Trainer(ABC):
    def __init__(
            self,
            agent: Agent,
            n_episodes: int,
            discount_factor: float = 0.99,
    ):
        self.agent = agent
        self.env = agent.env
        self.n_episodes = n_episodes
        self.discount_factor = discount_factor

    @abstractmethod
    def train(self):
        pass