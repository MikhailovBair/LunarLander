from abc import ABC, abstractmethod

import gymnasium as gym
from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo

from agent import Agent
from config import visualizer_path, video_record_period


class Trainer(ABC):
    def __init__(
            self,
            agent: Agent,
            env: gym.Env,
            n_episodes: int,
            discount_factor: float = 0.99,
            video_path=visualizer_path,
            record_period=video_record_period,
    ):
        self.agent = agent
        self.env=RecordVideo(env=env, video_folder=video_path + "/video_training",
                             episode_trigger=lambda x: x % record_period == 0)
        self.env=RecordEpisodeStatistics(self.env)
        self.n_episodes = n_episodes
        self.discount_factor = discount_factor

    @abstractmethod
    def train(self):
        pass