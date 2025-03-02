from agent import Agent
from abc import ABC, abstractmethod
from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo
import gymnasium as gym
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
        self.env=RecordEpisodeStatistics(self.env, buffer_length=n_episodes)
        self.n_episodes = n_episodes
        self.discount_factor = discount_factor

    @abstractmethod
    def train(self):
        pass