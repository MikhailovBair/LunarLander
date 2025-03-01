import matplotlib.pyplot as plt
import numpy as np
from config import visualizer_path, rolling_window

class Visualizer:
    def __init__(self,
                 n_episodes: int,
                 total_rewards: list[float]):
        self.n_episodes = n_episodes
        self.total_rewards = total_rewards

    def plot_rewards(self):
        plt.figure(figsize=(16, 9))
        rmean = np.convolve(self.total_rewards, np.ones(rolling_window), 'valid') / rolling_window
        plt.plot(np.arange(0, self.n_episodes), self.total_rewards, label='Total rewards', zorder=10)
        plt.plot(np.arange(rolling_window - 1, self.n_episodes), rmean, label='Rolling Mean', zorder=20)
        plt.xlabel('Episodes')
        plt.ylabel('Total reward')
        plt.legend()
        plt.savefig(visualizer_path + "/rewards.png")

