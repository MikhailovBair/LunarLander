import matplotlib.pyplot as plt
import numpy as np
from config import visualizer_path, rolling_window

class Visualizer:
    def __init__(self,
                 save_path=visualizer_path):
        self.save_path = save_path

    def plot_rewards(self, n_episodes, total_rewards, custom_name=""):
        plt.figure(figsize=(16, 9))
        rmean = np.convolve(total_rewards, np.ones(rolling_window), 'valid') / rolling_window
        plt.plot(np.arange(0, n_episodes), total_rewards, label='Total rewards', zorder=10)
        plt.plot(np.arange(rolling_window - 1, n_episodes), rmean, label='Rolling Mean', zorder=20)
        plt.xlabel('Episodes')
        plt.ylabel('Total reward')
        plt.title(label="Training rewards " + custom_name)
        plt.legend()
        plt.savefig(self.save_path + "/training_rewards_" + custom_name + ".png")
        plt.close()

