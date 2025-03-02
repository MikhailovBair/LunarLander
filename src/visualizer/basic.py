from agent import PolicyAgent
from evaluator import Evaluator
from config import device, visualizer_path

import matplotlib.pyplot as plt
import matplotlib.animation
import gymnasium as gym
import torch
import numpy as np
import seaborn as sns
import matplotlib.patheffects as path_effects
from config import rolling_window
from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo


def add_median_labels(ax: plt.Axes, fmt: str = ".1f") -> None:
    """Add text labels to the median lines of a seaborn boxplot.

    Args:
        ax: plt.Axes, e.g. the return value of sns.boxplot()
        fmt: format string for the median value
    """
    lines = ax.get_lines()
    boxes = [c for c in ax.get_children() if "Patch" in str(c)]
    start = 4
    if not boxes:  # seaborn v0.13 => fill=False => no patches => +1 line
        boxes = [c for c in ax.get_lines() if len(c.get_xdata()) == 5]
        start += 1
    lines_per_box = len(lines) // len(boxes)
    for median in lines[start::lines_per_box]:
        x, y = (data.mean() for data in median.get_data())
        # choose value depending on horizontal or vertical plot orientation
        value = x if len(set(median.get_xdata())) == 1 else y
        text = ax.text(x, y, f'{value:{fmt}}', ha='center', va='center',
                       fontweight='bold', color='white')
        # create median-colored border around white text for contrast
        text.set_path_effects([
            path_effects.Stroke(linewidth=3, foreground=median.get_color()),
            path_effects.Normal(),
        ])


class Visualizer:
    def __init__(self,
                 environment: gym.Env,
                 agent: PolicyAgent,
                 save_path=visualizer_path,
                 ):
        self.env = environment
        self.agent = agent
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

    def visualize_game(self, custom_name=""):
        rec_env = RecordVideo(env=self.env, video_folder=self.save_path + "/video_final",
                              name_prefix=custom_name, episode_trigger=lambda x: True)
        evaluator = Evaluator(self.agent, rec_env)
        evaluator.play_game()
        rec_env.close()

    def visualize_evaluation(self, num_times, custom_name=""):
        evaluator = Evaluator(self.agent, self.env)
        median, rewards = evaluator.evaluate_agent(num_times)
        plt.figure(figsize=(16, 9))
        box_plot = sns.boxplot(x=rewards, label="Total rewards")
        add_median_labels(box_plot)
        plt.xlabel('Reward')
        plt.title("Policy Reward Distribution " + custom_name)
        plt.legend()
        plt.savefig(self.save_path + "/policy_reward_distribution" + custom_name + ".png")
        plt.close()

