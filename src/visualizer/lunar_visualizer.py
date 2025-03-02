from cProfile import label

from agent import PolicyAgent
from evaluator import LunarEvaluator
from visualizer import Visualizer
from config import device, visualizer_path

import matplotlib.pyplot as plt
import matplotlib.animation
import gymnasium as gym
import torch
import seaborn as sns
import matplotlib.patheffects as path_effects


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


class LunarVisualizer(Visualizer):
    def __init__(self,
                 environment: gym.Env,
                 agent: PolicyAgent,
                 save_path=visualizer_path,
                 ):
        super().__init__(save_path)
        self.env = environment
        self.agent = agent
        self.evaluator = LunarEvaluator(self.agent, self.env)

    def visualize_game(self, custom_name=""):
        state, _ = self.env.reset()
        done = False
        frames = []
        while not done:
            frames.append(self.env.render())
            state_tensor = torch.tensor(state, dtype=torch.float32, device=device)
            action, _ = self.agent.get_action(observation=state_tensor)
            next_state, reward, terminated, truncated, _ = self.env.step(action.cpu().item())
            done = terminated or truncated
            state = next_state

        fig = plt.figure(figsize=(5, 5))
        plt.axis('off')
        im = plt.imshow(frames[0])

        def animate(i):
            im.set_array(frames[i])
            return im,

        anim = matplotlib.animation.FuncAnimation(fig, animate, frames=len(frames))
        writer_video = matplotlib.animation.FFMpegWriter(fps=60)
        anim.save(self.save_path + "/demo_" + custom_name + ".mp4", writer=writer_video)
        plt.close()

    def visualize_evaluation(self, num_times, custom_name=""):
        evaluator = LunarEvaluator(self.agent, self.env)
        median, rewards = evaluator.evaluate_agent(num_times)
        plt.figure(figsize=(16, 9))
        box_plot = sns.boxplot(x=rewards, label="Total rewards")
        add_median_labels(box_plot)
        plt.xlabel('Reward')
        plt.title("Policy Reward Distribution " + custom_name)
        plt.legend()
        plt.savefig(self.save_path + "/policy_reward_distribution" + custom_name + ".png")
        plt.close()

