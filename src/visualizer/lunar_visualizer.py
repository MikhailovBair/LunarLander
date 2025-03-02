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
        mean, rewards = evaluator.evaluate_agent(num_times)
        plt.figure(figsize=(16, 9))
        sns.boxplot(x=rewards, label="Total rewards")
        plt.xlabel('Reward')
        plt.title("Policy Reward Distribution")
        plt.legend()
        plt.savefig(self.save_path + "/policy_reward_distribution" + custom_name + ".png")

