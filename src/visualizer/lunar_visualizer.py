from agent import PolicyAgent
from visualizer import Visualizer
from config import device, visualizer_path

import matplotlib.pyplot as plt
import matplotlib.animation
import gymnasium as gym
import torch

class LunarVisualizer(Visualizer):
    def __init__(self,
                 environment: gym.Env,
                 agent: PolicyAgent,
                 n_episodes: int,
                 total_rewards: list[float]
                 ):
        super().__init__(n_episodes, total_rewards)
        self.env = environment
        self.agent = agent

    def visualize_game(self):
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
        anim.save(visualizer_path + "/demo.mp4", writer=writer_video)
        plt.close()