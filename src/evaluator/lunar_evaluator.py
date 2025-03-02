from agent import PolicyAgent
from evaluator import Evaluator
from config import device
import gymnasium as gym
import torch
import numpy as np
from tqdm.auto import tqdm

class LunarEvaluator(Evaluator):
    def __init__(self,
                 agent: PolicyAgent,
                 env: gym.Env):
        super().__init__(agent, env)
        self.agent = agent

    def evaluate_agent(self, num_times: int):
        rewards = []
        for _ in tqdm(range(num_times)):
            rewards.append(self.play_game())
        mean_reward = np.mean(rewards)
        return mean_reward, rewards

    def play_game(self):
        state, _ = self.env.reset()
        done = False
        total_reward = 0
        while not done:
            state_tensor = torch.tensor(state, dtype=torch.float32, device=device)
            action, _ = self.agent.get_action(observation=state_tensor)
            next_state, reward, terminated, truncated, _ = self.env.step(action.cpu().item())
            total_reward += reward
            done = terminated or truncated
            state = next_state
        return total_reward

