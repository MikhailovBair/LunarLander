import gymnasium as gym
import torch
from tqdm.auto import tqdm
import numpy as np

from agent import PolicyAgent
from config import (device, checkpoint_path,
                    video_record_period, visualizer_path,
                    update_interval, save_interval)
from trainer import Trainer


class REINFORCETrainer(Trainer):
    def __init__(self,
                 agent: PolicyAgent,
                 env: gym.Env,
                 n_steps: int,
                 discount_factor: float,
                 learning_rate: float,
                 optimizer_class,
                 info_frequency: int = 100,
                 update_interval_:int=update_interval,
                 save_interval_: int=save_interval,
                 checkpoint_path_:str=checkpoint_path,
                 video_path = visualizer_path,
                 record_period = video_record_period,
                 ):
        super().__init__(agent, env, n_steps, discount_factor, video_path, record_period)
        self.agent = agent
        self.optimizer = optimizer_class(self.agent.policy.parameters(), lr=learning_rate)
        self.gamma = discount_factor
        self.device = device
        self.info_frequency = info_frequency
        self.checkpoint_path = checkpoint_path_
        self.update_interval = update_interval_
        self.save_interval = save_interval_

    def train(self):
        total_rewards = []
        total_lengths = []
        for step in tqdm(range(self.n_steps)):
            step_returns = torch.empty(0, dtype=torch.float64, device=device)
            step_log_probs = torch.empty(0, dtype=torch.float64, device=device)
            step_rewards = []
            step_lengths = []
            for episode in range(self.update_interval):
                state, _ = self.env.reset()
                rewards = []
                done = False
                while not done:
                    state_tensor = torch.tensor(state, dtype=torch.float32,
                                                device=self.device).to(device)
                    action, log_prob_action = self.agent.get_action(state_tensor)
                    next_state, reward, terminated, truncated, _ = self.env.step(action.cpu().item())

                    step_log_probs = torch.concatenate((step_log_probs,log_prob_action))
                    rewards.append(reward)
                    done = terminated or truncated
                    state = next_state

                episode_returns = self.calculate_returns(rewards)
                step_returns = torch.concatenate((step_returns, episode_returns))
                step_rewards.append(self.env.return_queue[-1])
                step_lengths.append(self.env.length_queue[-1])

            loss = self.calculate_loss(step_returns, step_log_probs)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_rewards.append(np.mean(step_rewards))
            total_lengths.append(np.mean(step_lengths))

            if (step + 1) % self.info_frequency == 0:
                print(f'Episode {step}, Total Reward: {total_rewards[-1]:.2f}')

            if (step + 1) % self.save_interval == 0:
                torch.save({
                    'episode': step,
                    'model_state_dict': self.agent.policy.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                }, f'{self.checkpoint_path}/checkpoint_{step}.tar')

        last_policy = self.agent.policy.state_dict()
        self.env.close()
        return last_policy, np.array(total_rewards), np.array(total_lengths)


    def calculate_loss(self, returns, log_probs):
        # log_probs = torch.stack(log_probs)
        loss = -(log_probs * returns).sum() ### Is it really minus?
        return loss


    def calculate_returns(self, rewards):
        returns = []
        current_return = 0
        for r in reversed(rewards):
            current_return = r + self.gamma * current_return
            returns.append(current_return)
        returns = list(reversed(returns))
        returns = torch.tensor(returns, dtype=torch.float32, device=self.device)
        # returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        return returns







