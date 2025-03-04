import gymnasium as gym
import torch
from tqdm.auto import tqdm
import numpy as np

from agent import PolicyAgent
from config import (device, checkpoint_path,
                    video_record_period, visualizer_path,
                    update_interval, save_interval, num_envs)
from trainer import Trainer


class REINFORCETrainer(Trainer):
    def __init__(self,
                 agent: PolicyAgent,
                 env,
                 n_steps: int,
                 discount_factor: float,
                 learning_rate: float,
                 optimizer_class,
                 info_frequency: int = 100,
                 update_interval_:int=update_interval,
                 save_interval_: int=save_interval,
                 num_envs_:int=num_envs,
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
        self.num_envs = num_envs_

    def train(self):
        total_rewards = []
        total_lengths = []
        for step in tqdm(range(self.n_steps)):
            step_returns = torch.empty((self.num_envs, 0), dtype=torch.float64, device=device)
            step_log_probs = torch.empty((self.num_envs, 0), dtype=torch.float64, device=device)
            step_rewards = []
            step_lengths = []
            for episode in range(self.update_interval):
                state, _ = self.env.reset()
                episode_rewards = np.empty((self.num_envs, 0), dtype=np.float64)
                done = [False] * self.num_envs
                while not all(done):
                    state_tensor = torch.tensor(state, dtype=torch.float32,
                                                device=self.device).to(device)
                    actions, log_probs = self.agent.get_action(state_tensor)

                     # Take a step in the environment for all agents
                    next_state, reward, terminated, truncated, _ = self.env.step(actions.detach().cpu().numpy())  # Assuming `next_state` is batched

                    # Collect log probabilities, rewards, and update dones
                    step_log_probs = torch.cat((step_log_probs, log_probs), dim = 1)
                    for  num, (term, trunc) in enumerate(zip(terminated.tolist(), truncated.tolist())):
                        if term or trunc:
                            next_state[num], _ = self.env.envs[num].reset()

                    for i in range(len(done)):
                        if done[i]:
                            reward[i] = 0

                    done = [d or t or r for d, t, r in zip(done, terminated.tolist(), truncated.tolist())]
                    episode_rewards = np.concatenate((episode_rewards, reward[..., np.newaxis]), axis=1)

                    state = next_state

                # episode_returns = self.calculate_returns(episode_rewards)
                average_reward_sum = np.mean(np.sum(episode_rewards, axis=1))
                average_episode_len = np.mean(np.count_nonzero(episode_rewards, axis=1))

                step_returns = torch.cat((step_returns, self.calculate_returns(episode_rewards)), dim=1)
                
                step_rewards.append(average_reward_sum)  # Store the mean reward for this episode
                step_lengths.append(average_episode_len)

            loss = self.calculate_loss(step_returns, step_log_probs)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_rewards.append(np.mean(step_rewards))
            total_lengths.append(np.mean(step_lengths))

            if (step + 1) % self.info_frequency == 0:
                print(f'Episode {step}, Total Reward: {total_rewards[-1]:.2f}, Average Episode Duration: {total_lengths[-1]:.2f}')

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
        total_returns = torch.empty(rewards.shape, dtype=torch.float32, device=self.device)
        for i in range(self.num_envs):
            returns = []
            current_return = 0
            for r in reversed(rewards[i].tolist()):
                if r != -np.inf:
                    current_return = r + self.gamma * current_return
                returns.append(current_return)
            returns = list(reversed(returns))
            total_returns[i] = torch.tensor(returns, dtype=torch.float32, device=self.device)
        # returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        return total_returns







