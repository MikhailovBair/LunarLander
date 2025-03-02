from trainer import Trainer
from agent import PolicyAgent
from config import device, checkpoint_path
from tqdm.auto import tqdm
import torch

class REINFORCETrainer(Trainer):
    def __init__(self,
                 agent: PolicyAgent,
                 n_episodes: int,
                 discount_factor: float,
                 learning_rate: float,
                 optimizer_class,
                 info_frequency: int = 100,
                 checkpoint_path_:str=checkpoint_path
                 ):
        super().__init__(agent, n_episodes, discount_factor)
        self.agent = agent
        self.optimizer = optimizer_class(self.agent.policy.parameters(), lr=learning_rate)
        self.gamma = discount_factor
        self.device = device
        self.info_frequency = info_frequency
        self.checkpoint_path = checkpoint_path_


    def train(self):
        total_rewards = []
        policies = []
        for episode in tqdm(range(self.n_episodes)):
            state, _ = self.env.reset()

            rewards = []
            log_probs = []

            done = False
            while not done:
                state_tensor = torch.tensor(state, dtype=torch.float32,
                                            device=self.device)
                action, log_prob_action = self.agent.get_action(state_tensor)
                next_state, reward, terminated, truncated, _ = self.env.step(action.cpu().item())

                log_probs.append(log_prob_action)
                rewards.append(reward)

                done = terminated or truncated
                state = next_state

            loss = self.calculate_loss(rewards, log_probs)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_rewards.append(sum(rewards))
            policies.append(self.agent.policy.state_dict())
            if (episode + 1) % self.info_frequency == 0:
                print(f'Episode {episode}, Total Reward: {total_rewards[-1]:.2f}')
                torch.save({
                    'episode': episode,
                    'model_state_dict': self.agent.policy.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                }, f'{self.checkpoint_path}/checkpoint_{episode}.tar')
        self.env.close()
        return policies[-1], policies, total_rewards


    def calculate_loss(self, rewards, log_probs):
        returns = self.calculate_returns(rewards)
        log_probs = torch.stack(log_probs)
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







