from agent import Agent
from config import device
from policy import Policy


class PolicyAgent(Agent):
    def __init__(self, policy: Policy):
        self.policy = policy.float().to(device)

    def get_action(self, observation):
        distribution = self.policy(observation)
        action = distribution.sample()
        action_log_probability = distribution.log_prob(action)
        return action, action_log_probability.unsqueeze(-1)

