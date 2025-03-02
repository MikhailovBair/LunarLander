from policy import Policy

from agent import Agent
import gymnasium as gym
import torch


class PolicyAgent(Agent):
    def __init__(self, policy: Policy):
        self.policy = policy

    def get_action(self, observation):
        distribution = self.policy(observation)
        action = distribution.sample()
        action_log_probability = distribution.log_prob(action)
        return action, action_log_probability

