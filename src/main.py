import gymnasium as gym
import torch.optim

from policy import FCPolicy
from agent import PolicyAgent
from trainer import REINFORCETrainer
from visualizer import LunarVisualizer
from config import (device, hidden_size, learning_rate,
                    n_episodes, discount_factor, info_frequency,
                    evaluation_time)

if __name__ == "__main__":
    env = gym.make("LunarLander-v3", render_mode="rgb_array")
    policy = FCPolicy(input_size=env.observation_space.shape[0],
                      hidden_size=hidden_size,
                      output_size=env.action_space.n,
                      ).to(device)
    agent = PolicyAgent(env=env, policy=policy)
    Optimizer_class = torch.optim.Adam
    trainer = REINFORCETrainer(agent=agent, n_episodes=n_episodes,
                               discount_factor=discount_factor,
                               learning_rate=learning_rate,
                               optimizer_class=Optimizer_class,
                               info_frequency=info_frequency,
                               )
    last_policy, policies, rewards = trainer.train()

    policy.load_state_dict(last_policy)
    final_agent = PolicyAgent(env, policy)
    visualizer = LunarVisualizer(environment=env,
                                 agent=final_agent)
    visualizer.plot_rewards(n_episodes=n_episodes,
                            total_rewards=rewards)
    visualizer.visualize_game()
    visualizer.visualize_evaluation(evaluation_time, "")


