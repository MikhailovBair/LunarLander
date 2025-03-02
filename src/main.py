import gymnasium as gym
import torch.optim

from policy import FCPolicy
from agent import PolicyAgent
from trainer import REINFORCETrainer
from visualizer import Visualizer
from config import (device, hidden_size, learning_rate,
                    n_episodes, discount_factor, info_frequency,
                    evaluation_time, visualizer_path,
                    video_record_period, checkpoint_path)

if __name__ == "__main__":
    env = gym.make("LunarLander-v3", render_mode="rgb_array")
    policy = FCPolicy(input_size=env.observation_space.shape[0],
                      hidden_size=hidden_size,
                      output_size=env.action_space.n,
                      ).to(device)
    agent = PolicyAgent(policy=policy)
    Optimizer_class = torch.optim.Adam
    trainer = REINFORCETrainer(agent=agent,
                               env=env,
                               n_episodes=n_episodes,
                               discount_factor=discount_factor,
                               learning_rate=learning_rate,
                               optimizer_class=Optimizer_class,
                               info_frequency=info_frequency,
                               checkpoint_path_=checkpoint_path,
                               video_path=visualizer_path,
                               record_period=video_record_period
                               )
    last_policy, policies, rewards, lengths = trainer.train()

    policy.load_state_dict(last_policy)
    final_agent = PolicyAgent(policy)
    visualizer = Visualizer(environment=env,
                            agent=final_agent)
    visualizer.plot_statistics(n_episodes=n_episodes,
                               total_rewards=rewards,
                               lengths=lengths,
                               custom_name="")
    visualizer.visualize_game()
    visualizer.visualize_evaluation(evaluation_time, "")


