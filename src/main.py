import gymnasium as gym
import numpy as np
import torch.optim

from agent import PolicyAgent
from config import (device, hidden_size, learning_rate,
                    n_steps, discount_factor, info_frequency,
                    evaluation_time, visualizer_path,
                    video_record_period, checkpoint_path, num_runs, data_path, rolling_window, num_envs)
from policy import FCPolicy
from trainer import REINFORCETrainer
from visualizer import Visualizer
from visualizer.basic import plot_comparison

if __name__ == "__main__":
    all_rewards = []
    all_lengths = []
    for run in range(num_runs):
        final_env = gym.make("LunarLander-v3", render_mode="rgb_array")
        env = gym.make_vec('LunarLander-v3', num_envs, render_mode =  "rgb_array")
        output_size = final_env.action_space.n
        policy = FCPolicy(input_size=env.observation_space.shape[-1],
                          hidden_size=hidden_size,
                          output_size=output_size,
                          ).to(device)
        # checkpoint = torch.load("../local_results/checkpoints_final/checkpoint_899.tar", weights_only=True)
        # policy.load_state_dict(checkpoint['model_state_dict'])
        agent = PolicyAgent(policy=policy)
        Optimizer_class = torch.optim.Adam
        trainer = REINFORCETrainer(agent=agent,
                                   env=env,
                                   n_steps=n_steps,
                                   discount_factor=discount_factor,
                                   learning_rate=learning_rate,
                                   optimizer_class=Optimizer_class,
                                   info_frequency=info_frequency,
                                   num_envs_=num_envs,
                                   checkpoint_path_=checkpoint_path,
                                   video_path=visualizer_path,
                                   record_period=video_record_period
                                   )
        last_policy, rewards, lengths = trainer.train()

        all_lengths.append(lengths)
        all_rewards.append(rewards)

        policy.load_state_dict(last_policy)
        final_agent = PolicyAgent(policy)

        custom_name = f"_run_{run}"

        final_env = gym.make("LunarLander-v3", render_mode="rgb_array")
        visualizer = Visualizer(environment=final_env,
                                agent=final_agent)
        visualizer.plot_statistics(n_steps=n_steps,
                                   total_rewards=rewards,
                                   lengths=lengths,
                                   custom_name=custom_name)
        visualizer.visualize_game(custom_name=custom_name)
        visualizer.visualize_evaluation(evaluation_time, custom_name)

    plot_comparison(all_rewards, "Rewards", window=rolling_window, dir_path=data_path)
    plot_comparison(all_lengths, "Lengths", window=rolling_window, dir_path=data_path)
    all_rewards = np.stack(all_lengths)
    all_lengths = np.stack(all_lengths)
    np.save(data_path + "/rewards.npy", all_rewards)
    np.save(data_path + "/lengths.npy", all_lengths)


