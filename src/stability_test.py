import gymnasium as gym
import matplotlib.pyplot as plt
import seaborn as sns
import torch.optim
import os
from tqdm.auto import tqdm
from evaluator import LunarEvaluator
from policy import FCPolicy
from agent import PolicyAgent
from trainer import REINFORCETrainer
from visualizer import LunarVisualizer
from config import (device, hidden_size, learning_rate,
                    discount_factor, info_frequency,
                    evaluation_time)

medians = []
for i in range(10):
    env = gym.make("LunarLander-v3", render_mode="rgb_array")
    policy = FCPolicy(input_size=env.observation_space.shape[0],
                      hidden_size=hidden_size,
                      output_size=env.action_space.n,
                      ).to(device)
    agent = PolicyAgent(env=env, policy=policy)
    Optimizer_class = torch.optim.Adam

    cp_path = f"../local_results/checkpoints_{i}/"
    img_path = f"../local_results/img/"
    os.makedirs(name=cp_path, exist_ok=True)
    n_episodes = 1000

    trainer = REINFORCETrainer(agent=agent, n_episodes=n_episodes,
                               discount_factor=discount_factor,
                               learning_rate=learning_rate,
                               optimizer_class=Optimizer_class,
                               info_frequency=info_frequency,
                               checkpoint_path_=cp_path
                               )
    last_policy, policies, rewards = trainer.train()

    policy.load_state_dict(last_policy)
    final_agent = PolicyAgent(env, policy)
    visualizer = LunarVisualizer(environment=env,
                                 agent=final_agent,
                                 save_path=img_path)
    visualizer.plot_rewards(n_episodes=n_episodes,
                            total_rewards=rewards)
    visualizer.visualize_game(custom_name=f"run_{i}")
    visualizer.visualize_evaluation(evaluation_time, f"run_{i}")
    evaluator = LunarEvaluator(agent, env)
    median, results = evaluator.evaluate_agent(evaluation_time)
    medians.append(median)
    print(f"\nCurrent median is {median}")

plt.figure(figsize=(16, 9))
sns.rugplot(medians)
plt.savefig("../local_results/img/median_plot.png")
print(medians)