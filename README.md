# LunarLander
![demo](https://gymnasium.farama.org/_images/lunar_lander.gif)

## Table of Contents
- [Environment description](#environment-description)
- [Rewards](#rewards)
- [Method](#method-applied)
- [Hyperparameter tuning](#hyperparameter-tuning)
- [Results](#results)

## Environment description
### Overview

This environment is a classic rocket trajectory optimization problem. According to Pontryagin’s maximum principle, it is optimal to fire the engine at full throttle or turn it off. This is the reason why this environment has discrete actions: engine on or off.

There are two environment versions: discrete or continuous. The landing pad is always at coordinates (0,0). The coordinates are the first two numbers in the state vector. Landing outside of the landing pad is possible. Fuel is infinite, so an agent can learn to fly and then land on its first attempt.

### Action Space

There are four discrete actions available:

- 0: do nothing

- 1: fire left orientation engine

- 2: fire main engine

- 3: fire right orientation engine

### Observation Space

The state is an 8-dimensional vector: the coordinates of the lander in x & y, its linear velocities in x & y, its angle, its angular velocity, and two booleans that represent whether each leg is in contact with the ground or not.

## Rewards

After every step a reward is granted. The total reward of an episode is the sum of the rewards for all the steps within that episode.

For each step, the reward:



- is increased/decreased the closer/further the lander is to the landing pad.

- is increased/decreased the slower/faster the lander is moving.

- is decreased the more the lander is tilted (angle not horizontal).

- is increased by 10 points for each leg that is in contact with the ground.

- is decreased by 0.03 points each frame a side engine is firing.

- Starting Stateis decreased by 0.3 points each frame the main engine is firing.

The episode receive an additional reward of -100 or +100 points for crashing or landing safely respectively.

An episode is considered a solution if it scores at least 200 points.

## Method applied

**REINFORCE**


REINFORCE is a Monte Carlo-based policy gradient algorithm used in Reinforcement Learning (RL) to optimize a policy directly. REINFORCE algorithm falls under the class of on-policy methods, meaning it updates the policy based on the actions taken during the current policy's execution.

REINFORCE algorithm improves the policy by adjusting the probabilities of actions taken in each state based on the cumulative rewards (or returns) obtained after those actions. Unlike value-based methods, which rely on estimating state-action values, REINFORCE directly learns the policy that maps states to actions, making it well-suited for environments with continuous action spaces or complex tasks where value estimation is challenging.

**The REINFORCE algorithm works in the following steps:**

- **Collect Episodes:** The agent interacts with the environment for a fixed number of steps or until an episode is complete, following the current policy. This generates a trajectory consisting of states, actions, and rewards.

- **Calculate Returns:** For each time step $t$, calculate the return $G_t$​​, which is the total reward obtained from time $t$ onwards. Typically, this is the discounted sum of rewards:

$$ G_t = \sum_{k=t}^{T} \gamma^{k-t} R_k $$

Where $\gamma$ is the discount factor, $\{T}$ is the final time step of the episode, and $R_k​$ is the reward received at time step $k$.

- **Policy Gradient Update:** The policy parameters $θ$ are updated using the following formula:

$$ \theta_{t+1} = \theta_t + \alpha \nabla_{\theta} \log \pi_{\theta}(a_t | s_t) G_t $$

Where:

$\alpha$ is the learning rate.

$\pi_{\theta}(a_t | s_t)$ is the probability of taking action $a_t​$ at state  $s_t​$, according to the policy.

$G_t$​ is the return or cumulative reward obtained from time step $t$ onwards.

The gradient $\nabla_{\theta} \log \pi_{\theta}(a_t | s_t)$ represents how much the policy probability for action atat​​ at state stst​ should be adjusted based on the obtained return.

- **Repeat:** This process is repeated for several episodes, iteratively updating the policy in the direction of higher rewards.

## Hyperparameter tuning

To increase stability of training process and reduce gradient variance we evaluated influence of changing a policy update interval. Simple replay buffer was implemented, with size provided in `update_interval` argument in configuration file. In section `Results` you can find training process visulization for update intervals 1, 5 and 10.

## Results
Demonsration of learned policy:

![demo](https://github.com/user-attachments/assets/5507156b-c9aa-45e3-bde1-5ea017c9176d)

Training rewards with update interval 1:
![Results](https://github.com/MikhailovBair/LunarLander/blob/development/results/img/rewards_1.png)
Training rewards with update interval 5:
![Results](https://github.com/MikhailovBair/LunarLander/blob/development/results/img/rewards_5.png)
Training rewards with update interval 10:
![Results](https://github.com/MikhailovBair/LunarLander/blob/development/results/img/rewards_10.png)


