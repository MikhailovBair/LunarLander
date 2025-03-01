import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
checkpoint_path = "../results/checkpoints"
visualizer_path = "../results/img"
hidden_size = 128
learning_rate = 0.001
n_episodes = 2000
discount_factor = 0.99
info_frequency = 100
rolling_window = 100