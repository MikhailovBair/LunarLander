import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
checkpoint_path = "../results/checkpoints"
visualizer_path = "../results/img"
data_path = "../results/training_data"
video_record_period=100
hidden_size = 128
learning_rate = 0.001
n_steps = 200
discount_factor = 0.99
info_frequency = 100
rolling_window = 100
evaluation_time = 20
update_interval = 5
save_interval = 100
num_runs = 1