import argparse
import torch
import matplotlib.pyplot as plt
import numpy as np
import os

def smooth_curve(data, window_size=50):
    """Applies a moving average to smooth out noisy RL data."""
    if len(data) < window_size:
        return data
    ret = np.cumsum(data, dtype=float)
    ret[window_size:] = ret[window_size:] - ret[:-window_size]
    return ret[window_size - 1:] / window_size

def plot_training_results(checkpoints_data):
    """
    Plots Rewards, Winrates, Actor Loss, and Critic Loss.
    checkpoints_data is a list of tuples: [(method_name, data_dict), ...]
    """
    fig, axs = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('Reinforcement Learning Training Metrics', fontsize=16)

    for method, data in checkpoints_data:
        # Extract data (default to empty list if key is missing for some reason)
        rewards = data.get('rewards', [])
        winrates = data.get('winrates', [])
        winrate_epochs = data.get('winrate_epochs', [])
        actor_losses = data.get('actor_losses', [])
        critic_losses = data.get('critic_losses', [])

        # X-axes for step=1 metrics
        episodes = np.arange(len(rewards))
        
        # 1. Plot Rewards
        # Plot faint raw data in the background, bold smoothed data on top
        axs[0, 0].plot(episodes, rewards, alpha=0.2)
        if len(rewards) >= 50:
            smoothed_rewards = smooth_curve(rewards, window_size=50)
            axs[0, 0].plot(np.arange(49, len(rewards)), smoothed_rewards, label=f"{method} (Smoothed)", linewidth=2)
        else:
            axs[0, 0].plot(episodes, rewards, label=method, linewidth=2)

        # 2. Plot Winrates (using specific epochs as X-axis)
        if len(winrates) > 0 and len(winrate_epochs) > 0:
            axs[0, 1].plot(winrate_epochs, np.array(winrates)*100, marker='o', linestyle='-', label=method, linewidth=2)

        # 3. Plot Actor Losses
        if len(actor_losses) >= 50:
            smoothed_al = smooth_curve(actor_losses, window_size=50)
            axs[1, 0].plot(np.arange(49, len(actor_losses)), smoothed_al, label=method, linewidth=2)
        else:
            axs[1, 0].plot(np.arange(len(actor_losses)), actor_losses, alpha=0.8, label=method)

        # 4. Plot Critic Losses
        if len(critic_losses) >= 50:
            smoothed_cl = smooth_curve(critic_losses, window_size=50)
            axs[1, 1].plot(np.arange(49, len(critic_losses)), smoothed_cl, label=method, linewidth=2)
        else:
            axs[1, 1].plot(np.arange(len(critic_losses)), critic_losses, alpha=0.8, label=method)

    # Formatting Subplots
    axs[0, 0].set_title('Episode Rewards')
    axs[0, 0].set_xlabel('Episode')
    axs[0, 0].set_ylabel('Total Reward')
    axs[0, 0].legend()
    axs[0, 0].grid(True, alpha=0.3)

    axs[0, 1].set_title('Winrate vs. Evaluation Epoch')
    axs[0, 1].set_xlabel('Episode')
    axs[0, 1].set_ylabel('Winrate (%)')
    axs[0, 1].legend()
    axs[0, 1].grid(True, alpha=0.3)

    axs[1, 0].set_title('Actor Loss')
    axs[1, 0].set_xlabel('Episode')
    axs[1, 0].set_ylabel('Loss')
    axs[1, 0].legend()
    axs[1, 0].grid(True, alpha=0.3)

    axs[1, 1].set_title('Critic Loss')
    axs[1, 1].set_xlabel('Episode')
    axs[1, 1].set_ylabel('Loss')
    axs[1, 1].legend()
    axs[1, 1].grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to make room for suptitle
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Visualize RL Training Checkpoints")
    # Using nargs='+' allows you to pass multiple files and methods
    parser.add_argument('--files', type=str, nargs='+', required=True, help="List of paths to checkpoint files (.pt or .pth)")
    parser.add_argument('--methods', type=str, nargs='+', required=True, help="List of method names corresponding to the files")
    
    args = parser.parse_args()

    if len(args.files) != len(args.methods):
        print("Error: The number of files must match the number of method labels.")
        return

    loaded_data = []

    for file_path, method_name in zip(args.files, args.methods):
        if not os.path.exists(file_path):
            print(f"Warning: File not found -> {file_path}. Skipping.")
            continue
        
        print(f"Loading {method_name} from {file_path}...")
        # map_location='cpu' ensures it loads safely even if trained on GPU but visualized on a standard laptop
        checkpoint = torch.load(file_path, map_location=torch.device('cpu'), weights_only=False)
        loaded_data.append((method_name, checkpoint))

    if not loaded_data:
        print("No valid data loaded. Exiting.")
        return

    print("Generating plots...")
    plot_training_results(loaded_data)

if __name__ == '__main__':
    # python visualize.py --files ppo_ckpt.pt --methods "PPO"
    # python visualize.py --files ppo_ckpt.pt a2c_shared_ckpt.pth a2c_separate_ckpt.pth --methods "PPO" "Shared A2C" "Separate A2C"
    main()