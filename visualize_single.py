import argparse
import torch
import matplotlib.pyplot as plt
import numpy as np
import os

def smooth_curve(data, window_size=50):
    """Applies a moving average to smooth out noisy RL data."""
    if len(data) < window_size:
        return np.array(data)
    ret = np.cumsum(data, dtype=float)
    ret[window_size:] = ret[window_size:] - ret[:-window_size]
    return ret[window_size - 1:] / window_size

def symlog(x):
    return np.sign(x) * np.log10(np.abs(x))

def plot_detailed_results(file_path, method_name):
    # Load data
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} not found.")
        return

    checkpoint = torch.load(file_path, map_location=torch.device('cpu'), weights_only=False)
    
    # Extract metrics
    rewards = checkpoint.get('rewards', [])
    winrates = checkpoint.get('winrates', [])
    winrate_epochs = checkpoint.get('winrate_epochs', [])
    actor_losses = checkpoint.get('actor_losses', [])
    critic_losses = checkpoint.get('critic_losses', [])
    entropy_losses = checkpoint.get('entropy_losses', []) # Specific to PPO or entropy-enabled A2C

    # Setup the figure (2 rows, 3 columns)
    fig, axs = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f'Detailed Training Analysis: {method_name}', fontsize=18, fontweight='bold')

    # Define dynamic smoothing based on data length
    smooth_step = max(1, len(rewards) // 100)
    
    # --- 1. Episode Rewards ---
    ax = axs[0, 0]
    ax.plot(rewards, color='skyblue', alpha=0.3)
    if len(rewards) > smooth_step:
        ax.plot(np.arange(smooth_step-1, len(rewards)), smooth_curve(rewards, smooth_step), color='blue', linewidth=1.5)
    ax.set_title('Total Episode Rewards')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward')
    ax.grid(True, alpha=0.3)

    # --- 2. Winrates ---
    ax = axs[0, 1]
    smooth_step_winrates = max(1, len(winrates) // 100)
    x_axis = winrate_epochs if winrate_epochs else np.arange(len(winrates))
    ax.plot(x_axis, np.array(winrates) * 100, color='palegreen', alpha=0.3)
    if len(winrates) > 0:
        # If winrate_epochs isn't provided, assume standard log intervals
        ax.plot(x_axis[smooth_step_winrates-1:], smooth_curve(np.array(winrates) * 100, smooth_step_winrates), color='green', linestyle='-', linewidth=1)
        ax.set_title('Winrate Percentage (%)')
        ax.set_ylabel('Winrate %')
    else:
        ax.text(0.5, 0.5, 'No Winrate Data', ha='center')
    ax.grid(True, alpha=0.3)

    # --- 3. Entropy (The "Curiosity" metric) ---
    ax = axs[0, 2]
    if len(entropy_losses) > 0:
        ax.plot(entropy_losses, color='orange', alpha=0.3)
        if len(entropy_losses) > smooth_step:
            ax.plot(np.arange(smooth_step-1, len(entropy_losses)), smooth_curve(entropy_losses, smooth_step), color='darkorange', linewidth=1.5)
        ax.set_title('Policy Entropy (Exploration)')
        ax.set_ylabel('Entropy Value')
    else:
        ax.text(0.5, 0.5, 'No Entropy Data\n(Typical for basic A2C)', ha='center')
    ax.grid(True, alpha=0.3)

    # --- 4. Actor Loss ---
    ax = axs[1, 0]
    ax.plot(symlog(actor_losses), color='salmon', alpha=0.3)
    if len(actor_losses) > smooth_step:
        ax.plot(np.arange(smooth_step-1, len(actor_losses)), symlog(smooth_curve(actor_losses, smooth_step)), color='red', linewidth=1.5)
    ax.set_title('Actor Loss (Policy Gradient)')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Symlog Loss')
    # ax.set_yscale('log')
    ax.grid(True, alpha=0.3)

    # --- 5. Critic Loss ---
    ax = axs[1, 1]
    ax.plot(critic_losses, color='lightgray', alpha=0.3)
    if len(critic_losses) > smooth_step:
        ax.plot(np.arange(smooth_step-1, len(critic_losses)), smooth_curve(critic_losses, smooth_step), color='black', linewidth=1.5)
    ax.set_title('Critic Loss (Value MSE)')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Loss')
    ax.set_yscale('log') # Losses in RL often span orders of magnitude
    ax.grid(True, alpha=0.3)

    # --- 6. Summary Stats / Placeholder ---
    ax = axs[1, 2]
    ax.axis('off')
    summary_text = (
        f"Method: {method_name}\n"
        f"Total Episodes: {len(rewards)}\n"
        f"Final Avg Reward: {np.mean(rewards[-100:]):.2f}\n"
        f"Max Winrate: {max(winrates)*100 if winrates else 0:.1f}%\n"
        f"Final Winrate: {winrates[-1]*100 if winrates else 0:.1f}%"
    )
    ax.text(0.1, 0.5, summary_text, fontsize=12, verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save option
    save_name = f"{method_name.lower().replace(' ', '_')}_detailed.png"
    plt.savefig(save_name)
    print(f"Plot saved as {save_name}")
    plt.show()

if __name__ == '__main__':
    # python visualize_single.py --file ppo/ppo_20k_wallpoint_cnn.pt --name "ppo"
    parser = argparse.ArgumentParser(description="Single-Method Detailed RL Visualizer")
    parser.add_argument('--file', type=str, required=True, help="Path to checkpoint (.pt)")
    parser.add_argument('--name', type=str, default="RL Agent", help="Display name for the method")
    
    args = parser.parse_args()
    plot_detailed_results(args.file, args.name)