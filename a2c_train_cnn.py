import argparse
import torch
torch.set_float32_matmul_precision('high')

import torch.optim as optim
import torch.nn as nn
import time
import numpy as np
import os
from models_cnn import ActorCriticNet
from env_cnn import HarryPotterEnv

def train_a2c(env, model, args, device):
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    rewards_hist = []
    actor_loss_hist = []
    critic_loss_hist = []
    winrates, winrate_epochs = [], []
    start_episode = 0

    # --- CHECKPOINT LOADING ---
    if args.load_path and os.path.isfile(args.load_path):
        print(f"Loading checkpoint from {args.load_path}...")
        checkpoint = torch.load(args.load_path, map_location=device, weights_only=False)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Restore history
        rewards_hist = checkpoint.get('rewards', [])
        winrates = checkpoint.get('winrates', [])
        winrate_epochs = checkpoint.get('winrate_epochs', [])
        actor_loss_hist = checkpoint.get('actor_losses', [])
        critic_loss_hist = checkpoint.get('critic_losses', [])
        start_episode = len(rewards_hist)
        
        print(f"Resuming from episode {start_episode}")
    elif args.load_path:
        print(f"Warning: Checkpoint file {args.load_path} not found. Starting from scratch.")

    wins = 0
    t0 = time.time()
    # Adjusted range to account for resumed episodes
    for ep in range(start_episode, start_episode + args.episodes):
        start_time = time.time()
        obs, _ = env.reset()
        
        # Lists to store trajectory data
        log_probs, values, rewards = [], [], []
        next_obses, masks = [], [] # Need masks to handle terminal states
        done = False
        
        # 1. Collect Trajectory
        while not done:
            obs_tensor = torch.FloatTensor(obs).to(device) # Shape: (C, H, W) or (D,)
            # get_action expects a batch dimension, so we use .unsqueeze(0)
            action, log_prob, value = model.get_action(obs_tensor.unsqueeze(0))
            
            next_obs, reward, done, _, info = env.step(action.detach().cpu().squeeze().numpy())
            
            log_probs.append(log_prob)
            values.append(value)
            rewards.append(reward)
            next_obses.append(next_obs)
            masks.append(1.0 - done) # 0.0 if state is terminal, 1.0 otherwise
            
            obs = next_obs

        # 2. Vectorize Data (Convert lists to tensors in one go)
        # We use .view(-1) to ensure they are 1D arrays
        log_probs_t = torch.cat(log_probs).view(-1)
        values_t = torch.cat(values).view(-1)
        rewards_t = torch.FloatTensor(rewards).to(device)
        masks_t = torch.FloatTensor(masks).to(device)
        
        # Stack all next_obs into one big batch: Shape (Steps, Channels, H, W)
        next_obs_batch = torch.FloatTensor(np.array(next_obses)).to(device)

        # 3. Get all Next Values in ONE forward pass
        # We don't need gradients for the "target" side of the equation
        with torch.no_grad():
            _, _, next_values_t = model(next_obs_batch)
            next_values_t = next_values_t.view(-1)

        # 4. Calculate TD-Target and Advantage
        # Note: if it's the last step (mask=0), target is just the reward
        td_targets = rewards_t + args.gamma * next_values_t * masks_t
        advantages = td_targets - values_t

        # 5. Compute Final Losses (Vectorized)
        # We detach advantages so we don't backprop through the baseline
        actor_loss = -(log_probs_t * advantages.detach()).mean()
        
        # Critic loss: how close was our estimate V(s_t) to the TD-Target?
        critic_loss = nn.MSELoss()(values_t, td_targets)
        
        loss = actor_loss + critic_loss
        
        # Perform backprop...
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        ep_time = time.time() - start_time
        ep_reward = sum(rewards)
        
        rewards_hist.append(ep_reward)
        actor_loss_hist.append(actor_loss.item())
        critic_loss_hist.append(critic_loss.item())

        if info.get('result') == "escaped": wins += 1
        
        if ep % args.log_interval == 0:
            print(f"Shared A2C | Ep: {ep} | Reward: {ep_reward:.2f} | Winrate {wins/args.log_interval:.2f} | Time: {ep_time:.3f}s | ",
                  f"Critic loss {critic_loss.item():.2f} | Actor loss {actor_loss.item():.2f}"
                #  "| Result: {info.get('result', 'N/A')}"
                )
            winrates.append(wins/args.log_interval)
            winrate_epochs.append(ep)
            t0 = time.time()
            wins = 0

        # Periodic auto-save
        if ep % 1000 == 0 and ep > 0:
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'rewards': rewards_hist,
                'winrates': winrates,
                'winrate_epochs': winrate_epochs,
                'actor_losses': actor_loss_hist,
                'critic_losses': critic_loss_hist,
                'args': vars(args)
            }, args.save_path)

    # --- FINAL SAVE ---
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'rewards': rewards_hist,
        'winrates': winrates,
        'winrate_epochs': winrate_epochs,
        'actor_losses': actor_loss_hist,
        'critic_losses': critic_loss_hist,
        'args': vars(args)
    }
    torch.save(checkpoint, args.save_path)
    print(f"Final model and history saved to {args.save_path}")

if __name__ == '__main__':
    # Run with
    # python a2c_train_cnn.py --episodes 20000 --save_path a2c/a2c_20k_cnn.pt --device cuda
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--log_interval', type=int, default=100)
    parser.add_argument('--save_path', type=str, default='a2c/a2c.pt')
    parser.add_argument('--load_path', type=str, default=None, help="Path to existing .pt file")
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()

    device = args.device
    print(f"Using device: {device}")

    env = HarryPotterEnv()
    env.reset()
    # Get the 3D shape: (4, 64, 64)
    obs_shape = env.observation_space.shape
    model = ActorCriticNet(obs_shape, 2).to(device)
    train_a2c(env, model, args, device)