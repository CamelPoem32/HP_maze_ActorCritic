import argparse
import torch
torch.set_float32_matmul_precision('high')

import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
import time
import os

from env_cnn import HarryPotterEnv
from models_cnn import ActorNet, CriticNet, SharedACNet

def train_a2c_separate(env, model, args, device):
    # actor_opt = optim.Adam(actor.parameters(), lr=args.lr)
    # critic_opt = optim.Adam(critic.parameters(), lr=args.lr)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    rewards_history = []
    actor_loss_history = []
    critic_loss_history = []
    entropy_loss_history = []
    winrates, winrate_epochs = [], []
    start_episode = 0

    # --- CHECKPOINT LOADING ---
    if args.load_path and os.path.isfile(args.load_path):
        print(f"Loading checkpoint from {args.load_path}...")
        checkpoint = torch.load(args.load_path, map_location=device, weights_only=False)
        
        # actor.load_state_dict(checkpoint['actor_state_dict'])
        # critic.load_state_dict(checkpoint['critic_state_dict'])
        # actor_opt.load_state_dict(checkpoint['actor_opt_state_dict'])
        # critic_opt.load_state_dict(checkpoint['critic_opt_state_dict'])
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        rewards_history = checkpoint.get('rewards', [])
        winrates = checkpoint.get('winrates', [])
        winrate_epochs = checkpoint.get('winrate_epochs', [])
        actor_loss_history = checkpoint.get('actor_losses', [])
        critic_loss_history = checkpoint.get('critic_losses', [])
        entropy_loss_history = checkpoint.get('entropy_losses', [])
        start_episode = len(rewards_history)
        
        print(f"Resuming from episode {start_episode}")
    elif args.load_path:
        print(f"Warning: {args.load_path} not found. Starting fresh.")

    wins = 0
    t0 = time.time()
    for ep in range(start_episode, start_episode + args.episodes):
        start_time = time.time()
        obs, _ = env.reset()
        # Buffers for the trajectory
        log_probs, values, rewards = [], [], []
        next_obses, masks, entropies = [], [], []
        done = False
        
        # 1. Collect Trajectory (Environment Loop)
        while not done:
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
            
            # Separate Forward Passes
            # mean, std = actor(obs_tensor)
            # value = critic(obs_tensor)
            mean, std, value = model(obs_tensor)
            
            dist = torch.distributions.Normal(mean, std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=-1)
            entropy = dist.entropy().sum(dim=-1)
            
            next_obs, reward, done, _, info = env.step(action.detach().cpu().squeeze().numpy())
            
            # Store data
            log_probs.append(log_prob)
            values.append(value)
            rewards.append(reward)
            next_obses.append(next_obs)
            masks.append(1.0 - done) 
            entropies.append(entropy)
            
            obs = next_obs
            
        # 2. Vectorize Data (Conversion to Tensors)
        log_probs_t = torch.cat(log_probs).view(-1)
        values_t = torch.cat(values).view(-1)
        rewards_t = torch.FloatTensor(rewards).to(device)
        masks_t = torch.FloatTensor(masks).to(device)
        entropy_t = torch.cat(entropies).mean()
        
        # Batch all next observations for a single Critic call
        next_obs_batch = torch.FloatTensor(np.array(next_obses)).to(device)
        
        # 3. Vectorized TD-Target Calculation
        with torch.no_grad():
            # Only need the critic for the TD target
            # next_values_t = critic(next_obs_batch).view(-1)
            _, _, next_values_t = model(next_obs_batch)
            next_values_t = next_values_t.view(-1)
            
        # TD Target: r + gamma * V(s_t+1)
        # We multiply by masks_t so terminal state value is exactly 0
        td_targets = rewards_t + args.gamma * next_values_t * masks_t
        
        # Advantage: (Actual Outcome) - (Our Baseline)
        advantages = td_targets - values_t

        # 4. Separate Loss Calculation
        # Actor Loss: Log_prob * Advantage + Entropy Bonus
        # Note: We detach advantage so it doesn't try to train the Critic through this loss
        actor_loss = -(log_probs_t * advantages.detach()).mean() - 0.01 * entropy_t
        
        # Critic Loss: MSE between current value estimate and TD-Target
        critic_loss = torch.nn.MSELoss()(values_t, td_targets)

        loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy_t

        # # 5. Optimization
        # # If you have separate optimizers:
        # actor_opt.zero_grad()
        # actor_loss.backward()
        # torch.nn.utils.clip_grad_norm_(actor.parameters(), 0.5)
        # actor_opt.step()
        # critic_opt.zero_grad()
        # critic_loss.backward()
        # torch.nn.utils.clip_grad_norm_(critic.parameters(), 0.5)
        # critic_opt.step()

        # 5. Single Optimization Step
        optimizer.zero_grad()
        loss.backward()
        # Clip gradients to prevent CNN weights from exploding
        nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        
        ep_time = time.time() - start_time
        ep_reward = sum(rewards)
        
        rewards_history.append(ep_reward)
        actor_loss_history.append(actor_loss.item())
        critic_loss_history.append(critic_loss.item())
        entropy_loss_history.append(entropy_t.item())

        if info.get('result') == "escaped": wins += 1
        
        if ep % args.log_interval == 0:
            print(f"Separate A2C | Ep: {ep} | Reward: {ep_reward:.2f} | Winrate {wins/args.log_interval:.2f} | Time: {ep_time:.3f}s | ",
                  f"Critic loss {critic_loss.item():.2f} | Actor loss {actor_loss.item():.2f} | Entropy loss {entropy_t.item():.2f}"
            )
            winrates.append(wins/args.log_interval)
            winrate_epochs.append(ep)
            t0 = time.time()
            wins = 0

        # Periodic save
        if ep % 1000 == 0 and ep > 0:
            torch.save({
                # 'actor_state_dict': actor.state_dict(),
                # 'critic_state_dict': critic.state_dict(),
                # 'actor_opt_state_dict': actor_opt.state_dict(),
                # 'critic_opt_state_dict': critic_opt.state_dict(),
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'rewards': rewards_history,
                'winrates': winrates,
                'winrate_epochs': winrate_epochs,
                'actor_losses': actor_loss_history,
                'critic_losses': critic_loss_history,
                'entropy_losses': entropy_loss_history,
            }, args.save_path)

    # --- FINAL SAVE ---
    checkpoint = {
        # 'actor_state_dict': actor.state_dict(),
        # 'critic_state_dict': critic.state_dict(),
        # 'actor_opt_state_dict': actor_opt.state_dict(),
        # 'critic_opt_state_dict': critic_opt.state_dict(),
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'rewards': rewards_history,
        'winrates': winrates,
        'winrate_epochs': winrate_epochs,
        'actor_losses': actor_loss_history,
        'critic_losses': critic_loss_history,
        'entropy_losses': entropy_loss_history,
        'args': vars(args)
    }
    torch.save(checkpoint, args.save_path)
    print(f"Final model saved to {args.save_path}")

if __name__ == '__main__':
    # Run with
    # python a2c_separate_cnn.py --episodes 20000 --save_path a2c_separate/a2c_separate_20k_cnn.pt --device cuda
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--log_interval', type=int, default=100)
    parser.add_argument('--save_path', type=str, default='a2c_separate/a2c_separate.pt')
    parser.add_argument('--load_path', type=str, default=None)
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()

    device = args.device
    print(f"Using device: {device}")

    env = HarryPotterEnv()
    env.reset()
    # Get the 3D shape: (4, 64, 64)
    obs_shape = env.observation_space.shape
    
    # actor = ActorNet(obs_shape, 2).to(device)
    # critic = CriticNet(obs_shape).to(device)
    ac = SharedACNet(obs_shape, 2).to(device)
    
    # train_a2c_separate(env, actor, critic, args, device)
    train_a2c_separate(env, ac, args, device)