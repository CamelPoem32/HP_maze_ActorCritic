import argparse
import torch
torch.set_float32_matmul_precision('high')

import torch.optim as optim
import torch.nn as nn
from torch.distributions import Normal
import numpy as np
import time
import os
from models_cnn import ActorCriticNet
from env_cnn import HarryPotterEnv

def train_ppo(env, model, args, device):
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    rewards_hist = []
    actor_loss_hist = []
    critic_loss_hist = []
    entropy_loss_hist = []
    winrates, winrate_epochs = [], []
    start_episode = 0

    # --- CHECKPOINT LOADING ---
    if args.load_path and os.path.isfile(args.load_path):
        print(f"Loading checkpoint from {args.load_path}...")
        checkpoint = torch.load(args.load_path, map_location=device, weights_only=False)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load history if it exists
        rewards_hist = checkpoint.get('rewards', [])
        winrates = checkpoint.get('winrates', [])
        winrate_epochs = checkpoint.get('winrate_epochs', [])
        actor_loss_hist = checkpoint.get('actor_losses', [])
        critic_loss_hist = checkpoint.get('critic_losses', [])
        entropy_loss_hist = checkpoint.get('entropy_losses', [])
        start_episode = len(rewards_hist)
        
        print(f"Resuming from episode {start_episode}")
    elif args.load_path:
        print(f"Warning: Checkpoint file {args.load_path} not found. Starting from scratch.")

    # Adjust loop to handle start_episode
    wins = 0
    t0 = time.time()
    for ep in range(start_episode, start_episode + args.episodes):
        start_time = time.time()
        obs, _ = env.reset()
        states, actions, log_probs, values, rewards = [], [], [], [], []
        done = False
        
        while not done:
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
            action, log_prob, value = model.get_action(obs_tensor)
            
            # Detach and move to CPU for the environment
            next_obs, reward, done, _, info = env.step(action.detach().cpu().squeeze().numpy())
            
            states.append(obs)
            actions.append(action)
            log_probs.append(log_prob)
            values.append(value)
            rewards.append(reward)
            obs = next_obs
            
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + args.gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns).to(device).float()
        if returns.std() > 1e-5:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        old_states = torch.FloatTensor(np.array(states)).to(device).float()
        old_actions = torch.cat(actions).to(device).float()
        old_log_probs = torch.cat(log_probs).detach().to(device).float()
        old_values = torch.cat(values).detach().squeeze().to(device).float()
        advantages = (returns - old_values).detach() # Advantage shouldn't propagate gradients back to old values
        
        ep_actor_loss, ep_critic_loss, ep_entropy_loss = 0, 0, 0
        
        for _ in range(args.k_epochs):
            mean, std, current_values = model(old_states)
            dist = Normal(mean, std)
            current_log_probs = dist.log_prob(old_actions).sum(dim=-1)
            
            ratios = torch.exp(current_log_probs - old_log_probs)
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - args.eps_clip, 1 + args.eps_clip) * advantages
            
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = nn.MSELoss()(current_values.squeeze(), returns)
            entropy_loss = dist.entropy().sum(dim=-1).mean()
            
            c_critic = torch.tensor(0.5, dtype=torch.float32, device=device)
            c_entropy = torch.tensor(0.0075, dtype=torch.float32, device=device)
            # Force each component to float32 and ensure they are on the right device
            actor_loss = actor_loss.to(device=device, dtype=torch.float32)
            critic_loss = critic_loss.to(device=device, dtype=torch.float32)
            entropy_loss = entropy_loss.to(device=device, dtype=torch.float32)
            
            # print(f"Actor: {actor_loss.dtype}, Critic: {critic_loss.dtype}, Entropy: {entropy_loss.dtype}")
            loss = actor_loss + c_critic * critic_loss - c_entropy * entropy_loss
            # print(f"Loss: {loss.dtype}")
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()
            
            ep_actor_loss += actor_loss.item()
            ep_critic_loss += critic_loss.item()
            ep_entropy_loss += entropy_loss.item()
            
        ep_time = time.time() - start_time
        ep_reward = sum(rewards)
        
        rewards_hist.append(ep_reward)
        actor_loss_hist.append(ep_actor_loss / args.k_epochs)
        critic_loss_hist.append(ep_critic_loss / args.k_epochs)
        entropy_loss_hist.append(ep_entropy_loss / args.k_epochs)

        if info.get('result') == "escaped": wins += 1
            
        if ep % args.log_interval == 0:
            print(f"PPO | Ep: {ep} | Reward: {ep_reward:.2f} | Winrate {wins/args.log_interval:.2f} | Time: {ep_time:.3f}s | ",
                  f"Critic loss {critic_loss.item():.2f} | Actor loss {actor_loss.item():.2f} | Entropy loss {entropy_loss.item():.2f}"
            )
            winrates.append(wins/args.log_interval)
            winrate_epochs.append(ep)
            t0 = time.time()
            wins = 0

        # Intermediate save every 100 episodes
        if ep % 1000 == 0 and ep > 0:
            temp_checkpoint = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'rewards': rewards_hist,
                'winrates': winrates,
                'winrate_epochs': winrate_epochs,
                'actor_losses': actor_loss_hist,
                'critic_losses': critic_loss_hist,
                'entropy_losses': entropy_loss_hist,
                'args': vars(args)
            }
            torch.save(temp_checkpoint, args.save_path)

    # Final Save
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'rewards': rewards_hist,
        'winrates': winrates,
        'winrate_epochs': winrate_epochs,
        'actor_losses': actor_loss_hist,
        'critic_losses': critic_loss_hist,
        'entropy_losses': entropy_loss_hist,
        'args': vars(args)
    }
    torch.save(checkpoint, args.save_path)
    print(f"Final model and history saved to {args.save_path}")

if __name__ == '__main__':
    # Run with
    # python ppo_train_cnn.py --episodes 20000 --save_path ppo/ppo_20k_cnn.pt --device cuda
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--eps_clip', type=float, default=0.2)
    parser.add_argument('--k_epochs', type=int, default=4)
    parser.add_argument('--log_interval', type=int, default=100)
    parser.add_argument('--save_path', type=str, default='ppo/ppo_ckpt.pt')
    parser.add_argument('--load_path', type=str, default=None, help="Path to load a checkpoint from")
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()

    device = args.device
    print(f"Using device: {device}")

    env = HarryPotterEnv()
    env.reset()
    
    # Get the 3D shape: (4, 64, 64)
    obs_shape = env.observation_space.shape
    
    # Model now takes a shape tuple instead of a 1D length
    model = ActorCriticNet(obs_shape, 2).to(device)
    
    train_ppo(env, model, args, device)