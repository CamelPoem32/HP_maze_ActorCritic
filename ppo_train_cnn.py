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
from env_cnn_njit import HarryPotterEnv

def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    """
    rewards: [T]
    values:  [T + 1]  (IMPORTANT: includes bootstrap value)
    dones:   [T]  (1 if done, 0 otherwise)
    """
    advantages = torch.zeros_like(rewards)
    gae = 0.0

    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * values[t + 1] * (1 - dones[t]) - values[t]
        gae = delta + gamma * lam * (1 - dones[t]) * gae
        advantages[t] = gae

    returns = advantages + values[:-1]
    return advantages, returns

def train_ppo(env, model, args, device, gamma=0.99, lam=0.95):
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    rewards_hist = []
    actor_loss_hist = []
    critic_loss_hist = []
    entropy_loss_hist = []
    kl_div_hist = []  # <--- New metric history
    winrates, winrate_epochs = [], []
    start_episode = 0

    # --- CHECKPOINT LOADING ---
    if args.load_path and os.path.isfile(args.load_path):
        print(f"Loading checkpoint from {args.load_path}...")
        checkpoint = torch.load(args.load_path, map_location=device, weights_only=False)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        rewards_hist = checkpoint.get('rewards', [])
        winrates = checkpoint.get('winrates', [])
        winrate_epochs = checkpoint.get('winrate_epochs', [])
        actor_loss_hist = checkpoint.get('actor_losses', [])
        critic_loss_hist = checkpoint.get('critic_losses', [])
        entropy_loss_hist = checkpoint.get('entropy_losses', [])
        kl_div_hist = checkpoint.get('kl_divs', []) # <--- Load KL history
        start_episode = len(rewards_hist)
        
        print(f"Resuming from episode {start_episode}")

    wins = 0
    for ep in range(start_episode, start_episode + args.episodes):
        start_time = time.time()
        obs, _ = env.reset()
        states, actions, log_probs, values, rewards, dones = [], [], [], [], [], []
        
        # 1. Trajectory Collection
        for step in range(args.rollout_steps):
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
            action, log_prob, value = model.get_action(obs_tensor)
            
            next_obs, reward, done, _, info = env.step(action.detach().cpu().squeeze().numpy())
            
            states.append(obs)
            actions.append(action.detach())   # Detach to avoid RuntimeError
            log_probs.append(log_prob.detach())
            values.append(value.detach())
            rewards.append(reward)
            dones.append(done)
            obs = next_obs
            if done:
                obs, _ = env.reset()

        # 2. Preparation for Update
        with torch.no_grad():
            _, _, last_value = model.get_action(torch.FloatTensor(next_obs).unsqueeze(0).to(device))

        values = torch.cat(values)
        values = torch.cat([values, last_value], dim=0).squeeze(-1)
        
        rewards = torch.FloatTensor(np.array(rewards)).to(device)
        old_states = torch.FloatTensor(np.array(states)).to(device)
        old_actions = torch.cat(actions).to(device)
        old_log_probs = torch.cat(log_probs).detach().to(device)
        dones = torch.BoolTensor(np.array(dones)).to(device).float()

        advantages, returns = compute_gae(rewards, values, dones, gamma, lam)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        ep_actor_loss, ep_critic_loss, ep_entropy_loss, ep_kl = 0, 0, 0, 0
        
        # 3. PPO Update Loop
        for _ in range(args.k_epochs):
            mean, std, current_values = model(old_states)
            dist = Normal(mean, std)
            current_log_probs = dist.log_prob(old_actions).sum(dim=-1)
            
            # --- KL Divergence Calculation ---
            log_ratio = current_log_probs - old_log_probs
            ratio = torch.exp(log_ratio)
            with torch.no_grad():
                # Approx KL: http://joschu.net/blog/kl-approx.html
                approx_kl = ((ratio - 1) - log_ratio).mean().item()
            
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - args.eps_clip, 1 + args.eps_clip) * advantages
            
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = nn.MSELoss()(current_values.squeeze(), returns)
            entropy_loss = dist.entropy().sum(dim=-1).mean()
            
            loss = actor_loss + 0.5 * critic_loss - 0.05 * entropy_loss
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()
            
            ep_actor_loss += actor_loss.item()
            ep_critic_loss += critic_loss.item()
            ep_entropy_loss += entropy_loss.item()
            ep_kl += approx_kl
            
        # 4. Logging & History
        ep_reward = sum(rewards).item()
        rewards_hist.append(ep_reward)
        actor_loss_hist.append(ep_actor_loss / args.k_epochs)
        critic_loss_hist.append(ep_critic_loss / args.k_epochs)
        entropy_loss_hist.append(ep_entropy_loss / args.k_epochs)
        kl_div_hist.append(ep_kl / args.k_epochs)

        if info.get('result') == "escaped": wins += 1
            
        ep_time = time.time() - start_time
        if ep % args.log_interval == 0:
            avg_kl = ep_kl / args.k_epochs
            print(f"PPO | Ep: {ep} | Reward: {ep_reward:.2f} | Winrate: {wins/args.log_interval:.2f} | Time: {ep_time:.3f}s | "
                  f"KL: {avg_kl:.5f} | Critic: {ep_critic_loss/args.k_epochs:.2f} | Ent: {ep_entropy_loss/args.k_epochs:.2f}")
            winrates.append(wins/args.log_interval)
            winrate_epochs.append(ep)
            wins = 0

        # Periodic Save
        if ep % 100 == 0:
            checkpoint_data = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'rewards': rewards_hist,
                'winrates': winrates,
                'winrate_epochs': winrate_epochs,
                'actor_losses': actor_loss_hist,
                'critic_losses': critic_loss_hist,
                'entropy_losses': entropy_loss_hist,
                'kl_divs': kl_div_hist, # <--- Persist KL history
                'args': vars(args)
            }
            torch.save(checkpoint_data, args.save_path)

    print(f"Final model saved to {args.save_path}")

if __name__ == '__main__':
    # Run with
    # python ppo_train_cnn.py --episodes 20000 --save_path ppo/ppo_20k_cnn.pt --device cuda
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--eps_clip', type=float, default=0.2)
    parser.add_argument('--k_epochs', type=int, default=4)
    parser.add_argument('--rollout_steps', type=int, default=1024)
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