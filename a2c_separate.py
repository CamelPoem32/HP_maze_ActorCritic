import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
import time
import os

from env import HarryPotterEnv
from models import ActorNet, CriticNet

def train_a2c_separate(env, actor, critic, args, device):
    actor_opt = optim.Adam(actor.parameters(), lr=args.lr)
    critic_opt = optim.Adam(critic.parameters(), lr=args.lr)
    
    rewards_history = []
    actor_loss_history = []
    critic_loss_history = []
    start_episode = 0

    # --- CHECKPOINT LOADING ---
    if args.load_path and os.path.isfile(args.load_path):
        print(f"Loading checkpoint from {args.load_path}...")
        checkpoint = torch.load(args.load_path, map_location=device, weights_only=False)
        
        actor.load_state_dict(checkpoint['actor_state_dict'])
        critic.load_state_dict(checkpoint['critic_state_dict'])
        actor_opt.load_state_dict(checkpoint['actor_opt_state_dict'])
        critic_opt.load_state_dict(checkpoint['critic_opt_state_dict'])
        
        rewards_history = checkpoint.get('rewards', [])
        actor_loss_history = checkpoint.get('actor_losses', [])
        critic_loss_history = checkpoint.get('critic_losses', [])
        start_episode = len(rewards_history)
        
        print(f"Resuming from episode {start_episode}")
    elif args.load_path:
        print(f"Warning: {args.load_path} not found. Starting fresh.")

    wins = 0
    t0 = time.time()
    for ep in range(start_episode, start_episode + args.episodes):
        start_time = time.time()
        obs, _ = env.reset()
        log_probs, values, rewards = [], [], []
        done = False
        
        while not done:
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
            
            mean, std = actor(obs_tensor)
            value = critic(obs_tensor)
            
            dist = Normal(mean, std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=-1)
            
            # Environment step on CPU
            next_obs, reward, done, _, info = env.step(action.detach().cpu().squeeze().numpy())
            
            log_probs.append(log_prob)
            values.append(value)
            rewards.append(reward)
            obs = next_obs
            
        # Discounted Returns
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + args.gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns).to(device)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # Loss Calculation
        total_actor_loss, total_critic_loss = 0, 0
        for log_prob, value, R in zip(log_probs, values, returns):
            advantage = R - value.item()
            total_actor_loss -= log_prob * advantage
            target = R.unsqueeze(0).to(device)
            total_critic_loss += nn.MSELoss()(value.view(-1), target.view(-1))
            
        # Update Actor
        actor_opt.zero_grad()
        total_actor_loss.backward()
        actor_opt.step()
        
        # Update Critic
        critic_opt.zero_grad()
        total_critic_loss.backward()
        critic_opt.step()
        
        ep_time = time.time() - start_time
        ep_reward = sum(rewards)
        
        rewards_history.append(ep_reward)
        actor_loss_history.append(total_actor_loss.item())
        critic_loss_history.append(total_critic_loss.item())

        if info.get('result') == "escaped": wins += 1
        
        if ep % args.log_interval == 0:
            print(f"Separate A2C | Ep: {ep} | Reward: {ep_reward:.2f} | Winrate {wins/args.log_interval:.2f} | Time: {ep_time:.3f}s | Result: {info.get('result', 'N/A')}")
            t0 = time.time()
            wins = 0

        # Periodic save
        if ep % 1000 == 0 and ep > 0:
            torch.save({
                'actor_state_dict': actor.state_dict(),
                'critic_state_dict': critic.state_dict(),
                'actor_opt_state_dict': actor_opt.state_dict(),
                'critic_opt_state_dict': critic_opt.state_dict(),
                'rewards': rewards_history,
                'actor_losses': actor_loss_history,
                'critic_losses': critic_loss_history,
            }, args.save_path)

    # --- FINAL SAVE ---
    checkpoint = {
        'actor_state_dict': actor.state_dict(),
        'critic_state_dict': critic.state_dict(),
        'actor_opt_state_dict': actor_opt.state_dict(),
        'critic_opt_state_dict': critic_opt.state_dict(),
        'rewards': rewards_history,
        'actor_losses': actor_loss_history,
        'critic_losses': critic_loss_history,
        'args': vars(args)
    }
    torch.save(checkpoint, args.save_path)
    print(f"Final model saved to {args.save_path}")

if __name__ == '__main__':
    # Run with
    # python a2c_separate.py --episodes 10000 --save_path a2c_separate/a2c_separate_10k.pt --device cpu
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
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    
    actor = ActorNet(obs_dim, act_dim).to(device)
    critic = CriticNet(obs_dim).to(device)
    
    train_a2c_separate(env, actor, critic, args, device)