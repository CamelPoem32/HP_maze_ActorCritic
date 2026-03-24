import argparse
import torch
import torch.optim as optim
import torch.nn as nn
from torch.distributions import Normal
import numpy as np
import time
import os
from models import ActorCriticNet
from env import HarryPotterEnv

def train_ppo(env, model, args, device):
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    rewards_hist = []
    actor_loss_hist = []
    critic_loss_hist = []
    start_episode = 0

    # --- CHECKPOINT LOADING ---
    if args.load_path and os.path.isfile(args.load_path):
        print(f"Loading checkpoint from {args.load_path}...")
        checkpoint = torch.load(args.load_path, map_location=device, weights_only=False)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load history if it exists
        rewards_hist = checkpoint.get('rewards', [])
        actor_loss_hist = checkpoint.get('actor_losses', [])
        critic_loss_hist = checkpoint.get('critic_losses', [])
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
        returns = torch.tensor(returns).to(device)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        old_states = torch.FloatTensor(np.array(states)).to(device)
        old_actions = torch.cat(actions).to(device)
        old_log_probs = torch.cat(log_probs).detach().to(device)
        old_values = torch.cat(values).detach().squeeze().to(device)
        advantages = (returns - old_values).detach() # Advantage shouldn't propagate gradients back to old values
        
        ep_actor_loss, ep_critic_loss = 0, 0
        
        for _ in range(args.k_epochs):
            mean, std, current_values = model(old_states)
            dist = Normal(mean, std)
            current_log_probs = dist.log_prob(old_actions).sum(dim=-1)
            
            ratios = torch.exp(current_log_probs - old_log_probs)
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - args.eps_clip, 1 + args.eps_clip) * advantages
            
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = nn.MSELoss()(current_values.squeeze(), returns)
            
            loss = actor_loss + 0.5 * critic_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            ep_actor_loss += actor_loss.item()
            ep_critic_loss += critic_loss.item()
            
        ep_time = time.time() - start_time
        ep_reward = sum(rewards)
        
        rewards_hist.append(ep_reward)
        actor_loss_hist.append(ep_actor_loss / args.k_epochs)
        critic_loss_hist.append(ep_critic_loss / args.k_epochs)

        if info.get('result') == "escaped": wins += 1
            
        if ep % args.log_interval == 0:
            print(f"PPO | Ep: {ep} | Reward: {ep_reward:.2f} | Winrate {wins/args.log_interval:.2f} | Time: {(time.time() - t0)/args.log_interval:.3f}s | Result: {info.get('result', 'N/A')}")
            t0 = time.time()
            wins = 0

        # Intermediate save every 100 episodes
        if ep % 1000 == 0 and ep > 0:
            temp_checkpoint = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'rewards': rewards_hist,
                'actor_losses': actor_loss_hist,
                'critic_losses': critic_loss_hist,
                'args': vars(args)
            }
            torch.save(temp_checkpoint, args.save_path)

    # Final Save
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'rewards': rewards_hist,
        'actor_losses': actor_loss_hist,
        'critic_losses': critic_loss_hist,
        'args': vars(args)
    }
    torch.save(checkpoint, args.save_path)
    print(f"Final model and history saved to {args.save_path}")

if __name__ == '__main__':
    # Run with
    # python ppo_train.py --episodes 10000 --save_path ppo/ppo_10k.pt --device cpu
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
    # Observation space is 10, Action space is 2
    model = ActorCriticNet(10, 2).to(device)
    
    train_ppo(env, model, args, device)