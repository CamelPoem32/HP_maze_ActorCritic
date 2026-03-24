import argparse
import torch
import torch.optim as optim
import torch.nn as nn
import time
import os
from models import ActorCriticNet
from env import HarryPotterEnv

def train_a2c(env, model, args, device):
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
        
        # Restore history
        rewards_hist = checkpoint.get('rewards', [])
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
        log_probs, values, rewards = [], [], []
        done = False
        
        while not done:
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
            action, log_prob, value = model.get_action(obs_tensor)
            
            # Step the environment
            next_obs, reward, done, _, info = env.step(action.detach().cpu().squeeze().numpy())
            
            log_probs.append(log_prob)
            values.append(value)
            rewards.append(reward)
            obs = next_obs
            
        # Calculate Returns
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + args.gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns).to(device)
        # Standardize returns for stability
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        actor_loss, critic_loss = 0, 0
        for log_prob, value, R in zip(log_probs, values, returns):
            # Advantage = Return - Value Baseline
            advantage = R - value.item()
            actor_loss -= log_prob * advantage
            target = R.unsqueeze(0).to(device)
            critic_loss += nn.MSELoss()(value.view(-1), target.view(-1))
            
        loss = actor_loss + critic_loss
        
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
            print(f"Shared A2C | Ep: {ep} | Reward: {ep_reward:.2f} | Winrate {wins/args.log_interval:.2f} | Time: {ep_time:.3f}s | Result: {info.get('result', 'N/A')}")
            t0 = time.time()
            wins = 0

        # Periodic auto-save
        if ep % 1000 == 0 and ep > 0:
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'rewards': rewards_hist,
                'actor_losses': actor_loss_hist,
                'critic_losses': critic_loss_hist,
                'args': vars(args)
            }, args.save_path)

    # --- FINAL SAVE ---
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
    # python a2c_train.py --episodes 10000 --save_path a2c/a2c_10k.pt --device cpu
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
    model = ActorCriticNet(10, 2).to(device)
    train_a2c(env, model, args, device)