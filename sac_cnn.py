import argparse
import torch
torch.set_float32_matmul_precision('high')

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import time
import os
import random
from collections import deque

from env_cnn_njit import HarryPotterEnv
from models_cnn import SACNet 

class ReplayBuffer:
    def __init__(self, capacity, obs_shape, action_dim, storage_device, train_device):
        self.capacity = capacity
        self.storage_device = storage_device
        self.train_device = train_device # Device where gradients are calculated
        self.ptr = 0
        self.size = 0

        # Optimization: Store as uint8 to save 4x memory regardless of device
        if 'cuda' in storage_device:
            # Pre-allocate directly on GPU
            self.states = torch.zeros((capacity, *obs_shape), dtype=torch.uint8, device=storage_device)
            self.next_states = torch.zeros((capacity, *obs_shape), dtype=torch.uint8, device=storage_device)
            self.actions = torch.zeros((capacity, action_dim), dtype=torch.float32, device=storage_device)
            self.rewards = torch.zeros((capacity, 1), dtype=torch.float32, device=storage_device)
            self.dones = torch.zeros((capacity, 1), dtype=torch.float32, device=storage_device)
        else:
            # Pre-allocate on System RAM
            self.states = np.zeros((capacity, *obs_shape), dtype=np.uint8)
            self.next_states = np.zeros((capacity, *obs_shape), dtype=np.uint8)
            self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
            self.rewards = np.zeros((capacity, 1), dtype=np.float32)
            self.dones = np.zeros((capacity, 1), dtype=np.float32)

    def push(self, state, action, reward, next_state, done):
        # Normalize and convert to uint8 before storing
        # Use as_tensor to handle both numpy and torch inputs efficiently
        s_fixed = (torch.as_tensor(state) * 255).to(torch.uint8)
        ns_fixed = (torch.as_tensor(next_state) * 255).to(torch.uint8)

        if 'cuda' in self.storage_device:
            self.states[self.ptr] = s_fixed.to(self.storage_device)
            self.next_states[self.ptr] = ns_fixed.to(self.storage_device)
            self.actions[self.ptr] = torch.as_tensor(action, device=self.storage_device)
            self.rewards[self.ptr] = torch.tensor([reward], device=self.storage_device)
            self.dones[self.ptr] = torch.tensor([done], device=self.storage_device)
        else:
            self.states[self.ptr] = s_fixed.numpy()
            self.next_states[self.ptr] = ns_fixed.numpy()
            self.actions[self.ptr] = action
            self.rewards[self.ptr] = reward
            self.dones[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        if 'cuda' in self.storage_device:
            idxs = torch.randint(0, self.size, (batch_size,), device=self.storage_device)
        else:
            idxs = np.random.randint(0, self.size, size=batch_size)

        # Retrieve and ensure everything is on the training device
        s = torch.as_tensor(self.states[idxs], device=self.train_device).float() / 255.0
        ns = torch.as_tensor(self.next_states[idxs], device=self.train_device).float() / 255.0
        a = torch.as_tensor(self.actions[idxs], device=self.train_device)
        r = torch.as_tensor(self.rewards[idxs], device=self.train_device)
        d = torch.as_tensor(self.dones[idxs], device=self.train_device)

        return s, a, r, ns, d

def train_sac(env, model, args, device):
    # Optimizers
    actor_params = list(model.actor.parameters()) + list(model.cnn.parameters())
    critic_params = list(model.critic1.parameters()) + list(model.critic2.parameters()) + list(model.cnn.parameters())
    actor_opt = optim.Adam(actor_params, lr=args.lr)
    critic_opt = optim.Adam(critic_params, lr=args.lr) # Slower LR for critics can help stabilize training
    
    # Adaptive Entropy (Alpha) Setup
    target_entropy = -0.5 * np.prod(env.action_space.shape).item() # Heuristic: -dim(A)
    log_alpha = torch.zeros(1, requires_grad=True, device=device)
    alpha_opt = optim.Adam([log_alpha], lr=args.lr)
    
    obs_shape = env.observation_space.shape
    # Use np.prod to handle multi-dimensional action spaces (like [2,] or [1,2])
    action_dim = np.prod(env.action_space.shape)
    replay_buffer = ReplayBuffer(
        capacity=args.buffer_size, 
        obs_shape=env.observation_space.shape, 
        action_dim=np.prod(env.action_space.shape),
        storage_device=args.buffer_device, # Where it sits
        train_device=device)
    
    rewards_history = []
    actor_loss_history = []
    critic_loss_history = []
    alpha_loss_history = []
    
    # Diagnostic Histories
    entropy_history = []
    q_mean_history = []
    q_gap_history = []      # Difference between Q1 and Q2
    target_q_history = []   # What the critic is trying to reach
    std_mean_history = []   # How "random" the policy is
    
    winrates, winrate_epochs = [], []
    start_episode = 0
    wins = 0

    global_step = 0

    # --- CHECKPOINT LOADING ---
    if args.load_path and os.path.isfile(args.load_path):
        print(f"Loading checkpoint from {args.load_path}...")
        checkpoint = torch.load(args.load_path, map_location=device, weights_only=False)

        global_step = checkpoint['global_step']
        model.load_state_dict(checkpoint['model_state_dict'])
        actor_opt.load_state_dict(checkpoint['actor_opt_state_dict'])
        critic_opt.load_state_dict(checkpoint['critic_opt_state_dict'])
        alpha_opt.load_state_dict(checkpoint['alpha_opt_state_dict'])
        log_alpha = checkpoint['log_alpha']
        
        rewards_history = checkpoint.get('rewards', [])
        winrates = checkpoint.get('winrates', [])
        winrate_epochs = checkpoint.get('winrate_epochs', [])
        actor_loss_history = checkpoint.get('actor_losses', [])
        critic_loss_history = checkpoint.get('critic_losses', [])
        alpha_loss_history = checkpoint.get('alpha_losses', [])
        start_episode = len(rewards_history)
        # New Diagnostics
        entropy_history = checkpoint.get('entropies', [])
        q_mean_history = checkpoint.get('q_means', [])
        q_gap_history = checkpoint.get('q_gaps', [])
        target_q_history = checkpoint.get('target_qs', [])
        std_mean_history = checkpoint.get('std_means', [])
        
        print(f"Resuming from episode {start_episode}")
    elif args.load_path:
        print(f"Warning: {args.load_path} not found. Starting fresh.")

    obs, _ = env.reset()
    episode_reward = 0.0
    episode_length = 0
    episode_idx = start_episode
    update_step = 0

    start_time = time.time()
    last_log_step = 0

    while episode_idx < args.episodes:

        # ===== ACTION SELECTION =====
        if global_step < args.warmup_steps:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
                action, _, _ = model.get_action(obs_tensor)
                action = action.cpu().numpy()[0]

        # ===== ENV STEP =====
        next_obs, reward, done, _, info = env.step(action)

        replay_buffer.push(obs, action, reward, next_obs, float(done))

        obs = next_obs
        episode_reward += reward
        episode_length += 1
        global_step += 1

        # ===== TRAINING =====
        if (
            global_step >= args.warmup_steps
            and replay_buffer.size > args.batch_size
            and global_step % args.train_freq == 0
        ):
            for _ in range(args.gradient_steps):
                b_states, b_actions, b_rewards, b_next_states, b_dones = replay_buffer.sample(args.batch_size)

                alpha = log_alpha.exp().detach()

                # --- CRITIC ---
                with torch.no_grad():
                    next_actions, next_log_probs, _ = model.get_action(b_next_states)
                    target_q1, target_q2 = model.get_target_q(b_next_states, next_actions)
                    # Ensure shapes match [batch_size, 1]
                    next_log_probs = next_log_probs.view(-1, 1) 
                    target_q1 = target_q1.view(-1, 1)
                    target_q2 = target_q2.view(-1, 1)
                    target_v = torch.min(target_q1, target_q2) - alpha * next_log_probs
                    target_q = b_rewards + (1 - b_dones) * args.gamma * target_v

                current_q1, current_q2 = model.get_q(b_states, b_actions)
                current_q1 = current_q1.view(-1, 1)
                current_q2 = current_q2.view(-1, 1)
                critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

                critic_opt.zero_grad()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(critic_params, 0.5)
                critic_opt.step()

                # --- ACTOR ---
                curr_actions, curr_log_probs, _ = model.get_action(b_states)
                q1_pi, q2_pi = model.get_q(b_states, curr_actions)
                min_q_pi = torch.min(q1_pi, q2_pi)

                actor_loss = ((alpha * curr_log_probs) - min_q_pi).mean()

                actor_opt.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(actor_params, 0.5)
                actor_opt.step()

                # --- ALPHA ---
                alpha_loss = -(log_alpha * (curr_log_probs + target_entropy).detach()).mean()

                alpha_opt.zero_grad()
                alpha_loss.backward()
                alpha_opt.step()

                # --- DIAGNOSTIC METRIC CALCULATION ---
                with torch.no_grad():
                    # 1. Entropy: SAC tries to keep this high
                    entropy_history.append(-curr_log_probs.mean().item())

                    # 2. Q-Values: Track if they are exploding or collapsing
                    q_mean_history.append(current_q1.mean().item())
                    target_q_history.append(target_q.mean().item())

                    # 3. Q-Gap: Are the two critics diverging?
                    # Calculation: mean(abs(Q1 - Q2))
                    q_gap = (current_q1 - current_q2).abs().mean().item()
                    q_gap_history.append(q_gap)

                    # 4. Policy Std: How "brave" is the agent?
                    _, _, curr_mean = model.get_action(b_states) 
                    std_mean_history.append(curr_mean.mean().item())

                model.soft_update_targets(args.tau)

                update_step += 1

                # store metrics
                actor_loss_history.append(actor_loss.item())
                critic_loss_history.append(critic_loss.item())
                alpha_loss_history.append(alpha_loss.item())

        # ===== EPISODE END =====
        if done:
            rewards_history.append(episode_reward)

            if info.get('result') == "escaped":
                wins += 1

            episode_idx += 1

            # logging
            if episode_idx % args.log_interval == 0:
                avg_reward = np.mean(rewards_history[-args.log_interval:])
                current_alpha = log_alpha.exp().item()

                print(
                    f"SAC | Ep: {episode_idx} | "
                    f"AvgR: {avg_reward:.2f} | "
                    f"Winrate: {wins/args.log_interval:.2f} | "
                    f"Alpha: {current_alpha:.4f} | "
                    f"Steps: {global_step}"
                )

                winrates.append(wins / args.log_interval)
                winrate_epochs.append(episode_idx)
                wins = 0

            # reset
            obs, _ = env.reset()
            episode_reward = 0.0
            episode_length = 0

        # ===== SAVE =====
        if global_step % args.save_interval == 0 and global_step > args.warmup_steps:
            torch.save({
                'global_step': global_step,
                'model_state_dict': model.state_dict(),
                'actor_opt_state_dict': actor_opt.state_dict(),
                'critic_opt_state_dict': critic_opt.state_dict(),
                'alpha_opt_state_dict': alpha_opt.state_dict(),
                'log_alpha': log_alpha,
                'rewards': rewards_history,
                'winrates': winrates,
                'winrate_epochs': winrate_epochs,
                'actor_losses': actor_loss_history,
                'critic_losses': critic_loss_history,
                'alpha_losses': alpha_loss_history,
                'entropies': entropy_history,
                'q_means': q_mean_history,
                'q_gaps': q_gap_history,
                'target_qs': target_q_history,
                'std_means': std_mean_history,
            }, args.save_path)
        
    # --- FINAL SAVE ---
    print(f"Training finished. Final model saved to {args.save_path}")

if __name__ == '__main__':
    # Run with
    # python sac_cnn.py --episodes 20000 --device cuda --buffer_device cuda --save_path sac/sac_cnn_20k.pt --gradient_steps 5 --tau 0.001
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=20000)
    parser.add_argument('--warmup_steps', type=int, default=20_000, help="Random actions before training starts")
    parser.add_argument('--buffer_size', type=int, default=100_000)
    parser.add_argument('--train_freq', type=int, default=1)
    parser.add_argument('--gradient_steps', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--tau', type=float, default=0.005, help="Target network soft update rate")
    parser.add_argument('--log_interval', type=int, default=10)
    parser.add_argument('--save_interval', type=int, default=100)
    parser.add_argument('--save_path', type=str, default='sac/sac_cnn.pt')
    parser.add_argument('--load_path', type=str, default=None)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--buffer_device', type=str, default='cpu', help="Where to store the replay buffer (cpu or cuda)")
    args = parser.parse_args()

    device = args.device
    print(f"Using device: {device}")

    env = HarryPotterEnv()
    env.reset()
    obs_shape = env.observation_space.shape
    action_dim = np.prod(env.action_space.shape) # Adjust based on your env
    
    # Initialize your combined SAC model
    model = SACNet(obs_shape, action_dim).to(device)
    
    train_sac(env, model, args, device)