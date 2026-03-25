import argparse
import os
import random
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from env import HarryPotterEnv


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class TransitionBatch:
    obs: torch.Tensor
    act: torch.Tensor
    rew: torch.Tensor
    next_obs: torch.Tensor
    done: torch.Tensor


class ReplayBuffer:
    def __init__(self, obs_dim: int, act_dim: int, capacity: int, device: str):
        self.capacity = int(capacity)
        self.device = device
        self.obs = np.zeros((self.capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((self.capacity, obs_dim), dtype=np.float32)
        self.act = np.zeros((self.capacity, act_dim), dtype=np.float32)
        self.rew = np.zeros((self.capacity, 1), dtype=np.float32)
        self.done = np.zeros((self.capacity, 1), dtype=np.float32)
        self.ptr = 0
        self.size = 0

    def add(self, obs, act, rew, next_obs, done):
        self.obs[self.ptr] = obs
        self.act[self.ptr] = act
        self.rew[self.ptr] = rew
        self.next_obs[self.ptr] = next_obs
        self.done[self.ptr] = float(done)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> TransitionBatch:
        idx = np.random.randint(0, self.size, size=batch_size)
        obs = torch.as_tensor(self.obs[idx], device=self.device)
        act = torch.as_tensor(self.act[idx], device=self.device)
        rew = torch.as_tensor(self.rew[idx], device=self.device)
        next_obs = torch.as_tensor(self.next_obs[idx], device=self.device)
        done = torch.as_tensor(self.done[idx], device=self.device)
        return TransitionBatch(obs=obs, act=act, rew=rew, next_obs=next_obs, done=done)


class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


LOG_STD_MIN = -20
LOG_STD_MAX = 2


class SquashedGaussianActor(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.backbone = MLP(obs_dim, hidden_dim, hidden_dim=hidden_dim)
        self.mu = nn.Linear(hidden_dim, act_dim)
        self.log_std = nn.Linear(hidden_dim, act_dim)

    def forward(self, obs):
        h = self.backbone(obs)
        mu = self.mu(h)
        log_std = self.log_std(h)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)
        return mu, std

    def sample(self, obs):
        mu, std = self.forward(obs)
        dist = torch.distributions.Normal(mu, std)
        eps = dist.rsample()
        action = torch.tanh(eps)
        log_prob = dist.log_prob(eps).sum(dim=-1, keepdim=True)
        # Tanh correction (from SAC paper / standard impls)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6).sum(dim=-1, keepdim=True)
        return action, log_prob

    @torch.no_grad()
    def act(self, obs, deterministic: bool = False):
        mu, std = self.forward(obs)
        if deterministic:
            a = torch.tanh(mu)
            return a
        dist = torch.distributions.Normal(mu, std)
        eps = dist.sample()
        return torch.tanh(eps)


class QNetwork(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.q = MLP(obs_dim + act_dim, 1, hidden_dim=hidden_dim)

    def forward(self, obs, act):
        x = torch.cat([obs, act], dim=-1)
        return self.q(x)


def soft_update(source: nn.Module, target: nn.Module, tau: float):
    with torch.no_grad():
        for p, tp in zip(source.parameters(), target.parameters()):
            tp.data.mul_(1.0 - tau).add_(p.data, alpha=tau)


def ensure_parent_dir(path: str):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def train_sac(args):
    device = args.device
    print(f"Using device: {device}")

    set_seed(args.seed)

    env = HarryPotterEnv()
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    actor = SquashedGaussianActor(obs_dim, act_dim, hidden_dim=args.hidden_dim).to(device)
    q1 = QNetwork(obs_dim, act_dim, hidden_dim=args.hidden_dim).to(device)
    q2 = QNetwork(obs_dim, act_dim, hidden_dim=args.hidden_dim).to(device)
    q1_t = QNetwork(obs_dim, act_dim, hidden_dim=args.hidden_dim).to(device)
    q2_t = QNetwork(obs_dim, act_dim, hidden_dim=args.hidden_dim).to(device)
    q1_t.load_state_dict(q1.state_dict())
    q2_t.load_state_dict(q2.state_dict())

    actor_opt = optim.Adam(actor.parameters(), lr=args.lr)
    q_opt = optim.Adam(list(q1.parameters()) + list(q2.parameters()), lr=args.lr)

    # Automatic entropy temperature tuning
    target_entropy = -float(act_dim) if args.target_entropy is None else float(args.target_entropy)
    log_alpha = torch.tensor(np.log(args.init_alpha), device=device, requires_grad=True)
    alpha_opt = optim.Adam([log_alpha], lr=args.lr)

    rb = ReplayBuffer(obs_dim, act_dim, capacity=args.buffer_size, device=device)

    rewards_hist = []
    winrates, winrate_epochs = [], []
    actor_loss_hist, q_loss_hist, alpha_hist = [], [], []

    start_episode = 0
    total_steps = 0

    if args.load_path and os.path.isfile(args.load_path):
        print(f"Loading checkpoint from {args.load_path}...")
        ckpt = torch.load(args.load_path, map_location=device, weights_only=False)
        actor.load_state_dict(ckpt["actor_state_dict"])
        q1.load_state_dict(ckpt["q1_state_dict"])
        q2.load_state_dict(ckpt["q2_state_dict"])
        q1_t.load_state_dict(ckpt.get("q1_target_state_dict", ckpt["q1_state_dict"]))
        q2_t.load_state_dict(ckpt.get("q2_target_state_dict", ckpt["q2_state_dict"]))
        actor_opt.load_state_dict(ckpt["actor_opt_state_dict"])
        q_opt.load_state_dict(ckpt["q_opt_state_dict"])
        log_alpha = torch.tensor(ckpt.get("log_alpha", float(log_alpha.item())), device=device, requires_grad=True)
        alpha_opt = optim.Adam([log_alpha], lr=args.lr)
        if "alpha_opt_state_dict" in ckpt:
            alpha_opt.load_state_dict(ckpt["alpha_opt_state_dict"])

        rewards_hist = ckpt.get("rewards", [])
        winrates = ckpt.get("winrates", [])
        winrate_epochs = ckpt.get("winrate_epochs", [])
        actor_loss_hist = ckpt.get("actor_losses", [])
        q_loss_hist = ckpt.get("q_losses", [])
        alpha_hist = ckpt.get("alphas", [])
        start_episode = len(rewards_hist)
        total_steps = ckpt.get("total_steps", 0)
        print(f"Resuming from episode {start_episode}, total_steps={total_steps}")
    elif args.load_path:
        print(f"Warning: Checkpoint file {args.load_path} not found. Starting from scratch.")

    wins = 0
    t0 = time.time()

    for ep in range(start_episode, start_episode + args.episodes):
        obs, _ = env.reset()
        done = False
        ep_reward = 0.0

        while not done:
            total_steps += 1

            if total_steps < args.start_steps:
                act = env.action_space.sample()
            else:
                obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                act_t = actor.act(obs_t, deterministic=False).squeeze(0).cpu().numpy()
                act = np.clip(act_t, -1.0, 1.0)

            next_obs, rew, done, _, info = env.step(act)
            ep_reward += float(rew)

            # Optional reward scaling to keep critic targets reasonable
            rew_scaled = float(rew) * args.reward_scale
            rb.add(obs, act, rew_scaled, next_obs, done)
            obs = next_obs

            if rb.size >= args.batch_size and total_steps % args.update_every == 0:
                for _ in range(args.updates_per_step):
                    batch = rb.sample(args.batch_size)
                    alpha = log_alpha.exp().detach()

                    with torch.no_grad():
                        next_a, next_logp = actor.sample(batch.next_obs)
                        q1_next = q1_t(batch.next_obs, next_a)
                        q2_next = q2_t(batch.next_obs, next_a)
                        q_next = torch.min(q1_next, q2_next) - alpha * next_logp
                        target_q = batch.rew + (1.0 - batch.done) * args.gamma * q_next

                    q1_pred = q1(batch.obs, batch.act)
                    q2_pred = q2(batch.obs, batch.act)
                    q_loss = F.mse_loss(q1_pred, target_q) + F.mse_loss(q2_pred, target_q)

                    q_opt.zero_grad()
                    q_loss.backward()
                    if args.grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(list(q1.parameters()) + list(q2.parameters()), args.grad_clip)
                    q_opt.step()

                    # Actor update
                    a, logp = actor.sample(batch.obs)
                    q1_pi = q1(batch.obs, a)
                    q2_pi = q2(batch.obs, a)
                    q_pi = torch.min(q1_pi, q2_pi)
                    actor_loss = (alpha * logp - q_pi).mean()

                    actor_opt.zero_grad()
                    actor_loss.backward()
                    if args.grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(actor.parameters(), args.grad_clip)
                    actor_opt.step()

                    # Alpha / temperature update
                    alpha_loss = -(log_alpha * (logp + target_entropy).detach()).mean()
                    alpha_opt.zero_grad()
                    alpha_loss.backward()
                    alpha_opt.step()

                    soft_update(q1, q1_t, args.tau)
                    soft_update(q2, q2_t, args.tau)

                    actor_loss_hist.append(float(actor_loss.item()))
                    q_loss_hist.append(float(q_loss.item()))
                    alpha_hist.append(float(log_alpha.exp().item()))

        rewards_hist.append(ep_reward)
        if info.get("result") == "escaped":
            wins += 1

        if ep % args.log_interval == 0:
            winrate = wins / args.log_interval
            print(
                f"SAC | Ep: {ep} | Reward: {ep_reward:.2f} | Winrate {winrate:.2f} | "
                f"Steps: {total_steps} | Alpha: {log_alpha.exp().item():.3f} | "
                f"Time: {(time.time() - t0)/args.log_interval:.3f}s | Result: {info.get('result', 'N/A')}"
            )
            winrates.append(winrate)
            winrate_epochs.append(ep)
            wins = 0
            t0 = time.time()

        if args.save_path and ep % args.save_interval == 0 and ep > 0:
            ensure_parent_dir(args.save_path)
            torch.save(
                {
                    "actor_state_dict": actor.state_dict(),
                    "q1_state_dict": q1.state_dict(),
                    "q2_state_dict": q2.state_dict(),
                    "q1_target_state_dict": q1_t.state_dict(),
                    "q2_target_state_dict": q2_t.state_dict(),
                    "actor_opt_state_dict": actor_opt.state_dict(),
                    "q_opt_state_dict": q_opt.state_dict(),
                    "alpha_opt_state_dict": alpha_opt.state_dict(),
                    "log_alpha": float(log_alpha.item()),
                    "total_steps": total_steps,
                    "rewards": rewards_hist,
                    "winrates": winrates,
                    "winrate_epochs": winrate_epochs,
                    "actor_losses": actor_loss_hist,
                    "q_losses": q_loss_hist,
                    "alphas": alpha_hist,
                    "args": vars(args),
                },
                args.save_path,
            )

    if args.save_path:
        ensure_parent_dir(args.save_path)
        torch.save(
            {
                "actor_state_dict": actor.state_dict(),
                "q1_state_dict": q1.state_dict(),
                "q2_state_dict": q2.state_dict(),
                "q1_target_state_dict": q1_t.state_dict(),
                "q2_target_state_dict": q2_t.state_dict(),
                "actor_opt_state_dict": actor_opt.state_dict(),
                "q_opt_state_dict": q_opt.state_dict(),
                "alpha_opt_state_dict": alpha_opt.state_dict(),
                "log_alpha": float(log_alpha.item()),
                "total_steps": total_steps,
                "rewards": rewards_hist,
                "winrates": winrates,
                "winrate_epochs": winrate_epochs,
                "actor_losses": actor_loss_hist,
                "q_losses": q_loss_hist,
                "alphas": alpha_hist,
                "args": vars(args),
            },
            args.save_path,
        )
        print(f"Final SAC checkpoint saved to {args.save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Soft Actor-Critic training for HarryPotterEnv")
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--hidden_dim", type=int, default=256)

    parser.add_argument("--buffer_size", type=int, default=200000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--start_steps", type=int, default=2000, help="Pure random actions for exploration")
    parser.add_argument("--update_every", type=int, default=1)
    parser.add_argument("--updates_per_step", type=int, default=1)

    parser.add_argument("--reward_scale", type=float, default=0.001, help="Multiply env reward before storing in replay")

    parser.add_argument("--init_alpha", type=float, default=0.2)
    parser.add_argument("--target_entropy", type=float, default=None, help="Default: -act_dim")

    parser.add_argument("--grad_clip", type=float, default=1.0)

    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--save_interval", type=int, default=500)
    parser.add_argument("--save_path", type=str, default="sac/sac_ckpt.pt")
    parser.add_argument("--load_path", type=str, default=None)

    args = parser.parse_args()
    train_sac(args)
