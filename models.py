import torch
import torch.nn as nn
from torch.distributions import Normal

@torch.compile
class ActorCriticNet(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super(ActorCriticNet, self).__init__()

        self.hidden_dim = 128
        
        # Shared feature extractor
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, self.hidden_dim),
            nn.Tanh(), # Tanh is often more stable than LeakyReLU in continuous PPO
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Tanh()
        )
        
        # Actor head (Mean and log standard deviation for continuous actions)
        self.actor_mean = nn.Linear(self.hidden_dim, act_dim)
        self.actor_logstd = nn.Parameter(torch.zeros(1, act_dim))
        
        # Critic head (Value estimation)
        self.critic = nn.Linear(self.hidden_dim, 1)

    def forward(self, obs):
        features = self.shared(obs)
        value = self.critic(features)
        
        action_mean = torch.tanh(self.actor_mean(features)) # bounded to [-1, 1]
        action_std = torch.exp(self.actor_logstd).expand_as(action_mean)
        
        return action_mean, action_std, value

    def get_action(self, obs):
        mean, std, value = self.forward(obs)
        dist = Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action, log_prob, value
    
    
@torch.compile
class ActorNet(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super(ActorNet, self).__init__()

        self.hidden_dim = 128

        self.net = nn.Sequential(
            nn.Linear(obs_dim, self.hidden_dim),
            nn.Tanh(), # Tanh is often more stable than LeakyReLU in continuous PPO
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Tanh()
        )

        self.mean = nn.Linear(self.hidden_dim, act_dim)
        self.logstd = nn.Parameter(torch.zeros(1, act_dim))

    def forward(self, obs):
        features = self.net(obs)
        action_mean = torch.tanh(self.mean(features))
        action_std = torch.exp(self.logstd).expand_as(action_mean)
        return action_mean, action_std

@torch.compile
class CriticNet(nn.Module):
    def __init__(self, obs_dim):
        super(CriticNet, self).__init__()

        self.hidden_dim = 64

        self.net = nn.Sequential(
            nn.Linear(obs_dim, self.hidden_dim),
            nn.Tanh(), # Tanh is often more stable than LeakyReLU in continuous PPO
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Tanh()
        )
        self.value = nn.Linear(self.hidden_dim, 1)

    def forward(self, obs):
        x = self.net(obs)
        value = self.value(x)
        return value