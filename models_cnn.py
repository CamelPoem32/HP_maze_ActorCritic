import torch
import torch.nn as nn
from torch.distributions import Normal

@torch.compile
class ActorCriticNet(nn.Module):
    def __init__(self, obs_shape, act_dim):
        super(ActorCriticNet, self).__init__()
        
        channels = obs_shape[0] # Should be 4
        
        # 1. CNN Feature Extractor
        self.cnn = nn.Sequential(
            # Input: (Batch, 4, 64, 64)
            nn.Conv2d(in_channels=channels, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            # Output: (Batch, 32, 15, 15)
            
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            # Output: (Batch, 64, 6, 6)
            
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            # Output: (Batch, 64, 4, 4)
            
            nn.Flatten()
        )
        
        # The output of the CNN is 64 * 4 * 4 = 1024 features
        cnn_out_dim = 64 * 4 * 4
        self.hidden_dim = 128
        
        # 2. Shared MLP on top of the CNN
        self.shared_mlp = nn.Sequential(
            nn.Linear(cnn_out_dim, self.hidden_dim),
            nn.Tanh(), # Tanh for continuous PPO stability
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Tanh(),
        )
        
        # Actor head
        self.actor_mean = nn.Linear(self.hidden_dim, act_dim)
        self.actor_logstd = nn.Parameter(torch.zeros(1, act_dim))
        
        # Critic head
        self.critic = nn.Linear(self.hidden_dim, 1)

    def forward(self, obs):
        # Ensure obs is a 4D tensor (Batch, Channels, Height, Width)
        if obs.dim() == 3:
            obs = obs.unsqueeze(0)
            
        cnn_features = self.cnn(obs)
        features = self.shared_mlp(cnn_features)
        
        value = self.critic(features)
        action_mean = torch.tanh(self.actor_mean(features))
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
    def __init__(self, obs_shape, act_dim):
        super(ActorNet, self).__init__()
        
        channels = obs_shape[0] # Should be 4
        
        # 1. CNN Feature Extractor
        self.cnn = nn.Sequential(
            # Input: (Batch, 4, 64, 64)
            nn.Conv2d(in_channels=channels, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            # Output: (Batch, 32, 15, 15)
            
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            # Output: (Batch, 64, 6, 6)
            
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            # Output: (Batch, 64, 4, 4)
            
            nn.Flatten()
        )
        
        # The output of the CNN is 64 * 4 * 4 = 1024 features
        cnn_out_dim = 64 * 4 * 4
        self.hidden_dim = 128

        self.net = nn.Sequential(
            nn.Linear(cnn_out_dim, self.hidden_dim),
            nn.Tanh(), # Tanh for continuous PPO stability
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Tanh(),
        )

        self.mean = nn.Linear(self.hidden_dim, act_dim)
        self.logstd = nn.Parameter(torch.zeros(1, act_dim))

    def forward(self, obs):
        if obs.dim() == 3:
            obs = obs.unsqueeze(0)
            
        cnn_features = self.cnn(obs)
        features = self.net(cnn_features)
        action_mean = torch.tanh(self.mean(features))
        action_std = torch.exp(self.logstd).expand_as(action_mean)
        return action_mean, action_std

@torch.compile
class CriticNet(nn.Module):
    def __init__(self, obs_shape):
        super(CriticNet, self).__init__()
        
        channels = obs_shape[0] # Should be 4
        
        # 1. CNN Feature Extractor
        self.cnn = nn.Sequential(
            # Input: (Batch, 4, 64, 64)
            nn.Conv2d(in_channels=channels, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            # Output: (Batch, 32, 15, 15)
            
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            # Output: (Batch, 64, 6, 6)
            
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            # Output: (Batch, 64, 4, 4)
            
            nn.Flatten()
        )
        
        # The output of the CNN is 64 * 4 * 4 = 1024 features
        cnn_out_dim = 64 * 4 * 4
        self.hidden_dim = 64

        self.net = nn.Sequential(
            nn.Linear(cnn_out_dim, self.hidden_dim),
            nn.Tanh(), # Tanh for continuous PPO stability
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Tanh(),
        )
        self.value = nn.Linear(self.hidden_dim, 1)

    def forward(self, obs):
        if obs.dim() == 3:
            obs = obs.unsqueeze(0)
            
        cnn_features = self.cnn(obs)
        x = self.net(cnn_features)
        value = self.value(x)
        return value