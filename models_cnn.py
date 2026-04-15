import torch
import torch.nn as nn
from torch.distributions import Normal
import torch.nn.functional as F

from env_cnn import WIN_REWARD

@torch.compile
class ActorCriticNet(nn.Module):
    def __init__(self, obs_shape, act_dim):
        super(ActorCriticNet, self).__init__()
        
        channels = obs_shape[0] # Should be 4
        
        # 1. CNN Feature Extractor
        self.cnn = nn.Sequential(
            # First Block
            nn.Conv2d(channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # 64x64 -> 32x32
            
            # Second Block
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # 32x32 -> 16x16
            
            # Third Block
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # 16x16 -> 8x8
            
            nn.Flatten()
        )
        
        # The output of the CNN is 64 * 4 * 4 = 1024 features
        # cnn_out_dim = 64 * 4 * 4
        cnn_out_dim = 64 * 8 * 8
        self.hidden_dim = 128
        
        # 2. Shared MLP on top of the CNN
        self.shared_mlp = nn.Sequential(
            nn.Linear(cnn_out_dim, self.hidden_dim),
            nn.Tanh(), # Tanh for continuous PPO stability
            # nn.Linear(self.hidden_dim, self.hidden_dim),
            # nn.Tanh(),
            # nn.Linear(self.hidden_dim, self.hidden_dim),
            # nn.Tanh(),
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
        
        value = self.critic(features) * -WIN_REWARD
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
            # First Block
            nn.Conv2d(channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # 64x64 -> 32x32
            
            # Second Block
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # 32x32 -> 16x16
            
            # Third Block
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # 16x16 -> 8x8
            
            nn.Flatten()
        )
        
        # The output of the CNN is 64 * 4 * 4 = 1024 features
        # cnn_out_dim = 64 * 4 * 4
        cnn_out_dim = 64 * 8 * 8
        self.hidden_dim = 128

        self.net = nn.Sequential(
            nn.Linear(cnn_out_dim, self.hidden_dim),
            nn.Tanh(), # Tanh for continuous PPO stability
            # nn.Linear(self.hidden_dim, self.hidden_dim),
            # nn.Tanh(),
            # nn.Linear(self.hidden_dim, self.hidden_dim),
            # nn.Tanh(),
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
            # First Block
            nn.Conv2d(channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # 64x64 -> 32x32
            
            # Second Block
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # 32x32 -> 16x16
            
            # Third Block
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # 16x16 -> 8x8
            
            nn.Flatten()
        )
        
        # The output of the CNN is 64 * 4 * 4 = 1024 features
        # cnn_out_dim = 64 * 4 * 4
        cnn_out_dim = 64 * 8 * 8
        self.hidden_dim = 64

        self.net = nn.Sequential(
            nn.Linear(cnn_out_dim, self.hidden_dim),
            nn.Tanh(), # Tanh for continuous PPO stability
            # nn.Linear(self.hidden_dim, self.hidden_dim),
            # nn.Tanh(),
            # nn.Linear(self.hidden_dim, self.hidden_dim),
            # nn.Tanh(),
        )
        self.value = nn.Linear(self.hidden_dim, 1)

    def forward(self, obs):
        if obs.dim() == 3:
            obs = obs.unsqueeze(0)
            
        cnn_features = self.cnn(obs)
        x = self.net(cnn_features)
        value = self.value(x) * -WIN_REWARD
        return value

@torch.compile
class SharedACNet(nn.Module):
    def __init__(self, obs_shape, act_dim):
        super(SharedACNet, self).__init__()
        channels = obs_shape[0]
        
        # 1. THE SHARED EYES (CNN)
        self.cnn = nn.Sequential(
            # First Block
            nn.Conv2d(channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # 64x64 -> 32x32
            
            # Second Block
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # 32x32 -> 16x16
            
            # Third Block
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # 16x16 -> 8x8
            
            nn.Flatten()
        )
        
        # Calculate this based on your final pooling size
        cnn_out_dim = 64 * 8 * 8 
        self.hidden_dim = 128
        
        # 2. THE ACTOR HEAD
        self.actor_mlp = nn.Sequential(
            nn.Linear(cnn_out_dim, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, act_dim),
        )
        self.log_std = nn.Parameter(torch.zeros(1, act_dim))
        
        # 3. THE CRITIC HEAD
        self.critic_mlp = nn.Sequential(
            nn.Linear(cnn_out_dim, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, 1),
        )

    def forward(self, x):
        features = self.cnn(x)
        
        # Actor branch
        mu = torch.tanh(self.actor_mlp(features))
        std = torch.exp(self.log_std).expand_as(mu)
        
        # Critic branch
        value = self.critic_mlp(features)
        
        return mu, std, value




@torch.compile
class ActorMLP(nn.Module):
    def __init__(self, input_dim, act_dim, hidden_dim):
        super(ActorMLP, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mu = nn.Linear(hidden_dim, act_dim)
        self.log_std = nn.Linear(hidden_dim, act_dim)

    def forward(self, x):
        x = self.fc(x)
        mu = self.mu(x)
        log_std = torch.clamp(self.log_std(x), -20, 2)
        return mu, log_std

@torch.compile
class CriticMLP(nn.Module):
    def __init__(self, input_dim, act_dim, hidden_dim):
        super(CriticMLP, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim + act_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, features, action):
        x = torch.cat([features, action], dim=1)
        return self.fc(x)

@torch.compile
class SACNet(nn.Module):
    def __init__(self, obs_shape, act_dim, hidden_dim=256):
        super(SACNet, self).__init__()
        channels = obs_shape[0]
        
        # 1. Shared CNN Feature Extractor
        self.cnn = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten()
        )
        
        # Assuming 64x64 input -> after 3 MaxPools (8x8) -> 64 * 8 * 8 = 4096
        cnn_out_dim = 64 * 8 * 8 

        # 2. Heads as distinct fields
        self.actor = ActorMLP(cnn_out_dim, act_dim, hidden_dim)
        self.critic1 = CriticMLP(cnn_out_dim, act_dim, hidden_dim)
        self.critic2 = CriticMLP(cnn_out_dim, act_dim, hidden_dim)
        
        # 3. Target Networks
        self.target_critic1 = CriticMLP(cnn_out_dim, act_dim, hidden_dim)
        self.target_critic2 = CriticMLP(cnn_out_dim, act_dim, hidden_dim)
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())

    def get_action(self, obs, epsilon=1e-6):
        features = self.cnn(obs)
        mu, log_std = self.actor(features)
        std = log_std.exp()

        dist = Normal(mu, std)
        z = dist.rsample() # Reparameterization trick
        action = torch.tanh(z)

        # Correction for Tanh squashing
        log_prob = dist.log_prob(z) - torch.log(1 - action.pow(2) + epsilon)
        return action, log_prob.sum(-1, keepdim=True), torch.tanh(mu)

    def get_q(self, obs, action):
        features = self.cnn(obs)
        return self.critic1(features, action), self.critic2(features, action)

    def get_target_q(self, obs, action):
        with torch.no_grad():
            features = self.cnn(obs)
            return self.target_critic1(features, action), self.target_critic2(features, action)

    def soft_update_targets(self, tau):
        for param, target_param in zip(self.critic1.parameters(), self.target_critic1.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)
        for param, target_param in zip(self.critic2.parameters(), self.target_critic2.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)