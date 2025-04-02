import ptan
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
torch.manual_seed(42)

import torch.optim as optim
from collections import OrderedDict


class Policy(nn.Module):
    """
    The policy that will be trained to learn to predict the training ratio.
    This can be replaced with any continous action model e.g., DQN, DDPG, PPO
    """
    def __init__(self, obs_size, hidden_size, n_actions):
        super(Policy, self).__init__()
        self.policy = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions)
        )
    
    def forward(self, x):
        return self.agent(x)
        
HID_SIZE = 128
class ModelA2C(nn.Module):
    def __init__(self, obs_size, act_size):
        super(ModelA2C, self).__init__()

        self.base = nn.Sequential(
            nn.Linear(obs_size, HID_SIZE),
            nn.ReLU(),
        )
        self.mu = nn.Sequential(
            nn.Linear(HID_SIZE, act_size),
            nn.Softplus(),
        )
        self.var = nn.Sequential(
            nn.Linear(HID_SIZE, act_size),
            nn.Softplus(),
        )
        self.value = nn.Linear(HID_SIZE, 1)

    def forward(self, x):
        base_out = self.base(x)
        return self.mu(base_out), self.var(base_out), self.value(base_out)

class AgentA2C(ptan.agent.BaseAgent):
    def __init__(self, net, device="cpu"):
        self.net = net
        self.device = device

    def __call__(self, states):
        #if state tensors are on gpu then transfer to cpu for numpy conversion
        if states.device.type == 'mps':
            states = states.cpu()
        states_v = ptan.agent.float32_preprocessor(states)
        states_v = states_v.to(self.device)

        mu_v, var_v, _ = self.net(states_v)
        mu = mu_v.data.cpu().numpy()
        sigma = torch.sqrt(var_v).data.cpu().numpy()
        action = [np.random.normal(i, j) for i, j in zip(mu, sigma)]
        action = torch.clip(torch.tensor(action, dtype=torch.float32), 0.1, 0.8)
        mu, var_v = torch.tensor(mu_v, dtype=torch.float32), torch.tensor(var_v, dtype=torch.float32)
        action = action.to(self.device)
        return mu, var_v, action