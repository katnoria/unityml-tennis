import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class Actor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=400, fc2_units=300, use_bn=False):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.use_bn = use_bn
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)            
        self.fc3 = nn.Linear(fc2_units, action_size)
        if self.use_bn:
            self.bn1 = nn.BatchNorm1d(state_size)
            self.bn2 = nn.BatchNorm1d(fc1_units)
            self.bn3 = nn.BatchNorm1d(fc2_units)

        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)
        self.fc1.bias.data.fill_(0.1)
        self.fc2.bias.data.fill_(0.1)
        self.fc3.bias.data.fill_(0.1)

    def forward(self, state):
        """Build an actor (policy) network that maps states -> actions."""
        if self.use_bn:
            x = self.fc1(self.bn1(state))
        else:            
            x = self.fc1(state)
            
        x = F.relu(x)
        if self.use_bn:
            x = self.bn2(x)
        x = self.fc2(x)
        x = F.relu(x)
        if self.use_bn:
            x = self.bn3(x)
        return F.tanh(self.fc3(x))


class Critic(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=400, fc2_units=300, use_bn=False):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in the first hidden layer
            fc2_units (int): Number of nodes in the second hidden layer
        """
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.use_bn = use_bn
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units+action_size, fc2_units)
        self.fc3 = nn.Linear(fc2_units, 1)
        if self.use_bn:
            self.bn1 = nn.BatchNorm1d(fc1_units)
            self.bn2 = nn.BatchNorm1d(fc2_units)
        
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)
        self.fc1.bias.data.fill_(0.1)
        self.fc2.bias.data.fill_(0.1)
        self.fc3.bias.data.fill_(0.1)

    def forward(self, state, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        x  = self.fc1(state)
        xs = F.relu(x)
        if self.use_bn:
            x = self.bn1(x)
        x = torch.cat((xs, action), dim=1)
        x = self.fc2(x)
        x = F.relu(x)
        if self.use_bn:
            x = self.bn2(x)
        return self.fc3(x)

class CentralCritic(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size, seed, num_agents=1, fc1_units=64, fc2_units=32, use_bn=False):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in the first hidden layer
            fc2_units (int): Number of nodes in the second hidden layer
        """
        super(CentralCritic, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.use_bn = use_bn
        import pdb; pdb.set_trace()
        self.fc1 = nn.Linear(state_size * num_agents, fc1_units * num_agents)
        self.fc2 = nn.Linear(fc1_units * num_agents + action_size * num_agents, fc2_units * num_agents)
        self.fc3 = nn.Linear(fc2_units * num_agents, 1 * num_agents)
        if self.use_bn:
            self.bn1 = nn.BatchNorm1d(fc1_units * num_agents)
            self.bn2 = nn.BatchNorm1d(fc2_units * num_agents)
        
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)
        self.fc1.bias.data.fill_(0.1)
        self.fc2.bias.data.fill_(0.1)
        self.fc3.bias.data.fill_(0.1)

    def forward(self, state, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        x  = self.fc1(state)
        xs = F.relu(x)
        if self.use_bn:
            x = self.bn1(x)
        x = torch.cat((xs, action), dim=1)
        x = self.fc2(x)
        x = F.relu(x)
        if self.use_bn:
            x = self.bn2(x)
        return self.fc3(x)        