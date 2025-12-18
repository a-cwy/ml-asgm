import os
import torch as T 
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical
import numpy as np

class CriticNetwork(nn.Module):
    def __init__(self, beta, input_dims, n_actions, fc1_dims=256, fc2_dims=256, fc3_dims=256, fc4_dims=256,
            name='critic', chkpt_dir='tmp/sac'):
        super(CriticNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.fc3_dims = fc3_dims
        self.fc4_dims = fc4_dims
        self.n_actions = n_actions
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name + '_sac.pth')

        # critic takes state and action (for discrete action we will pass one-hot)
        self.fc1 = nn.Linear(self.input_dims[0] + n_actions, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)        
        self.fc3 = nn.Linear(self.fc2_dims, self.fc3_dims)
        self.fc4 = nn.Linear(self.fc3_dims, self.fc4_dims)
        self.q = nn.Linear(self.fc4_dims, 1)

        self.optimizer = optim.AdamW(self.parameters(), lr=beta, weight_decay=1e-5)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)
    
    def forward(self, state, action):
        action_value = self.fc1(T.cat([state, action], dim=1))
        action_value = F.relu(action_value)
        action_value = self.fc2(action_value)
        action_value = F.relu(action_value)
        action_value = self.fc3(action_value)
        action_value = F.relu(action_value)
        action_value = self.fc4(action_value)
        action_value = F.relu(action_value)
        q = self.q(action_value)
        return q

    def save_checkpoint(self):
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))


class ValueNetwork(nn.Module):
    def __init__(self, beta, input_dims, fc1_dims=256, fc2_dims=256, fc3_dims=256, fc4_dims=256, name='value', chkpt_dir='tmp/sac'):
        super(ValueNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.fc3_dims = fc3_dims
        self.fc4_dims = fc4_dims
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name + '_sac.pth')

        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, fc2_dims)
        self.fc3 = nn.Linear(self.fc2_dims, fc3_dims)
        self.fc4 = nn.Linear(self.fc3_dims, fc4_dims)
        self.v = nn.Linear(self.fc4_dims, 1)

        self.optimizer = optim.AdamW(self.parameters(), lr=beta, weight_decay=1e-5)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        state_value = self.fc1(state)
        state_value = F.relu(state_value)
        state_value = self.fc2(state_value)
        state_value = F.relu(state_value)
        state_value = self.fc3(state_value)
        state_value = F.relu(state_value)
        state_value = self.fc4(state_value)
        state_value = F.relu(state_value)
        v = self.v(state_value)
        return v

    def save_checkpoint(self):
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))


class ActorNetwork(nn.Module):
    def __init__(self, alpha, input_dims, max_action, fc1_dims=256, fc2_dims=256, fc3_dims=256, fc4_dims=256, n_actions=2, name='actor', chkpt_dir='tmp/sac'):
        super(ActorNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.fc3_dims = fc3_dims
        self.fc4_dims = fc4_dims
        self.n_actions = n_actions
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name + '_sac.pth')
        self.max_action = max_action  # if None -> treat as discrete
        self.reparam_noise = 1e-6

        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.fc3 = nn.Linear(self.fc2_dims, self.fc3_dims)
        self.fc4 = nn.Linear(self.fc3_dims, self.fc4_dims)

        # outputs
        # For continuous: mu and sigma
        # For discrete: logits (use mu as logits)
        self.mu = nn.Linear(self.fc4_dims, self.n_actions)
        self.sigma = nn.Linear(self.fc4_dims, self.n_actions)

        self.optimizer = optim.AdamW(self.parameters(), lr=alpha, weight_decay=1e-5)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        prob = self.fc1(state)
        prob = F.relu(prob)
        prob = self.fc2(prob)
        prob = F.relu(prob)
        prob = self.fc3(prob)
        prob = F.relu(prob)
        prob = self.fc4(prob)
        prob = F.relu(prob)
        mu = self.mu(prob)
        sigma = self.sigma(prob)
        sigma = T.clamp(sigma, min=self.reparam_noise, max=1)
        return mu, sigma

    def sample_normal(self, state, reparameterize=True):
        """
        Returns a triple (action_for_critic, log_prob, discrete_index_or_None)

        - For discrete: action_for_critic = one-hot tensor, log_prob shape (batch,1), index tensor shape (batch,)
        - For continuous: action_for_critic = scaled action tensor, log_prob shape (batch,1), index = None
        """
        mu, sigma = self.forward(state)

        # If max_action is None treat as discrete action space (Categorical)
        if self.max_action is None:
            # mu are logits for discrete case
            logits = mu
            probs = F.softmax(logits, dim=-1)
            cat = Categorical(probs)
            # sampling
            indices = cat.sample()  # shape: (batch,)
            # one-hot encode chosen actions for critic input
            action_one_hot = F.one_hot(indices, num_classes=self.n_actions).float().to(self.device)
            # log_prob of chosen action (shape -> (batch,1))
            log_prob = cat.log_prob(indices).unsqueeze(1)
            return action_one_hot, log_prob, indices

        # Continuous action (original behavior)
        probabilities = Normal(mu, sigma)
        if reparameterize:
            actions = probabilities.rsample()
        else:
            actions = probabilities.sample()

        # tanh transform and correct log-prob with Jacobian of tanh
        tanh_actions = T.tanh(actions)
        max_action_tensor = T.tensor(self.max_action).to(self.device)
        action = tanh_actions * max_action_tensor
        log_probs = probabilities.log_prob(actions)
        # use tanh_actions (pre-scale) for jacobian correction to keep values in (-1,1)
        log_probs -= T.log(1 - tanh_actions.pow(2) + self.reparam_noise)
        log_probs = log_probs.sum(1, keepdim=True)
        return action, log_probs, None

    def save_checkpoint(self):
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))