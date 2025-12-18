import os
import numpy as np
import torch as T
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical

class ActorNetwork(nn.Module):
    def __init__(self, alpha, input_dims, max_action, fc1_dims=256, fc2_dims=256, fc3_dims=256, fc4_dims=256, n_actions=4, name='actor', chkpt_dir='dash/actor_sac.pth'):
        super(ActorNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.fc3_dims = fc3_dims
        self.fc4_dims = fc4_dims
        self.n_actions = n_actions
        self.name = name
        self.checkpoint_file = chkpt_dir
        self.max_action = max_action  # if None -> treat as discrete
        self.reparam_noise = 1e-6

        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.fc3 = nn.Linear(self.fc2_dims, self.fc3_dims)
        self.fc4 = nn.Linear(self.fc3_dims, self.fc4_dims)

        # outputs
        # For continuous: mu and sigma
        # For discrete: logits (use mu as logits)
        self.mu = nn.Linear(self.fc2_dims, self.n_actions)
        self.sigma = nn.Linear(self.fc2_dims, self.n_actions)

        self.optimizer = optim.AdamW(self.parameters(), lr=alpha, weight_decay=1e-5)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        prob = self.fc1(state)
        prob = F.relu(prob)
        prob = self.fc2(prob)
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

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))

class SACAgent():
    def __init__(
            self, 
            alpha = 0.0003, 
            input_dims = [6],
            env = None,
        ):

        self.actor = ActorNetwork(
            alpha, 
            input_dims, 
            max_action = None, 
            n_actions = 4, 
            name='actor'
        )
        
        self.actor.load_checkpoint()

    def flatten_observation(self, obs):
        """Convert gymnasium (Dict/Tuple) observation to 1D numpy array."""
        if isinstance(obs, dict):
            vec = np.array(list(obs.values()), dtype=np.float32).flatten()
        else:
            vec = np.array(obs, dtype=np.float32).flatten()
        return vec


    def choose_action(self, observation, deterministic=False):
        """
        Returns an action usable by env.step():
        - For discrete env: returns scalar int (sampled category if deterministic=False,
          otherwise argmax index)
        - For continuous env: returns numpy array (sampled if deterministic=False, otherwise mean action)
        """
        state = T.Tensor([observation]).to(self.actor.device)

        # For discrete case, actor.sample_normal returns (one_hot, log_prob, indices)
        actions_for_critic, logp, indices = self.actor.sample_normal(state, reparameterize=False)

        if indices is not None:
            if deterministic:
                # deterministic: use logits -> argmax
                mu, _ = self.actor.forward(state)
                probs = F.softmax(mu, dim=-1)
                return int(probs.argmax(dim=-1).cpu().numpy()[0])
            else:
                # stochastic: sampled index
                return int(indices.cpu().numpy()[0])
        else:
            # continuous
            if deterministic:
                # return mean action (mu) clipped as in forward
                mu, sigma = self.actor.forward(state)
                max_action_tensor = T.tensor(self.actor.max_action).to(self.actor.device)
                action = T.tanh(mu) * max_action_tensor
                return action.cpu().detach().numpy()[0]
            else:
                return actions_for_critic.cpu().detach().numpy()[0]