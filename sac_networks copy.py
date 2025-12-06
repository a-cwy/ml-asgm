import os # handle model check point
import torch as T 
import torch.nn.functional as F
import torch.nn as nn # for layers
import torch.optim as optim # optimizer
from torch.distributions.normal import Normal # normal distribution
import numpy as np

# Determine action valuable or not
class CriticNetwork(nn.Module):
    # beta = learning rate
    # fc1/2_dim = dimension of hidden layer [256 is from the paper]
    # name for model checkpoint
    def __init__(self, beta, input_dims, n_actions, fc1_dims=256, fc2_dims=256,
            name='critic', chkpt_dir='tmp/sac'):
        super(CriticNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_sac')

        # define [hidden] network - critic evaluate the value of a 'state and action' pair

        # Linear layer = fully-connected layer
        self.fc1 = nn.Linear(self.input_dims[0]+n_actions, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        # output layer
        self.q = nn.Linear(self.fc2_dims, 1)

        # self.parameters() come from nn.modules(base class[parent class])
        self.optimizer = optim.Adam(self.parameters(), lr=beta)

        # Device computation utilize if got
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        # send entire network to hard device above
        self.to(self.device)
    
    def forward(self, state, action):

        """
        Passes through to 1st and 2nd hidden layer - 
        
        This function feed forward of concatenates(torch.cat()) a sequence of tensors along an existing dimension.

        What is tensor?
        [fundamental data structure - store and manipulate data, multi-dimension arrays - scalars(0D), vector(1D), matrices(2D) 
        --GPU accelerated better than NumPy{If you got GPU}
        --Enable complex data structures = images/videos]
        
        e.g.: torch.tensor([1,2], [3, 4]) =>2 dimension"""
        action_value = self.fc1(T.cat([state, action], dim=1))
        action_value = F.relu(action_value)
        action_value = self.fc2(action_value)
        action_value = F.relu(action_value)

        q = self.q(action_value)

        return q

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))

class ValueNetwork(nn.Module):
    def __init__(self, beta, input_dims, fc1_dims=256, fc2_dims=256,
            name='value', chkpt_dir='tmp/sac'):
        # No action value needed, this VN focuses on estimation value state without emphasize the value of action taken
        super(ValueNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_sac')

        # Simple neuron network
        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, fc2_dims)
        # Output layer here
        self.v = nn.Linear(self.fc2_dims, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, state):
        state_value = self.fc1(state)
        state_value = F.relu(state_value)
        state_value = self.fc2(state_value)
        state_value = F.relu(state_value)

        # Output and return it
        v = self.v(state_value)

        return v

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))

# Hard part of the problem compared with others
# policy network = is a probability distribution - prob of select any action in action state based on given state/set of states = based on env's action space = discrete(simple)/continuous(may need Gaussian)
class ActorNetwork(nn.Module):
    # alpha = learning rate
    """
    Max action = DNN constraint - restricting the policy sampling to be plus or minus 1 on the tangent hyperbolic function
    
    But environment may have >1 action, therefore value of action * max_action for calculation following
    """
    def __init__(self, alpha, input_dims, max_action, fc1_dims=256, 
            fc2_dims=256, n_actions=2, name='actor', chkpt_dir='tmp/sac'):
        super(ActorNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_sac')
        self.max_action = max_action
        """
        Reparameterization noise handle the calculation of policies - serve the number of functions
        
        To make sure without taking log of 0 [undefined, software will break]
        """ 
        self.reparam_noise = 1e-6

        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)

        # 2 output, self.mu = mean of the distribution of our policy; self.sigma = standard deviation
        self.mu = nn.Linear(self.fc2_dims, self.n_actions)
        self.sigma = nn.Linear(self.fc2_dims, self.n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, state):
        prob = self.fc1(state)
        prob = F.relu(prob)
        prob = self.fc2(prob)
        prob = F.relu(prob)
        mu = self.mu(prob)
        sigma = self.sigma(prob)

        # To avoid distribution of policy become arbitraily broad - make it a finite and constraint value

        """
        The torch.clamp() function in PyTorch is used to **limit** the values within a tensor to a **specified range**. It takes an input tensor and ensures that all its elements fall within a given minimum and maximum value.

        Behavior:
            - If an element in the input tensor is **less than min**, it is **replaced by min**.
            - If an element in the input tensor is **greater than max**, it is **replaced by max**.
            - If an element is **within the range [min, max]**, it remains **unchanged**.
        
        """
        # Lower end = -20, Higher end = +2 in paper, to avoid sigma of zero too
        # At here is lower bound = 10^-6
        # Sigmoid function also can be used = non-linear activation between 0 and 1 [But slow in computation speed, this is faster]
        sigma = T.clamp(sigma, min=self.reparam_noise, max=1)

        return mu, sigma

    
    def sample_normal(self, state, reparameterize=True):
        mu, sigma = self.forward(state)
        #
        probabilities = Normal(mu, sigma)

        if reparameterize:
            # rsample = random sample of normal distribution [without noise]
            # or sample with noise **exploration factor to the actions
            actions = probabilities.rsample()
        else:
            # 
            actions = probabilities.sample()

        # For having same data type of the entire graph[send to device], has to be cuda tensor
        action = T.tanh(actions)*T.tensor(self.max_action).to(self.device)
        # For calculate loss function[updating weight]
        log_probs = probabilities.log_prob(actions) # plural cuz singular cases is proportional to the tan hyperbolic & action of the environment[just sampled from the pd], not actually drawing the tanh
        # Could be 0 if 1-1 so + noise, come from paper's appendix - handling the scale of the action
        log_probs -= T.log(1-action.pow(2)+self.reparam_noise)
        #scalar quatity and calculate loss
        # below make no. of components = no. of actions
        # pytorch output dimensionality corresponding to the no. of components of actions(c=a), also for scalar quantity for loss
        log_probs = log_probs.sum(1, keepdim=True)

        return action, log_probs

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))
