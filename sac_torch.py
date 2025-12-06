import os
import torch as T
import torch.nn.functional as F
import numpy as np
from sac_buffer import ReplayBuffer
from sac_networks import ActorNetwork, CriticNetwork, ValueNetwork
import gymnasium as gym

class Agent():
    def __init__(self, alpha=0.0003, beta=0.0003, input_dims=[8],
            env=None, gamma=0.99, n_actions=2, max_size=1000000, tau=0.005,
            layer1_size=256, layer2_size=256, batch_size=256, reward_scale=2):
        self.gamma = gamma
        self.tau = tau
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        self.batch_size = batch_size
        self.n_actions = n_actions

        chkpt_dir = 'models/sac'

        # detect discrete action space
        self.discrete = isinstance(env.action_space, gym.spaces.Discrete)

        # set max_action for continuous, else None for discrete
        if not self.discrete and hasattr(env.action_space, 'high'):
            max_action = env.action_space.high
        else:
            max_action = None

        self.actor = ActorNetwork(alpha, input_dims, max_action=max_action,
                    n_actions=n_actions, name='actor', chkpt_dir=chkpt_dir)
        
        self.critic_1 = CriticNetwork(beta, input_dims, n_actions=n_actions,
                    name='critic_1', chkpt_dir=chkpt_dir)
        self.critic_2 = CriticNetwork(beta, input_dims, n_actions=n_actions,
                    name='critic_2', chkpt_dir=chkpt_dir)
        self.value = ValueNetwork(beta, input_dims, name='value', chkpt_dir=chkpt_dir)
        self.target_value = ValueNetwork(beta, input_dims, name='target_value', chkpt_dir=chkpt_dir)

        self.scale = reward_scale
        self.update_network_parameters(tau=1)

    def choose_action(self, observation):
        state = T.Tensor([observation]).to(self.actor.device)
        actions, _ = self.actor.sample_normal(state, reparameterize=False)

        # For discrete: actions is one-hot tensor (batch, n_actions)
        if self.discrete:
            # return index (int)
            idx = actions.argmax(dim=-1).cpu().detach().numpy()[0]
            return int(idx)
        else:
            # continuous case (action array)
            return actions.cpu().detach().numpy()[0]

    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        target_value_params = self.target_value.named_parameters()
        value_params = self.value.named_parameters()

        target_value_state_dict = dict(target_value_params)
        value_state_dict = dict(value_params)

        # soft update: target = tau*value + (1-tau)*target
        for name in value_state_dict:
            value_state_dict[name] = tau * value_state_dict[name].clone() + \
                    (1 - tau) * target_value_state_dict[name].clone()

        self.target_value.load_state_dict(value_state_dict)

    def save_models(self):
        print('.... saving models ....')
        os.makedirs(self.actor.checkpoint_dir, exist_ok=True)
        self.actor.save_checkpoint()
        self.value.save_checkpoint()
        self.target_value.save_checkpoint()
        self.critic_1.save_checkpoint()
        self.critic_2.save_checkpoint()

    def load_models(self):
        print('.... loading models ....')
        self.actor.load_checkpoint()
        self.value.load_checkpoint()
        self.target_value.load_checkpoint()
        self.critic_1.load_checkpoint()
        self.critic_2.load_checkpoint()

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return

        state, action, reward, new_state, done = \
                self.memory.sample_buffer(self.batch_size)

        reward = T.tensor(reward, dtype=T.float32).to(self.actor.device)
        done = T.tensor(done, dtype=T.bool).to(self.actor.device)
        state_ = T.tensor(new_state, dtype=T.float32).to(self.actor.device)
        state = T.tensor(state, dtype=T.float32).to(self.actor.device)

        # convert discrete actions (indices) to one-hot for critic input
        action = T.tensor(action, dtype=T.long).to(self.actor.device)
        action_one_hot = F.one_hot(action, num_classes=self.n_actions).float()

        value = self.value(state).view(-1)
        value_ = self.target_value(state_).view(-1)
        # zero out the value for terminal states
        value_[done] = 0.0

        # Value network update
        actions_new, log_probs = self.actor.sample_normal(state, reparameterize=False)
        # For discrete actor actions_new is one-hot, for continuous shaped actions
        q1_new_policy = self.critic_1.forward(state, actions_new)
        q2_new_policy = self.critic_2.forward(state, actions_new)
        critic_value = T.min(q1_new_policy, q2_new_policy).view(-1)

        log_probs = log_probs.view(-1)
        self.value.optimizer.zero_grad()
        value_target = critic_value - log_probs
        value_loss = 0.5 * F.mse_loss(value, value_target.detach())
        value_loss.backward(retain_graph=True)
        self.value.optimizer.step()

        # Actor update
        actions_new, log_probs = self.actor.sample_normal(state, reparameterize=True)
        q1_new_policy = self.critic_1.forward(state, actions_new)
        q2_new_policy = self.critic_2.forward(state, actions_new)
        critic_value = T.min(q1_new_policy, q2_new_policy).view(-1)

        actor_loss = log_probs.view(-1) - critic_value
        actor_loss = T.mean(actor_loss)
        self.actor.optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)
        self.actor.optimizer.step()

        # Critics update (use stored actions -> convert to one-hot)
        self.critic_1.optimizer.zero_grad()
        self.critic_2.optimizer.zero_grad()
        q_hat = self.scale * reward + self.gamma * value_
        q1_old_policy = self.critic_1.forward(state, action_one_hot).view(-1)
        q2_old_policy = self.critic_2.forward(state, action_one_hot).view(-1)
        critic_1_loss = 0.5 * F.mse_loss(q1_old_policy, q_hat.detach())
        critic_2_loss = 0.5 * F.mse_loss(q2_old_policy, q_hat.detach())

        critic_loss = critic_1_loss + critic_2_loss
        critic_loss.backward()
        self.critic_1.optimizer.step()
        self.critic_2.optimizer.step()

        self.update_network_parameters()