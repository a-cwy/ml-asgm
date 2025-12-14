import os
import torch as T
import torch.nn.functional as F
import numpy as np
from sac_buffer import ReplayBuffer
from sac_networks import ActorNetwork, CriticNetwork, ValueNetwork

class Agent():
    def __init__(self, alpha=0.0003, beta=0.0003, input_dims=[8],
        env=None, gamma=0.99, n_actions=2, max_size=1000000, tau=0.005,
        layer1_size=256, layer2_size=256, batch_size=256, reward_scale=5.0,
        ent_coef=0.2):
        self.gamma = gamma
        self.tau = tau
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        self.batch_size = batch_size
        self.n_actions = n_actions

        chkpt_dir = 'models/sac'

        # Detect continuous vs discrete:
        if hasattr(env.action_space, 'high'):
            max_action = env.action_space.high
        else:
            max_action = None

        self.actor = ActorNetwork(alpha, input_dims, max_action=max_action, n_actions=n_actions,
                    name='actor', chkpt_dir=chkpt_dir)
        
        self.critic_1 = CriticNetwork(beta, input_dims, n_actions=n_actions,
                    name='critic_1', chkpt_dir=chkpt_dir)
        self.critic_2 = CriticNetwork(beta, input_dims, n_actions=n_actions,
                    name='critic_2', chkpt_dir=chkpt_dir)
        self.value = ValueNetwork(beta, input_dims, name='value', chkpt_dir=chkpt_dir)
        self.target_value = ValueNetwork(beta, input_dims, name='target_value', chkpt_dir=chkpt_dir)

        self.scale = reward_scale
        # Entropy coefficient (temperature) for SAC losses
        # Smaller values reduce entropy regularization; tune for your env.
        self.entropy_alpha = ent_coef
        self.update_network_parameters(tau=1.0)

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

    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        # Polyak update: target = tau * source + (1-tau) * target
        for target_param, param in zip(self.target_value.parameters(), self.value.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

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

        # Value network update with gradient clipping
        actions_new, log_probs, _ = self.actor.sample_normal(state, reparameterize=False)
        # ensure actions_new is in correct device/type for critic
        if actions_new is None:
            raise RuntimeError("actor.sample_normal returned None for actions_new")
        q1_new_policy = self.critic_1.forward(state, actions_new)
        q2_new_policy = self.critic_2.forward(state, actions_new)
        critic_value = T.min(q1_new_policy, q2_new_policy).view(-1)

        log_probs = log_probs.view(-1)
        self.value.optimizer.zero_grad()
        # value target: Q - alpha * logp
        value_target = critic_value - self.entropy_alpha * log_probs
        value_loss = 0.5 * F.mse_loss(value, value_target.detach())
        value_loss.backward(retain_graph=True)

        # Clip gradients to prevent explosion
        T.nn.utils.clip_grad_norm_(self.value.parameters(), max_norm=1.0)
        self.value.optimizer.step()

        # Actor update with gradient clipping
        actions_new, log_probs, _ = self.actor.sample_normal(state, reparameterize=True)
        q1_new_policy = self.critic_1.forward(state, actions_new)
        q2_new_policy = self.critic_2.forward(state, actions_new)
        critic_value = T.min(q1_new_policy, q2_new_policy).view(-1)

        # actor loss: minimize (alpha * logp - Q)
        actor_loss = self.entropy_alpha * log_probs.view(-1) - critic_value
        actor_loss = T.mean(actor_loss)
        self.actor.optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)

        T.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        self.actor.optimizer.step()

        # Critics update with gradient clipping
        self.critic_1.optimizer.zero_grad()
        self.critic_2.optimizer.zero_grad()
        q_hat = self.scale * reward + self.gamma * value_
        q1_old_policy = self.critic_1.forward(state, action_one_hot).view(-1)
        q2_old_policy = self.critic_2.forward(state, action_one_hot).view(-1)
        critic_1_loss = 0.5 * F.mse_loss(q1_old_policy, q_hat.detach())
        critic_2_loss = 0.5 * F.mse_loss(q2_old_policy, q_hat.detach())

        critic_loss = critic_1_loss + critic_2_loss
        critic_loss.backward()

        T.nn.utils.clip_grad_norm_(self.critic_1.parameters(), max_norm=1.0)
        T.nn.utils.clip_grad_norm_(self.critic_2.parameters(), max_norm=1.0)

        self.critic_1.optimizer.step()
        self.critic_2.optimizer.step()

        self.update_network_parameters()