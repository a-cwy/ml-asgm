import os
import numpy as np
import torch as T
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical #using categorical because actions are discrete (0-3) as stated in environment


# MEMORY BUFFER

class PPOMemory:
    def __init__(self, batch_size):
        self.states = []
        self.probs = []
        self.vals = []
        self.actions = []
        self.rewards = []
        self.dones = []

        self.batch_size = batch_size

    def generate_batches(self):
        n_states = len(self.states) #total number of states collected during rollout
        batch_start = np.arange(0, n_states, self.batch_size) #used to split rollout data into mini batches

        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
#each states, actions, etc. are split into mini batches representing with different indices for training to avoid bias over time by shuffling the batch indices
        batches = [indices[i:i+self.batch_size] for i in batch_start] 

        return np.array(self.states), \
               np.array(self.actions), \
               np.array(self.probs), \
               np.array(self.vals), \
               np.array(self.rewards), \
               np.array(self.dones), \
               batches

# Store memory during rollout, trajectory collection held here so PPO can perform on-policy updates
    def store_memory(self, state, action, probs, vals, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(probs)
        self.vals.append(vals)
        self.rewards.append(reward)
        self.dones.append(done)

#Clears stored rollout data after policy optimization. PPO is an on-policy algorithm, so once the policy is updated, previously collected trajectories become invalid and must be discarded
    def clear_memory(self):
        self.states = []
        self.probs = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.vals = []


# ACTOR NETWORK


class ActorNetwork(nn.Module):
    def __init__(self, n_actions, input_dims, alpha,
                 fc1_dims=256, fc2_dims=256, chkpt_dir='models/ppo'):
        super(ActorNetwork, self).__init__()

        os.makedirs(chkpt_dir,exist_ok=True)
        self.checkpoint_file = os.path.join(chkpt_dir, 'actor_best.pth')

        self.actor = nn.Sequential(
            nn.Linear(*input_dims, fc1_dims),
            nn.ReLU(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.ReLU(),
            nn.Linear(fc2_dims, n_actions),
            nn.Softmax(dim=-1)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        probs = self.actor(state)
        dist = Categorical(probs)
        return dist

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))


# CRITIC NETWORK

class CriticNetwork(nn.Module):
    def __init__(self, input_dims, alpha,
                 fc1_dims=256, fc2_dims=256, chkpt_dir='tmp/ppo'):
        super(CriticNetwork, self).__init__()

        os.makedirs(chkpt_dir, exist_ok=True)
        self.checkpoint_file = os.path.join(chkpt_dir, 'critic_best.pth')

        self.critic = nn.Sequential(
            nn.Linear(*input_dims, fc1_dims),
            nn.ReLU(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.ReLU(),
            nn.Linear(fc2_dims, 1)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        value = self.critic(state)
        return value

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))



class Agent:
    def __init__(
        self,
        n_actions,
        input_dims,
        gamma=0.995,
        alpha=1e-4,
        entropy_coef=0.01,
        gae_lambda=0.95,
        policy_clip=0.2,
        batch_size=512,
        N=2048,
        n_epochs=4
    ):
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.gae_lambda = gae_lambda
        self.N = N

        self.actor = ActorNetwork(n_actions, input_dims, alpha)
        self.critic = CriticNetwork(input_dims, alpha)
        self.memory = PPOMemory(batch_size)

        for param in self.actor.actor[0].parameters():
            nn.init.uniform_(param, -0.1, 0.1)


    def _process_obs(self, obs):
        return np.array([
            obs["day"],
            obs["time"],
            obs["waterTemperature"],
            obs["targetTemperature"],
            obs["timeSinceSterilization"],
            obs["forecast"]
        ], dtype=np.float32)


    def choose_action(self, observation):
        if isinstance(observation, dict):
            observation = self._process_obs(observation)

        state = T.tensor(observation, dtype=T.float32).unsqueeze(0).to(self.actor.device)

        dist = self.actor(state)
        value = self.critic(state)
        action = dist.sample()

        prob = T.squeeze(dist.log_prob(action)).item()
        value = T.squeeze(value).item()
        action = action.item()

        return action, prob, value


    # Store memory

    def remember(self, state, action, prob, val, reward, done):
        if isinstance(state, dict):
            state = self._process_obs(state)
        self.memory.store_memory(state, action, prob, val, reward, done)

    def save_models(self):
        print("... saving models ...")
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()

    def load_models(self):
        print("... loading models ...")
        self.actor.load_checkpoint()
        self.critic.load_checkpoint()

# Learning phase from the rollout
    def learn(self):
        if len(self.memory.states) < self.memory.batch_size:
            return

#
        for _ in range(self.n_epochs):
            state_arr, action_arr, old_prob_arr, vals_arr, reward_arr, dones_arr, batches = \
                self.memory.generate_batches()

            values = vals_arr
            reward_arr = (reward_arr - reward_arr.mean()) / (reward_arr.std() + 1e-8)
            urgency_penalty = (reward_arr < 0) * (np.abs(reward_arr) * 0.1)
            reward_arr = reward_arr - urgency_penalty

            # COMPUTE ADVANTAGE USING GAE
            advantages = np.zeros(len(reward_arr), dtype=np.float32)
            lastgaelam = 0

            for t in reversed(range(len(reward_arr) - 1)):
                next_nonterminal = 1 - dones_arr[t]
                delta = reward_arr[t] + self.gamma * values[t+1] * next_nonterminal - values[t]
                advantages[t] = lastgaelam = delta + self.gamma * self.gae_lambda * next_nonterminal * lastgaelam

            hygiene_mask = (reward_arr < 0) 
            advantages[hygiene_mask] *= 3.0 
                    

            advantage = T.tensor(advantages, dtype=T.float32).to(self.actor.device)
            values = T.tensor(values, dtype=T.float32).to(self.actor.device)

            # TRAIN ON BATCHES
            for batch in batches:
                batch_states = T.tensor(state_arr[batch], dtype=T.float32).to(self.actor.device)
                batch_actions = T.tensor(action_arr[batch]).to(self.actor.device)
                batch_old_probs = T.tensor(old_prob_arr[batch], dtype=T.float32).to(self.actor.device)
                batch_advantage = advantage[batch]
                batch_advantage = (batch_advantage - batch_advantage.mean()) / (batch_advantage.std() + 1e-8)
                batch_returns = batch_advantage + values[batch]

                dist = self.actor(batch_states)
                critic_value = self.critic(batch_states).squeeze()

                new_probs = dist.log_prob(batch_actions)
                prob_ratio = (new_probs - batch_old_probs).exp()

                weighted_probs = batch_advantage * prob_ratio
                clipped_probs = batch_advantage * T.clamp(prob_ratio, 1 - self.policy_clip, 1 + self.policy_clip)

                actor_loss = -T.min(weighted_probs, clipped_probs).mean()
                critic_loss = (batch_returns - critic_value).pow(2).mean()
                entropy = dist.entropy().mean()
                loss = actor_loss + 0.5 * critic_loss - self.entropy_coef * entropy

                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()
                loss.backward()

                nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                self.actor.optimizer.step()
                self.critic.optimizer.step()

        self.memory.clear_memory()

    def act(self, env, max_steps=2*7*96):
        """
        Evaluate the trained PPO policy (NO learning).
        Runs one deterministic episode.
        """
        obs, _ = env.reset()
        state = self._process_obs(obs)

        terminated = False
        truncated = False
        steps = 0
        total_reward = 0.0

        rewards_breakdown = {
            "comfort": 0.0,
            "hygiene": 0.0,
            "energy": 0.0,
            "safety": 0.0
        }
        while not terminated and not truncated and steps < max_steps:
            state_t = T.tensor(state, dtype=T.float32).unsqueeze(0).to(self.actor.device)

        # Deterministic action (no exploration)
        with T.no_grad():
            dist = self.actor(state_t)
            action = dist.probs.argmax(dim=-1).item()

        next_obs, reward, terminated, truncated, info = env.step(action)

        total_reward += reward
        for k in rewards_breakdown:
            rewards_breakdown[k] += info["rewards"][k]

        state = self._process_obs(next_obs)
        steps += 1

        return total_reward, rewards_breakdown
