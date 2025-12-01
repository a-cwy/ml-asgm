import gymnasium as gym
import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
import os
import utils


# Actor network (PyTorch)
class Actor(nn.Module):
    def __init__(self, state_size, action_size, hidden=24):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_size),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.net(x)


class Critic(nn.Module):
    def __init__(self, state_size, hidden=24):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )

    def forward(self, x):
        return self.net(x)


class A2CAgent():
    """
    Simple A2C agent using PyTorch. Supports n-step returns.
    If `actor_model` or `critic_model` are provided they should be nn.Module instances;
    otherwise defaults are created using observation shape inferred from the env.
    """
    def __init__(
            self,
            env: gym.Env,
            actor_model: nn.Module = None,
            critic_model: nn.Module = None,
            discount_factor: float = 0.99,
            learning_rate: float = 1e-3,
            gamma: float = 0.99,
            n_steps: int = 50,
            entropy_coef: float = 0.01,
            value_loss_coef: float = 0.5,
            max_grad_norm: float = 0.5
        ):
        self.env = env
        self.discount_factor = discount_factor
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.n_steps = n_steps
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.max_grad_norm = max_grad_norm

        # Build a flattened state vector from a sample observation
        sample_obs, _ = self.env.reset()
        sample_vec = self._flatten_obs(sample_obs)
        self.state_size = sample_vec.shape[0]
        self.action_size = self.env.action_space.n

        # Create models if not provided
        self.actor_model = actor_model or Actor(self.state_size, self.action_size)
        self.critic_model = critic_model or Critic(self.state_size)

        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor_model.to(self.device)
        self.critic_model.to(self.device)

        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor_model.parameters(), lr=self.learning_rate)
        self.critic_optimizer = optim.Adam(self.critic_model.parameters(), lr=self.learning_rate)

    def _flatten_obs(self, obs):
        """Convert gym `Dict` observation to 1D numpy float32 array (same as DQN)."""
        if isinstance(obs, dict):
            vec = np.array(list(obs.values()), dtype=np.float32).flatten()
        else:
            vec = np.array(obs, dtype=np.float32).flatten()
        return vec

    def choose_action(self, state_vec):
        """Sample an action from the actor given a flattened state vector."""
        state = torch.tensor(state_vec, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            probs = self.actor_model(state)
        dist = Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action), dist.entropy()

    def act(self):
        """Run one episode using the current policy (deterministic argmax)."""
        obs, _ = self.env.reset()
        state = self._flatten_obs(obs)
        done = False
        total_reward = 0.0
        while True:
            state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            with torch.no_grad():
                probs = self.actor_model(state_t)
                action = probs.argmax(dim=-1).item()
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            state = self._flatten_obs(next_obs)
            if terminated or truncated:
                break
        print(f"Episode total reward: {total_reward}")

    def save_models(self, policy_path, value_path):
        os.makedirs(os.path.dirname(policy_path), exist_ok=True)
        torch.save(self.actor_model.state_dict(), policy_path)
        torch.save(self.critic_model.state_dict(), value_path)

    def load_models(self, policy_path, value_path, map_location=None):
        if map_location is None:
            map_location = self.device
        self.actor_model.load_state_dict(torch.load(policy_path, map_location=map_location))
        self.critic_model.load_state_dict(torch.load(value_path, map_location=map_location))

    def compute_n_step_returns(self, rewards, next_value, masks):
        """Compute discounted n-step returns (list) in reverse order."""
        R = next_value
        returns = []
        for r, m in zip(reversed(rewards), reversed(masks)):
            R = r + self.gamma * R * m
            returns.insert(0, R)
        return returns

    def train(self, episodes: int, save_every: int = 50, policy_path=None, value_path=None, max_steps_per_episode: int = 2000):
        """Train the actor-critic using n-step returns."""
        rewards_per_episode = []

        for ep in range(episodes):
            obs, _ = self.env.reset()
            state = self._flatten_obs(obs)
            ep_reward = 0.0

            log_probs = []
            values = []
            rewards = []
            entropies = []
            masks = []  # 1.0 for not done, 0.0 for done

            for step in range(max_steps_per_episode):
                state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)

                probs = self.actor_model(state_t)
                dist = Categorical(probs)
                action = dist.sample()

                value = self.critic_model(state_t)
                log_prob = dist.log_prob(action)
                entropy = dist.entropy()

                next_obs, reward, terminated, truncated, info = self.env.step(action.item())
                done = bool(terminated or truncated)

                log_probs.append(log_prob)
                values.append(value)
                rewards.append(float(reward))
                entropies.append(entropy)
                masks.append(0.0 if done else 1.0)

                ep_reward += reward
                state = self._flatten_obs(next_obs)

                # if we have enough steps or episode finished, perform update
                if len(rewards) >= self.n_steps or done:
                    # bootstrap value
                    if done:
                        next_value = 0.0
                    else:
                        next_state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
                        with torch.no_grad():
                            next_value = self.critic_model(next_state_t).item()

                    returns = self.compute_n_step_returns(rewards, next_value, masks)

                    returns_t = torch.tensor(returns, dtype=torch.float32, device=self.device)
                    values_t = torch.cat(values).squeeze(-1)
                    log_probs_t = torch.cat(log_probs)
                    entropies_t = torch.cat(entropies)

                    advantages = returns_t - values_t

                    actor_loss = -(log_probs_t * advantages.detach()).mean() - self.entropy_coef * entropies_t.mean()
                    critic_loss = F.mse_loss(values_t, returns_t)

                    # optimize actor
                    self.actor_optimizer.zero_grad()
                    actor_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.actor_model.parameters(), self.max_grad_norm)
                    self.actor_optimizer.step()

                    # optimize critic
                    self.critic_optimizer.zero_grad()
                    (self.value_loss_coef * critic_loss).backward()
                    torch.nn.utils.clip_grad_norm_(self.critic_model.parameters(), self.max_grad_norm)
                    self.critic_optimizer.step()

                    # clear buffers
                    log_probs = []
                    values = []
                    rewards = []
                    entropies = []
                    masks = []

                if done:
                    break

            rewards_per_episode.append(ep_reward)

            if ep % 10 == 0:
                avg_last_10 = np.mean(rewards_per_episode[-10:])
                print(f"Episode {ep}/{episodes} - Reward: {ep_reward:.2f} - AvgLast10: {avg_last_10:.2f}")
            else:
                print(f"Episode {ep}/{episodes} - Reward: {ep_reward:.2f}")

            # save checkpoint
            if policy_path and value_path and (ep % save_every == 0):
                self.save_models(policy_path, value_path)

        return rewards_per_episode


#### TRAINING CONFIG
from datetime import datetime
LOAD_PRETRAINED = False
USE_DATESTAMP = True

# Get the current date and time
now = datetime.now()

# Format the datetime object into a string
# YYYYMMDD_HHMMSS is a common and sortable format
version = now.strftime("%Y%m%d_%H%M%S") if USE_DATESTAMP else "v1"

VERSION_NUM = f"{version}"
EPISODES = 100
POLICY_DIR = f"./models/a2c/a2c-{VERSION_NUM}-e{EPISODES}-policy.pth"
VALUE_DIR = f"./models/a2c/a2c-{VERSION_NUM}-e{EPISODES}-value.pth"


if __name__ == "__main__":
    utils.init()
    env = gym.make("WaterHeater-v0")
    agent = A2CAgent(env)
    
    #ENSURE directory EXISTS
    if LOAD_PRETRAINED and os.path.exists(POLICY_DIR) and os.path.exists(VALUE_DIR):
        agent.load_models(POLICY_DIR, VALUE_DIR)
        agent.act()
    else:
        rewards = agent.train(EPISODES, save_every=50, policy_path=POLICY_DIR, value_path=VALUE_DIR)
        print(rewards)
        if not os.path.exists(os.path.dirname(POLICY_DIR)):
            os.makedirs(os.path.dirname(POLICY_DIR), exist_ok=True)
        agent.save_models(POLICY_DIR, VALUE_DIR)