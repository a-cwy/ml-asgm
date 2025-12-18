import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class Actor(nn.Module):
    def __init__(self, state_size, action_size, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_size), 
            nn.Softmax(dim=-1)
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
        ):
        self.env = env

        self.actor_model = Actor(6, 4)
        self.actor_model.load_state_dict(torch.load('dash/actor_a2c.pth'))

        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.actor_model.to(self.device)

        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor_model.parameters(), lr=0.0001)

    def _flatten_obs(self, obs):
        """Convert gym `Dict` observation to 1D numpy float32 array (same as DQN)."""
        if isinstance(obs, dict):
            vec = np.array(list(obs.values()), dtype=np.float32).flatten()
        else:
            vec = np.array(obs, dtype=np.float32).flatten()
        return vec



    def act(self):
        """Run one episode using the current policy."""
        #breakdown rewards
        rewards_breakdown = [[0.0, 0.0, 0.0, 0.0]]
        obs, _ = self.env.reset()
        state = self._flatten_obs(obs)
        terminated = False
        truncated = False
        total_reward = 0.0
        while (not terminated and not truncated):
            state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            with torch.no_grad():
                probs = self.actor_model(state_t)
                action = probs.argmax(dim=-1).item()

            next_obs, reward, terminated, truncated, info = self.env.step(action)
            state = self._flatten_obs(next_obs)
            rewards_breakdown.append(list(info["rewards"].values()))

        return rewards_breakdown