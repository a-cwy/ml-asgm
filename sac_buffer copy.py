import numpy as np

# Agent memory

class ReplayBuffer():

    def __init__(self, max_size, input_shape, n_actions):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, *input_shape)) #unpack the tensors
        self.new_state_memory = np.zeros((self.mem_size, *input_shape))
        self.action_memory = np.zeros((self.mem_size, n_actions))
        self.reward_memory = np.zeros((self.mem_size))
        # Used to store done flag, in boolean
        self.terminal_memory = np.zeros(self.mem_size, dtype = np.bool)

    def stor_transition(self, state, action, reward, state_, done):
        # position -defined by counter and memory size
        index = self.mem_cntr % self.mem_size

        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done

        # update to new position for next time
        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_contr, self.mem_size)

        # batch size is the number taken for each training (of samples), not necessary to be power of 2(If do is a good practice also)
        batch = np.random.choice(max_mem, batch_size)

        # go ahead on sample of memory
        states = self.state_memory[batch]
        states_ = self.new_state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        dones = self.terminal_memory[batch]

        return states, actions, rewards, states_, dones