import random
import numpy as np
import torch
from collections import deque

class ExperienceReplay(object):
    def __init__(self, max_capacity):
        self.memory = deque(maxlen=max_capacity)
    
    def append(self, exp):
        self.memory.append(exp)
    
    def sample(self, batch_size):
        mini_batch = random.sample(self.memory, batch_size)
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = [], [], [], [], []

        for item in mini_batch:
            s, a, r, n_s, d = item[0:4], item[4], item[5], item[6:10], item[10]
            state_batch.append(s)
            action_batch.append(a)
            reward_batch.append(r)
            next_state_batch.append(n_s)
            done_batch.append(d)
        
        state_batch = torch.FloatTensor(np.array(state_batch).astype("float32")).view(batch_size, -1)
        action_batch = torch.LongTensor(np.array(action_batch).astype("float32")).view(batch_size, -1)
        reward_batch = torch.FloatTensor(np.array(reward_batch).astype("float32")).view(batch_size, -1)
        next_state_batch = torch.FloatTensor(np.array(next_state_batch).astype("float32")).view(batch_size, -1)
        done_batch = torch.FloatTensor(np.array(done_batch).astype("float32")).view(batch_size, -1)
        
        return state_batch, action_batch, reward_batch, next_state_batch, done_batch
    
    def __len__(self):
        return len(self.memory)

