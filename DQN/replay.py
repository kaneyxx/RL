import random
import numpy as np
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
            s, a, r, n_s, d = item
            state_batch.append(s)
            action_batch.append(a)
            reward_batch.append(r)
            next_state_batch.append(n_s)
            done_batch.append(d)
        
        state_batch = np.array(state_batch).astype("float32")
        action_batch = np.array(action_batch).astype("float32")
        reward_batch = np.array(reward_batch).astype("float32")
        next_state_batch = np.array(next_state_batch).astype("float32")
        done_batch = np.array(done_batch).astype("float32")
        
        return state_batch, action_batch, reward_batch, next_state_batch, done_batch
    
    def __len__(self):
        return len(self.memory)

