import random
from collections import deque
import numpy as np


class ReplayBuffer(object):
    def __init__(self, buffer_size, random_seed=123):
        self.buffer_size = buffer_size # how many memories we can store
        self.count = 0 # how many memories are currently stored
        self.buffer = deque() # the actual storage (automatically removes oldest when full)
        random.seed(random_seed) # makes randomness repeatable for debugging

    def add(self, s, a, r, t, s2):
        # s = state/what robot saw, a = action/what it did, r = reward/was it good?,
        # t = terminated/did episode end?, s2 = next state/what it saw after
        experience = (s, a, r, t, s2)
        if self.count < self.buffer_size:
            self.buffer.append(experience) # still have room, add it
            self.count += 1
        else:
            self.buffer.popleft() # full, remove oldest memory
            self.buffer.append(experience)

    def size(self):
        return self.count # how many memories does robot have?

    def sample_batch(self, batch_size):
        # pick random memories to learn from, random because it prevents robot from forgetting old lessons
        if self.count < batch_size:
            batch = random.sample(self.buffer, self.count) # not enough yet, take all we have
        else:
            batch = random.sample(self.buffer, batch_size) # pick random batch_size memories

        s_batch = np.array([_[0] for _ in batch]) # all states
        a_batch = np.array([_[1] for _ in batch]) # all actions
        r_batch = np.array([_[2] for _ in batch]).reshape(-1, 1) # all rewards
        t_batch = np.array([_[3] for _ in batch]).reshape(-1, 1) # all terminations
        s2_batch = np.array([_[4] for _ in batch]) # all next states

        return s_batch, a_batch, r_batch, t_batch, s2_batch

    def clear(self):
        self.buffer.clear() # forget everything, start fresh
        self.count = 0
    
    def save(self, filepath):
    	# save memories to disk so we can resume training later
        """Save buffer to file"""
        import pickle
        data = {
            'buffer': list(self.buffer),
            'count': self.count,
            'buffer_size': self.buffer_size
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        print(f"Buffer saved to {filepath} (size: {self.count})")
    
    def load(self, filepath):
    	# load previously saved memories from disk
        """Load buffer from file"""
        import pickle
        import os
        if not os.path.exists(filepath):
            print(f"Buffer file not found: {filepath}")
            return False
        
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.buffer = deque(data['buffer'])
        self.count = data['count']
        self.buffer_size = data['buffer_size']
        print(f"Buffer loaded from {filepath} (size: {self.count})")
        return True
