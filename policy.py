import numpy as np

from scipy import stats
from multiprocessing.pool import ThreadPool

# much left to implement here

class Policy:
    def __init__(self, parameters, alpha=0.1 ,gamma=0.98, seq_len=20, batch_size=64):
        self.theta = parameters
        self.alpha = alpha
        self.gamma = gamma
        self.seq_len = seq_len
        self.batch_size = batch_size
        
    def policy(self):
        pass
    
    def probs(self):
        pass
    
    def action_batch(self, distributions):
        pass
    
    def discounted_rewards(self, rewards):
        discounted_rewards = np.zeros(len(seq_len))
        discounted_rewards[i] = rewards[0]
        for i in range(1,seq_len):
            discounted_rewards[i] = discounted_rewards[i-1] * self.gamma**2 + rewards[i]
        return discounted_rewards
    
    def update(self):
        pass