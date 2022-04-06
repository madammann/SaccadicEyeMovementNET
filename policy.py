import numpy as np
import tensorflow as tf

from scipy import stats
from multiprocessing.pool import ThreadPool

class PolicyGradient:
    def __init__(self, ccel, gamma=0.98, seq_len=20, batch_size=64):
        '''DESC
Arguments:
    ccel (list): A list of the categorical crossentropy loss batch-averages for each step except the first (len=seq_len-1).'''
        
        self.ccel = ccel
        self.seq_len = seq_len-1
        self.gamma = gamma
        self.rewards = []
        self.discounted_rewards = []
        self.error = None
        
    def generate_rewards(self):
        for ccel in self.ccel:
            self.rewards += [ccel.numpy()**-1]
    
    def generate_disc_rewards(self):
        self.discounted_rewards = np.zeros((self.seq_len))
        for i in range(self.seq_len):
            for j in range(self.seq_len):
                self.discounted_rewards[j] += self.rewards[i]*self.gamma**i
    
    def get_eye_loss(self, avg_states, with_entropy=True):
        self.generate_rewards()
        self.generate_disc_rewards()
        if with_entropy:
            self.error = np.mean([self.discounted_rewards[i]*avg_states[0][i]+avg_states[1][i] for i in range(self.seq_len)])
            return self.error
        else:
            self.gradient = np.mean([self.discounted_rewards[i]*avg_states[0][i] for i in range(self.seq_len)])
            return self.error
        
    # def generate_gradient(self, parameters, alpha=0.1):
    #     return tf.add(parameters,alpha*self.gradient)

class Environment:
    def __init__(self,image_batch,batch_size=64,seq_len=20):
        self.image_batch = image_batch
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.distributions = []
        self.coordinates = []
        self.action_probs = []
        self.entropy = []
        self.sequence = self.first_look()
        
    def terminal(self):
        if self.seq_len == len(self.sequence):
            return True
        else:
            return False
    
    def first_look(self):
        return np.expand_dims(np.array([patch for patch in ThreadPool(self.batch_size).starmap(self._get_image_patch, [(self.image_batch[i],None) for i in range(self.batch_size)])]),axis=0)
    
    def look_next(self):
        next_elem = np.expand_dims(np.array([patch for patch in ThreadPool(self.batch_size).starmap(self._get_image_patch, [(self.image_batch[i],self.coordinates[-1][i]) for i in range(self.batch_size)])]),axis=0)
        self.sequence = np.concatenate([self.sequence,next_elem],axis=0)
        return next_elem
    
    def _get_image_patch(self, image, coordinates=None):
        size = image.shape[:2]
        if type(coordinates) == type(None):
            coordinates = [int(size[0]/2),int(size[1]/2)]
        if coordinates[0] < 16:
            coordinates[0] = 16
        elif coordinates[0] > size[0] - 15:
            coordinates[0] = size[0]-15
        if coordinates[1] < 16:
            coordinates[1] = 16
        elif coordinates[1] > size[1] - 15:
            coordinates[1] = size[1]-15
        return image[coordinates[0]-16:coordinates[0]+16,coordinates[1]-16:coordinates[1]+16]
    
    def action(self, distributions):
        self.distributions += [[stats.multivariate_normal(mean=distributions[i][0:2],cov=distributions[i][2:]) for i in range(self.batch_size)]]
        self.coordinates += [[self.distributions[-1][i].rvs().astype('int32') for i in range(self.batch_size)]]
        self.action_probs += [[self.distributions[-1][i].pdf(self.coordinates[-1][i]) for i in range(self.batch_size)]]
        self.entropy += [[self.distributions[-1][i].entropy() for i in range(self.batch_size)]]
        self.look_next()
        return self.sequence[-1]
    
    def get_states(self, average=True):
        if average:
            avg_action_prob = [np.mean(self.action_probs[step]) for step in range(self.seq_len-1)]
            avg_entropy = [np.mean(self.entropy[step]) for step in range(self.seq_len-1)]
            return np.array([avg_action_prob,avg_entropy])
        else:
            states = np.zeros((self.seq_len-1,self.batch_size,2))
            for step in range(self.seq_len-1):
                for index in range(self.batch_size):
                    states[step][index][0] = self.action_probs[step][index]
                    states[step][index][1] = self.entropy[step][index]
            return states