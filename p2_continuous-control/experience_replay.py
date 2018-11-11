import numpy as np
import random
import torch
from collections import deque

'''
Creating an experience replay class that handles the circular buffer
cleaning and random sampling
'''
class ExperienceReplay(object):
    def __init__(self, buffer_size=16):
        self.buffer = deque(maxlen=buffer_size)
        self.buffer_size = buffer_size
        self.full = False
        self.index = 0

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size=256):
        experiences = random.sample(self.buffer, k=batch_size)
        state = torch.stack([e[0] for e in experiences],dim=0)
        action = torch.stack([e[1] for e in experiences],dim=0)
        reward = torch.stack([e[2] for e in experiences],dim=0)
        next_state = torch.stack([e[3] for e in experiences],dim=0)
        done = torch.stack([e[4] for e in experiences],dim=0)
        return state, action, reward, next_state, done

    def clear(self, buffer_size=None):
        if(buffer_size!=None):
            self.buffer_size = buffer_size
        self.buffer = deque(maxlen=self.buffer_size)
