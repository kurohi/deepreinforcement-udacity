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

    def sample(self):
        index = np.random.randint(0, len(self.buffer))
        return self.buffer[index]

    def clear(self, buffer_size=None):
        if(buffer_size!=None):
            self.buffer_size = buffer_size
        self.buffer = deque(maxlen=self.buffer_size)
