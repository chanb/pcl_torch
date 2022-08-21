import numpy as np


class ReplayBuffer(object):
    def __init__(self, max_len=100000, alpha=1, min_episodes=4):
        self.max_len = max_len
        self.alpha = alpha
        self.min_episodes = min_episodes
        self.buffer = []

    def add(self, episode):
        self.buffer.append(episode)
        if len(self.buffer) > self.max_len:
            delete_ind = np.random.randint(len(self.buffer))
            del self.buffer[delete_ind]

    def sample(self):
        return np.random.choice(self.buffer)

    @property
    def trainable(self):
        return len(self.buffer) > self.min_episodes
