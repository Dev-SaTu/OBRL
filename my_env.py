import numpy as np
import pandas as pd
import torch

class OBRLEnv:

    def __init__(self):
        self.observation_space = torch.zeros(size=(121,))
        self.action_space = torch.zeros(size=(3,))
        self.state = np.pad(pd.read_csv('data.csv').to_numpy(), ((0, 0), (0, 1)), 'constant', constant_values=0)
        self.page = 0
        self.entry = False

    def reset(self):
        self.page = 0
        self.entry = False
        return self.state[self.page]

    def step(self, action: int):
        self.page = self.page + 1
        next_state = self.state[self.page]
        
        reward = 0
        
        if action == 1:
            if not self.entry:
                next_state[-1] = next_state[0]
                self.entry = True

        elif action == 2:
            if self.entry:
                reward = 1
                next_state[-1] = 0
                self.entry = False

        done = self.page >= len(self.state) - 1
        return next_state, reward, done
    
    def close(self):
        pass
