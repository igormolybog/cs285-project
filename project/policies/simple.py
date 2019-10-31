import numpy as np
import random
from .base_policy import BasePolicy

class RandomPolicy(BasePolicy):
    
     def __init__(self, policy_params):
        super(RandomPolicy, self).__init__()
          
        self.discrete = policy_params['discrete']
        
        self.ac_dim = policy_params['ac_dim']
        
     def fit(self):
        print("Dont try to fit a Random Policy >.< ")
         
     def get_action(self, obs):
         
        # Select a random action
        #action = env.action_space.sample()
        if self.discrete:
            action = random.randint(0, self.ac_dim - 1)
            
        return action
