import numpy as np
import time
import copy

from .base_reward_function import BaseRewardFunction
from project.infrastructure.lookuptable import LookUpTable

############################################
############################################

class MazeRewardFunction(BaseRewardFunction):
         
     def __init__(self, params):
        super(MazeRewardFunction, self).__init__()
        
        
        self.params = params
        
        self.parametric_reward = LookUpTable((self.ob_dim, self.ac_dim))
        
       
     def __call__(self, obs, acs):
                          
        if obs == self.params['maze_goal']:
            return 1
         
        else:  
            base_reward = -0.1 / (self.params['maze_goal'][0]*self.params['maze_goal'][1])
            
            total_reward = base_reward + self.parametric_reward(obs)[acs]
            return total_reward
      
        
     def update_reward_function(self):
         raise NotImplementedError   
          
    

         