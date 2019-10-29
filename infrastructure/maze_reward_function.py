import numpy as np
import time
import copy

from .base_reward_function import BaseRewardFunction
from .utils import *
############################################
############################################

class MazeRewardFunction(BaseRewardFunction):
         
     def __init__(self, params):
        super(MazeRewardFunction, self).__init__()
        
     
        self.ob_dim = params['ob_dim']
        self.ac_dim = params['ac_dim']
        
        self.maze_size = params['maze_size']
        self.maze_end = params['maze_end']
        
        self.parametric_reward = LookUpTable((self.ob_dim, self.ac_dim))
        
       
     def __call__(self, obs, acs):
                          
        if obs == self.maze_end:
            return 1
         
        else:  
            base_reward = -0.1 / (self.maze_end[0]*self.maze_end[1])
            
            total_reward = base_reward + self.parametric_reward(obs)[acs]
            return total_reward
      
        
     def update_reward_function(self):
         raise NotImplementedError   
          
    

         