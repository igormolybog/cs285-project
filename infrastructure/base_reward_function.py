import numpy as np
import time
import copy

############################################
############################################

class BaseRewardFunction(object):
     
     def __init__(self, **kwargs):
        super(BaseRewardFunction, self).__init__(**kwargs)
        
     
     def compute_reward(self):
        raise NotImplementedError
         
     def update_reward_function(self):
         raise NotImplementedError
         