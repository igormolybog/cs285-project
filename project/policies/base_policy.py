import numpy as np
import time
import copy

############################################
############################################

class BasePolicy(object):
     
     def __init__(self, **kwargs):
        super(BasePolicy, self).__init__(**kwargs)
        
     
     def fit(self):
        raise NotImplementedError
         
     def get_action(self):
         raise NotImplementedError
         