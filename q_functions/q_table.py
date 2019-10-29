import numpy as np


class Q_table(np.ndarray):
    def __new__(cls, *args, **kwargs):
        '''called before __init__'''
        return super(Q_table, cls).__new__(cls, *args, **kwargs)

    def __call__(self, obs):
        '''
        observations are supposed to be the first indexes
        obs is either a list or tuple or ndarray
        '''
        buffer = self
        if isinstance(obs, (list, tuple, np.ndarray)):
            for i in obs:
                buffer = buffer[i,...]
            return buffer
        else :
            return self[obs]
