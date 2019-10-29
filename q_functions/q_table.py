import numpy as np


class Q_table(np.ndarray):
    def __new__(cls, *args, **kwargs):
        '''called before __init__'''
        return super(Q_table, cls).__new__(cls, *args, **kwargs)

    def __call__(self, obs):
        '''
        observations are supposed to be the first indexes
        '''
        return self[obs]
