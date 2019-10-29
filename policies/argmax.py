import numpy as np

class ArgMax(object):

    def __init__(self, q_function):
        '''
            q_function - function that returns an np.array of size Nactions or
            a function that takes in the actions
        '''
        self.q_function = q_function

    def get_action(self, obs):
        q_function(obs) = q_obs
        if isinstance(q_obs, np.ndarray):
            return np.argmax(q_function(obs))
        else:
            raise Exception("q_function did not return an array. continuous control is not implemented")
