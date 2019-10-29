import numpy as np


class Q_table(np.ndarray):
    '''
        usage:
            q_function = Q_table((3, 4, 2))
        or
            arr = np.array(range(24)).reshape((3,4,2))
            q_function = Q_table.cast(arr)
    '''
    def __new__(cls, *args, **kwargs):
        '''called before __init__'''
        return super(Q_table, cls).__new__(cls, *args, **kwargs)

    def __init__(self, *args, **kwargs):
        '''
            q_function - function that returns an np.array of size Nactions or
            a function that takes in the actions
        '''
        self.NUM_ACTIONS = self.shape[-1]

    def __call__(self, obs, acs=None):
        '''
        observations are supposed to be the first indexes
        A SINGLE OBSERVATION MUST BE A TUPLE
        '''
        if len(np.array(obs).shape) == 1:
            return self[obs][acs]
        else:
            q_obs = [self[ob] for ob in obs]
            if acs is None:
                return np.array(q_obs)
            else:
                return np.array([q_obs[i][ac] for i, ac in enumerate(acs)])

    @classmethod
    def cast(cls, ndarray):
        return ndarray.view(cls)

    def __add__(self):
        pass



def test():
    q_function = Q_table.cast(np.array(range(24)).reshape((3,4,2)))
    assert (q_function == np.array(range(24)).reshape((3,4,2))).all()
    assert q_function((1,2), 0) == 12
    assert (q_function((1,2)) == [12, 13]).all()
    assert (q_function([(1,2),(1,3)], (0, 1)) == [12, 15]).all()
