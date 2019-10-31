import numpy as np


class LookUpTable(np.ndarray):
    '''
        usage:
            q_function = Lookuptable((3, 4, 2))
        or
            arr = np.array(range(24)).reshape((3,4,2))
            q_function = Lookuptable.cast(arr)
    '''
    def __new__(cls, *args, **kwargs):
        '''called before __init__'''
        return super(Lookuptable, cls).__new__(cls, *args, **kwargs)

    def __init__(self, *args, **kwargs):
        '''
            q_function - function that returns an np.array of size Nactions or
            a function that takes in the actions
        '''
        self.ob_dim = self.shape[:-1]
        self.ac_dim = self.shape[-1]

    def __call__(self, obs, acs=None):
        '''
        observations are supposed to be the first indexes
        A SINGLE OBSERVATION MUST BE A TUPLE, not a list
        '''
        if len(np.array(obs).shape) == 1:
            return self[obs][acs]
        else:
            q_obs = [self[ob] for ob in obs]
            if acs is None:
                return np.array(q_obs)
            else:
                return np.array([q_obs[i][ac] for i, ac in enumerate(acs)])

    def assign(self, ob, ac, value):
        try:
            temp = (*ob, ac)
            self[temp] = value
        except:
            self[ob, ac] = value


    @classmethod
    def cast(cls, ndarray):
        return ndarray.view(cls)



def test():
    q_function = Q_table.cast(np.array(range(24)).reshape((3,4,2)))
    assert (q_function == np.array(range(24)).reshape((3,4,2))).all()
    assert q_function((1,2), 0) == 12
    assert (q_function((1,2)) == [12, 13]).all()
    assert (q_function([(1,2),(1,3)], (0, 1)) == [12, 15]).all()
