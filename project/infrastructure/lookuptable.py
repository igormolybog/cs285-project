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
        return super(LookUpTable, cls).__new__(cls, *args, **kwargs)

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
        if acs is None:
            if len(np.array(obs).shape) == 1:
                return self._ob(obs)
            else:
                return np.array([self._ob(ob) for ob in obs])
        else:
            if len(np.array(obs).shape) == 1:
                return self._ob(obs, acs)
            else:
                return np.array([self._ob(ob, ac) for ob, ac in zip(obs, acs)])

    def _ob(self, ob, ac=None):
        ob = to_format(ob)
        if ac is None:
            return self[ob]
        else:
            return self[ob][to_format(ac)]

    def assign(self, ob, ac, value):
        ob = to_format(ob)
        ac = to_format(ac)
        try:
            temp = (*ob, ac)
            self[temp] = value
        except:
            self[ob, ac] = value


    @classmethod
    def cast(cls, ndarray):
        return ndarray.view(cls)

def to_format(ob_or_ac):
    if ob_or_ac is not None:
        try:
            ob_or_ac = tuple(np.array(ob_or_ac).astype(int))
        except:
            ob_or_ac = int(ob_or_ac)
    return ob_or_ac


def test():
    q_function = LookUpTable.cast(np.array(range(24)).reshape((3,4,2)))
    assert (q_function == np.array(range(24)).reshape((3,4,2))).all()
    assert q_function((1,2), 0) == 12
    assert (q_function((1,2)) == [12, 13]).all()
    assert (q_function([(1,2),(1,3)], (0, 1)) == [12, 15]).all()

if __name__ == "__main__":
    test()
