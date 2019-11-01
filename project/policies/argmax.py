import numpy as np
from project.policies.base_policy import BasePolicy

class ArgMax(BasePolicy):

    def __init__(self, q_function, *args, **kwargs):
        '''
            q_function - function that returns an np.array of size Nactions or
            a function that takes in the actions
        '''
        super(ArgMax, self).__init__(*args, **kwargs)
        self.q_function = q_function

    def get_action(self, obs, t=None):
        return self._get_action(obs)

    def fit(self, ob_batch, ac_batch, re_batch, next_ob_batch, terminal_batch):
        self.q_function.fit(ob_batch, ac_batch, re_batch, next_ob_batch, terminal_batch)
        return None

    def _get_action(self, obs):
        q_obs = self.q_function(obs)
        if isinstance(q_obs, np.ndarray):
            return np.argmax(self.q_function(obs))
        else:
            raise Exception("q_function did not return an array. continuous control is not implemented")

class EpsilonGreedyAM(ArgMax):

    def __init__(self, exploration_schedule, *args, **kwargs):
        super(EpsilonGreedyAM, self).__init__( *args, **kwargs)
        self.exploration = exploration_schedule

    def get_action(self, obs, t):
        eps = self.exploration.value(t)
        if np.random.random() < eps:
            return int(q_function.NUM_ACTIONS*np.random.random())
        else:
            return self._get_action(self, obs)
