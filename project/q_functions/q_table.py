import numpy as np
from project.infrastructure.lookuptable import LookUpTable


class Q_table(LookUpTable):
    def __new__(cls, *args, **kwargs):
        '''called before __init__'''
        return super(Q_table, cls).__new__(cls, *args, **kwargs)

    def fit(self, ob_batch, ac_batch, re_batch, next_ob_batch, terminal_batch):
        for ob_t, ac_t, r_t, ob_tplus1, terminal_t in zip(ob_batch, ac_batch, re_batch, next_ob_batch, terminal_batch):
            self.update(ob_t, ac_t, r_t, ob_tplus1, terminal_t)

    def update(self, ob_t, ac_t, r_t, ob_tplus1, terminal_t):
        print('observed: '+str(ob_t))
        print('q(obs): '+str(self(ob_t)))
        print('action: '+str(ac_t))
        print('reward: '+str(r_t))
        new_val = r_t + max(self(ob_tplus1))
        self.assign(ob_t, ac_t, new_val)
        print('next will observe: '+str(ob_tplus1))
        print('q_new(obs): '+str(self(ob_t)))
