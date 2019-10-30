import numpy as np
from infrastructure.lookuptable import LookUpTable


class Q_table(LookUpTable):
    def __new__(cls, *args, **kwargs):
        '''called before __init__'''
        return super(Q_table, cls).__new__(cls, *args, **kwargs)

    def fit(self, batch):
        for item in batch:
            self.update(*item)

    def update(self, ob_t, ac_t, r_t, ob_tplus1, terminal_t):
        new_val = r_t + max(self(ob_tplus1))
        self.assign(ob_t, ac_t, new_val)
