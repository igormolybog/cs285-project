import numpy as np
from infrastructure.lookuptable import LookUpTable


class r_table(LookUpTable):
    def __new__(cls, *args, **kwargs):
        '''called before __init__'''
        return super(Q_table, cls).__new__(cls, *args, **kwargs)
