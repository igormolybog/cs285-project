import numpy as np
import time
import copy

############################################
############################################

class LookUpTable(np.ndarray):
    
    def __init__(self):
        '''called before __init__'''
        return super(LookUpTable,self).__init__()
        
        self.fill(0)
        
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
        
    def __add__(self, arr): 
        
        buffer = self
              
        return buffer  + arr 
    
    
def sample_bernoulli(p):
    
     if np.random.random() <= p   :
         return True
     else:
        return False     


def normalize(data, mean, std, eps=1e-8):
    return (data-mean)/(std+eps)

def unnormalize(data, mean, std):
    return data*std+mean


def add_noise(data_inp, noiseToSignal=0.01):
    data = copy.deepcopy(data_inp)  # (num data points, dim)

    # mean of data
    mean_data = np.mean(data, axis=0)

    # if mean is 0,
    # make it 0.001 to avoid 0 issues later for dividing by std
    mean_data[mean_data == 0] = 0.000001

    # width of normal distribution to sample noise from
    # larger magnitude number = could have larger magnitude noise
    std_of_noise = mean_data * noiseToSignal
    for j in range(mean_data.shape[0]):
        data[:, j] = np.copy(data[:, j] + np.random.normal(
            0, np.absolute(std_of_noise[j]), (data.shape[0],)))

    return data
