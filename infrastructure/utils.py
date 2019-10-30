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
    

############################################
############################################

def sample_trajectory(env, agent, max_path_length, render=False, render_mode=('rgb_array')):
 
    # initialize env for the beginning of a new rollout
    ob = env.reset() # HINT: should be the output of resetting the env

    # init vars
    obs, acs, rewards, next_obs, terminals = [], [], [], [], []
    steps = 0
    while True:

        # use the most recent ob to decide what to do
        obs.append(ob)
        ac = agent.compute_action(ob) 
        ac = ac[0]
        acs.append(ac)

        # take that action and record results
        ob_new, _, done, _ = env.step(ac)
        
        rew = agent.reward_function(ob, ac)
        ob = ob_new
        
        # record result of taking that action
        steps += 1
        next_obs.append(ob)
        rewards.append(rew)

        # End the rollout if the rollout ended 
        # Note that the rollout can end due to done, or due to max_path_length
        if(done is True or steps == max_path_length):
            rollout_done = True # HINT: this is either 0 or 1
        else:
            rollout_done = False
        terminals.append(rollout_done)
        
        if rollout_done: 
            break

    return Path(obs, acs, rewards, next_obs, terminals)

def sample_trajectories(env, policy, min_timesteps_per_batch, max_path_length, render=False, render_mode=('rgb_array')):

    timesteps_this_batch = 0
    paths = []
    while timesteps_this_batch < min_timesteps_per_batch:
        path = sample_trajectory(env, policy, max_path_length)
        timesteps_this_batch += get_pathlength(path)
        paths.append(path)
        
    return paths, timesteps_this_batch


############################################
############################################

def Path(obs, acs, rewards, next_obs, terminals):
    """
        Take info (separate arrays) from a single rollout
        and return it in a single dictionary
    """
    return {"observation" : np.array(obs, dtype=np.float32),
            "reward" : np.array(rewards, dtype=np.float32),
            "action" : np.array(acs, dtype=np.float32),
            "next_observation": np.array(next_obs, dtype=np.float32),
            "terminal": np.array(terminals, dtype=np.float32)}


def convert_listofrollouts(paths):
    """
        Take a list of rollout dictionaries
        and return separate arrays,
        where each array is a concatenation of that array from across the rollouts
    """
    observations = np.concatenate([path["observation"] for path in paths])
    actions = np.concatenate([path["action"] for path in paths])
    next_observations = np.concatenate([path["next_observation"] for path in paths])
    terminals = np.concatenate([path["terminal"] for path in paths])
    concatenated_rewards = np.concatenate([path["reward"] for path in paths])
    unconcatenated_rewards = [path["reward"] for path in paths]
    return observations, actions, next_observations, terminals, concatenated_rewards, unconcatenated_rewards

############################################
############################################

def get_pathlength(path):
    return len(path["reward"])
    
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
