import numpy as np
import time
import copy

############################################
############################################

def sample_trajectory(env, agent, max_path_length, render=False):

    # initialize env for the beginning of a new rollout
    agent.reset()
    agent.set_ob(env.reset())
    # init vars
    obs ,acs, rews, n_obs, terms = [], [], [], [], []

    while True:
       
        current_ob = copy.deepcopy(agent.get_ob())
        
        action = agent.get_action(current_ob)
        
        # take that action and record results
        next_ob, _, done, _ = env.step(env.ACTION[action])

        reward = agent.reward(current_ob, action)
        if done is True:
            reward += agent.reward(next_ob, action) #collecting terminal reward
        
        obs.append(current_ob)
        acs.append(action)
        rews.append(reward)
        n_obs.append(next_ob)
        terms.append(done)
        
        # Rendering
        if render:
            env.render()

        # End the rollout if the rollout ended
        agent.accumulate_reward(reward)
        if(done is True or agent.current_t == max_path_length):       
            break
        else:   
            agent.advance_time()
            agent.set_ob(next_ob)
            
    return Path(obs, acs, rews, n_obs, terms)

def sample_trajectories(agent, env, min_timesteps_per_batch, max_path_length, render=False):

    timesteps_this_batch = 0
    paths = []
    while timesteps_this_batch < min_timesteps_per_batch:
        path = sample_trajectory(env, agent, max_path_length, render)
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


def Transition(obs, acs, rewards, next_obs, terminals):
    
    assert(len(rewards) == 1)
    
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
    #if len(paths) == 1:
    #    path = paths[0]
    #    observations = path["observation"]
    #    actions = path["action"]
    #    next_observations = path["next_observation"]
    #    terminals = path["terminal"]
    #    concatenated_rewards = path["reward"]
    #    unconcatenated_rewards = path["reward"] 
    #else:
    observations = np.concatenate([path["observation"] for path in paths])
    actions = np.concatenate([path["action"] for path in paths])
    rewards = np.concatenate([path["reward"] for path in paths])
    next_observations = np.concatenate([path["next_observation"] for path in paths])
    terminals = np.concatenate([path["terminal"] for path in paths])

    #unconcatenated_rewards = [path["reward"] for path in paths]
        
    return observations, actions, rewards, next_observations, terminals

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
