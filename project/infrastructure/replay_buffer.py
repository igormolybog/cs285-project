import numpy as np
import copy

from project.infrastructure.utils import convert_listofrollouts, add_noise, Path, Transition

class NewReplayBuffer(object):

    def __init__(self, max_size=1000000):

        self.max_size = max_size
        self.current_size = 0
        self.path_store = []
        self.current_path = None
        
        self.obs = None
        self.acs = None
        self.concatenated_rews = None
        self.unconcatenated_rews = None
        self.next_obs = None
        self.terminals = None

    def store_path(self):
        
        path = copy.deepcopy(self.current_path)
            
        # Store path
        self.path_store.append(path)
        
        # Reset current path
        self.current_path = None
        
        return
    
    def add_transition_to_path(self, transition):
             
        if self.current_path is None:
            self.current_path = transition
            
        else:
            self.current_path["observation"] = np.append(self.current_path["observation"], transition["observation"])
            self.current_path["action"] = np.append(self.current_path["action"], transition["action"])
            
            #print(self.current_path["reward"],transition["reward"])
            self.current_path["reward"] = np.append(self.current_path["reward"], transition["reward"])
            #print(self.current_path["reward"])
            #input("Inside add_transition_to_path")
            
            self.current_path["next_observation"] = np.append(self.current_path["next_observation"], transition["next_observation"])
            self.current_path["terminal"] = np.append(self.current_path["terminal"], transition["terminal"])
        
        return
        
        
    def add_transition(self, transition, noised=False):
        
        #assert(len(transition['reward']) == 1)

        if transition["terminal"] == 1: #finish the current path     
            self.add_transition_to_path(transition)
            self.store_path()
        
        else:
            self.add_transition_to_path(transition)
            
    
        # Adding to the buffer:     
        observations = np.array(transition["observation"])
        actions = np.array(transition["action"])
        rewards = np.array(transition["reward"])
        next_observations = np.array(transition["next_observation"])
        terminals = np.array(transition["terminal"])
        
        if noised:
            observations = add_noise(observations)
            next_observations = add_noise(next_observations)

        if self.obs is None: #i.e: if there is nothing currently in the buffer
            self.obs = observations[-self.max_size:]
            self.acs = actions[-self.max_size:]
            self.rewards = rewards[-self.max_size:]
            self.next_obs = next_observations[-self.max_size:]
            self.terminals = terminals[-self.max_size:]         
        else:
            self.obs = np.concatenate([self.obs, observations])[-self.max_size:]
            self.acs = np.concatenate([self.acs, actions])[-self.max_size:]
            self.rewards = np.concatenate([self.rewards, rewards])[-self.max_size:]
            self.next_obs = np.concatenate([self.next_obs, next_observations])[-self.max_size:]
            self.terminals = np.concatenate([self.terminals, terminals])[-self.max_size:]
            
        return 
    
    def add_rollouts(self, paths, noised=False):

        # add new rollouts into our list of rollouts
        
        for path in paths:
            self.path_store.append(path)

        # convert new rollouts into their component arrays, and append them onto our arrays
        observations, actions, rewards, next_observations, terminals = convert_listofrollouts(paths)
        
        #print("obs: " +str(observations) + "acs: " +str(actions) + "next_obs: " +str(next_observations))
        #print("adding into buffer: " +str(len(paths)) + " path(s)")
        
        if noised:
            observations = add_noise(observations)
            next_observations = add_noise(next_observations)

        if self.obs is None: #i.e: if there is nothing currently in the buffer
            self.obs = observations[-self.max_size:]
            self.acs = actions[-self.max_size:]
            self.rewards = rewards[-self.max_size:]
            self.next_obs = next_observations[-self.max_size:]
            self.terminals = terminals[-self.max_size:]         
            #self.unconcatenated_rews = unconcatenated_rews[-self.max_size:]
        else:
            self.obs = np.concatenate([self.obs, observations])[-self.max_size:]
            self.acs = np.concatenate([self.acs, actions])[-self.max_size:]
            self.rewards = np.concatenate([self.rewards, rewards])[-self.max_size:]
            self.next_obs = np.concatenate([self.next_obs, next_observations])[-self.max_size:]
            self.terminals = np.concatenate([self.terminals, terminals])[-self.max_size:]

    ########################################
    ########################################
    
    def sample_random_rollouts(self, num_rollouts):
        rand_indices = np.random.permutation(len(self.path_store))[:num_rollouts]
        return self.path_store[rand_indices]

    def sample_recent_rollouts(self, num_rollouts=1):
        return self.path_store[-num_rollouts:]
    

    def sample_random_data(self, batch_size):
        assert self.obs.shape[0] == self.acs.shape[0] == self.concatenated_rews.shape[0] == self.next_obs.shape[0] == self.terminals.shape[0]
        rand_indices = np.random.permutation(self.obs.shape[0])[:batch_size]
        return self.obs[rand_indices], self.acs[rand_indices], self.concatenated_rews[rand_indices], self.next_obs[rand_indices], self.terminals[rand_indices]

    def sample_recent_data(self, batch_size=1, concat_rew=True):
        return self.obs[-batch_size:], self.acs[-batch_size:], self.rewards[-batch_size:], self.next_obs[-batch_size:], self.terminals[-batch_size:]

'''   
class ReplayBuffer(object):

    def __init__(self, max_size=1000000):

        self.max_size = max_size
        self.paths = []
        self.obs = None
        self.acs = None
        self.concatenated_rews = None
        self.unconcatenated_rews = None
        self.next_obs = None
        self.terminals = None

    def add_rollouts(self, paths, noised=False):

        # add new rollouts into our list of rollouts
        
        for path in paths:
            self.paths.append(path)

        # convert new rollouts into their component arrays, and append them onto our arrays
        observations, actions, rewards, next_observations, terminals = convert_listofrollouts(paths)
        
        #print("obs: " +str(observations) + "acs: " +str(actions) + "next_obs: " +str(next_observations))
        #print("adding into buffer: " +str(len(paths)) + " path(s)")
        
        if noised:
            observations = add_noise(observations)
            next_observations = add_noise(next_observations)

        if self.obs is None: #i.e: if there is nothing currently in the buffer
            self.obs = observations[-self.max_size:]
            self.acs = actions[-self.max_size:]
            self.rewards = rewards[-self.max_size:]
            self.next_obs = next_observations[-self.max_size:]
            self.terminals = terminals[-self.max_size:]         
            #self.unconcatenated_rews = unconcatenated_rews[-self.max_size:]
        else:
            self.obs = np.concatenate([self.obs, observations])[-self.max_size:]
            self.acs = np.concatenate([self.acs, actions])[-self.max_size:]
            self.rewards = np.concatenate([self.rewards, rewards])[-self.max_size:]
            self.next_obs = np.concatenate([self.next_obs, next_observations])[-self.max_size:]
            self.terminals = np.concatenate([self.terminals, terminals])[-self.max_size:]
            
            #if isinstance(unconcatenated_rews, list):
            #    self.unconcatenated_rews += unconcatenated_rews
            #else:
            #    self.unconcatenated_rews.append(unconcatenated_rews)

    ########################################
    ########################################

    def sample_random_rollouts(self, num_rollouts):
        rand_indices = np.random.permutation(len(self.paths))[:num_rollouts]
        return self.paths[rand_indices]

    def sample_recent_rollouts(self, num_rollouts=1):
        return self.paths[-num_rollouts:]

    ########################################
    ########################################

    def sample_random_data(self, batch_size):

        assert self.obs.shape[0] == self.acs.shape[0] == self.concatenated_rews.shape[0] == self.next_obs.shape[0] == self.terminals.shape[0]
        rand_indices = np.random.permutation(self.obs.shape[0])[:batch_size]
        return self.obs[rand_indices], self.acs[rand_indices], self.concatenated_rews[rand_indices], self.next_obs[rand_indices], self.terminals[rand_indices]

    def sample_recent_data(self, batch_size=1, concat_rew=True):

        return self.obs[-batch_size:], self.acs[-batch_size:], self.rewards[-batch_size:], self.next_obs[-batch_size:], self.terminals[-batch_size:]
    

        if concat_rew:
            return self.obs[-batch_size:], self.acs[-batch_size:], self.concatenated_rews[-batch_size:], self.next_obs[-batch_size:], self.terminals[-batch_size:]
        else:
            num_recent_rollouts_to_return = 0
            num_datapoints_so_far = 0
            index = -1
            while num_datapoints_so_far < batch_size:
                recent_rollout = self.paths[index]
                index -=1
                num_recent_rollouts_to_return +=1
                num_datapoints_so_far += get_pathlength(recent_rollout)
            rollouts_to_return = self.paths[-num_recent_rollouts_to_return:]
            observations, actions, next_observations, terminals, concatenated_rews, unconcatenated_rews = convert_listofrollouts(rollouts_to_return)
            return observations, actions, unconcatenated_rews, next_observations, terminals
'''