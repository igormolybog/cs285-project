import tensorflow as tf
import numpy as np

from maze_reward_functon import *

class MazeAgent(object):
    def __init__(self, initial_state, agent_params):


        self.current_obs = initial_state

        self.ac_dim = agent_params['ac_dim']
        
        self.learning_starts = agent_params['learning_starts']
        self.learning_freq = agent_params['learning_freq']
        self.target_update_freq = agent_params['target_update_freq']


        self.optimizer_spec = agent_params['optimizer_spec']
        #self.critic = DQNCritic(sess, agent_params, self.optimizer_spec)
        self.policy = None

        self.current_t = 0
        self.num_param_updates = 0
        
        self.reward_function = MazeRewardFunction(agent_params['reward_spec'])
        
    def compute_action(self, obs):
       
        self.current_obs = obs
       
        action = self.policy.get_action(self.current_obs)
        action = action[0]
        
        reward = self.reward_function.compute_reward(self.current_obs, action)
        
        return [reward, action]
    

    def train(self):

        pass
        
        return
