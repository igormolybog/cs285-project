import tensorflow as tf
import numpy as np

from project.rewards.maze_reward_function import MazeRewardFunction

class MazeAgent(object):
    def __init__(self, initial_state, agent_params):


        # Agents info: current state and current time in the rollout
        self.current_obs = initial_state
        self.current_t = 0
        
        self.ob_dim = self.current_obs.shape
        self.ac_dim = agent_params['ac_dim']
        
        #self.optimizer_spec = agent_params['optimizer_spec']
        #self.critic = DQNCritic(sess, agent_params, self.optimizer_spec)
        self.policy = agent_params['policy']
     
        self.reward_function = MazeRewardFunction(agent_params['reward_params'])
        
    def get_action(self, obs):
       
        self.current_obs = obs
       
        action = self.policy.get_action(self.current_obs)
               
        return action
    

    def train(self, ob_batch, ac_batch, re_batch, next_ob_batch, terminal_batch):

        #self.policy.fit(ob_batch, ac_batch, re_batch, next_ob_batch, terminal_batch)
        
        return 0
