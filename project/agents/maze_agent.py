import tensorflow as tf
import numpy as np
import copy

from project.rewards.maze_reward_function import MazeRewardFunction

class MazeAgent(object):
    def __init__(self, initial_state, policy, reward_function):

        # Agents info: current state and current time in the rollout
        self.current_ob = initial_state
        self.current_t = 0
        self.accumulated_reward = 0.0
        
        self.policy = policy
        self.reward = reward_function

    def get_action(self, obs):
        action = self.policy.get_action(obs)
        return action

    def advance_time(self):
        self.current_t += 1
        
    def reset(self): #does not currently reset obs!
       self.current_t = 0
       self.accumulated_reward = 0
    
    def get_time(self):
        return self.current_t
    
    def set_ob(self, ob):
        self.current_ob = copy.deepcopy(ob)

    def get_ob(self):
        return self.current_ob

    def accumulate_reward(self, reward):
        self.accumulated_reward += copy.deepcopy(reward) 
      
    def get_accumulated_reward(self):
        return self.accumulated_reward
    
    def train(self, ob_batch, ac_batch, re_batch, next_ob_batch, terminal_batch):

        self.policy.fit(ob_batch, ac_batch, re_batch, next_ob_batch, terminal_batch)
        return 0
