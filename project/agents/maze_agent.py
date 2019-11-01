import tensorflow as tf
import numpy as np

from project.rewards.maze_reward_function import MazeRewardFunction

class MazeAgent(object):
    def __init__(self, initial_state, policy, reward_function):

        # Agents info: current state and current time in the rollout
        self.current_ob = initial_state
        self.current_t = 0
        self.policy = policy
        self.reward = reward_function

    def get_action(self, obs):
        action = self.policy.get_action(obs)
        return action

    def advance_time():
        self.current_t += 1
        
   def reset_time():
       self.current_t = 0
    
    def set_ob(self, ob):
        self.current_ob = ob

    def get_ob(self):
        return self.current_ob

    def train(self, ob_batch, ac_batch, re_batch, next_ob_batch, terminal_batch):

        self.policy.fit(ob_batch, ac_batch, re_batch, next_ob_batch, terminal_batch)
        return 0
