import tensorflow as tf
import numpy as np

from project.rewards.maze_reward_function import MazeRewardFunction

class MazeAgent(object):
    def __init__(self, initial_state, policy, reward_function):


        # Agents info: current state and current time in the rollout
        self.current_obs = initial_state
        self.current_t = 0
        self.policy = policy
        self.reward = reward_function

    def get_action(self, obs):

        self.current_obs = obs

        action = self.policy.get_action(self.current_obs)
        
        return action


    def train(self, ob_batch, ac_batch, re_batch, next_ob_batch, terminal_batch):

        self.policy.fit(ob_batch, ac_batch, re_batch, next_ob_batch, terminal_batch)
        return 0
