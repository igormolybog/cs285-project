# -*- coding: utf-8 -*-
"""
Simple script to run our project code
"""

import os
import time
import numpy as np

from project.infrastructure.maze_trainer import  MazeTrainer
from project.agents.maze_agent import MazeAgent
from project.policies.simple import RandomPolicy
from project.policies.argmax import ArgMax
from project.q_functions.q_table import Q_table
from project.rewards.r_table import R_table


# Gym registration (?)
import gym
import gym_maze
import gym_maze.envs

from gym import wrappers
from gym.wrappers.monitor import Monitor


class Objective(object):

    def __init__(self, params):

        self.params = params

        # ALl batches used for training
        self.params['train_batch_size'] = params['batch_size']

        # Additions:
        self.params['render'] = True
        self.params['special'] = {'solved_t': None,
                                  'streak_to_end': None,
                                  'maze_size': None,
                                  'maze_goal': None
                                 }



        # Monitor
        # if  self.params['enable_recording']:
        #     self.monitor = Monitor(self.params['env'], self.params['recording_folder'], force=True)

        # Agent
        self.agent_factory = lambda initial_state, reward_list, shape: MazeAgent(initial_state,
                                                        ArgMax(Q_table.cast(np.zeros(shape))),
                                                        R_table.cast(np.array(reward_list).reshape(shape)))

        # Environment
        # Make the gym environment
        self.env_factory = lambda: gym.make(self.params.pop('env_name')).env

        self.trainer = MazeTrainer(self.params)



    def __call__(self, argument):
        '''
            argument is a list that will be cast into a table
        '''
        env = self.env_factory()
        env.seed(self.params['seed'])

        # Ideally, we would create here an ob_placeholder and ac_placeholder,
        # and pass it to the agent (instead of shape)
        ob_dim = env.observation_space.shape
        ac_dim = env.action_space.n
        maze_size = tuple((env.observation_space.high + np.ones(env.observation_space.shape)).astype(int))
        shape = maze_size+(ac_dim,)

        agent = self.agent_factory(env.reset(), argument, shape)

        self.trainer.train(agent, env, self.params['n_iter'])

        self.trainer.evaluate(agent, env)

        # TODO: WHAT DO WE RETURN??
        real_value = 0
        return real_value

def main():

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name',  default='maze-random-10x10-plus-v0',
                        choices=('maze-random-10x10-plus-v0',
                                 'Maze_Dragons-v0')
                        )
    parser.add_argument('--exp_name', type=str, default='First_Test')

    parser.add_argument('--ep_len', type=int, default=200)
    parser.add_argument('--n_iter', type=int, default = 1)

    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--eval_batch_size', type=int, default=1000)

    parser.add_argument('--num_agent_train_steps_per_iter', type=int, default=1)
    parser.add_argument('--start_training', default = 0)
    parser.add_argument('--evaluate_after_each_iter', default = False)

    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--use_gpu', '-gpu', action='store_true')
    #parser.add_argument('--which_gpu', '-gpu_id', default=0)
    #parser.add_argument('--scalar_log_freq', type=int, default=int(1e4))

    parser.add_argument('--online', default=True)
    parser.add_argument('--enable_recording', default = True)

    parser.add_argument('--agent_type', default = 'DQN')

    parser.add_argument('--save_params', action='store_true')
    args = parser.parse_args()

    # convert to dictionary
    params = vars(args)

    # Set random seeds
    seed = params['seed']
    np.random.seed(seed)
    #tf.set_random_seed(self.params['seed'])
    #self.mean_episode_reward = -float('nan')
    #self.best_mean_episode_reward = -float('inf')

    #Some additional parameters
    params['training_begins'] = 0
    params['recording_folder'] ="C:/Repositories/cs285-project/data"

    objective = Objective(params)

    # Seting some of the parameters
    MAZE_SIZE = (10, 10)
    MAX_T = np.prod(MAZE_SIZE, dtype=int) * 100
    NUM_EPISODES = 50000


    objective.params['n_iter'] = NUM_EPISODES
    objective.params['ep_len'] = MAX_T
    objective.params['batch_size'] = 1
    objective.params['eval_batch_size'] = NUM_EPISODES

    # objective['special']['solved_t'] = np.prod(MAZE_SIZE, dtype=int)
    # objective['special']['streak_to_end'] = 120
    #
    # objective['special']['maze_size'] = MAZE_SIZE
    # objective['special']['maze_goal'] = MAZE_SIZE - np.array((1, 1))

    from project.optimizers.initialize import default_reward_table
    objective(list(default_reward_table(MAZE_SIZE+(4,)).flatten()))


if __name__ == "__main__":
    main()
