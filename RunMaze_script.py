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

# Gym registration (?)
import gym
import gym_maze
import gym_maze.envs

from gym import wrappers
from gym.wrappers.monitor import Monitor

class Obj(object):

    def __init__(self, params):

        self.params = params

        # ALl batches used for training
        self.params['train_batch_size'] = params['batch_size']
        #self.params['env_wrappers'] = self.agent_params['env_wrappers']

        # Environment
         # Make the gym environment
        env_wrap = gym.make(self.params['env_name'])
        self.params['env'] = env_wrap.env

        # Monitor
        if  self.params['enable_recording']:
            self.monitor = Monitor(self.params['env'], self.params['recording_folder'], force=True)

        # Agent
<<<<<<< HEAD

        ob_dim = self.params['env'].observation_space.shape
        ac_dim = self.params['env'].action_space.n
        shape = ob_dim+(ac_dim,)
        agent_factory = lambda initial_state, reward_list: MazeAgent(initial_state,
                                                        ArgMax(Q_table(shape)),
                                                        Q_reward.cast(np.array(reward_list).reshape(shape)))
        # self.params['agent_class'] = MazeAgent
        # self.params['agent_params'] = {'agent_type':    params['agent_type'],
        #                                'discrete':      True,  # Is this env continuous, or self.discrete?
        #                                'ac_dim':        self.params['env'].action_space.n,
        #                                'ob_dim':        self.params['env'].observation_space.shape
        #                               }
        # Policy
        # self.params['agent_params']['policy'] = RandomPolicy(self.params['agent_params']) # (!)(!)(!)
=======
        self.params['agent_class'] = MazeAgent
        self.params['agent_params'] = {'agent_type':    params['agent_type'],
                                       'discrete':      True,  # Is this env continuous, or self.discrete?
                                       'ac_dim':        self.params['env'].action_space.n,
                                       'ob_dim':        self.params['env'].observation_space.shape
                                      }

        # Policy
        self.params['agent_params']['policy'] = RandomPolicy(self.params['agent_params']) # (!)(!)(!)
>>>>>>> 844e4949081f4b061a7ed95f75167b5a2bb2f4b4


        # Additions:
        self.params['render'] = True
        self.params['special'] = {'solved_t': None,
                                 'streak_to_end': None
                                 }

        # Trainer
        self.rl_trainer = MazeTrainer(self.params)


    def run_training_loop(self):
        self.rl_trainer.run_training_loop( self.params['n_iter'],
                                           collect_policy = self.rl_trainer.agent,
                                         )

    def run_evaluation_loop(self):
        self.rl_trainer.evaluate_trainer(eval_policy = self.rl_trainer.agent)

    def __call__(self, argument):
        real_value = 0 #TODO
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

    parser.add_argument('--batch_size', type=int, default=50)
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


    #Some additional parameters
    params['recording_folder'] ="C:/Repositories/cs285-project/data"
    params['env'] = None

    # Defining our RL_OBJECT:
    RL_OBJ = Obj(params)

    # Seting some of the parameters
    MAZE_SIZE = tuple((RL_OBJ.params['env'].observation_space.high + np.ones(RL_OBJ.params['env'].observation_space.shape)).astype(int))
    MAX_T = np.prod(MAZE_SIZE, dtype=int) * 100
    NUM_EPISODES = 50000


    RL_OBJ.params['n_iter'] = NUM_EPISODES
    RL_OBJ.params['ep_len'] = MAX_T
    RL_OBJ.params['batch_size'] = 1
    RL_OBJ.params['eval_batch_size'] = NUM_EPISODES

    RL_OBJ['special']['solved_t'] = np.prod(MAZE_SIZE, dtype=int)
    RL_OBJ['special']['streak_to_end'] = 120

    RL_OBJ.run_training_loop()


if __name__ == "__main__":
    main()
