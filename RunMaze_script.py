# -*- coding: utf-8 -*-
"""
Simple script to run our project code
"""

import os
import time

from infrastructure.rl_trainer import  Maze_Trainer
from agents.maze_agent import MazeAgent

class Obj(object):

    def __init__(self, params):
        self.params = params

        train_args = {
            'num_agent_train_steps_per_iter': params['num_agent_train_steps_per_iter'],
            'train_batch_size': params['batch_size'],
        }

        self.agent_params = {**train_args, **params}

        self.params['agent_class'] = MazeAgent
        self.params['agent_params'] = self.agent_params
        self.params['train_batch_size'] = params['batch_size']
        self.params['env_wrappers'] = self.agent_params['env_wrappers']


        # Trainer
        self.rl_trainer = MazeTrainer(self.params)
        
        # Monitor
        self.monitor = None
        
    def run_training_loop(self):
        self.rl_trainer.run_training_loop(
            self.agent_params['num_timesteps'],
            collect_policy = self.rl_trainer.agent,
            eval_policy = self.rl_trainer.agent,
            )

def main():

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name',  default='Maze_Dragons-v0',
                        choices=('Maze_Dragons-v0',
                                 'Maze_Dragons-v1')
                        )

    parser.add_argument('--ep_len', type=int, default=200)
    parser.add_argument('--exp_name', type=str, default='todo')

    parser.add_argument('--eval_batch_size', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_agent_train_steps_per_iter', type=int, default=1)
    parser.add_arggument('start_training', default = 100)
    
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--use_gpu', '-gpu', action='store_true')
    parser.add_argument('--which_gpu', '-gpu_id', default=0)
    parser.add_argument('--scalar_log_freq', type=int, default=int(1e4))
    
    parser.add_argument('--online', default=True)
    parser.add_argument('--enable_recording', default = True)
    
    parser.add_argument('--save_params', action='store_true')

    args = parser.parse_args()

    # convert to dictionary
    params = vars(args)


    RL_OBJ = Obj(params)
    
    RL_OBJ.run_training_loop()


if __name__ == "__main__":
    main()