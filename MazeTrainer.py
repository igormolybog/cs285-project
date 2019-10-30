import time

from collections import OrderedDict
import pickle
import numpy as np
import tensorflow as tf
import gym
import os
import sys

from gym import wrappers
from gym.wrappers.monitor import Monitor

from infrastructure.utils import *
from infrastructure.tf_utils import create_tf_session
from infrastructure.logger import Logger
from infrastructure.replay_buffer import ReplayBuffer

#register all of our envs
import cs285.envs

from cs285.agents.dqn_agent import DQNAgent
from cs285.infrastructure.dqn_utils import get_wrapper_by_name

# how many rollouts to save as videos to tensorboard
MAX_NVIDEO = 2
MAX_VIDEO_LEN = 40 # we overwrite this in the code below


class RL_Trainer(object):

    def __init__(self, params):

        #############
        ## INIT
        #############

        # Get params, create logger, create TF session
        self.params = params
        self.logger = Logger(self.params['logdir'])
        self.sess = create_tf_session(self.params['use_gpu'], which_gpu=self.params['which_gpu'])
        
        # Set Replay Buffer
        self.replay_buffer = ReplayBuffer()
        
        # Set random seeds
        self.env.seed(self.params['seed'])
        tf.set_random_seed(self.params['seed'])
        np.random.seed(self.params['seed'])
                 
        self.mean_episode_reward = -float('nan')
        self.best_mean_episode_reward = -float('inf')
        
        
        #############
        ## ENV
        #############

        # Make the gym environment
        env_wrap = gym.make(self.params['env_name'])
        self.env = env_wrap.env

        
        # Recording or Not
        if  self.params['enable_recording']:
            self.monitor = Monitor(self.env, self.params['recording_folder'], force=True)
            
        # Maximum length for episodes
        self.params['ep_len'] = self.params['ep_len']
    
        # Enviroment Dimension details
        # Is this env continuous, or self.discrete?
        self.params['agent_params']['discrete'] = True

        # Observation and action sizes
        self.params['agent_params']['ac_dim'] = self.env.action_space.n
        self.params['agent_params']['ob_dim'] = self.env.observation_space.shape

        #############
        ## AGENT
        #############
        agent_class = self.params['agent_class']
        self.agent = agent_class(self.params['agent_params'])

        #############
        ## INIT VARS
        #############
        tf.global_variables_initializer().run(session=self.sess)


    def step_env(self):
            
        # Query action from agent
        action = self.agent.compute_action(self.agent.current_obs)
       
        # Make a step in the enviroment
        next_obs, _, done, info = self.env.step(action)
       
        # Compute the reward
        reward = self.agent.reward_function(self.agent.current_obs, action)
        
        # Store transition in the buffer
        transition = []
        transition. append(Path(self.agent.current_obs, action, reward, next_obs, done))
        self.replay_buffer.add_rollouts(transition) 

        # Update agent current position
        if done is True:
            self.agent.current_obs = self.env.reset()
        else:
            self.agent.current_obs = next_obs
        
        return
    
    def run_training_loop(self, n_iter, collect_policy, eval_policy,
                          initial_expertdata=None, relabel_with_expert=False,
                          start_relabel_with_expert=1, expert_policy=None):
        """
        :param n_iter:  number of (dagger) iterations
        :param collect_policy:
        :param eval_policy:
        :param initial_expertdata:
        :param relabel_with_expert:  whether to perform dagger
        :param start_relabel_with_expert: iteration at which to start relabel with expert
        :param expert_policy:
        """

        # init vars at beginning of training
        self.total_envsteps = 0
        self.start_time = time.time()

        for itr in range(n_iter):
            print("\n\n********** Iteration %i ************"%itr)

            # Collect Training Transitions using the Agent
            # If online then we make a step in the enviroment and record the transition, else we simulate a 'batch_size' number of trajectories
            if self.params['online']:
                self.step_env
                envsteps_this_batch = 1
            else: 
                envsteps_this_batch = self.collect_training_trajectories(itr, self.agent, self.params['batch_size'])
                            
            self.total_envsteps += envsteps_this_batch


            # Train agent (using sampled data from replay buffer)
            all_losses = self.train_agent()

            #if self.params['logdir'].split('/')[-1][:2] == 'mb' and itr==0:
            if itr == 0:
                self.log_model_predictions(itr, all_losses)

            # log/save
            if self.logmetrics:
                print('\nBeginning logging procedure...')
                self.perform_logging(itr, paths, eval_policy, train_video_paths, all_losses)

            return

    ####################################
    ####################################

    def collect_training_trajectories(self, itr, collect_policy, batch_size):
        
        print("\nCollecting data to be used for training...")
        
        # Simulate Trajectories
        paths, envsteps_this_batch = sample_trajectories(self.env, collect_policy, batch_size, self.params['ep_len'])

        # Add Simulated Sample Paths to Replay Buffer
        self.replay_buffer.add_rollouts(paths)
        
        return envsteps_this_batch


    def train_agent(self):
        print('\nTraining agent using sampled data from replay buffer...')
        total_loss = []
        for train_step in range(self.params['num_agent_train_steps_per_iter']):

            ob_batch, ac_batch, re_batch, next_ob_batch, terminal_batch =  self.replay_buffer.sample_recent_data(self.params['train_batch_size'])
                     
            loss = self.agent.train(ob_batch, ac_batch, re_batch, next_ob_batch, terminal_batch)
            
            total_loss.append(loss)
        
        return total_loss

    ####################################
    ####################################

    def perform_logging(self, itr, paths, eval_policy, train_video_paths, all_losses):

        loss = all_losses[-1]

        # collect eval trajectories, for logging
        print("\nCollecting data for eval...")
        eval_paths, eval_envsteps_this_batch = sample_trajectories(self.env, eval_policy, self.params['eval_batch_size'], self.params['ep_len'])

        # save eval rollouts as videos in tensorboard event file
        if self.logvideo and train_video_paths != None:
            print('\nCollecting video rollouts eval')
            eval_video_paths = sample_n_trajectories(self.env, eval_policy, MAX_NVIDEO, MAX_VIDEO_LEN, True)

            #save train/eval videos
            print('\nSaving train rollouts as videos...')
            self.logger.log_paths_as_videos(train_video_paths, itr, fps=self.fps, max_videos_to_save=MAX_NVIDEO,
                                            video_title='train_rollouts')
            self.logger.log_paths_as_videos(eval_video_paths, itr, fps=self.fps,max_videos_to_save=MAX_NVIDEO,
                                             video_title='eval_rollouts')

        # save eval metrics
        if self.logmetrics:
            # returns, for logging
            train_returns = [path["reward"].sum() for path in paths]
            eval_returns = [eval_path["reward"].sum() for eval_path in eval_paths]

            # episode lengths, for logging
            train_ep_lens = [len(path["reward"]) for path in paths]
            eval_ep_lens = [len(eval_path["reward"]) for eval_path in eval_paths]

            # decide what to log
            logs = OrderedDict()
            logs["Eval_AverageReturn"] = np.mean(eval_returns)
            logs["Eval_StdReturn"] = np.std(eval_returns)
            logs["Eval_MaxReturn"] = np.max(eval_returns)
            logs["Eval_MinReturn"] = np.min(eval_returns)
            logs["Eval_AverageEpLen"] = np.mean(eval_ep_lens)

            logs["Train_AverageReturn"] = np.mean(train_returns)
            logs["Train_StdReturn"] = np.std(train_returns)
            logs["Train_MaxReturn"] = np.max(train_returns)
            logs["Train_MinReturn"] = np.min(train_returns)
            logs["Train_AverageEpLen"] = np.mean(train_ep_lens)

            logs["Train_EnvstepsSoFar"] = self.total_envsteps
            logs["TimeSinceStart"] = time.time() - self.start_time
            if isinstance(loss, dict):
                logs.update(loss)
            else:
                logs["Training loss"] = loss

            if itr == 0:
                self.initial_return = np.mean(train_returns)
            logs["Initial_DataCollection_AverageReturn"] = self.initial_return

            # perform the logging
            for key, value in logs.items():
                print('{} : {}'.format(key, value))
                self.logger.log_scalar(value, key, itr)
            print('Done logging...\n\n')

            self.logger.flush()


    def log_model_predictions(self, itr, all_losses):
        # model predictions

        import matplotlib.pyplot as plt
        self.fig = plt.figure()

        # sample actions
        action_sequence = self.agent.actor.sample_action_sequences(num_sequences=1, horizon=10) #20 reacher
        action_sequence = action_sequence[0]

        # calculate and log model prediction error
        mpe, true_states, pred_states = calculate_mean_prediction_error(self.env, action_sequence, self.agent.dyn_models, self.agent.actor.data_statistics)
        assert self.params['agent_params']['ob_dim'] == true_states.shape[1] == pred_states.shape[1]
        ob_dim = self.params['agent_params']['ob_dim']

        # skip last state for plotting when state dim is odd
        if ob_dim%2 == 1:
            ob_dim -= 1

        # plot the predictions
        self.fig.clf()
        for i in range(ob_dim):
            plt.subplot(ob_dim/2, 2, i+1)
            plt.plot(true_states[:,i], 'g')
            plt.plot(pred_states[:,i], 'r')
        self.fig.suptitle('MPE: ' + str(mpe))
        self.fig.savefig(self.params['logdir']+'/itr_'+str(itr)+'_predictions.png', dpi=200, bbox_inches='tight')

        # plot all intermediate losses during this iteration
        np.save(self.params['logdir']+'/itr_'+str(itr)+'_losses.npy', all_losses)
        self.fig.clf()
        plt.plot(all_losses)
        self.fig.savefig(self.params['logdir']+'/itr_'+str(itr)+'_losses.png', dpi=200, bbox_inches='tight')