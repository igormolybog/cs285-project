import time
import os
import sys
import numpy as np
from collections import OrderedDict

from project.infrastructure.utils import Path, sample_trajectories, sample_trajectory, get_pathlength
from project.infrastructure.replay_buffer import ReplayBuffer


#import tensorflow as tf
#from infrastructure.logger import Logger


class MazeTrainer(object):

    def __init__(self, params):

        # Initializations
        
        # Get params, create logger, create TF session
        self.params = params
        #self.logger = Logger(self.params['logdir'])
        #self.sess = create_tf_session(self.params['use_gpu'], which_gpu=self.params['which_gpu'])
        
        # Assign the gym environment
        self.env = self.params['env']
        
        # Set Replay Buffer
        self.replay_buffer = ReplayBuffer()
        
        # Set random seeds
        self.env.seed(self.params['seed'])
        np.random.seed(self.params['seed'])
        #tf.set_random_seed(self.params['seed'])      
        #self.mean_episode_reward = -float('nan')
        #self.best_mean_episode_reward = -float('inf')
            
        # Initializing Agent
        self.agent = None

        # Initializing TF variables
        #tf.global_variables_initializer().run(session=self.sess)

    def run_training_loop(self, n_iter, collect_policy):

        
        # Rendering the Maze
        if self.params['render']:
            self.env.render()
                        
        # init vars at beginning of training
        self.total_envsteps = 0
        #self.start_time = time.time()

        for itr in range(n_iter):
            print("\n\n********** Iteration %i ************"%itr)

            # Collect Training Transitions using the Agent
            # If online then we make a step in the enviroment and record the transition, else we simulate a 'batch_size' number of trajectories
            if self.params['online']:
                self.step_env()
                envsteps_this_batch = 1
            else: 
                paths, envsteps_this_batch = self.collect_simulation_trajectories(itr, self.agent, self.params['batch_size'])
                self.replay_buffer.add_rollouts(paths)
                
            self.total_envsteps += envsteps_this_batch
        
            # Train agent (using sampled data from replay buffer)
            if itr >= self.params['training_begins']:
                start_train = True
            else:
                start_train= False
                
            if start_train:
                all_losses = self.train_agent()

            # log/save
            if start_train and self.logmetrics:
                self.simple_training_log_function(itr, all_losses)

            if self.params['evaluate_after_each_iter']:
                self.evaluate_trainer(collect_policy)
              
        print("Training Finished!")    
        return

    def step_env(self):
            
        # Query action from agent
        action = self.agent.get_action(self.agent.current_obs)
       
        # Make a step in the enviroment
        next_obs, _, done, info = self.env.step(action)
       
        # Compute the reward
        reward = self.agent.reward_function(self.agent.current_obs, action)
        
        # Store transition in the buffer
        transition = []
        transition.append(Path(self.agent.current_obs, action, reward, next_obs, done))
        self.replay_buffer.add_rollouts(transition) 

        # Rendering the Maze
        if self.params['render']:
            self.env.render()
                
        # Update agent current position    
        if done is True:          
            reward += self.agent.reward_function(next_obs, action) #collecting terminal reward
            self.agent.current_t = 0
            self.agent.current_obs = self.env.reset()
        else:
            self.agent.current_t += 1
            self.agent.current_obs = next_obs
        return
    
    
    ####################################
    ####################################

    def collect_simulation_trajectories(self, itr, collect_policy, batch_size):
        
        print("\nCollecting data to be used for training...")
        
        # Simulate Trajectories
        paths, envsteps_this_batch = sample_trajectories(self.env, collect_policy, batch_size, self.params['ep_len'], self.params['render'])

        # Add Simulated Sample Paths to Replay Buffer
        
        return paths, envsteps_this_batch


    def train_agent(self):
        print('\nTraining agent using sampled data from replay buffer...')

        for train_step in range(self.params['num_agent_train_steps_per_iter']):

            ob_batch, ac_batch, re_batch, next_ob_batch, terminal_batch =  self.replay_buffer.sample_recent_data(self.params['train_batch_size'])
                     
            loss = self.agent.train(ob_batch, ac_batch, re_batch, next_ob_batch, terminal_batch)
            
        return loss

    ####################################
    ####################################

    def simple_training_log_function(self, itr, loss):
        
        print("\n Current Training Major Iteration: ", itr, " Current Training Loss: ", loss)
        
        return
    
    
    def evaluate_trainer(self, eval_policy):
        
         num_streaks = 0
         for episode in range(self.params['eval_batch_size']):
             
              path = sample_trajectory(self.env, eval_policy, self.params['ep_len'], self.params['render'])
         
              envsteps_this_episode = get_pathlength(path)
              
              
              if envsteps_this_episode <= self.params['special']['solved_t']:
                    num_streaks += 1
              else:
                    num_streaks = 0
            
              total_reward = path["reward"].sum()
              print("")
              print("======================================")
              print("Episode: %d" % episode)
              print("Total Reward: %d" % total_reward)
              print("Streaks: %d" % num_streaks)
              print("")
                                  
              if envsteps_this_episode >= self.params['ep_len'] - 1:
                print("Episode %d timed out at %d with total reward = %f." % (episode, envsteps_this_episode, total_reward))
              

              # It's considered done when it's solved over 120 times consecutively
              if num_streaks > self.params['special']['streak_to_end']:
                  break

         
         #eval_returns = [eval_path["reward"].sum() for eval_path in eval_paths]
         #eval_ep_lens = [len(eval_path["reward"]) for eval_path in eval_paths]

         #print("Avg Evaluation Returns: ", np.mean(eval_returns))
         #print("Evaluation STD on Returns: ", np.std(eval_returns))
         #print("Maximum Return on Evaluation Runs: ", np.max(eval_returns))
         #print("Minimum Return on Evaluation Runs: ",np.min(eval_returns))
         #print("Average Episode Length", np.mean(eval_ep_lens))
         
         return
