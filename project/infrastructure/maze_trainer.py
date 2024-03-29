import time
import os
import sys
import numpy as np
from collections import OrderedDict
import copy

from project.infrastructure.utils import Path, Transition, sample_trajectories, sample_trajectory, get_pathlength
from project.infrastructure.replay_buffer import NewReplayBuffer #, ReplayBuffer


#import tensorflow as tf
#from infrastructure.logger import Logger


class MazeTrainer(object):

    def __init__(self, params):

        # Initializations

        # Get params, create logger, create TF session
        self.params = params
        #self.logger = Logger(self.params['logdir'])
        #self.sess = create_tf_session(self.params['use_gpu'], which_gpu=self.params['which_gpu'])

        # # Assign the gym environment
        # self.env = env
        #
        # # Initializing Agent
        # self.agent = agent

        # Set Replay Buffer
        #self.replay_buffer = ReplayBuffer()
        self.replay_buffer = NewReplayBuffer()
 
        # Initializing TF variables
        #tf.global_variables_initializer().run(session=self.sess)

    def train(self, agent, env, n_iter):
    
        # Rendering the Maze
        if self.params['render']:
            env.render()

        # init vars at beginning of training
        self.total_envsteps = 0
        self.success_counter = 0
        #self.start_time = time.time()

        for itr in range(n_iter):
            print("\n\n********** Iteration %i ************"%itr)

            # Collect Training Transitions using the Agent
            # If online then we make a step in the enviroment and record the transition, else we simulate a 'batch_size' number of trajectories
            if self.params['online']:
                self.step_env(agent, env)
                envsteps_this_batch = 1                                   
            else:
                paths, envsteps_this_batch = self.collect_simulation_trajectories(itr, agent, env, self.params['batch_size'])
                self.replay_buffer.add_rollouts(paths)

            self.total_envsteps += envsteps_this_batch

            # Train agent (using sampled data from replay buffer)
            if itr >= self.params['training_begins']:
                start_train = True
            else:
                start_train = False

            if start_train:
                all_losses = self.run_learner(agent)

            # log/save
            # if start_train and self.logmetrics:
            if start_train:
                self.simple_training_log_function(itr, all_losses)

            if self.params['evaluate_after_each_iter']:
                self.evaluate_agent(agent, env)
            
            if self.success_counter >= self.params['streaks_to_end']:
                print("Agent has solved the maze " +str(self.params['streaks_to_end']) + " times in a row. Exiting training... ")
                break
          
            
        print("Training Finished!")
        print(" ")
        input("Continue to Evaluation...")
        
        return

    def step_env(self, agent, env):

        current_ob = copy.deepcopy(agent.get_ob())

        #print('STEP')

        #print('observe: '+str(current_ob))
        #print('q(obs): '+str(agent.policy.q_function(current_ob)))
        # Query action from agent
        action = agent.get_action(current_ob)
        #print('action: '+str(env.ACTION[action]))
        # Make a step in the enviroment
        next_ob, _, done, _ = env.step(env.ACTION[action])
        #print('next will observe: '+str(next_ob))
        #print(' ')
        # Compute the reward
        reward = agent.reward(current_ob, action)  
        if done is True:
            reward += agent.reward(next_ob, action) #collecting terminal reward
            
        # Store transition in the Buffer
        ob ,ac, rew, n_ob, term = [], [], [], [], []     
        ob.append(current_ob)
        ac.append(action)
        rew.append(reward)
        n_ob.append(next_ob)
        term.append(done)
              
        transition = Transition(ob, ac, rew, n_ob, term)
        self.replay_buffer.add_transition(transition)

        # Rendering the Maze
        if self.params['render']:
            env.render()

        # Update agent current position
        agent.accumulate_reward(reward)
        if (done is True or agent.current_t == self.params['ep_len']):   
                    
            if done:
                 print("Episode finished sucessfully after " +str(agent.get_time()) + " time steps with total reward = " +str(agent.get_accumulated_reward()) )   
                 
                 if agent.current_t <= self.params['max_sucess_time']:
                     self.success_counter += 1
                 else:
                     self.success_counter = 0 #reset the counter
                 
            else:
                print("Agent timed out after %f time steps with total reward = %f" % (agent.get_time(), agent.get_accumulated_reward()) )   
                
            agent.reset()
            agent.set_ob(env.reset())
                
        else:
            print("Agent at time step %f, with running reward = %f" % (agent.get_time(), agent.get_accumulated_reward()) )   
            
            agent.advance_time()
            agent.set_ob(next_ob)
        
        return

    ####################################
    ####################################

    def collect_simulation_trajectories(self, itr, agent, env, batch_size):

        print("\nCollecting data to be used for training...")

        # Simulate Trajectories
        paths, envsteps_this_batch = sample_trajectories(env, agent,  batch_size, self.params['ep_len'], self.params['render'])

        # Add Simulated Sample Paths to Replay Buffer

        return paths, envsteps_this_batch


    def run_learner(self, agent):
        #print('\nTraining agent using sampled data from replay buffer...')

        for train_step in range(self.params['num_agent_train_steps_per_iter']):

            ob_batch, ac_batch, re_batch, next_ob_batch, terminal_batch =  self.replay_buffer.sample_recent_data(self.params['train_batch_size'])
            
            #print(" ")
            #print('Debug Check on Learner:')
            #print("ob: " + str(ob_batch[0]) + " ac: " +str(ac_batch[0]) + " next_ob: " + str(next_ob_batch[0]) )
            #print(" ")
            
            loss = agent.train(ob_batch, ac_batch, re_batch, next_ob_batch, terminal_batch)

        return loss

    ####################################
    ####################################

    def simple_training_log_function(self, itr, loss):
        
        #print("\n Current Training Major Iteration: ", itr, " Current Training Loss: ", loss)
        print(" ")
        
        return

    def evaluate_training(self):
        
        num_of_paths = len(self.replay_buffer.path_store)
        
        path_lengths = []
        for path in self.replay_buffer.path_store:
            path_lengths.append(len(path['reward']))
              
        Summary = {  "max_path_length" :  max(path_lengths),
                     "min_path_length" :  min(path_lengths),
                     "num_of_env_steps" : sum(path_lengths),
                     "avg_path_length":  int(sum(path_lengths) / num_of_paths)
                  }
       
        print("")
        print("====================================")
        print("Number of Env steps until training finished: " +str(Summary["num_of_env_steps"]))
        print("Maximum number of steps until time out/success: " +str(Summary["max_path_length"]))
        print("Minimum number of steps until time out/success: " +str(Summary["min_path_length"]))
        print("Average number of rollout length in training: " +str(Summary["avg_path_length"]))
        print("")
     
        return Summary
   
    def evaluate_agent(self, agent, env):

         num_streaks = 0
         shortest_path = self.params['ep_len']
         
         for episode in range(self.params['eval_batch_size']):

              path = sample_trajectory(env, agent, self.params['ep_len'], self.params['render'])

              envsteps_this_episode = get_pathlength(path) - 1
                         
              if envsteps_this_episode <= self.params['max_sucess_time']:
                    num_streaks += 1
              else:
                    num_streaks = 0
            
              total_reward = path["reward"].sum()
              
              print("")
              print("======================================")
              print("Episode: " + str(episode))
              print("Total number of env steps: " +str(envsteps_this_episode))
              print("Total Reward: " + str(total_reward))
              print("Streaks: " + str(num_streaks))
              print("")

              if envsteps_this_episode >= self.params['ep_len'] - 1:
                print("Episode " +str(episode) + " timed out at " + str(envsteps_this_episode) + "with total reward = " + str(total_reward))

              if envsteps_this_episode < shortest_path:
                 shortest_path =  envsteps_this_episode
      
        
              # It's considered done when it's solved over 120 times consecutively
              if num_streaks > self.params['streaks_to_end']:
                  print("Agent has solved the maze " + str(self.params['streaks_to_end']) + " times in a row. Exiting evaluation... ")
                  break

         
         print("After the Evaluation, the current Shortest Path has lenght = " + str(shortest_path))
         print("Evaluation Finished!")
         
         return
     
         #eval_returns = [eval_path["reward"].sum() for eval_path in eval_paths]
         #eval_ep_lens = [len(eval_path["reward"]) for eval_path in eval_paths]

         #print("Avg Evaluation Returns: ", np.mean(eval_returns))
         #print("Evaluation STD on Returns: ", np.std(eval_returns))
         #print("Maximum Return on Evaluation Runs: ", np.max(eval_returns))
         #print("Minimum Return on Evaluation Runs: ",np.min(eval_returns))
         #print("Average Episode Length", np.mean(eval_ep_lens))