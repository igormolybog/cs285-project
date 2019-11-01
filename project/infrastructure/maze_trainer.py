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

        # # Assign the gym environment
        # self.env = env
        #
        # # Initializing Agent
        # self.agent = agent

        # Set Replay Buffer
        self.replay_buffer = ReplayBuffer()

        # Initializing TF variables
        #tf.global_variables_initializer().run(session=self.sess)

    def train(self, agent, env, n_iter):


        # Rendering the Maze
        if self.params['render']:
            env.render()

        # init vars at beginning of training
        self.total_envsteps = 0
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
                self.evaluate(agent, env)

        print("Training Finished!")
        return

    def step_env(self, agent, env):

        current_ob = agent.get_ob()

        print('STEP')

        print('observe: '+str(current_ob))
        print('q(obs): '+str(agent.policy.q_function(current_ob)))
        # Query action from agent
        action = agent.get_action(current_ob)
        print('action: '+str(env.ACTION[action]))
        # Make a step in the enviroment
        next_ob, _, done, info = env.step(env.ACTION[action])
        print('next will observe: '+str(next_ob))
        # Compute the reward
        reward = agent.reward(current_ob, action)

        # Store transition in the buffer
        transition = []
        transition.append(Path([current_ob], [action], [reward], [next_ob], [done]))
        self.replay_buffer.add_rollouts(transition)

        # Rendering the Maze
        if self.params['render']:
            env.render()

        # Update agent current position
        if done is True:
            reward += agent.reward(next_ob, action) #collecting terminal reward ???? this does not do anywhere
            agent.reset_time()
            agent.set_ob(env.reset())
        else:
            agent.advance_time()
            agent.set_ob(next_ob)
        return # ?????

    ####################################
    ####################################

    def collect_simulation_trajectories(self, itr, agent, env, batch_size):

        print("\nCollecting data to be used for training...")

        # Simulate Trajectories
        paths, envsteps_this_batch = sample_trajectories(agent, env, batch_size, self.params['ep_len'], self.params['render'])

        # Add Simulated Sample Paths to Replay Buffer

        return paths, envsteps_this_batch


    def run_learner(self, agent):
        print('\nTraining agent using sampled data from replay buffer...')

        for train_step in range(self.params['num_agent_train_steps_per_iter']):

            ob_batch, ac_batch, re_batch, next_ob_batch, terminal_batch =  self.replay_buffer.sample_recent_data(self.params['train_batch_size'])
            
            print(len(ob_batch))
            print("")
            check = [ob_batch[0], ac_batch[0], re_batch[0], next_ob_batch[0], terminal_batch[0]]
            print(check)
            input('Debug. Please wait...')
            
            loss = agent.train(ob_batch, ac_batch, re_batch, next_ob_batch, terminal_batch)

        return loss

    ####################################
    ####################################

    def simple_training_log_function(self, itr, loss):

        print("\n Current Training Major Iteration: ", itr, " Current Training Loss: ", loss)

        return

    def evaluate(self, agent, env):

         num_streaks = 0
         for episode in range(self.params['eval_batch_size']):

              path = sample_trajectory(env, agent, self.params['ep_len'], self.params['render'])

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
