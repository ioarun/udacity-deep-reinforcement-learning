import torch
from tennis_env import TennisEnv
from maddpg_agent import MADDPGAgent
from constants import *
import argparse
from collections import deque
import numpy as np
import json
import time

parser = argparse.ArgumentParser('This is a Tennis Project.')


def MADDPG_Training(env, maddpg_agent, n_episodes=2500, t_max=1000):
    scores_deque = deque(maxlen=100)
    scores_list = []
    scores_list_100_avg = []
    for i_episode in range(EPISODE_START, n_episodes+1):
        states = env.reset()    
                          
        maddpg_agent.reset() #reset the maddpg_agent OU Noise
        scores = np.zeros(NUM_AGENTS)                          # initialize the score (for each agent in MADDPG)
        num_steps = 0
        for _ in range(t_max):
            actions = maddpg_agent.act(states, i_episode)
            next_states, rewards, dones = env.step(actions)           # send all actions to the environment
            scores += rewards                                  # update the score (for each agent in MADDPG)
            maddpg_agent.step(i_episode, states, actions, rewards, next_states, dones) #train the MADDPG_obj           
            states = next_states                               # roll over states to next time step
            num_steps += 1
            if np.any(dones):                                  # exit loop if episode finished
                break
            #print('Total score (averaged over agents) this episode: {}'.format(np.mean(score)))
        
        scores_deque.append(np.max(scores))
        scores_list.append(np.max(scores))
        scores_list_100_avg.append(np.mean(scores_deque))
        
        print('Episode {}\tAverage Score: {:.2f}\tCurrent Score: {}'.format(i_episode, np.mean(scores_deque), np.max(scores)))
        print('Noise Scaling: {}, Memory size: {} and Num Steps: {}'.format(maddpg_agent.agents[0].noise_scale, len(maddpg_agent.replay_buffer), num_steps))
        
        if i_episode % 500 == 0:
            maddpg_agent.save_maddpg()
            print('Saved Model: Episode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))
            
        if np.mean(scores_deque) > 0.5 and len(scores_deque) >= 100:
            maddpg_agent.save_maddpg()
            print('Saved Model: Episode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))
            break
            
    return scores_list, scores_list_100_avg

def MADDPG_Testing(env, maddpg_agent, n_episodes=10, t_max=1000):
	maddpg_agent.load_maddpg()
	scores_list = []
	for i_episode in range(n_episodes+1):
	    states = env.reset()    
	                      
	    maddpg_agent.reset() #reset the maddpg_agent OU Noise
	    scores = np.zeros(NUM_AGENTS)                          # initialize the score (for each agent in MADDPG)
	    num_steps = 0
	    for _ in range(t_max):
	        actions = maddpg_agent.act(states, i_episode, add_noise=False)
	        next_states, rewards, dones = env.step(actions)           # send all actions to the environment
	        scores += rewards                                  # update the score (for each agent in MADDPG)
	        states = next_states                               # roll over states to next time step
	        num_steps += 1
	        
	        if np.any(dones):                                  # exit loop if episode finished
	            break
	        #print('Total score (averaged over agents) this episode: {}'.format(np.mean(score)))
	    
	    scores_list.append(np.max(scores))
	    
	    print('Episode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(np.array(scores_list))))
	    

	return scores_list

def main(args):
	render = args.render
	train = args.train
	env = TennisEnv(PATH_TO_ENV, render=render)
	maddpg_agent = MADDPGAgent(STATE_SIZE, ACTION_SIZE, 123)
	
	if train:
	    scores_list, scores_list_100_avg = MADDPG_Training(env, maddpg_agent, n_episodes=NUM_TRAIN_EPISODES)
	    with open("scores.json", "w") as write_file:
	        json.dump((scores_list, scores_list_100_avg), write_file)
	else:
		scores_list = MADDPG_Testing(env, maddpg_agent, n_episodes=NUM_TEST_EPISODES)

if __name__=='__main__':
    parser.add_argument('--train', action='store_true', default=False)
    parser.add_argument('--render', action='store_true', default=False)
    args = parser.parse_args()
    main(args)