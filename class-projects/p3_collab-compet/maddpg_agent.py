import numpy as np
import random
import copy
from collections import namedtuple, deque

from model import Actor, Critic
from ddpg_agent import DDPGAgent
from replay_buffer import ReplayBuffer
from constants import *

import torch
import torch.nn.functional as F
import torch.optim as optim

class MADDPGAgent():
    def __init__(self, state_size, action_size, random_seed, train=False):
        torch.manual_seed(random_seed)
        self.train = train
        self.state_size = state_size
        self.action_size = action_size
        self.action_size_full = NUM_AGENTS*action_size
        self.replay_buffer = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE) # memory
        self.agents = [DDPGAgent(state_size, action_size, NUM_AGENTS, _, train) for _ in range(NUM_AGENTS)]
        self.episodes_before_training = EPISODES_BEFORE_TRAINING
        
    def reset(self):
        for agent in self.agents:
            agent.reset()

    def step(self, i_episode, states, actions, rewards, next_states, dones):
        """Save experience in replay memory, and use random sample from buffer to learn."""

        full_states = np.reshape(states, newshape=(-1))
        full_next_states = np.reshape(next_states, newshape=(-1))

        # Save experience / reward
        self.replay_buffer.add(full_states, states, actions, rewards, full_next_states, next_states, dones)

        # Learn, if enough samples are available in memory
        if len(self.replay_buffer) > BATCH_SIZE and i_episode > self.episodes_before_training:
            for _ in range(LEARN_NUM): #learn multiple times at every step
                for agent_id in range(NUM_AGENTS):
                    samples = self.replay_buffer.sample()
                    self.learn(samples, agent_id, GAMMA)
                self.soft_update_all()

    def learn(self, samples, agent_id, gamma):
        #for learning MADDPG
        full_states, states, actions, rewards, full_next_states, next_states, dones = samples

        critic_full_next_actions = torch.zeros(states.shape[:2] + (self.action_size,), dtype=torch.float, device=DEVICE)
        for agent_idx, agent in enumerate(self.agents):
            agent_next_state = next_states[:,agent_idx,:]
            critic_full_next_actions[:,agent_idx,:] = agent.actor_target.forward(agent_next_state)
        critic_full_next_actions = critic_full_next_actions.view(-1, self.action_size_full)

        agent = self.agents[agent_id]
        agent_state = states[:,agent_id,:]
        actor_full_actions = actions.clone() #create a deep copy
        actor_full_actions[:,agent_id,:] = agent.actor_local.forward(agent_state)
        actor_full_actions = actor_full_actions.view(-1, self.action_size_full)

        full_actions = actions.view(-1,self.action_size_full)

        agent_rewards = rewards[:,agent_id].view(-1,1) #gives wrong result without doing this
        agent_dones = dones[:,agent_id].view(-1,1) #gives wrong result without doing this
        experiences = (full_states, actor_full_actions, full_actions, agent_rewards, \
                       agent_dones, full_next_states, critic_full_next_actions)
        agent.learn(experiences, gamma)

    def soft_update_all(self):
        #soft update all the agents            
        for agent in self.agents:
            agent.soft_update_all()


    def act(self, full_states, i_episode, add_noise=True):
        # all actions between -1 and 1
        actions = []
        for agent_id, agent in enumerate(self.agents):
            action = agent.act(np.reshape(full_states[agent_id,:], newshape=(1,-1)), i_episode, add_noise)
            action = np.reshape(action, newshape=(1,-1))            
            actions.append(action)
        actions = np.concatenate(actions, axis=0)
        return actions

    def save_maddpg(self):
        for agent_id, agent in enumerate(self.agents):
            torch.save(agent.actor_local.state_dict(), 'checkpoint/checkpoint_actor_local_' + str(agent_id) + '.pth')
            torch.save(agent.critic_local.state_dict(), 'checkpoint/checkpoint_critic_local_' + str(agent_id) + '.pth')

    def load_maddpg(self):
        for agent_id, agent in enumerate(self.agents):
            #Since the model is trained on gpu, need to load all gpu tensors to cpu:
            agent.actor_local.load_state_dict(torch.load('checkpoint/checkpoint_actor_local_' + str(agent_id) + '.pth', map_location=lambda storage, loc: storage))
            agent.critic_local.load_state_dict(torch.load('checkpoint/checkpoint_critic_local_' + str(agent_id) + '.pth', map_location=lambda storage, loc: storage))

            agent.noise_scale = NOISE_END #initialize to the final epsilon value upon training





