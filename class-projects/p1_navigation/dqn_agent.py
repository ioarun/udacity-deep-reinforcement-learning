import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from model import QNetwork
from collections import namedtuple, deque
import random
import numpy as np
from os import path
from constants import *
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ReplayBuffer:
    def __init__(self):
        '''
        Initialize replay buffer
        '''
        self.buffer_size = REPLAY_BUFFER_SIZE
        self.buffer = []
        self.seed = random.seed(0)
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

    def append(self, experience):
        '''
        Add experience [s, a, r, s'] in the replay buffer
        '''
        if len(self.buffer) > self.buffer_size:
            self.buffer.pop(0)
        e = self.experience(experience[0], experience[1], experience[2], experience[3], \
        experience[4])
        self.buffer.append(e)

    def sample(self, n):
        '''
        Sample n [s, a, r, s'] transitions from replay buffer
        '''
        samples = random.sample(self.buffer, n)
        return samples

class DQNAgent:
    def __init__(self, env, train=True, seed=0):
        '''
        Initialize the DQN Agent.
        Train or Test depending on whether train=True or not.
        '''
        self.train = train
        self.seed = random.seed(seed)
        self.env = env
        self.action_size = self.env.action_space.n
        self.state_size = self.env.observation_space.n

        self.epsilon = EPSILON
        self.epsilon_decay = EPSILON_DECAY
        self.epsilon_min = EPSILON_MIN
        self.gamma = GAMMA
        self.batch_size = BATCH_SIZE
        self.lr = LEARNING_RATE
        self.tau = TAU

        self.running_episode = 0
        self.reward_per_episode = []
        self.qnetwork_local = QNetwork(self.state_size, self.action_size).to(device)
        self.qnetwork_target = QNetwork(self.state_size, self.action_size).to(device)

        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=self.lr)
        self.replay_buffer =  ReplayBuffer()

        # TRAIN
        if self.train:
            if path.exists(CHECKPOINT_PATH):
                model_dict, running_episode, reward_per_episode = self.load_models('train')
                self.qnetwork_local.load_state_dict(model_dict)
                self.qnetwork_target.load_state_dict(model_dict)
                self.running_episode = running_episode
                self.reward_per_episode = reward_per_episode
                print ("Models loaded successfully!")
            self.train_agent()

        # TEST
        elif (not self.train):
            if path.exists(TRAINED_MODEL_PATH):
                model_dict, running_episode, reward_per_episode = self.load_models('test')
                self.qnetwork_local.load_state_dict(model_dict)
                print ("Model loaded successfully!")
                self.test_agent()
            else:
                print ("Model not found! Check the MODEL_PATH in constants.py.")

    def save_models(self):
        '''
        Save models to a checkpoint.
        '''
        checkpoint = {
        'input_size': self.qnetwork_local.input_size,
        'output_size': self.qnetwork_local.output_size,
        'hidden_layers': self.qnetwork_local.hidden_sizes,
        'state_dict': self.qnetwork_local.state_dict(),
        'running_episode': self.running_episode,
        'reward_per_episode': self.reward_per_episode
        } 
        checkpoint['state_dict'] = self.qnetwork_target.state_dict()
        torch.save(checkpoint, CHECKPOINT_PATH)

    def load_models(self, train_or_test):
        '''
        Load models from the CHECKPOINT_PATH or
        TRAINED_MODEL_PATH depending on whether the
        task is to train or to test.
        '''
        print ("Loading models ...")
        if train_or_test == 'train':
        	checkpoint = torch.load(CHECKPOINT_PATH)
        else:
        	checkpoint = torch.load(TRAINED_MODEL_PATH)

        model_state_dict = checkpoint['state_dict']
        running_episode = checkpoint['running_episode']
        reward_per_episode = checkpoint['reward_per_episode']
        return model_state_dict, running_episode, reward_per_episode

    def epsilon_greedy_policy(self, state, epsilon):
        '''
        Choose action e-greedily. 
        '''
        if random.random() < epsilon: # exploration
            action = self.env.action_space.sample()
        else:
            self.qnetwork_local.eval()

            state = torch.from_numpy(state).float().unsqueeze(0).to(device)
            with torch.no_grad():
                qvalues = self.qnetwork_local(state).cpu().data.numpy()
            self.qnetwork_local.train()

            action = np.argmax(qvalues)

        return action

    def greedy_policy(self, state):
        self.qnetwork_local.eval()
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        with torch.no_grad():
            qvalues = self.qnetwork_local(state).cpu().data.numpy()
        action = np.argmax(qvalues)
        return action

    def update_networks(self, experiences):
        states, actions, rewards, next_states, dones = \
        torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device), \
        torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device), \
        torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device), \
        torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device), \
        torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        self.update_local_network(states, actions, rewards, next_states, dones)
        self.update_target_network(self.qnetwork_local, self.qnetwork_target)

    def update_local_network(self, states, actions, rewards, next_states, dones):
        current_expected_q_values = self.qnetwork_local(states).gather(1, actions) # (N, 1)-dim 
        target_q_values = rewards + \
        self.gamma*(self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1))*(1 - dones)
        loss = F.mse_loss(current_expected_q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


    def update_target_network(self, qnetwork_local, qnetwork_target):
        '''
        Soft update of the target network
        '''
        # self.qnetwork_target.load_state_dict(self.qnetwork_local.state_dict())
        for target_param, local_param in zip(qnetwork_target.parameters(), qnetwork_local.parameters()):
            target_param.data.copy_(self.tau*local_param.data + (1.0-self.tau)*target_param.data)

    def train_agent(self, n_episodes=N_TRAIN_EPISODES):
            update_models_every = UPDATE_MODELS_EVERY # time steps
            save_models_every = SAVE_MODELS_EVERY # episodes
            reward_buffer = deque(maxlen=REWARD_BUFFER_SIZE)
            time_step = 0
            for i in range(self.running_episode, n_episodes):
                state = self.env.reset()
                reward_sum = 0
                self.epsilon = max(self.epsilon*self.epsilon_decay, self.epsilon_min)

                while True:
                    time_step += 1
                    action = self.epsilon_greedy_policy(state, self.epsilon)
                    next_state, reward, done = self.env.step(action)
                    reward_sum += reward
                    self.replay_buffer.append([state, action, reward, next_state, done])
                    state = next_state
                    if (time_step) % update_models_every == 0:
                        if (len(self.replay_buffer.buffer) > self.batch_size):
                            sampled_experiences = self.replay_buffer.sample(self.batch_size)
                            self.update_networks(sampled_experiences)

                    if done:
                        break

                self.reward_per_episode.append(reward_sum)
                reward_buffer.append(reward_sum)
                avg_reward = np.mean(reward_buffer)
                # print('\rEpisode {}\t Average Score: {:.2f}'.format(i+1, np.mean(reward_buffer)), end="")

                self.running_episode += 1
                if (i+1) % REWARD_BUFFER_SIZE == 0:
                    print ("Average reward after {} episodes : {} ".format(i+1, avg_reward))

                # if ((i+1) % SAVE_MODELS_EVERY == 0):
                #     print ("Saving network models at episode {}".format(i+1))
                #     self.save_models()

                if (avg_reward >= AVG_REWARD_FOR_SUCCESS):
                    print ("Solved in {} episodes!".format(i+1))
                    print ("Saving network models at episode {}".format(i+1))
                    self.save_models()
                    self.plot(self.reward_per_episode, 'Training', 'images/training.png')
                    break

    def test_agent(self, n_episodes=N_TEST_EPISODES):
        reward_buffer = []
        for i in range(n_episodes):
            state = self.env.reset()
            reward_sum = 0
            while True:
                action = self.greedy_policy(state)
                next_state, reward, done = self.env.step(action)
                reward_sum += reward
                state = next_state
                if done:
                    reward_buffer.append(reward_sum)
                    self.reward_per_episode.append(reward_sum)
                    print('\rEpisode {}\t Reward: {:.2f}'.format(i+1, reward_sum), end="")
                    break
        self.plot(self.reward_per_episode, 'Testing', 'images/testing.png')
        print('\rAverage Reward: {:.2f}'.format(np.mean(np.array(reward_sum))), end="")

    def plot(self, reward_buffer, title, plot_name):
        plt.plot(reward_buffer, label='total reward')
        plt.xlabel('episode')
        plt.ylabel('total reward')
        plt.title(title)
        plt.legend()
        plt.savefig(plot_name)




