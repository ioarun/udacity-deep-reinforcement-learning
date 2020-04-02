import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from model import QNetwork
from collections import namedtuple, deque
import random
import numpy as np
from os import path

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ReplayBuffer:
    def __init__(self):
        self.buffer_size = 1e5
        self.buffer = []
        self.seed = random.seed(0)
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

    def append(self, experience):
        if len(self.buffer) > self.buffer_size:
            self.buffer.pop(0)
        e = self.experience(experience[0], experience[1], experience[2], experience[3], \
        experience[4])
        self.buffer.append(e)

    def sample(self, n):
        samples = random.sample(self.buffer, n)
        
        return samples


class DQNAgent:
    def __init__(self, env, train=True, seed=0):
        self.train = train
        self.seed = random.seed(seed)
        self.env = env
        self.action_size = self.env.action_space.n
        self.state_size = self.env.observation_space.n

        self.epsilon = 1.0
        self.epsilon_decay = 0.99
        self.epsilon_min = 0.01
        self.gamma = 0.99
        self.batch_size = 64
        self.lr = 0.0005
        self.tau = 1e-3

        self.running_episode = 0
        self.reward_per_episode = 0.0
        self.qnetwork_local = QNetwork(self.state_size, self.action_size).to(device)
        self.qnetwork_target = QNetwork(self.state_size, self.action_size).to(device)

        if path.exists('qnetwork_local_checkpoint.pth'):
            qnetwork_local_dict, qnetwork_target_dict, running_episode = self.load_models()
            self.qnetwork_local.load_state_dict(qnetwork_local_dict)
            self.qnetwork_target.load_state_dict(qnetwork_target_dict)
            self.running_episode = running_episode
            print ("Models loaded successfully!")
            
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=self.lr)

        self.replay_buffer =  ReplayBuffer()


    
    def save_models(self):
        qnetwork_local_checkpoint = {
        'input_size': self.qnetwork_local.input_size,
        'output_size': self.qnetwork_local.output_size,
        'hidden_layers': self.qnetwork_local.hidden_sizes,
        'state_dict': self.qnetwork_local.state_dict(),
        'running_episode': self.running_episode,
        'reward_per_episode': self.reward_per_episode
        } 
        qnetwork_target_checkpoint = qnetwork_local_checkpoint
        qnetwork_target_checkpoint['state_dict'] = self.qnetwork_target.state_dict()
        torch.save(qnetwork_local_checkpoint, 'qnetwork_local_checkpoint.pth')
        torch.save(qnetwork_target_checkpoint, 'qnetwork_target_checkpoint.pth')
    
    def load_models(self):
        print ("Loading models ...")
        local = torch.load('qnetwork_local_checkpoint.pth')['state_dict']
        target = torch.load('qnetwork_target_checkpoint.pth')
        return local, target['state_dict'], target['running_episode'] 

    def epsilon_greedy_policy(self, state, epsilon):
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

    def update_local_network(self, experiences):
        # experiences = np.array(experiences).reshape(len(experiences), len(experiences[0]))
        
        states, actions, rewards, next_states, dones = \
        torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device), \
        torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device), \
        torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device), \
        torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device), \
        torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        current_expected_q_values = self.qnetwork_local(states).gather(1, actions) # (N, 1)-dim 
        target_q_values = rewards + \
        self.gamma*(self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1))*(1 - dones)
        loss = F.mse_loss(current_expected_q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.update_target_network(self.qnetwork_local, self.qnetwork_target)

        return loss

    def update_target_network(self, qnetwork_local, qnetwork_target):
        # self.qnetwork_target.load_state_dict(self.qnetwork_local.state_dict())
        for target_param, local_param in zip(qnetwork_target.parameters(), qnetwork_local.parameters()):
            target_param.data.copy_(self.tau*local_param.data + (1.0-self.tau)*target_param.data)

    def train(self, n_episodes=5000):
            update_local_every = 5 # time steps
            update_target_every = 10 # time steps
            save_models_every = 10 # episodes
            reward_buffer = deque(maxlen=100)
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
                    if (time_step) % update_local_every == 0:
                        if (len(self.replay_buffer.buffer) > self.batch_size):
                            sampled_experiences = self.replay_buffer.sample(self.batch_size)
                            loss = self.update_local_network(sampled_experiences)
                            
                    if done:
                        self.reward_per_episode = reward_sum
                        reward_buffer.append(reward_sum)
                        avg_reward = np.mean(reward_buffer)
                        print('\rEpisode {}\t Average Score: {:.2f}'.format(i+1, np.mean(reward_buffer)), end="")

                        self.running_episode += 1
                        if (i+1) % 100 == 0:
                            print (" Average reward after {} episodes : {} ".format(i+1, avg_reward))
                        # reward_buffer = []
                        if (avg_reward >= 13.0):
                            print (" Solved in {} episodes!".format(i+1))
                            print (" Saving network models at episode {}".format(i+1))
                            self.save_models()

                        break

    def test(self, n_episodes=10):
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
                    print('\rEpisode {}\t Reward: {:.2f}'.format(i+1, reward_sum), end="")
                    break

        print('\rAverage Reward: {:.2f}'.format(np.mean(np.array(reward_sum))), end="")



