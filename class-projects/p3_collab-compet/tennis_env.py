# Wrapper for the Tennis Environment - OpenAI Gym style!

from unityagents import UnityEnvironment
import numpy as np
from constants import *

class ActionSpace:
    def __init__(self, num_agents, action_size):
        self.n = (num_agents, action_size)

    def sample(self):
        actions = np.random.randn(self.n[0], self.n[1])
        return np.clip(actions, -1, 1)
    

class ObservationSpace:
    def __init__(self, num_agents, obs_size):
        self.n = (num_agents, obs_size)

class TennisEnv:
    def __init__(self, path_to_env, train=True, render=False):
        if (render):
            no_graphics = False
        else:
            no_graphics = True
        self.env_ = UnityEnvironment(file_name=path_to_env, no_graphics=no_graphics)
        self.brain_name = self.env_.brain_names[0] # get the default brain
        self.brain = self.env_.brains[self.brain_name]
        self.action_space = ActionSpace(NUM_AGENTS, self.brain.vector_action_space_size)
        self.observation_space = ObservationSpace(NUM_AGENTS, STATE_SIZE)
        self.train = train
        
    def step(self, action):
        env_info = self.env_.step(action)[self.brain_name]
        next_state = env_info.vector_observations # get the next state
        reward = env_info.rewards # get the reward
        done = env_info.local_done # end of episode ?
        return next_state, reward, done

    def reset(self):
        env_info = self.env_.reset(train_mode=self.train)[self.brain_name]
        next_state = env_info.vector_observations
        return next_state
   
       
        