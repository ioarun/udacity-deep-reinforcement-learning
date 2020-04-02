# Wrapper for the Banana Environment - OpenAI Gym style!

from unityagents import UnityEnvironment
import numpy as np

class ActionSpace:
    def __init__(self, action_size):
        self.n = action_size

    def sample(self):
        return np.random.choice(np.arange(self.n))

class ObservationSpace:
    def __init__(self, obs_size):
        self.n = obs_size

class BananaEnv:
    def __init__(self, path_to_env, train=True, render=False):
        if render:
            no_graphics = False
        else:
            no_graphics = True
        self.env_ = UnityEnvironment(file_name=path_to_env, no_graphics=no_graphics)
        self.brain_name = self.env_.brain_names[0] # get the default brain
        self.brain = self.env_.brains[self.brain_name]
        self.action_space = ActionSpace(self.brain.vector_action_space_size)
        self.observation_space = ObservationSpace(37)
        self.train = train
        
    def step(self, action):
        env_info = self.env_.step(action)[self.brain_name]
        next_state = env_info.vector_observations[0] # get the next state
        reward = env_info.rewards[0] # get the reward
        done = env_info.local_done[0] # end of episode ?
        return next_state, reward, done

    def reset(self):
        env_info = self.env_.reset(train_mode=self.train)[self.brain_name]
        next_state = env_info.vector_observations[0]
        return next_state
   
       
        