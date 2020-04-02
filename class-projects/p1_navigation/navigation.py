from unityagents import UnityEnvironment
import numpy as np
from banana_env import BananaEnv
from dqn_agent import DQNAgent
import time
import torch
# please do not modify the line below
env = BananaEnv("Banana_Linux/Banana.x86_64", train=False, render=True)

agent = DQNAgent(env, train=False)


agent.test()