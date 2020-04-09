import torch

PATH_TO_ENV = 'Tennis_Linux/Tennis.x86_64'
NUM_AGENTS = 2
STATE_SIZE = 24
ACTION_SIZE = 2

NUM_TRAIN_EPISODES = 100000
NUM_TEST_EPISODES = 100

BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 256        # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 1e-3         # learning rate of the actor
LR_CRITIC = 3e-4        # learning rate of the critic
WEIGHT_DECAY = 0        # L2 weight decay
LEARN_NUM = 3          # number of learning passes

NOISE_START=1.0
NOISE_END=0.1
NOISE_REDUCTION=0.999
EPISODES_BEFORE_TRAINING = 300
EPISODE_START = 0

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")