ENV_PATH = 'Banana_Linux/Banana.x86_64'
CHECKPOINT_PATH = 'checkpoint/checkpoint.pth'
TRAINED_MODEL_PATH = 'trained_model/model.pth'

N_TRAIN_EPISODES = 5000
REPLAY_BUFFER_SIZE = 1e5
REWARD_BUFFER_SIZE = 100
BATCH_SIZE = 64
EPSILON = 1.0
EPSILON_DECAY = 0.99
EPSILON_MIN = 0.01
GAMMA = 0.99
LEARNING_RATE = 0.0005
TAU = 1e-3

UPDATE_MODELS_EVERY = 5 # every 5 time steps
SAVE_MODELS_EVERY = 10 # every 10 episodes

AVG_REWARD_FOR_SUCCESS = 13.0
N_TEST_EPISODES = 100

PLOT = True