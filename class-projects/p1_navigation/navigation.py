'''
The entry point of the project.
Usage :
	python navigation.py --train=True 
	python navigation.py --train=False
'''
from banana_env import BananaEnv
from dqn_agent import DQNAgent
import argparse
from constants import *

parser = argparse.ArgumentParser(description='This is a Deep Q Learning Project!')

def main(args):
	path_to_env = ENV_PATH
	train = args.train
	test = args.test
	render = args.render
	train_or_test = True if train else False # True if train, False otherwise
	env = BananaEnv(path_to_env, train=train_or_test, render=render)
	agent = DQNAgent(env, train=train_or_test)
	
if __name__=="__main__":
	parser.add_argument('--train', action='store_true', default=False, help='Train or Test? Default is True.')
	parser.add_argument('--test', action='store_true', default=False, help='Train or Test? Default is True.')
	parser.add_argument('--render', action='store_true', default=False)
	args = parser.parse_args()
	main(args)

