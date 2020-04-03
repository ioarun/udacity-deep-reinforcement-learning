# Project Report

## About
A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana.  Thus, the goal of your agent is to collect as many yellow bananas as possible while avoiding blue bananas.  

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction.  Given this information, the agent has to learn how to best select actions.  Four discrete actions are available, corresponding to:
- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

The task is episodic, and in order to solve the environment, your agent must get an average score of +13 over 100 consecutive episodes.

## Goal
The task is episodic, and in order to solve the environment, your agent must get an average score of +13 over 100 consecutive episodes.

## Algorithm - DQN
Explain

## Architecture

## Hyperparams

## Training log
![training_visual](images/training_visual.gif)
```

Average reward after 100 episodes : 0.6 
Average reward after 200 episodes : 4.14 
Average reward after 300 episodes : 9.28 
Average reward after 400 episodes : 12.43 
Average Score: 13.04 Solved in 419 episodes!
Saving network models at episode 419

```

## Training curve
![training_plot](images/training.png)

## Test output
![testing_visual](images/testing_visual.gif)

## Plot
