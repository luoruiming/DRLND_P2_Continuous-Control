## Introduction

This is my report of project 2: Continuous Control. In this project, I trained a double-jointed arm to move towards target locations and maintain its position for as many as time steps as possible.  The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with 4 numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1. 

I chose the second version of the environment which contains 20 identical agents. The problem is considered solved when the average (across 100 consecutive episodes) of those average scores from 20 agents is at least +30. 

## Algorithm

I solved the problem using Deep Deterministic Policy Gradient ([DDPG]( https://arxiv.org/abs/1509.02971 )).

Pseudocode of DDPG:

![ddpg](F:\DRLND\deep-reinforcement-learning\p2_continuous-control\DRLND_P2_Continuous-Control\pic\ddpg.png)

Hyperparameters:

buffer size: 1000000

batch size: 128

$\gamma$ (discounting rate): 0.99

$\tau$ (soft update): 0.001

learning rate for actor: 0.001

learning rate for critic: 0.001

number of time steps that model learns: 20

noise parameters: 0.2 ($\sigma$), 0.15 ($\theta$), 1.0 ($\epsilon$), 1e-6 ($\epsilon$ -decay)



Architecture of the actor network:

input layer (#33) -> hidden layer (#400) -> batch normalization -> hidden layer (#300) -> output layer (#4)

All fully connected layers use ReLU for activation except for the output layer which uses tanh for activation.



Architecture of the critic network:

input layer (#33) -> hidden layer (#400)  -> batch normalization ->  concatenate actions -> hidden layer (#300) -> output layer (#1)

All fully connected layers use ReLU for activation except for the output layer which doesn't use an  activation.

## Result

![curve](F:\DRLND\deep-reinforcement-learning\p2_continuous-control\DRLND_P2_Continuous-Control\pic\curve_20.png)

As shown above, the score achieved the target (30) in less than 40 episodes. And the agents stop improving after around 50th episode.

## Future work

Implement PPO, A3C and D4PG, and then compare the results.

