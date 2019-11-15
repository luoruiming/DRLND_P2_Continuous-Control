from unityagents import UnityEnvironment
import numpy as np
import torch
import argparse
from collections import deque
from ddpg_agent import Agent
import matplotlib.pyplot as plt

env = UnityEnvironment(file_name='Reacher_Windows_x86_64/Reacher.exe')

brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# reset the environment
env_info = env.reset(train_mode=True)[brain_name]

# number of agents
num_agents = len(env_info.agents)
print('Number of agents:', num_agents)

# size of each action
action_size = brain.vector_action_space_size
print('Size of each action:', action_size)

# examine the state space
states = env_info.vector_observations
state_size = states.shape[1]
print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))
print('The state for the first agent looks like:', states[0])

parser = argparse.ArgumentParser()
parser.add_argument('--train', dest='train', action='store_true', help='train agent locally')
parser.add_argument('--test', dest='test', action='store_true', help='test agent locally')
args = parser.parse_args()

agent = Agent(state_size=state_size, action_size=action_size, random_seed=0)

def ddpg(n_episodes=300, max_t=1000, solved_score=30.0, print_every=100):
    scores_window = deque(maxlen=print_every)
    scores = []
    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=True)[brain_name]      # reset the environment
        states = env_info.vector_observations                  # get the current state (for each agent)
        score = np.zeros(num_agents)                           # initialize the score (for each agent)
        agent.reset()
        for t in range(max_t):
            actions = agent.act(states, add_noise=True)        # select an action (for each agent)
            actions = np.clip(actions, -1, 1)                  # all actions between -1 and 1
            env_info = env.step(actions)[brain_name]           # send all actions to tne environment
            next_states = env_info.vector_observations         # get next state (for each agent)
            rewards = env_info.rewards                         # get reward (for each agent)
            dones = env_info.local_done                        # see if episode finished
            for state, action, reward, next_state, done in zip(states, actions, rewards, next_states, dones):
                agent.step(state, action, reward, next_state, done, t)
            score += rewards                                   # update the score (for each agent)
            states = next_states                               # roll over states to next time step
            if np.any(dones):                                  # exit loop if episode finished
                break
        scores_window.append(score)
        scores.append(score)
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        if i_episode % print_every == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
        if np.mean(scores_window) >= solved_score:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode - print_every,
                                                                                         np.mean(scores_window)))
            torch.save(agent.actor_local.state_dict(), 'saved_weights/checkpoint_actor.pth')
            torch.save(agent.critic_local.state_dict(), 'saved_weights/checkpoint_critic.pth')
            break
    return scores

if args.train:
    # train the agent with DDPG and save the result
    scores = ddpg()
    np.save("scores.npy", scores)

    # plot the scores
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.show()
    plt.savefig('pic/curve_20.png')

elif args.test:
    # load the saved weights
    agent.actor_local.load_state_dict(torch.load('saved_weights/checkpoint_actor.pth', map_location='cpu'))
    agent.critic_local.load_state_dict(torch.load('saved_weights/checkpoint_critic.pth', map_location='cpu'))

    env_info = env.reset(train_mode=False)[brain_name]
    states = env_info.vector_observations
    scores = np.zeros(num_agents)

    for _ in range(1000):
        actions = agent.act(states, add_noise=False)
        env_info = env.step(actions)[brain_name]
        next_states = env_info.vector_observations
        rewards = env_info.rewards
        dones = env_info.local_done
        scores += env_info.rewards
        states = next_states
        if np.any(dones):
            break
    print('Total score:', np.mean(scores))

env.close()

