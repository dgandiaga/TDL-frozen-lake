import argparse
import time
from itertools import count
import math

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import gym

from agent import TDLAgent

parser = argparse.ArgumentParser(description='Train an agent for different versions of frozen-lake')
parser.add_argument('--render', action='store_true', help='render the environment')
parser.add_argument('--model', type=str, help='model name', required=True)
parser.add_argument('--episodes', type=int, default=10, help='number of episodes')
parser.add_argument('--env_size', type=int, choices=[4, 8], help='size of the environment grid', default=8)
parser.add_argument('--deterministic', action='store_true', help='Make the env deterministic')
args = parser.parse_args()

if __name__ == "__main__":

    if args.env_size == 4:
        env = gym.make('FrozenLake-v1', is_slippery=(not args.deterministic))
    elif args.env_size == 8:
        env = gym.make('FrozenLake8x8-v1', is_slippery=(not args.deterministic))

    agent = TDLAgent(env.action_space.n, trainable=False)
    agent.load_policy(f'{args.model}')

    episode_rewards = []
    accumulated_reward = []
    results = pd.DataFrame(columns=['episode', 'reward', 'avg_reward'])

    for i_ep in range(args.episodes):
        episode_rewards.append(0)
        state = env.reset()
        action = agent.select_action(state)
        if args.render:
            env.render()
            time.sleep(1)

        for t in count():

            next_state, reward, done, _ = env.step(action)
            if args.render:
                env.render()
                if done:
                    if reward == 1:
                        print('SUCCESS')
                    else:
                        print('FAIL')
                time.sleep(1)

            next_action = agent.select_action(next_state)

            episode_rewards[-1] += reward
            state = next_state
            action = next_action

            if done:
                break

    print(f'Average reward: {np.sum(episode_rewards) / len(episode_rewards)}')
