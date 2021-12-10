import argparse
from itertools import count
import math

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import gym

from agent import TDLAgent

parser = argparse.ArgumentParser(description='Train an agent for different versions of frozen-lake')
parser.add_argument('--gamma', type=float, default=0.99, help='discount factor (default: 0.9)')
parser.add_argument('--render', action='store_true', help='render the environment')
parser.add_argument('--tag', type=str, required=True, help='tag for saving results')
parser.add_argument('--episodes', type=int, default=1000, help='number of episodes')
parser.add_argument('--env-size', type=int, choices=[4, 8], help='size of the environment grid', default=8)
parser.add_argument('--log-interval', type=int, help='interval between training status logs (default: episodes/10)')
parser.add_argument('--deterministic', action='store_true', help='Make the env deterministic')
parser.add_argument('--default-q-value', type=float, default=10, help='Default initialization for q values')
parser.add_argument('--alpha', type=float, default=0.2, help='learning rate')
parser.add_argument('--eps-decay', type=float, default=0.999, help='epsilon decay')
parser.add_argument('--eps-min', type=float, default=0.05, help='epsilon minimum')
args = parser.parse_args()


def train(env, episodes, render, default_q_value, eps_decay, eps_min, gamma, alpha, log_interval=0, tag='', save=True):

    # Initialize agent
    agent = TDLAgent(env.action_space.n, default_q_value, eps_decay, eps_min, gamma, alpha, trainable=True)

    # Initialize result arrays
    episode_rewards = []
    accumulated_reward = []
    results = pd.DataFrame(columns=['episode', 'reward', 'avg_reward'])
    best_accumulated_reward = 0

    for i_ep in range(episodes):

        # Start env and choose first action
        state = env.reset()
        action = agent.select_action(state)

        # Initialize reward for this episode
        episode_rewards.append(0)
        if render:
            env.render()

        for t in count():

            # Take action
            next_state, reward, done, _ = env.step(action)
            if render:
                env.render()

            # Choose next action
            next_action = agent.select_action(next_state)

            # Update model according to tuple (state, action, next_state, next_action, done)
            agent.update(state, action, reward, next_state, next_action, done)

            # Append reward and update state and action
            episode_rewards[-1] += reward
            state = next_state
            action = next_action

            # Finish
            if done:
                break

        # Append accumulated reward for plotting
        if len(episode_rewards) < 100:
            accumulated_reward.append(np.mean(episode_rewards))
        else:
            accumulated_reward.append(np.mean(episode_rewards[-100:]))

        # Save data from episode
        results.loc[len(results)] = [i_ep, episode_rewards[-1], accumulated_reward[-1]]

        # If accumulated reward over 100 episodes improves, save model to disk
        if (best_accumulated_reward < accumulated_reward[-1]) and save:
            agent.save_policy(tag)
            best_accumulated_reward = accumulated_reward[-1]

        # Print status and update accumulated reward plot
        if (i_ep % log_interval == 0) and save:
            plt.figure()
            plt.plot(episode_rewards, label='rewards')
            plt.plot(accumulated_reward, label='averaged_reward_over_100_runs')
            plt.title(f'Rewards over time - {tag}')
            plt.xlabel('Episode')
            plt.ylabel('Reward')
            plt.legend()
            plt.savefig(f'results/{tag}.png')
            print(
                f'Episode {i_ep} finished in {t} steps, reward = {reward}, epsilon = {agent.epsilon}'
                f', accumulated_reward = {accumulated_reward[-1]}')

            results.to_csv(f'results/{tag}.csv', index=False)

        # Update epsilon
        agent.update_epsilon()

    return episode_rewards, accumulated_reward


if __name__ == "__main__":

    # Initialize environment
    if args.env_size == 4:
        env = gym.make('FrozenLake-v1', is_slippery=(not args.deterministic))
    elif args.env_size == 8:
        env = gym.make('FrozenLake8x8-v1', is_slippery=(not args.deterministic))

    # Define log interval
    if args.log_interval is None:
        log_interval = math.floor(args.episodes / 10)
    else:
        log_interval = args.log_interval

    # Start training
    train(env, args.episodes, args.render, args.default_q_value, args.eps_decay, args.eps_min, args.gamma, args.alpha,
          log_interval, args.tag)
