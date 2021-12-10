import numpy as np
import pickle


class TDLAgent:

    def __init__(self, action_range, q_init=2, eps_decay=0.99, eps_min=0.1, gamma=0.99, alpha=0.1, trainable=True):

        self.policy = {}
        self.q_init = q_init
        self.action_range = action_range
        self.eps_decay = eps_decay
        self.eps_min = eps_min
        self.gamma = gamma
        self.alpha = alpha

        # If model is set as non-trainable, remove exploration
        if trainable:
            self.epsilon = 1
            self.eps_min = eps_min
        else:
            self.epsilon = 0
            self.eps_min = 0

    # Initialize q-values when a new state is visited
    def create_state_if_not_exists(self, state):
        if state not in self.policy.keys():
            self.policy[state] = [self.q_init] * self.action_range

    # Choose random action if random > epsilon, otherwise take greedy action
    def select_action(self, state):
        self.create_state_if_not_exists(state)
        if np.random.random() > self.epsilon:
            action = np.argmax(self.policy[state])
        else:
            action = np.random.choice(range(self.action_range))
        return action

    # Save policy to disk
    def save_policy(self, tag):
        save_file = open(f"models/{tag}.pkl", "wb")
        pickle.dump(self.policy, save_file)

    # Initialize policy from disk
    def load_policy(self, name):
        load_file = open(f"models/{name}.pkl", "rb")
        self.policy = pickle.load(load_file)

    # Update epsilon with epsilon-decay after each episode
    def update_epsilon(self):
        self.epsilon = np.max([self.epsilon * self.eps_decay, self.eps_min])

    # Update followint SARSA rule
    def update(self, state, action, reward, next_state, next_action, done):

        # Save current Q value for old state action pair
        old_q_value = self.policy[state][action]

        # New Q observed equals to reward if it's a terminal state, otherwise it's the reward plus gamma * Q value for
        # next state and next action
        if done:
            new_q_value = reward
        else:
            new_q_value = reward + self.gamma * self.policy[next_state][next_action]

        # We compute the error
        update = new_q_value - old_q_value

        # And apply the update weighted by learning rate
        self.policy[state][action] += self.alpha * update
