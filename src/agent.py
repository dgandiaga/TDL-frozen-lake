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

        if trainable:
            self.epsilon = 1
            self.eps_min = eps_min
        else:
            self.epsilon = 0
            self.eps_min = 0

    def create_state_if_not_exists(self, state):
        if state not in self.policy.keys():
            self.policy[state] = [self.q_init] * self.action_range

    def select_action(self, state):
        self.create_state_if_not_exists(state)
        if np.random.random() > self.epsilon:
            action = np.argmax(self.policy[state])
        else:
            action = np.random.choice(range(self.action_range))
        return action

    def save_policy(self, tag, date):
        save_file = open(f"models/{tag}{date}.pkl", "wb")
        pickle.dump(self.policy, save_file)

    def load_policy(self, name):
        load_file = open(f"models/{name}.pkl", "rb")
        self.policy = pickle.load(load_file)

    def update_epsilon(self):
        self.epsilon = np.max([self.epsilon * self.eps_decay, self.eps_min])

    def update(self, state, action, reward, next_state, next_action, done):

        old_q_value = self.policy[state][action]

        if done:
            new_q_value = reward
        else:
            new_q_value = reward + self.gamma * self.policy[next_state][next_action]
        # print(f'Updating state {state}:{action} given reward {reward} with value {self.policy[state][action]} with new q value {new_q_value}, update = {self.alpha * (new_q_value - old_q_value)}, state is terminal: {done}')
        update = new_q_value - old_q_value
        self.policy[state][action] += self.alpha * update
