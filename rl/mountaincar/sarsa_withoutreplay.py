import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import sgd
from ExperienceReplay import ExperienceReplay


def epsilon_greedy_policy(n_actions, max_action, epsilon):
    epsilon_policy = np.ones(n_actions, dtype=float) * epsilon / n_actions
    epsilon_policy[max_action] += (1.0 - epsilon)
    return epsilon_policy


class Sarsa(ExperienceReplay):
    """

    Args:
        discount: discount factor
    """

    def __init__(self, network_parameter, max_memory=100, discount=.99, n_steps=1, alpha=.1, num_actions=3, epsilon=.1):
        super().__init__(max_memory, discount)
        self.backup_memory = list()
        self.n_steps = n_steps
        self.alpha = alpha
        self.num_actions = num_actions
        self.epsilon = epsilon
        self.network_parameter = network_parameter
        self.model = Sequential()
        self.model.add(Dense(network_parameter.hidden_size, input_shape=(network_parameter.input_size, ), activation=network_parameter.activation_function))
        for i in range(network_parameter.n_hidden_layer):
            self.model.add(Dense(units=network_parameter.hidden_size, activation=network_parameter.activation_function))
        self.model.add(Dense(network_parameter.num_actions))
        self.model.compile(sgd(lr=0.001, momentum=0.01), loss='mse')  # rmsprop(lr=0.003)

    """
    state,action,reward, next_state, game_over
    sample state [[array([[ 0.,  0.]]), 0, -1, array([[-0.0034965, -0.0034965]])], False]
    """

    def remember(self, states, game_over):
        super().remember(states, game_over)
        self.backup_memory.append([states, game_over])
    """
    predicting  [[ -1.55888324e-09   6.32959574e-10  -8.20817991e-09]]  of state  [[-0.49917767  0.00082233]]
    """

    def expected_reward(self, state, next_time_steps, game_over):
        state_t, action_t, reward_t, _ = state
        targets = self.model.predict(state_t)
        inputs = state_t
        # sum all reward
        cumulative_rewards = sum([(row[0][2] * (self.discount ** idx)) for idx, row in enumerate(next_time_steps)])
        # game is not over, choose epsilon greedy action
        if game_over is False:
            state_tp1 = next_time_steps[-1][0][3]
            outcome = self.model.predict(state_tp1)[0]
            e_policy = epsilon_greedy_policy(self.num_actions, np.argmax(outcome), self.epsilon)
            action = np.random.choice(outcome, p=e_policy)
            cumulative_rewards = cumulative_rewards + ((self.discount ** self.n_steps) * action)
        targets[0, action_t] = targets[0, action_t] + self.alpha * (cumulative_rewards - targets[0, action_t])
        # print(inputs,targets)
        return inputs, targets

    def update_n_steps(self):
        """
        This methods calculates the value of state,action pair for the first element in the backup_memory list.
        """
        loss = 0
        game_over = self.backup_memory[-1][1]
        if game_over is False and len(self.backup_memory) < self.n_steps + 1:
            return loss
        if game_over:
            # update rest
            inputs = np.zeros((len(self.backup_memory), self.network_parameter.input_size))
            targets = np.zeros((len(self.backup_memory), self.network_parameter.num_actions))
            for i in range(len(self.backup_memory)):
                input_, target = self.expected_reward(self.backup_memory[i][0], self.backup_memory[min(i + 1, len(self.backup_memory) - 1):], True)
                inputs[i] = input_
                targets[i] = target
            loss = self.model.train_on_batch(inputs, targets)
            del self.backup_memory[:]
        else:
            # update first
            inputs, targets = self.expected_reward(self.backup_memory[0][0], self.backup_memory[1:], False)
            loss = self.model.train_on_batch(inputs, targets)
            del self.backup_memory[0]
        return loss

    def train_model(self):
        return self.update_n_steps()

    def get_action(self, state):
        if np.random.rand() <= self.epsilon:
            action = np.random.randint(0, self.num_actions, size=1)[0]
        else:
            q = self.model.predict(state)
            action = np.argmax(q[0])
        return action
