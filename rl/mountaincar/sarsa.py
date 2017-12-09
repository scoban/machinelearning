import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Dropout
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
        # for i in range(network_parameter.n_hidden_layer):
        self.model.add(Dense(units=network_parameter.hidden_size, activation=network_parameter.activation_function))
        self.model.add(Dropout(rate=.1, noise_shape=None, seed=None))
        self.model.add(Dense(units=network_parameter.hidden_size, activation=network_parameter.activation_function))
        self.model.add(Dense(network_parameter.num_actions))
        self.model.compile(sgd(lr=0.01, momentum=0.09), loss='mse')  # rmsprop(lr=0.003)

    """
    state,action,reward, next_state, game_over
    sample state [[array([[ 0.,  0.]]), 0, -1, array([[-0.0034965, -0.0034965]])], False]
    """
    """
    predicting  [[ -1.55888324e-09   6.32959574e-10  -8.20817991e-09]]  of state  [[-0.49917767  0.00082233]]
    """

    def cumulative_rewards(self, state, next_time_steps, game_over):
        state_t, action_t, reward_t, _ = state
        # sum all reward
        cumulative_rewards = sum([(row[0][2] * (self.discount ** idx)) for idx, row in enumerate(next_time_steps)])
        # game is not over, choose epsilon greedy action
        if game_over is False:
            state_tp1 = next_time_steps[-1][0][3]
            outcome = self.model.predict(state_tp1)[0]
            e_policy = epsilon_greedy_policy(self.num_actions, np.argmax(outcome), self.epsilon)
            action = np.random.choice(outcome, p=e_policy)
            cumulative_rewards = cumulative_rewards + ((self.discount ** self.n_steps) * action)
        #  # leave it to network
        # cumulative_rewards = self.alpha * (cumulative_rewards - targets[0, action_t])
        return cumulative_rewards

    def train_on_batch(self, inputs, targets):
        return self.model.train_on_batch(inputs, targets)

    def get_action(self, state):
        if np.random.rand() <= self.epsilon:
            action = np.random.randint(0, self.num_actions, size=1)[0]
        else:
            q = self.model.predict(state)
            action = np.argmax(q[0])
        return action

    def get_batch(self, batch_size=10):
        self.positive_sample = False
        len_memory = len(self.memory)
        num_actions = self.model.output_shape[-1]
        # env_dim = self.memory[0][0][0].shape[1]
        env_dim = self.memory[0][0][0].shape[1]
        inputs = np.zeros((min(len_memory, batch_size), env_dim))
        targets = np.zeros((inputs.shape[0], num_actions))
        """ sample_distribution = []
        if self.positive_sample is False:
            sample_distribution = self.memory[:][0]
        len_memory = len(sample_distribution) """
        # is_print = False
        for i, idx in enumerate(np.random.randint(0, len_memory, size=inputs.shape[0])):
            state_t, action_t, reward_t, state_tp1 = self.memory[idx][0]
            game_over = self.memory[idx][1]

            inputs[i: i + 1] = state_t
            # There should be no target values for actions not taken.
            # Thou shalt not correct actions not taken #deep
            targets[i] = self.model.predict(state_t)[0]
            # np.max(model.predict(state_tp1)[0])
            if game_over:  # if game_over is True
                targets[i, action_t] = reward_t
                # is_print = True
            else:
                # reward_t + gamma * max_a' Q(s', a')
                next_time_steps = self.memory[idx: idx + self.n_steps + 1]
                game_over_states = np.array([x[1] for x in next_time_steps])
                if len(game_over_states[game_over_states == True]) > 0:
                    # print("game over #########################################")
                    game_over = True
                idx_game_over = np.argmax(game_over_states == True)
                Q_sa = self.cumulative_rewards(self.memory[idx][0], next_time_steps[0: idx_game_over + 1], game_over)
                targets[i, action_t] = reward_t + self.discount * Q_sa
        # if is_print:
        #     print(inputs, targets)
        return inputs, targets
