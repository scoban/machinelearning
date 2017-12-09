import numpy as np
from ExperienceReplay import ExperienceReplay
from keras.models import Sequential
from keras.layers.core import Dense, Dropout
from keras.optimizers import sgd


def epsilon_greedy_policy(n_actions, max_action, epsilon):
    epsilon_policy = np.ones(n_actions, dtype=float) * epsilon / n_actions
    epsilon_policy[max_action] += (1.0 - epsilon)
    return epsilon_policy


class DoubleQLearning(ExperienceReplay):
    def __init__(self, network_parameter, max_memory=100, n_steps=1, discount=.99, alpha=.01, num_actions=3, epsilon=.1):
        super().__init__(max_memory, discount)
        self.alpha = alpha
        self.num_actions = num_actions
        self.epsilon = epsilon
        self.network_parameter = network_parameter
        self.random_update = 0
        self.model = Sequential()
        self.model.add(Dense(network_parameter.hidden_size, input_shape=(network_parameter.input_size, ), activation=network_parameter.activation_function))
        # for i in range(network_parameter.n_hidden_layer):
        self.model.add(Dense(units=network_parameter.hidden_size, activation=network_parameter.activation_function))
        self.model.add(Dropout(rate=.1, noise_shape=None, seed=None))
        self.model.add(Dense(units=network_parameter.hidden_size, activation=network_parameter.activation_function))
        self.model.add(Dense(network_parameter.num_actions))
        self.model.compile(sgd(lr=0.01, momentum=0.09), loss='mse')

        self.model1 = Sequential()
        self.model1.add(Dense(network_parameter.hidden_size, input_shape=(network_parameter.input_size, ), activation=network_parameter.activation_function))
        # for i in range(network_parameter.n_hidden_layer):
        self.model1.add(Dense(units=network_parameter.hidden_size, activation=network_parameter.activation_function))
        self.model1.add(Dropout(rate=.1, noise_shape=None, seed=None))
        self.model1.add(Dense(units=network_parameter.hidden_size, activation=network_parameter.activation_function))
        self.model1.add(Dense(network_parameter.num_actions))
        self.model1.compile(sgd(lr=0.01, momentum=0.09), loss='mse')

    def train_on_batch(self, inputs, targets):
        if self.random_update < 0.5:
            val = self.model.train_on_batch(inputs, targets)  # [0]
        else:
            val = self.model1.train_on_batch(inputs, targets)  # [0]
        return val

    def get_action(self, state):
        model_predict = self.model.predict(state)[0]
        model1_predict = self.model.predict(state)[0]
        outcome = model_predict + model1_predict
        e_policy = epsilon_greedy_policy(self.num_actions, np.argmax(outcome), self.epsilon)
        action = np.random.choice(len(outcome), p=e_policy)
        return action
    # train model

    def get_batch(self, batch_size=10):
        len_memory = len(self.memory)
        num_actions = self.model.output_shape[-1]
        env_dim = self.memory[0][0][0].shape[1]
        inputs = np.zeros((min(len_memory, batch_size), env_dim))
        targets = np.zeros((inputs.shape[0], num_actions))
        for i, idx in enumerate(np.random.randint(0, len_memory, size=inputs.shape[0])):
            state_t, action_t, reward_t, state_tp1 = self.memory[idx][0]
            inputs[i:i + 1] = state_t
            game_over = self.memory[idx][1]
            self.random_update = np.random.random(1)
            if self.random_update < .5:
                action_t_outcome = self.model.predict(state_t)[0][action_t]
                next_state_outcome = self.model.predict(state_tp1)[0]
                next_state_max_outcome = np.argmax(next_state_outcome)
                model1_next_state_outcome = self.model1.predict(state_tp1)[0][next_state_max_outcome]
            else:
                action_t_outcome = self.model1.predict(state_t)[0][action_t]
                next_state_outcome = self.model1.predict(state_tp1)[0]
                next_state_max_outcome = np.argmax(next_state_outcome)
                model1_next_state_outcome = self.model.predict(state_tp1)[0][next_state_max_outcome]

            if game_over:  # if game_over is True
                targets[i, action_t] = reward_t
            else:
                targets[i, action_t] = reward_t + self.discount * model1_next_state_outcome
        return inputs, targets
