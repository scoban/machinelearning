class NeuralNetworkParameter():
    def __init__(self, hidden_size=50, n_hidden_layer=2, input_size=2, activation_function="relu", batch_size=50, num_actions=3):
        self.hidden_size = hidden_size
        self.n_hidden_layer = n_hidden_layer
        self.input_size = input_size
        self.activation_function = activation_function
        self.batch_size = batch_size
        self.num_actions = num_actions
