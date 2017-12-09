#from sarsa import Sarsa
#from sarsa_v1 import Sarsa
#from expectedsarsa import ExpectedSarsa
from doubleqlearning import DoubleQLearning
from MountainCar import MountainCar
import json
from NeuralNetworkParameter import NeuralNetworkParameter

if __name__ == "__main__":
    # parameters
    epsilon = .1  # exploration
    num_actions = 3  # [move_left, stay, move_right]
    epoch = 20
    max_memory = 10000
    number_of_steps = 1
    network_parameter = NeuralNetworkParameter()
    Xrange = [-1.5, 0.55]
    Vrange = [-2.0, 2.0]
    start = [0.0, 0.0]
    goal = [0.45]

    # Define environment/game
    env = MountainCar(start, goal, Xrange, Vrange)

    # Initialize experience replay object
    experience_replay = DoubleQLearning(network_parameter, max_memory=max_memory, n_steps=number_of_steps, num_actions=num_actions, epsilon=epsilon)

    # Train
    win_cnt = 0
    for e in range(epoch):
        loss = 0.
        env.reset()
        game_over = False
        # get initial input
        input_t = env.observe()

        step = 0
        while (not game_over):
            input_tm1 = input_t
            step += 1
            # get next action epsilon greedy
            action = experience_replay.get_action(input_tm1)

            # apply action, get rewards and new state
            input_t, reward, game_over = env.act(action)
            if reward == 100:
                win_cnt += 1

            # store experience
            experience_replay.remember([input_tm1, action, reward, input_t], game_over)

            # if (number_of_steps + 1) > step and experience_replay not isinstance(DoubleQLearning):
            #    continue

            inputs, targets = experience_replay.get_batch(batch_size=50)
            # update model
            step_loss = experience_replay.train_on_batch(inputs, targets)

            loss += step_loss
            print("Step {} Epoch {:03d}/{:03d} | Loss {:.4f} | Step Loss {:.4f} | Win count {} | Pos {:.3f} | Act {}"
                  .format(step, e, epoch, loss, step_loss, win_cnt, input_t[0, 0], action - 1))

    # Save trained model weights and architecture, this will be used by the visualization code
    experience_replay.model.save_weights("model.h5", overwrite=True)
    with open("model.json", "w") as outfile:
        json.dump(experience_replay.model.to_json(), outfile)
