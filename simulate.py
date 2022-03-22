import random

import matplotlib.pyplot as plt
import torch

from settings import *


class Simulation:
    """ the environment defined as a Markov Decision Process """
    def __init__(self, MDP_settings):
        self.batch_size = MDP_settings['BATCH_SIZE']
        self.map_size = MDP_settings['STATE_SIZE'] - 3
        self.num_actions = MDP_settings['ACTION_SIZE']
        self.gamma = MDP_settings['GAMMA']
        self.num_episodes = MDP_settings['NUM_EPISODES']
        self.eps = MDP_settings['EPSILON_START']
        self.eps_decay = MDP_settings['EPSILON_DECAY']
        self.eps_end = MDP_settings['EPSILON_END']
        self.train_data = datasets.MNIST(
            root='data',
            train=True,
            transform=ToTensor(),
            download=True,
        )
        self.test_data = datasets.MNIST(
            root='data',
            train=False,
            transform=ToTensor()
        )
        _, self.W, self.H = self.train_data[0][0].size()  # all images are the same size
        self.current_env = None

    def reset(self, train):
        """ set the current observable to a random datapoint from the MNIST dataset """
        dataset = self.train_data if train else self.test_data
        self.current_env = dataset[torch.randint(len(dataset), size=(1,)).item()]
        # initial state: all-zero feature map, initial invalid prediction, location of the eye at the img center
        return [0.0] * self.map_size, -1, self.W/2, self.H/2

    def env_step(self, step, es, feature_map, x, y, d, theta, prediction):
        r = 0

        img, label = self.current_env

        # prediction evaluation at last time step
        if step == self.map_size - 1:
            r += (0.5 if prediction == label else -0.5)

        # eye movement
        dx = round(d.item()) * round(math.cos(theta.item()))
        dy = round(d.item()) * round(math.sin(theta.item()))

        r += 0.1 if dx != 0 or dy != 0 else -0.1  # eye movement evaluation

        if 0 <= x + dx < self.W - es and 0 <= y + dy < self.H - es:
            r += 0.1  # reward for valid looking location
            x, y = x + dx, y + dy
            feature = img[0][int(x):int(x + es), int(y):int(y + es)].sum().item()
            r += (0.1 if feature > 0 else -0.1)  # meaningful feature evaluation
            r += sum([(abs(feature - feature_map[i]) > 0) for i in range(step)]) * 0.1  # diverse feature evaluation
            feature_map[step] = feature
        else:
            r += -0.1  # penalize invalid looking location

        return feature_map, x, y, r

    def rollout_train(self, agent):
        """ train the agent """
        loss = []
        for e in range(self.num_episodes):
            feature_map, pred, x, y = self.reset(train=True)  # sample initial state and environment's setting
            epsilon = self.eps
            trajectory = []

            for step in range(self.map_size):
                if epsilon > self.eps_end:  # anneal epsilon
                    epsilon *= self.eps_decay

                # epsilon-greedy action sampling
                s = torch.tensor(feature_map + [pred, x, y])
                d, theta = agent.sample_action(s)
                d = d + random.uniform(1, 5) * epsilon
                theta = theta + random.uniform(-math.pi / 4, math.pi / 4) * epsilon
                a = torch.cat([d, theta])

                # get agent's current digit prediction
                pred = agent.predict_digit(s).argmax().item()

                # invoke dynamics with sampled action and prediction
                feature_map, x, y, r = self.env_step(step, agent.eye_size, feature_map, x, y, d, theta, pred)

                # save the actual label only at last time step prediction
                label = self.current_env[1] if step == (self.map_size - 1) else -1

                trajectory.append((s, a, r, label))

            agent.save_exp(trajectory)

            classify_loss, Q_loss, policy_loss = agent.train(self.batch_size, self.gamma)
            print(f'Episode {e}: '
                  f'classify loss( {round(classify_loss, 3)} ), '
                  f'Q loss( {round(Q_loss, 3)} ), '
                  f'policy loss( {round(policy_loss, 3)} )')

            loss.append(policy_loss)
        return loss

    def rollout_test(self, agent, visualize):
        """ test the agent """
        def render():
            img, label = self.current_env
            es = agent.eye_size
            plt.figure()
            plt.title(label)
            plt.imshow(img[0])
            plt.annotate('eye', xy=(x, y), xycoords='data',
                         xytext=(0, 0), textcoords='figure fraction',
                         arrowprops=dict(arrowstyle="->"))
            plt.scatter([x, x + es, x, x + es], [y, y, y + es, y + es], s=10, c='red', marker='o')
            plt.show()

        score = []
        for e in range(self.num_episodes):
            feature_map, pred, x, y = self.reset(train=False)

            if visualize:
                render()

            total_reward = 0
            for step in range(self.map_size):
                s = torch.tensor(feature_map + [pred, x, y])
                print(f'state {s}')
                d, theta = agent.sample_action(s)
                a = torch.cat([d, theta])
                print(f'action {a}')

                pred = agent.predict_digit(s).argmax().item()
                print(f'predict {pred}')

                feature_map, x, y, r = self.env_step(step, agent.eye_size, feature_map, x, y, d, theta, pred)

                total_reward += r

                if visualize:
                    render()

            print(f'Episode {e} score: {total_reward}')

            score.append(total_reward)
        return score
