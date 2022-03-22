import copy

import torch

from settings import *


class DDPG_Agent:
    """ Deep Deterministic Policy Gradient Agent """
    def __init__(self, configs):
        self.num_actions = configs['OUTPUT_SIZE']
        self.eye_size = configs['EYE_SIZE']

        self.policy_net = nn.Sequential(
            nn.Linear(configs['INPUT_SIZE'], configs['HID1_SIZE']),
            nn.Tanh(),
            nn.Linear(configs['HID1_SIZE'], configs['HID2_SIZE']),
            nn.Tanh(),
            nn.Linear(configs['HID2_SIZE'], self.num_actions),
            nn.Sigmoid()  # so we can rescale the output easily
        )
        self.policy_net_optimizer = Adam(self.policy_net.parameters(), lr=configs['LEARNING_RATE'])

        self.Q_net = nn.Sequential(
            nn.Linear(configs['INPUT_SIZE'] + self.num_actions, configs['HID1_SIZE']),
            nn.Tanh(),
            nn.Linear(configs['HID1_SIZE'], configs['HID2_SIZE']),
            nn.Tanh(),
            nn.Linear(configs['HID2_SIZE'], 1)
        )
        self.Q_net_optimizer = Adam(self.Q_net.parameters(), lr=configs['LEARNING_RATE'])
        self.target_Q_net = copy.deepcopy(self.Q_net)
        self.Q_net_loss = nn.MSELoss()

        self.classify_net = nn.Sequential(
            nn.Linear(configs['INPUT_SIZE'], configs['HID1_SIZE']),
            nn.Tanh(),
            nn.Linear(configs['HID1_SIZE'], configs['HID2_SIZE']),
            nn.Tanh(),
            nn.Linear(configs['HID2_SIZE'], 10),
            nn.Softmax(dim=-1)
        )
        self.classify_net_optimizer = Adam(self.classify_net.parameters(), lr=configs['LEARNING_RATE'])
        self.classify_net_loss = nn.CrossEntropyLoss()

        self.target_update = configs['TARGET_UPDATE']
        self.target_update_counter = 0

        self.experience_replay = []
        self.experience_replay_size = 0
        self.experience_replay_cap = configs['EXPERIENCE_REPLAY_CAPACITY']

    def sample_action(self, state):
        with torch.no_grad():
            out = self.policy_net(state)
            d = out[:1] * 10  # have to be within a finite positive range
            theta = out[1:2] * math.pi  # have to be in -pi to pi
            return d, theta

    def predict_digit(self, state):
        with torch.no_grad():
            out = self.classify_net(state)
            return out

    def save_exp(self, trajectory):
        if self.experience_replay_size == self.experience_replay_cap:
            self.experience_replay.pop(0)
        else:
            self.experience_replay_size += 1
        self.experience_replay.append(trajectory)

    def get_Q(self, state, action):
        out = self.Q_net(torch.cat([state, action], dim=1))
        return out

    def get_targetQ(self, state, action):
        out = self.target_Q_net(torch.cat([state, action], dim=1))
        return out

    def train(self, batch_size, gamma):
        self.policy_net.train()
        self.Q_net.train()

        # sample a batch of transitions from the replay
        batch_s = torch.tensor([])
        batch_g = torch.tensor([])
        batch_a = torch.tensor([])

        batch_s_last = torch.tensor([])
        batch_y = torch.tensor([], dtype=torch.long)

        batch = random.sample(self.experience_replay, min(batch_size, self.experience_replay_size))
        for trajectory in batch:
            T = len(trajectory)
            for time in range(T):
                s, a, _, y = trajectory[time]
                g = sum([gamma**(t - time) * trajectory[t][2] for t in range(time, T)])
                batch_s = torch.cat([batch_s, s[None]])
                batch_g = torch.cat([batch_g, torch.tensor(g)[None, None]])
                batch_a = torch.cat([batch_a, a[None]])
                if y != -1:
                    batch_s_last = torch.cat([batch_s_last, s[None]])
                    batch_y = torch.cat([batch_y, torch.tensor(y, dtype=torch.long)[None]])

        # update target networks with current behavior networks
        if self.target_update_counter == self.target_update:
            self.target_Q_net.load_state_dict(self.Q_net.state_dict())
            self.target_update_counter = 0
        self.target_update_counter += 1

        # update the classification network using CEL
        self.classify_net.zero_grad()
        classify_loss_value = self.classify_net_loss(self.classify_net(batch_s_last), batch_y)
        classify_loss_value.backward()
        self.classify_net_optimizer.step()

        # update Q network using Mote-Carlo returns
        self.Q_net_optimizer.zero_grad()
        Q_loss_value = self.Q_net_loss(self.get_Q(batch_s, batch_a), batch_g)
        Q_loss_value.backward()
        self.Q_net_optimizer.step()

        # update target policy network using target Q network
        self.policy_net_optimizer.zero_grad()
        policy_loss_value = -self.get_targetQ(batch_s, torch.cat([self.policy_net(torch.tensor(batch_s))])).mean()
        policy_loss_value.backward()
        self.policy_net_optimizer.step()

        return classify_loss_value.item(), Q_loss_value.item(), policy_loss_value.item()
