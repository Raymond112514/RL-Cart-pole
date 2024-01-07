import torch.nn as nn
import torch.optim as optim
import torch as T
import torch.nn.functional as F

class DeepQNetwork(nn.Module):

    def __init__(self, input_dim, n_actions, lr):
        super(DeepQNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, n_actions)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=(1 - 10 ** (-6)))
        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, input):
        x = F.relu(self.fc1(input))
        x = self.fc2(x)
        return x

