import torch
import torch.nn as nn

from environment import get_env_device

# this file contains common utilities used by both worker and client

device = get_env_device()


class TwoNN(torch.nn.Module):
    def __init__(self):
        super(TwoNN, self).__init__()

        self.fc1 = torch.nn.Linear(784, 200)
        self.fc2 = torch.nn.Linear(200, 200)
        self.fc3 = torch.nn.Linear(200, 10)

    def forward(self, x):

        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        x = torch.nn.functional.softmax(self.fc3(x), dim=1)

        return x


# calculates the accuracy score of a prediction y_hat and the ground truth y
I = torch.eye(10, 10)


def get_accuracy(y_hat, y):
    y_vec = torch.tensor([I[int(i)].tolist() for i in y]).to(device)
    dot = torch.dot(y_hat.flatten(), y_vec.flatten())
    return dot / torch.sum(y_hat)


# sets the parameters of a neural net
def set_parameters(net, params):
    current_params = list(net.parameters())
    for i, p in enumerate(params):
        current_params[i].data = p.data.clone()


def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)
