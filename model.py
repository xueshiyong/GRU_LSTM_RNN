import torch
import torch.nn as nn
import data.process_data
from data.process_data import GetFileNumpy, GetTrainTestData
from Config import config

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class MultiVariGRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MultiVariGRU, self).__init__()
        self.hidden_size = hidden_size
        self.GRU_layer = nn.GRU(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.linear_layer = nn.Sequential(
            nn.Linear(self.hidden_size * 13, 64),
            nn.Sigmoid(),
            nn.Linear(64, 16),
            nn.Sigmoid(),
            nn.Linear(16, 1)
        )
        # nn.GRU return shape (len, variable, hidden_size)
        self.hidden = 0

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = x.float()
        x, self.hidden = self.GRU_layer(x)
        x = x.reshape(x.shape[0], -1)
        x = self.linear_layer(x)
        return x, self.hidden

class MultiVariLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MultiVariLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.GRU_layer = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.linear_layer = nn.Sequential(
            nn.Linear(self.hidden_size * 13, 64),
            nn.Sigmoid(),
            nn.Linear(64, 16),
            nn.Sigmoid(),
            nn.Linear(16, 1)
        )
        # nn.GRU return shape (len, variable, hidden_size)
        self.hidden = 0

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = x.float()
        x, self.hidden = self.GRU_layer(x)
        x = x.reshape(x.shape[0], -1)
        x = self.linear_layer(x)
        return x, self.hidden


# filename = r"./read_data/solar_location_1.csv"
# df = GetFileNumpy(filename, config)
# data, label, train, val, test = GetTrainTestData(df, data_config = config)


