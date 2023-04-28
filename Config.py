import torch.nn as nn
import torch.optim as optim
import torch

class config:
    '''
    model parameters
    '''
    epochs = 2
    lr = 0.001
    momentum = 0.8
    criterion = nn.MSELoss()
    get_optimizer = optim.Adam
    input_size, hidden_size, output_size = 24, 32, 1

    seed = 42
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    '''
    data parameters
    '''
    train_ratio, val_ratio, test_ratio = 0.8, 0.1, 0.1  # divide train, val, test
    lags = 24  # change univariate to multivariate
    df_columns = list(range(1, 14))  # The dataset effectively training data content
    batch_size = 64  # data batch_size

    '''
    save net, load net
    '''
    save_net_path = r".\save_model"
    net_name = "gru" # gru, rnn, lstm