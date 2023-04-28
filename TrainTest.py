import torch
import time
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# import model
from model import MultiVariGRU, MultiVariLSTM
from data.process_data import GetTrainTestData, GetTrainTestLoader, GetFileNumpy
from Config import config
import matplotlib.pyplot as plt
import random
import os
import numpy as np
import matplotlib.pyplot as plt

def seed_torch(seed = 0):
    seed = int(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True

seed_torch(seed=config.seed)

filename = r'.\read_data\solar_location_1.csv'
df = GetFileNumpy(filename, config)
train_data, train_label, val_data, val_label, test_data, test_label = GetTrainTestData(df, config)

train_loader, val_loader, test_loader = \
    GetTrainTestLoader(train_data, train_label, val_data, val_label, test_data, test_label, config)

def GetNet(config):
    if config.net_name == 'gru':
        net = MultiVariGRU(config.input_size, config.hidden_size, config.output_size)
    elif config.net_name == 'lstm':
        net = MultiVariLSTM(config.input_size, config.hidden_size, config.output_size)

    net = net.to(config.device)

    return net

net = GetNet(config)
net = net.to(config.device)


def Train(train_loader, net, config):
    optimizer = optim.Adam(net.parameters(), lr = 0.001)
    # optimizer = config.get_optimizer(net.parameters(), lr = config.lr, momentum = config.momentum)
    criterion = config.criterion
    net = net.train()
    loss_arr = []

    for epoch in range(config.epochs):
        start = time.time()
        batch_loss_arr = []
        for idx, (X, y) in enumerate(train_loader):
            optimizer.zero_grad()
            out, _ = net(X)
            out = torch.squeeze(out, 1)
            y = y.float()
            loss = criterion(out, y)
            # loss = loss.float()
            loss.backward()
            optimizer.step()
            batch_loss_arr.append(loss.detach().cpu().numpy())
        batch_loss = sum(batch_loss_arr) / len(batch_loss_arr)
        loss_arr.append(batch_loss)
        print("epoch {:} / {:}, loss {:.4f}, time {:.3f}".format(epoch+1, config.epochs, batch_loss, time.time() - start))

    return net, loss_arr

net, loss_arr = Train(train_loader, net, config)

def save_model_pt(net, config):
    save_net = config.save_net_path + "\\" + str(config.epochs) + "_" + config.net_name + ".pt"
    torch.save(net.state_dict(), save_net)


def train_plot_loss(loss_arr, config):
    plt.plot(loss_arr, 'r')
    plt.xlabel('train_epoch')
    plt.ylabel('loss')
    plt.savefig("./img/seed_" + config.net_name +"_" + str(config.seed) +
                "_epoch_" + str(config.epochs) + "_train_loss.png")

# save_model_pt(net, config)
# train_plot_loss(loss_arr, config)
test_data = test_data.to(config.device)

out, _ = net(test_data)
out = torch.squeeze(out, 1)
pred_y = out.detach().cpu().numpy()
true_y = test_label.cpu().numpy()


print("pred: ", pred_y)
print("true: ", true_y)
print("MAE: ", mean_absolute_error(true_y, pred_y))



torch.cuda.empty_cache()


# test_net = MultiVariLSTM(config.input_size, config.hidden_size, config.output_size)
# save_net = config.save_net_path + "\\" + str(config.epochs) + "_" + config.net_name + ".pt"
# test_net.load_state_dict(torch.load(save_net))
# test_net.to(config.device)


# def Test(test_loader, net, config):
#     net.eval()
#     criterion = config.criterion
#     loss_arr = []
#     for idx, (X, y) in enumerate(test_loader):
#         out, _ = net(X)
#         out = torch.squeeze(out, 1)
#         y = y.float()
#         loss = criterion(out, y)
#         loss_arr.append(loss.cpu().numpy())

            

