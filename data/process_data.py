import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import torch
from Config import config


# class data_config:
#     train_ratio, val_ratio, test_ratio = 0.8, 0.1, 0.1          # divide train, val, test
#     lags = input_size = 24   # change univariate to multivariate
#     df_columns = list(range(1, 14)) # The dataset effectively training data content
#     batch_size = 64 # data batch_size
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'

'''
filename -> string, data_config -> class data_config
return df -> ndarray
'''
def GetFileNumpy(filename, data_config):
    df_columns = data_config.df_columns
    df = pd.read_csv(filename, encoding='utf-8').fillna(0)
    df = df.iloc[:, df_columns]
    df = df.values
    return df

# filename = r"../read_data/solar_location_1.csv"
# df = GetFileNumpy(filename, data_config)
# print(df.shape)


def NumpyNorm(df):
    scaler = MinMaxScaler()
    scaler = scaler.fit(df)
    df_norm = scaler.transform(df)
    return df_norm


'''
return tensor data (X), tensor label (y), 
train = TensorDataset(train_X, train_y),
test = TensorDataset(test_X, test_y)
val = TensorDataset(val_X, val_y)
train_X, test_X, val_X is 80% part of the data, 10% of the data, 10% of the data
train_y, test_y, val_y is 80% part of the label, 10 % of the label, 10% of the label  
'''
def GetTrainTestData(df, data_config = config):
    lags = data_config.lags
    train_ratio, val_ratio, test_ratio = \
        data_config.train_ratio,data_config.val_ratio, data_config.test_ratio

    '''
    Normalize all columns
    '''

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler = scaler.fit(df)
    df_norm = scaler.transform(df)
    # df_norm_reverse = scaler.inverse_transform(df_norm)  # get the inital data

    '''
    Create an array of input_size = lags
    '''
    total = []
    for i in range(lags, len(df)):
        total.append(df_norm[i - lags : i+1, :]) # total.shape = (19680, 25, 13)
    total = np.array(total)
    # total = total.astype(np.float)
    data, label = total[:, :-1, :], total[:, -1, -1] # data.shape, label.shape = (19680, 24, 13), (19680, 1, 1)
    data, label = torch.from_numpy(data), torch.from_numpy(label)
    # data, label = data.float(), label.float() # 模型用的是torch.float32, 而使用numpy默认得到的是torch.float64
    print("data.type is {:}".format(data.dtype))

    train_data = data[ : int(data.shape[0] * train_ratio), :, :]
    val_data = data[int(data.shape[0] * train_ratio) : int(data.shape[0] * (train_ratio + val_ratio)), :, :]
    test_data = data[int(data.shape[0] * (train_ratio + val_ratio)) :, :, :]

    train_label = label[ : int(label.shape[0] * train_ratio)]
    val_label = label[int(label.shape[0] * train_ratio) : int(label.shape[0] * (train_ratio + val_ratio))]
    test_label = label[int(label.shape[0] * (train_ratio + val_ratio)) :]

    return train_data, train_label, val_data, val_label, test_data, test_label



def GetTrainTestLoader(train_data, train_label, val_data, val_label, test_data, test_label, config):
    batch_size = config.batch_size

    device = config.device
    '''
    copy the data into cuda
    '''
    train_data = train_data.to(device)
    train_label = train_label.to(device)
    val_data = val_data.to(device)
    val_label = val_label.to(device)
    test_data = test_data.to(device)
    test_label = test_label.to(device)

    train = TensorDataset(train_data, train_label)
    val = TensorDataset(val_data, val_label)
    test = TensorDataset(test_data, test_label)

    train_loader = DataLoader(train, batch_size=64, shuffle=True)
    test_loader = DataLoader(test, batch_size=64)
    val_loader = DataLoader(val, batch_size=64)

    return train_loader, val_loader, test_loader