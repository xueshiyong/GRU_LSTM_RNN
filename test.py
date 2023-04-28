import torch
import torch.nn as nn

device = 'cuda' if torch.cuda.is_available() else 'cpu'
x = torch.randn(size=(64, 13, 24))
x = x.to(device)
print(x.device)
model = nn.GRU(24, 36, 1)
model = model.to(device)
print(model)
y, _ = model(x)
print(y.shape)