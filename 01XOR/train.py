import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(2, 4)
        self.fc2 = nn.Linear(4, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

net = Net()
print(net)

learning_rate = 0.1

def xor(a, b):
    if a*b > 0:
        return 1
    else:
        return -1

def data_provider():
    data_in = torch.randn(100,2)
    label = torch.Tensor(100,1)
    for i in range(100):
        label[i][0] = xor(data_in[i][0], data_in[i][1])
    return data_in, label

while True:
    data_in, gt = data_provider()
    data_out = net(data_in)
    # print(data_in)
    gt[0] = data_in[0][0] * data_in[0][1]
    if gt[0] > 0:
        gt[0] = 1
    else:
        gt[0] = 0
    gt = gt.view(1, -1)
    # print(gt)
    criterion = nn.L1Loss()
    loss = criterion(data_out, gt)
    net.zero_grad()
    loss.backward()
    for f in net.parameters():
        f.data.sub_(f.grad.data * learning_rate)
    learning_rate = learning_rate * 0.99
    if learning_rate < 0.00000001:
        break

for f in range(10):
    data_in = torch.randn(2)
    data_in = data_in.view(1, -1)
    data_out = net(data_in)
    print(data_in)
    print(data_out)
