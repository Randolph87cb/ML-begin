import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(2, 4)
        self.fc2 = nn.Linear(4, 10)
        self.fc3 = nn.Linear(10, 16)
        self.fc4 = nn.Linear(16, 10)
        self.fc5 = nn.Linear(10, 4)
        self.fc6 = nn.Linear(4, 2)

    def forward(self, x):
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        x = F.tanh(self.fc3(x))
        x = F.tanh(self.fc4(x))
        x = F.tanh(self.fc5(x))
        return self.fc6(x)

net = Net()
print(net)

batch_size = 200

def xor(a, b):
    # return torch.Tensor([a+b])
    if (a*b) > 0:
        return torch.Tensor([1])
    else:
        return torch.Tensor([0])

def data_provider():
    data_in = torch.rand(batch_size,2) * 2 - 1
    label = torch.Tensor(batch_size,1)
    for i in range(batch_size):
        label[i] = xor(data_in[i][0], data_in[i][1])
    return data_in, label

optimizer = optim.Adam(net.parameters(), lr=0.01)

for i in range(300):
    optimizer.zero_grad()
    data_in, gt = data_provider()
    data_out = net(data_in)
    criterion = nn.L1Loss()
    loss = criterion(data_out, gt)
    loss.backward()
    # print(data_in[0])
    # print(gt[0])
    print(loss)
    optimizer.step()

def check(a, b):
    if (a[0]*a[1]) > 0 and b[0] > 0.5:
        return True
    if (a[0]*a[1]) < 0 and b[0] < 0.5:
        return True
    return False

test_size = 1000
count = 0
for f in range(test_size):
    data_in = torch.rand(2) * 2 - 1
    data_in = data_in.view(1, -1)
    data_out = net(data_in)
    # print(data_in[0], data_out[0])
    if check(data_in[0], data_out[0]):
        count += 1
print(count/test_size)
