import torch.optim as optim
import torch.nn as nn
import torch
import os
from net import Net
from data import trainloader, testloader

save_path = 'model.pkl'
net = Net()
if os.path.isfile(save_path):
    device = torch.device('cpu')
    net.load_state_dict(torch.load(save_path, map_location=device))
net = nn.DataParallel(net)
net_gpu = net.cuda()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net_gpu.parameters(), lr=0.001, momentum=0.9)


for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs
        inputs, labels = data
        inputs, labels = inputs.cuda(), labels.cuda()

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net_gpu(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')

torch.save(net_gpu.module.state_dict(), save_path)
print('Save model_gpu')

