import os
import torch
from data import testloader
from net import Net

save_path = 'model.pkl'
net = Net()
if os.path.isfile(save_path):
    device = torch.device('cpu')
    net.load_state_dict(torch.load(save_path, map_location=device))
else:
    print('model dose not exist')
    exit()
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))