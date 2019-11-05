# take data path from input
import sys
testset_data_path=sys.argv[1]

print('load test data from {}'.format(testset_data_path))

import torch
import torchvision
import torchvision.transforms as transforms


transform = transforms.Compose(
    [
        transforms.Resize(120),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]
)

testset = torchvision.datasets.ImageFolder(
    root=testset_data_path,
    transform=transform
)
testloader = torch.utils.data.DataLoader(
    testset,
    batch_size=4,
    num_workers=1,
    shuffle=True
)

MODEL_PATH = './fire_or_water_net.pth'


from net import Net

net = Net()
net.load_state_dict(torch.load(MODEL_PATH))

correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        print(labels, predicted)
        total += labels.size(0)

        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the test images: %d %%' % (
    100 * correct / total))