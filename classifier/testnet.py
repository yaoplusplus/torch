import os
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import torchvision
import torchvision.transforms as transforms
import dataloader
import dataloader
import torch
import classifier

PATH = 'trained_model'


# functions to show an image

def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# get some random training images
dataiter = iter(dataloader.trainloader)
images, labels = dataiter.next()
# print(images.shape)
# show images
imshow(torchvision.utils.make_grid(images))
# print labels
print('GroundTruth:', ' '.join('%5s' % dataloader.classes[labels[j]] for j in range(4)))

# 从训练好的模型上加载参数作为新训练模型的初始化参数
assert os.path.join(PATH, 'cifar_net.pth')
net = classifier.Net()
net.load_state_dict(torch.load(os.path.join(PATH, 'cifar_net.pth')))
# net.load_state_dict(torch.load(PATH))
outputs = net(images)
_, predicted = torch.max(outputs, 1)

print('Predicted: ', ' '.join('%5s' % dataloader.classes[predicted[j]]
                              for j in range(4)))

correct = 0
total = 0
with torch.no_grad():
    for data in dataloader.testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))
