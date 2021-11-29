import torch.nn as nn
import torch


class TestForHook(nn.Module):
    def __init__(self):
        super().__init__()
        # TODO 这里的in/out_features 是什么
        self.linear_1 = nn.Linear(in_features=2, out_features=2)
        self.linear_2 = nn.Linear(in_features=2, out_features=1)
        self.relu = nn.ReLU()
        self.relu6 = nn.ReLU6()
        self.initialize()

    def forward(self, x):
        linear_1 = self.linear_1(x)
        linear_2 = self.linear_2(linear_1)
        relu = self.relu(linear_2)
        relu_6 = self.relu6(relu)
        layers_in = (x, linear_1, linear_2)
        layers_out = (linear_1, linear_2, relu)
        return relu_6, layers_in, layers_out

    def initialize(self):
        self.linear_1.weight = nn.Parameter(torch.FloatTensor([[1, 1], [1, 1]]))
        self.linear_1.bias = nn.Parameter(torch.FloatTensor([1, 1]))
        # 一个细小的错误: not [1,1] but [[1,1]]
        self.linear_2.weight = nn.Parameter(torch.FloatTensor([[1, 1]]))
        self.linear_2.bias = nn.Parameter(torch.FloatTensor([1]))
        return True


module_name = []
features_in_hook = []
features_out_hook = []


def hook(module, fea_in, fea_out):
    print('hooker working')
    module_name.append(module.__class__)
    features_in_hook.append(fea_in)
    features_out_hook.append(fea_out)
    return None


net = TestForHook()
net_children = net.children()
for child in net_children:
    if not isinstance(child, nn.ReLU6):
        child.register_forward_hook(hook=hook)

inputs = torch.FloatTensor([[0.1, 0.1], [0.1, 0.1]])

out, features_in_forward, features_out_forward = net(inputs)
print("*" * 5 + "forward return features" + "*" * 5)

print(features_in_forward)
print(features_out_forward)
print("*" * 5 + "forward return features" + "*" * 5)
