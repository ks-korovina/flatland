"""
Pytorch Modules for MNIST and CIFAR10

Code adapted from torchvision

TODOs;
* other models?

"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import math


def get_model(model_name, *args, **kwargs):
    """
    User interface to models.
    Use "vgg{11,13,16,19}"" for CIFAR10,
    and "lenet" for MNIST
    """
    if "vgg" in model_name.lower():
        return VGG(model_name.upper())
    elif model_name == "lenet":
        # TODO: arg check
        return LeNet(*args, **kwargs)
    else:
        raise ValueError("Unknown model {}".format(model_name))


class BaseModule(nn.Module):
    """ Implement common logic """
    def __init__(self):
        super(BaseModule, self).__init__()

    def save(self, check_name):
        checkpoint = self.state_dict()
        os.makedirs("./checkpoints/", exist_ok=True)
        torch.save(checkpoint, "./checkpoints/{}.pth".format(check_name))

    def load(self, check_name):
        """
        checkpoint - ./checkpoints/some_name.pt
        """
        checkpoint = torch.load("./checkpoints/{}.pth".format(check_name), map_location='cpu')
        self.to(DEVICE)
        self.load_state_dict(checkpoint)

    def load_params(self, params):
        self.load_state_dict(params)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def _initialize_weights_large(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(6. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.1)
                m.bias.data.zero_()


class LeNet(BaseModule):
    def __init__(self, in_channels=1, img_rows=28, num_classes=10):
        """ Parameters set for MNIST,
        change in constructor if used with other data """
        super(LeNet, self).__init__()
        self.model_name = 'LeNet'
        self.out_rows = ((img_rows - 4)//2 - 4)//2
        self.features = nn.Sequential(
                nn.Conv2d(in_channels, 20, 5),
                nn.MaxPool2d(2, 2),
                nn.ReLU(inplace=True),
                nn.Conv2d(20, 50, 5),
                nn.Dropout2d(),
                nn.MaxPool2d(2, 2),
                nn.ReLU(inplace=True),
                )
        self.classifier = nn.Sequential(
                nn.Linear(self.out_rows * self.out_rows * 50, 500),
                nn.ReLU(inplace=True),
                nn.Dropout2d(),
                nn.Linear(500, num_classes),
                nn.LogSoftmax(dim=-1),
                )
        self._initialize_weights_large()

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, self.out_rows*self.out_rows*50)
        x = self.classifier(x)
        return x


cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(BaseModule):
    def __init__(self, vgg_name):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, 10)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


class MnistMLP(BaseModule):
    """ MLP for MNIST """
    def __init__(self):
        super(MnistMLP, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 800)
        self.fc2 = nn.Linear(800, 320)
        self.fc3 = nn.Linear(320, 50)
        self.fc4 = nn.Linear(50, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return F.log_softmax(x, dim=1)

