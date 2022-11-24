import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

from layers import EuclideanColorInvariantConv2d, LearnedColorInvariantConv2d, GrayscaleConv2d, GrayscaleEuclideanColorInvariantConv2d


#Defining the convolutional neural network
class LeNet(nn.Module):
    def __init__(self, model_type, num_classes=10):
        super(ConvNeuralNet, self).__init__()
        if model_type == "normal_lenet":
            self.layer1 = nn.Sequential(
                nn.Conv2d(3, 6, kernel_size=5, stride=1, padding=0),
                nn.BatchNorm2d(6),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size = 2, stride = 2))
        elif "something else":
            pass
        self.layer2 = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.fc = nn.Linear(400, 120)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(120, 84)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(84, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        out = self.relu(out)
        out = self.fc1(out)
        out = self.relu1(out)
        out = self.fc2(out)
        return out
