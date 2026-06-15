import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

# use pre-trained weights
model = resnet18(weights=ResNet18_Weights.DEFAULT)

# modify the architecture not shring the 32x32 CIFAR pictures at the start (resnet was trained on 224x224)
model.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
model.maxpool = nn.Identity()
model.fc = nn.Linear(512, 10)