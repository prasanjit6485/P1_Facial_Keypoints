import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # Input image size - (1, 224, 224) with 32 output channels/feature maps, 5x5 square convolution kernel
        self.conv1 = nn.Conv2d(1,  32,  5)         # Output - (32, 220, 220), with Pooling layer - (32, 110, 110)
        self.conv2 = nn.Conv2d(32, 64,  5)         # Output - (64, 106, 106), with Pooling layer - (64, 53, 53)
        self.conv3 = nn.Conv2d(64, 128, 3)         # Output - (128, 51, 51),  with Pooling layer - (128, 25, 25)
        self.conv4 = nn.Conv2d(128, 256, 3)        # Output - (256, 23, 23),  with Pooling layer - (256, 11, 11)
        self.conv5 = nn.Conv2d(256, 512, 3)        # Output - (512, 9, 9),    with Pooling layer - (512, 4, 4)
        
        # maxpooling layer
        self.pool = nn.MaxPool2d(2, 2)
        
        # fully-connected layers
        self.fc1 = nn.Linear(512 * 4 * 4, 1024)    # Output - 1024 features
        self.fc2 = nn.Linear(1024, 512)            # Output - 512 features
        self.fc3 = nn.Linear(512, 136)             # Output - 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # dropout layer to avoid overfitting
        self.drop1 = nn.Dropout(0.2)

    def forward(self, x):
        # three conv/relu + pool layers
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = self.pool(F.relu(self.conv5(x)))

        # prep for linear layer
        # flatten all dimensions except batch
        x = torch.flatten(x, 1)
        
        # three linear layers with dropout in between
        x = F.relu(self.fc1(x))
        x = self.drop1(x)
        x = F.relu(self.fc2(x))
        x = self.drop1(x)
        x = self.fc3(x)
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x
