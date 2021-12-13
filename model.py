import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class Feature(nn.Module):
    def __init__(self):
        super(Feature,self).__init__()
            
        # 3 * 224 * 224
        self.conv1_1 = nn.Conv2d(3, 64, 3) # 64 * 222 * 222
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=(1, 1)) # 64 * 222* 222
        self.maxpool1 = nn.MaxPool2d((2, 2), padding=(1, 1)) # pooling 64 * 112 * 112
        
        self.conv2_1 = nn.Conv2d(64, 128, 3) # 128 * 110 * 110
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=(1, 1)) # 128 * 110 * 110
        self.maxpool2 = nn.MaxPool2d((2, 2), padding=(1, 1)) # pooling 128 * 56 * 56
        
        self.conv3_1 = nn.Conv2d(128, 256, 3) # 256 * 54 * 54
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=(1, 1)) # 256 * 54 * 54
        self.conv3_3 = nn.Conv2d(256, 256, 3, padding=(1, 1)) # 256 * 54 * 54
        self.maxpool3 = nn.MaxPool2d((2, 2), padding=(1, 1)) # pooling 256 * 28 * 28
        
        self.conv4_1 = nn.Conv2d(256, 512, 3) # 512 * 26 * 26
        self.conv4_2 = nn.Conv2d(512, 512, 3, padding=(1, 1)) # 512 * 26 * 26
        self.conv4_3 = nn.Conv2d(512, 512, 3, padding=(1, 1)) # 512 * 26 * 26
        self.maxpool4 = nn.MaxPool2d((2, 2), padding=(1, 1)) # pooling 512 * 14 * 14
        
        self.conv5_1 = nn.Conv2d(512, 512, 3) # 512 * 12 * 12
        self.conv5_2 = nn.Conv2d(512, 512, 3, padding=(1, 1)) # 512 * 12 * 12
        self.conv5_3 = nn.Conv2d(512, 512, 3, padding=(1, 1)) # 512 * 12 * 12
        self.maxpool5 = nn.MaxPool2d((2, 2), padding=(1, 1)) # pooling 512 * 7 * 7
  
    def forward(self, x):
        
        # x.size(0)即为batch_size
        in_size = x.size(0)
        
        out = self.conv1_1(x) # 222
        out = F.relu(out)
        out = self.conv1_2(out) # 222
        out = F.relu(out)
        out = self.maxpool1(out) # 112
        
        out = self.conv2_1(out) # 110
        out = F.relu(out)
        out = self.conv2_2(out) # 110
        out = F.relu(out)
        out = self.maxpool2(out) # 56
        
        out = self.conv3_1(out) # 54
        out = F.relu(out)
        out = self.conv3_2(out) # 54
        out = F.relu(out)
        out = self.conv3_3(out) # 54
        out = F.relu(out)
        out = self.maxpool3(out) # 28
        
        out = self.conv4_1(out) # 26
        out = F.relu(out)
        out = self.conv4_2(out) # 26
        out = F.relu(out)
        out = self.conv4_3(out) # 26
        out = F.relu(out)
        out = self.maxpool4(out) # 14
        
        out = self.conv5_1(out) # 12
        out = F.relu(out)
        out = self.conv5_2(out) # 12
        out = F.relu(out)
        out = self.conv5_3(out) # 12
        out = F.relu(out)
        out = self.maxpool5(out) # 7
        return out

class Model(nn.Module):
    def __init__(self, num_classes=2):
        super(Model, self).__init__()
        net = models.vgg16(pretrained=True) 
        net.classifier = nn.Sequential()
        self.features = net
        
        # self.features = Feature()       
        self.classifier = nn.Sequential(
                nn.Linear(512 * 7 * 7, 256),
                nn.ReLU(True),
                nn.Dropout(),
                # nn.Linear(512, 256),
                # nn.ReLU(True),
                # nn.Dropout(),
                nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x