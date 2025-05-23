import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvNet(nn.Module):
    
    def __init__(self):
        super(ConvNet,self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=3,out_channels=8,stride=1,kernel_size=(3,3),padding=1)
        self.conv2 = nn.Conv2d(in_channels=8,out_channels=32,kernel_size=(3,3),padding=1,stride=1)
        self.conv3 = nn.Conv2d(in_channels=32,out_channels=64,kernel_size=(3,3),padding=1,stride=1)
        self.conv4 = nn.Conv2d(in_channels=64,out_channels=128,kernel_size=(3,3),padding=1,stride=1)
        self.conv5 = nn.Conv2d(in_channels=128,out_channels=256,kernel_size=(3,3),stride=1)

        self.fc1 = nn.Linear(in_features=6*6*256,out_features=256)
        self.fc2 = nn.Linear(in_features=256,out_features=128)

        self.fc3 = nn.Linear(in_features=128,out_features=64)
        self.glu_full = nn.Linear(in_features=128 + 256 + 6 * 6 * 256,out_features=64)
        #self.glu_fc3 = nn.Linear(in_features=128,out_features=64)
        
        self.fc4 = nn.Linear(in_features=64,out_features=2)
        
        self.max_pool = nn.MaxPool2d(kernel_size=(2,2),stride=2)
        self.dropout = nn.Dropout2d(p=0.5)
        
    def forward(self,x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.max_pool(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = self.max_pool(x)
        x = self.conv5(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = x.view(-1,6*6*256)
        gx1 = x
        x = self.fc1(x)
        x = F.relu(x)
        gx2 = x
        x = self.fc2(x)
        x = F.relu(x)
        gx3 = x
        """
        h = self.fc3(x)
        hg = self.glu_full(torch.cat((gx1, gx2, gx3), dim=1))
        x = torch.mul(h, torch.sigmoid(hg))
        """
        x = self.fc3(x)
        x = F.relu(x)
        logits = self.fc4(x)
        
        return logits

class ClassifyModelCATSDOGS(nn.Module):
    def __init__(self):
        super(ClassifyModelCATSDOGS, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=[3, 3])
        self.maxpool_layer1 = nn.MaxPool2d(kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=[4, 4])
        self.maxpool_layer2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.linear1 = nn.Linear(32 * 12 * 12, 64)
        self.glu_linear1 = nn.Linear(32 * 12 * 12, 64)
        self.linear2 = nn.Linear(64, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.ReLU()(x)
        x = self.maxpool_layer1(x)
        x = self.conv2(x)
        x = nn.ReLU()(x)
        x = self.maxpool_layer2(x)
        x = x.view(-1, 32 * 12 * 12)
        #h = self.linear1(x)
        #hg = self.glu_linear1(x)
        #x = torch.mul(h, torch.sigmoid(hg))
        x = self.linear1(x)
        x = nn.Tanh()(x)
        x = self.linear2(x)
        x = nn.Softmax(dim=1)(x)
        return x 

class ClassifyModelMNIST(nn.Module):
    def __init__(self, h_only: bool = False, use_convnet = True):
        super(ClassifyModelMNIST, self).__init__()
        
        if use_convnet: 
            self.convnet_l1 = nn.Sequential(
                torch.nn.Conv2d(1, 64, kernel_size=[3, 3]),
                torch.nn.ReLU(),
                torch.nn.MaxPool2d(kernel_size=3, stride=1),                
            )
            self.convnet = nn.Sequential(
                torch.nn.Conv2d(64, 32, kernel_size=[4, 4]),
                torch.nn.ReLU(),
                torch.nn.MaxPool2d(kernel_size=3, stride=2),
                torch.nn.Flatten(1),
                torch.nn.Linear(32 * 10 * 10, 128),
                torch.nn.Tanh(),
                torch.nn.Linear(128, 10),
                torch.nn.Softmax(dim=1),
            )
            self.convnet_h = nn.Sequential(
                #torch.nn.Conv2d(1, 64, kernel_size=[3, 3]),
                #torch.nn.ReLU(),
                #torch.nn.MaxPool2d(kernel_size=3, stride=1),
                torch.nn.Conv2d(64, 32, kernel_size=[4, 4]),
                torch.nn.ReLU(),
                torch.nn.MaxPool2d(kernel_size=3, stride=2),
                torch.nn.Flatten(1),
            )
            """
            self.conv1 = nn.Conv2d(1, 64, kernel_size=[3, 3])
            self.maxpool_layer1 = nn.MaxPool2d(kernel_size=3, stride=1)
            self.conv2 = nn.Conv2d(64, 32, kernel_size=[4, 4])
            self.maxpool_layer2 = nn.MaxPool2d(kernel_size=3, stride=2)
            self.linear1 = nn.Linear(32 * 10 * 10, 128)
            #elf.glu_linear1 = nn.Linear(32 * 9 * 9, 128)
            self.linear2 = nn.Linear(128, 10)
            """
        else: 
            # dnn arch 
            self.linear1 = nn.Linear(784, 512)
            self.linear2 = nn.Linear(512, 128)
            self.linear3 = nn.Linear(128, 10)
            self.glu_linear2 = nn.Linear(512, 128)
            self.glu_linear3 = nn.Linear(128, 10)

        self.use_glu = False
        self.use_convnet = use_convnet
        self.h_only=h_only
        self.h_output_dim = 128

    def forward(self, x):
        if self.use_convnet: 
            return self.forward_convnet(x)
        else: 
            x = torch.flatten(x, start_dim=1)
            x = self.linear1(x)
            x = nn.ReLU()(x)
            
            if self.h_only and self.use_glu: 
                y = self.linear2(x)
                mask = self.glu_linear2(x)
                x = torch.mul(y, torch.sigmoid(mask))
            else: 
                x = self.linear2(x)

            x = nn.Tanh()(x)
            if self.h_only: 
                x = nn.Dropout(0.00)(x)
                return x
            x = self.linear3(x)
            x = nn.Softmax(dim=1)(x)
        return x

    def forward_convnet(self, x):
        if self.h_only: 
            l1_h = self.convnet_l1(x)
            x = self.convnet_h(l1_h)
        else: 
            l1_h = self.convnet_l1(x)
            x = self.convnet(l1_h)
        l1_h = nn.Flatten(1)(l1_h)
        return x, l1_h