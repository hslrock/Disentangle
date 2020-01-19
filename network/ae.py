    
import torch
from torch.autograd import Variable
from torch import nn

class DiscriminatorNet_feature(torch.nn.Module):
    def __init__(self,latent_size=99):
        super().__init__()
        self.latent_size=latent_size
        
        self.fc1 = nn.Linear(latent_size, 1000)
        self.dout1=nn.Dropout()
        self.fc2 = nn.Linear(1000, 1000)
        self.dout2=nn.Dropout()
        self.fc3 = nn.Linear(1000, 1)
        self.bn1 = nn.BatchNorm1d(1000, 1.e-3)
        self.bn2 = nn.BatchNorm1d(1000, 1.e-3)
        self.relu = nn.ReLU()
        self.sigmoid=nn.Sigmoid()
    def forward(self,x):
        x=self.relu(self.dout1(self.bn1(self.fc1(x))))
        x=self.relu(self.dout1(self.bn2(self.fc2(x))))
        return self.sigmoid(self.fc3(x))
        
        
        
class DiscriminatorNet_reconstruction(torch.nn.Module):
        def __init__(self,img_channel,img_size):
            self.img_channel=img_channel
            self.img_size=img_size
            super().__init__()
            self.conv1 = nn.Sequential(
                nn.Conv2d(
                    in_channels=3, out_channels=32, kernel_size=4,
                    stride=2, padding=1, bias=False
                ),
                nn.LeakyReLU(0.2, inplace=True)
            )
            self.conv2 = nn.Sequential(
                nn.Conv2d(
                    in_channels=32, out_channels=64, kernel_size=4,
                    stride=2, padding=1, bias=False
                ),
                nn.BatchNorm2d(64),
                nn.LeakyReLU(0.2, inplace=True)
            )
            self.conv3 = nn.Sequential(
                nn.Conv2d(
                    in_channels=64, out_channels=128, kernel_size=4,
                    stride=2, padding=1, bias=False
                ),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(0.2, inplace=True)
            )
            self.out = nn.Sequential(
                nn.Conv2d(in_channels=128, out_channels=1, kernel_size=1, stride=1, padding=0, bias=False),
                nn.Sigmoid(),
            ) 
        def forward(self, x):   
            x=x.view(-1,self.img_channel,self.img_size,self.img_size)
            h1 = self.conv1(x)
            h2 = self.conv2(h1)
            h3 = self.conv3(h2)

            return self.out(h3)

    