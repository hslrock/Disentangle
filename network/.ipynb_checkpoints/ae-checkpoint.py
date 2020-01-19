
import torch
from torch.autograd import Variable
from torch import nn
class EncoderNet(torch.nn.Module):
        def __init__(self, nc, ndf, latent_variable_size):
            super(EncoderNet, self).__init__() 
            self.nc = nc
            self.ndf = ndf
            self.latent_variable_size = latent_variable_size
            self.e1 = nn.Conv2d(nc, ndf, 4, 2, 1)
            self.bn1 = nn.BatchNorm2d(ndf)
            self.e2 = nn.Conv2d(ndf, ndf*2, 4, 2, 1)
            self.bn2 = nn.BatchNorm2d(ndf*2)
            self.e3 = nn.Conv2d(ndf*2, ndf*4, 4, 2, 1)
            self.bn3 = nn.BatchNorm2d(ndf*4)
            self.e4 = nn.Conv2d(ndf*4, ndf*8, 4, 2, 1)
            self.bn4 = nn.BatchNorm2d(ndf*8)
            self.e5 = nn.Conv2d(ndf*8, ndf*8, 4, 2, 1)
            self.bn5 = nn.BatchNorm2d(ndf*8)
            self.fc1 = nn.Linear(ndf*8*4*4, latent_variable_size)
            self.fc2 = nn.Linear(ndf*8*4*4, latent_variable_size)
            self.leakyrelu = nn.LeakyReLU(0.2)
            self.relu = nn.ReLU(),
            self.sigmoid = nn.Sigmoid()
      
        def encode(self, x):
            h1 = self.leakyrelu(self.bn1(self.e1(x)))
            h2 = self.leakyrelu(self.bn2(self.e2(h1)))
            h3 = self.leakyrelu(self.bn3(self.e3(h2)))
            h4 = self.leakyrelu(self.bn4(self.e4(h3)))
            h5 = self.leakyrelu(self.bn5(self.e5(h4)))
            h5 = h5.view(-1, self.ndf*8*4*4)
            return self.fc1(h5)
        def forward(self, x):
            x=x.view(-1,self.nc,self.ndf,self.ndf)
            z = self.encode(x)
            return z

class DecoderNet(torch.nn.Module):
        def __init__(self, nc, ngf, latent_variable_size):    
            super(DecoderNet, self).__init__()
            self.nc = nc
            self.ngf = ngf
            self.latent_variable_size = latent_variable_size
            self.leakyrelu = nn.LeakyReLU(0.2)
            self.relu = nn.ReLU()
            self.sigmoid = nn.Sigmoid()
            self.d1 = nn.Linear(latent_variable_size, ngf*8*2*4*4)
            self.up1 = nn.UpsamplingNearest2d(scale_factor=2)
            self.pd1 = nn.ReplicationPad2d(1)
            self.d2 = nn.Conv2d(ngf*8*2, ngf*8, 3, 1)
            self.bn6 = nn.BatchNorm2d(ngf*8, 1.e-3)
            self.up2 = nn.UpsamplingNearest2d(scale_factor=2)
            self.pd2 = nn.ReplicationPad2d(1)
            self.d3 = nn.Conv2d(ngf*8, ngf*4, 3, 1)
            self.bn7 = nn.BatchNorm2d(ngf*4, 1.e-3)
            self.up3 = nn.UpsamplingNearest2d(scale_factor=2)
            self.pd3 = nn.ReplicationPad2d(1)
            self.d4 = nn.Conv2d(ngf*4, ngf*2, 3, 1)
            self.bn8 = nn.BatchNorm2d(ngf*2, 1.e-3)
            self.up4 = nn.UpsamplingNearest2d(scale_factor=2)
            self.pd4 = nn.ReplicationPad2d(1)
            self.d5 = nn.Conv2d(ngf*2, ngf, 3, 1)
            self.bn9 = nn.BatchNorm2d(ngf, 1.e-3)
            self.up5 = nn.UpsamplingNearest2d(scale_factor=2)
            self.pd5 = nn.ReplicationPad2d(1)
            self.d6 = nn.Conv2d(ngf, nc, 3, 1)

        def decode(self, z):
            h1 = self.relu(self.d1(z))
            h1 = h1.view(-1, self.ngf*8*2, 4, 4)
            h2 = self.leakyrelu(self.bn6(self.d2(self.pd1(self.up1(h1)))))
            h3 = self.leakyrelu(self.bn7(self.d3(self.pd2(self.up2(h2)))))
            h4 = self.leakyrelu(self.bn8(self.d4(self.pd3(self.up3(h3)))))
            h5 = self.bn9(self.d5(self.pd4(self.up4(h4))))
            return self.sigmoid(self.d6(self.pd5(self.up5(h5))))
        def forward(self, x):
            x=x.view(-1,self.latent_variable_size)
            result = self.decode(x)
            return result

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
