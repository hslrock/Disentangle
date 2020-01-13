import torch
from torch.autograd import Variable
from torch import nn

class TransNet(torch.nn.Module):
    def __init__(self,nc,nd,channel1,channel2,channel3):
        super().__init__()
        self.fc1=nn.Linear(nc, nd)
        self.nc=nc
        self.fcchannel1=nn.Linear(nd,channel1)
        self.fcchannel2=nn.Linear(nd,channel2)
        self.fcchannel3=nn.Linear(nd,channel3)
        self.fc3=nn.Linear(nc,nd)
        self.fc4=nn.Linear(nd, nc)
        
    def disentangle(self,x):
        z1=self.fcchannel1(x)
        z2=self.fcchannel2(x)
        z3=self.fcchannel3(x)
            
        return z1,z2,z3
        
    def entangle(self,z1,z2,z3):
        z=torch.cat((z1,z2,z3),1)
        return self.fc4(self.fc3(z))
    def forward(self,x):
        x=x.view(-1,self.nc)   
        z=self.fc1(x) 
        z1,z2,z3=self.disentangle(z)
        z_hat=self.entangle(z1,z2,z3)
            
        return z_hat,z1,z2,z3
    
class BCE_Classification(torch.nn.Module):
    def __init__(self,nc):
        super().__init__()
        self.f1=nn.Linear(nc,nc*2)
        self.f2=nn.Linear(nc*2,1)
        self.sigmoid=nn.Sigmoid()
    def forward(self,x):
        x=self.sigmoid(self.f2(self.f1(x)))
        return x
        
class phi(torch.nn.Module):
    def __init__(self,nc,nd,channel1,channel2,channel3):
        super().__init__()
        self.fc1=nn.Linear(nc, nd)
        self.nc=nc
        self.fcchannel1=nn.Linear(nd,channel1)
        self.fcchannel2=nn.Linear(nd,channel2)
        self.fcchannel3=nn.Linear(nd,channel3)
    def disentangle(self,x):
        z1=self.fcchannel1(x)
        z2=self.fcchannel2(x)
        z3=self.fcchannel3(x)
            
        return z1,z2,z3
    def forward(self,x):
        x=x.view(-1,self.nc)   
        z=self.fc1(x) 
        z1,z2,z3=self.disentangle(z)
        
class invphi():
    def __init__(self,nc,nd,channel1,channel2,channel3):
        super().__init__()
        self.fc1=nn.Linear(nc, nd)
        self.fc2=nn.Linear(nd, nc)
        self.nc=nc
        
        
    def entangle(self,z1,z2,z3):
        z=torch.cat((z1,z2,z3),1)
        return self.fc4(self.fc3(z))
    def