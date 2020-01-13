import torch
from torch.autograd import Variable
from torch import nn


    
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
    def __init__(self,nc=33,nd=1000):
        super().__init__()
        self.fc1=nn.Linear(nc, nd)
        self.fc2=nn.Linear(nc, nd)
    def disentangle(self,x):
        z1=x[0:33]
        z2=x[0:66]
        z3=x[0:99]
        return z1,z2,z3
    def forward(self,x):
        x=x.view(-1,self.nc)   
        z=self.fc1(x) 
        z1,z2,z3=self.disentangle(z)
        return 
        
