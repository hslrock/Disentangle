import torch
from torch.autograd import Variable
from torch import nn
import math

    

        
class phi(torch.nn.Module):
    def __init__(self,nc=99,nd=1000,inv=False):
        super().__init__()
        self.nc=nc
        self.nd=nd
        self.fc1=nn.Linear(nc, nd)
        self.fc2=nn.Linear(nd, nc)
        self.inv=inv
    def disentangle(self,x):
        z1=x[:,:33]
        z2=x[:,33:66]
        z3=x[:,66:99]
        return z1,z2,z3
    def forward(self,x):
        x=x.view(-1,self.nc)   
        z=self.fc2(self.fc1(x) )
        if  self.inv:
            return z
        else:
            z1,z2,z3=self.disentangle(z)
            return z1,z2,z3
        
class full_phi(torch.nn.Module):    
    def __init__(self,phi, invphi,nc=99):
        
        super().__init__()
        self.nc=nc
        self.Encoder=phi
        self.Decoder=invphi
    def forward(self, x):
        x=x.view(-1,self.nc) 
        z = self.Encoder.forward(x)
        result = self.Decoder.forward(z)
        
        return result
