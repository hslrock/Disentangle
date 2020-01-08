import torch

def noise(size):
    n = torch.randn(size, 3,64,64)
    if torch.cuda.is_available(): return n.cuda() 
    return n