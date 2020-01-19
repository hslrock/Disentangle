#train_disentangle.py

import torch
from torch import nn,optim
import itertools
import time
import numpy as np
import random

from .loss import TripleletLoss,reconstruction_loss_phi,cyclic_loss


def train(model_list,train_load,num_epochs=40,device=None):
    Encoder=model_list[0]
    Decoder=model_list[1]
    phi=model_list[2]
    invphi=model_list[3]
    
    
    opt_phi = optim.Adam(phi.parameters(), lr=0.0001, betas=(0.9, 0.999))
    opt_invphi = optim.Adam(invphi.parameters(), lr=0.001, betas=(0.9, 0.999))  
    params = [phi.parameters(), invphi.parameters()]
    opt_transform=optim.Adam(itertools.chain(*params),lr=0.001,betas=(0.9,0.999))
    
    
    loss_matrix=None
    first=True
    t_start = time.time()
    Encoder.eval()
    Decoder.eval()
    phi.train()
    invphi.train()
    for epoch in range(num_epochs):
        print("Epoch:", epoch)
        for batch_i, (real_images, gender,glasses) in enumerate(train_load):
            batch_size = real_images.size(0)
            real_images=real_images.to(device,dtype=torch.float)
            latent_vector=Encoder(real_images).detach()
            glass_vector,gender_vector,remain=phi(latent_vector)
            
            #Reconstruction Loss
            z_tilde=invphi(torch.cat((glass_vector,gender_vector,remain),1))
            loss_reconstruction=reconstruction_loss_phi(latent_vector,z_tilde,opt_transform)
            
            
            #Task Loss
            opt_phi.zero_grad()       
            loss=TripleletLoss(glass_vector,glasses) +    TripleletLoss(gender_vector,gender)  
            loss.backward(retain_graph=True)
            opt_phi.step()

            
            #Cyclic Loss
            loss_cycle=cyclic_loss(glass_vector,gender_vector,remain,glasses,gender,opt_transform,model_list)
            
            
            if (batch_i) % 300 == 0:
                print("Batch: ", batch_i)
                print("Task Loss: ", loss.item())
                print("Reconstruction Loss: ",loss_reconstruction.item())
                print("Cyclic Loss: ",loss_cycle.item())
                if first:
                    loss_matrix=np.array((loss.item(),loss_reconstruction.item(),loss_cycle.item()))
                    first=False
                else:
                    loss_matrix=np.vstack((loss_matrix,np.array((loss.item(),loss_reconstruction.item(),loss_cycle.item()))))
        t_end = time.time()
        duration_avg = (t_end - t_start) / (epoch + 1.0)
        print("Elapsed Time: ",duration_avg)
        torch.save(phi,'Phi.h')
        torch.save(invphi,'invphi.h')
    return loss_matrix