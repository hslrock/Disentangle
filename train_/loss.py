#train_disentangle.py

import torch
from torch import nn,optim
import itertools
import time
import numpy as np
import random

from functional.functional import noise_vector,real_data_target,fake_data_target,real_feature_target,fake_feature_target

def TripleletLoss(batch,targetAttribute):
    def triplet(value, positive, negative, margin=0.2) : 
        d = nn.PairwiseDistance(p=2)
        distance = d(value, positive) - d(value, negative) + margin 
        loss = torch.mean(torch.max(distance, torch.zeros_like(distance))) 
        return loss
    
    def findtriplet(src,attribute):
        timeout_start = time.time()
        index_list=np.arange(len(attribute)).tolist()
        rand=random.sample(index_list,len(attribute))
        for i,posindex in enumerate(rand):
            if attribute[src]==attribute[posindex]:
                if src != posindex:
                        break      
            if i==len(attribute)-1:
                posindex=src            
        rand=random.sample(index_list,len(attribute))                
        for i,negindex in enumerate(rand):
            if(attribute[src] !=attribute[negindex]):
                break   
            if i==len(attribute)-1:
                negindex=src
                
        return posindex,negindex
    loss=0
    pos_pair=None
    for i,value in enumerate(batch):
        posindex,negindex=findtriplet(i,targetAttribute)

        if not i:

            pos_pair=batch[posindex].unsqueeze(0)
            neg_pair=batch[negindex].unsqueeze(0)
        else:
            pos_pair=torch.cat((pos_pair,batch[posindex].unsqueeze(0)),0)
            neg_pair=torch.cat((neg_pair,batch[negindex].unsqueeze(0)),0)


    return triplet(batch,pos_pair,neg_pair)

def reconstruction_loss_phi(z,z_tilde,optimizer):
    loss = nn.L1Loss()
    optimizer.zero_grad()
    error_recons=loss(z,z_tilde)
    error_recons.backward(retain_graph=True)
    optimizer.step()
    return error_recons

def _concat(z_list):
    return torch.cat((z_list[0],z_list[1],z_list[2]),1)

def cyclic_loss(z1,z2,z3,true_glasses,true_gender,opt_transform,model_list):
    Encoder=model_list[0]
    Decoder=model_list[1]
    phi=model_list[2]
    invphi=model_list[3]
    batch_size=z1.size(0)
    swapped_pos=torch.randperm(batch_size)   
    z1_hat = z1[swapped_pos]   #Permutation
    true_glasses=true_glasses[swapped_pos]  
    swapped_pos=torch.randperm(batch_size)
    true_gender=true_glasses[swapped_pos]
    z2_hat=z2[swapped_pos]
    swapped_pos=torch.randperm(batch_size)
    z3_hat=z3[swapped_pos]
    true_gender=true_glasses[true_glasses]
    z_aster=torch.cat((z1_hat,z2_hat,z3),1)
    recontructed_z_aster=_concat(phi(Encoder(Decoder(invphi(z_aster)))))

    
    #Cycle_Consistency,Loss                 
    opt_transform.zero_grad()
    loss = nn.MSELoss()                                                     
    consistency_loss = loss(z_aster,recontructed_z_aster)
    consistency_loss.backward(retain_graph=True)
    opt_transform.step()
    
    
    #attr_cycle_augmentation_loss
    opt_transform.zero_grad()
    augmentation_loss =TripleletLoss(z1_hat,true_glasses) + TripleletLoss(z2_hat,true_gender)
    augmentation_loss.backward()
    opt_transform.step()
 
    return consistency_loss +augmentation_loss       


def reconstruction_loss_ae(optimizer,real_data,Encoder,Decoder):
    reconstruction=Decoder(Encoder(real_data))
    loss = nn.L1Loss()
    optimizer.zero_grad()
    error_recons=loss(real_data,reconstruction)*0.9
    error_recons.backward()
    optimizer.step()
    return error_recons
def adv_feature_loss(minimizing,real_feature,fake_feature,Discriminator,optimizer1,optimizer2):
    loss=nn.BCELoss()
    if minimizing:
        optimizer1.zero_grad()
        prediction_real = Discriminator(real_feature)
        error_real = loss(prediction_real, real_feature_target(real_feature.size(0)))
        error_real.backward()   
        
        prediction_fake = Discriminator(fake_feature)
        error_fake = loss(prediction_fake, fake_feature_target(real_feature.size(0)))
        error_fake.backward()
        total_error=error_real+error_fake
        optimizer1.step()    
        return error_real+error_fake
    else:
        
        optimizer2.zero_grad()
        prediction_fake=Discriminator(fake_feature)
        error_fake = loss(prediction_fake, real_feature_target(real_feature.size(0)))
        error_fake.backward()
        optimizer2.step()    
        return error_fake


def adv_img_loss(minimizing,real_data,fake_data,Discriminator,optimizer1,optimizer2):
    loss = nn.BCELoss()
    def dloss_calc_adv(optimizer, real_data, fake_data,Discriminator):        
        optimizer.zero_grad()
        prediction_real = Discriminator(real_data)
        error_real = loss(prediction_real, real_data_target(real_data.size(0)))
        error_real.backward(retain_graph=True)
        prediction_fake = Discriminator(fake_data)
        error_fake = loss(prediction_fake, fake_data_target(real_data.size(0)))
        error_fake.backward(retain_graph=True)
        optimizer.step()
        return error_real + error_fake
    def gloss_calc_adv(optimizer, real_data,fake_data,Discriminator):
        optimizer.zero_grad()
        prediction_fake=Discriminator(fake_data_target)
        error_fake = loss(prediction_fake, real_data_target(real_data.size(0)))
        error_fake.backward(retain_graph=True)
        optimizer.step()
       
    if minimizing:
        return dloss_calc_adv(optimizer1,real_data,fake_data,Discriminator)
    else:
        return gloss_calc_adv(optimizer2,real_data,fake_data,Discriminator)
    
    
def gen_image_loss(minimizing,real_data, fake_data,Discriminator, optimizer1,optimizer2,weight=0.8):
    def dloss_calc(optimizer, real_data, fake_data,Discriminator):
        optimizer.zero_grad()
        prediction_real = Discriminator(real_data)
        error_real = loss(prediction_real, real_data_target(real_data.size(0)))*weight
        error_real.backward()
        prediction_fake = Discriminator(fake_data)
        error_fake = loss(prediction_fake, fake_data_target(real_data.size(0)))*weight
        error_fake.backward()
        optimizer.step()
        return error_real + error_fake, prediction_real, prediction_fake
    def gloss_calc(optimizer, fake_data,Discriminator):
        optimizer.zero_grad()
        prediction = Discriminator(fake_data)
        error = loss(prediction, real_data_target(prediction.size(0)))*weight
        error.backward()
        optimizer.step()
        return error
    
    loss = nn.BCELoss()
        
    if minimizing:
        return dloss_calc(optimizer1,real_data,fake_data,Discriminator)
    else:
        return gloss_calc(optimizer2,fake_data,Discriminator)
    
def full_reconstuction_loss(optimizer,real_data,reconstructed_data):
    loss = nn.L1Loss()
    optimizer.zero_grad()
    error_recons=loss(real_data,reconstructed_data)*0.9
    error_recons.backward()
    optimizer.step()
    return error_recons
        
