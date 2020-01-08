#!/usr/bin/env python
# coding: utf-8

# In[6]:




import torch
from torch import nn, optim
from torchvision import transforms, utils
import matplotlib.pyplot as plt
from modules import EncoderNet,DecoderNet,DiscriminatorNet_reconstruction,GeneratorNet


import time
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_default_tensor_type(torch.cuda.FloatTensor)
loss = nn.BCELoss()

def real_data_target(size):
    '''
    Tensor containing ones, with shape = size
    '''
    data = torch.ones(size, 1, 8, 8)
    if torch.cuda.is_available(): return data.cuda()
    return data
def fake_data_target(size):
    '''
    Tensor containing zeros, with shape = size
    '''
    data = torch.zeros(size, 1, 8, 8)
    if torch.cuda.is_available(): return data.cuda()
    return data


# In[8]:


def dloss_calc(optimizer, real_data, fake_data,Discriminator):
    # Reset gradients
    optimizer.zero_grad()
    prediction_real = Discriminator(real_data)
    error_real = loss(prediction_real, real_data_target(real_data.size(0)))
    error_real.backward()

    # 1.2 Train on Fake Data
    prediction_fake = Discriminator(fake_data)
    # Calculate error and backpropagate
    # Error is measured against fake targets
    error_fake = loss(prediction_fake, fake_data_target(real_data.size(0)))
    error_fake.backward()
    
    # 1.3 Update weights with gradients
    optimizer.step()
    
    # Return error
    return error_real + error_fake, prediction_real, prediction_fake

def gloss_calc(optimizer, fake_data,Discriminator):
    # 2. Train Generator
    # Reset gradients
    optimizer.zero_grad()
    # Sample noise and generate fake data
    prediction = Discriminator(fake_data)
    # Calculate error and backpropagate
    # Note that here the error is pretending to be real
    error = loss(prediction, real_data_target(prediction.size(0)))
    error.backward()
    # Update weights with gradients
    optimizer.step()
    # Return error
    return error


def noise(size):
    n = torch.randn(size, 3,64,64)
    if torch.cuda.is_available(): return n.cuda() 
    return n

def eval_generate(Generator,num_images):
    Generator.eval()
    noi_input=noise(num_images)
    output=Generator(noi_input)
    output=output.detach().cpu()
    plt.figure(figsize=(16, 16))
    batch_size = len(output)
    grid_border_size = 2
    grid = utils.make_grid(output)
    
    plt.imshow(grid.numpy().transpose((1, 2, 0)))
    plt.axis('off')
    plt.ioff()
    plt.show()
    
    
def train(num_epochs,Generator,Discriminator,d_optimizer,g_optimizer,train_load):
    t_start = time.time()
    duration_avg = 0.0
    Generator.train()
    Discriminator.train()
    for epoch in range(num_epochs):
        print("Epoch:", epoch)
        for batch_i, (real_images, gender,glasses) in enumerate(train_load):        
 
            batch_size = real_images.size(0)
            real_images=real_images.to(device,dtype=torch.float)
            noi = noise(real_images.size(0))
            
            # 1.Train Discriminator
            fake_data = Generator(noi).detach()
   

            d_error, d_pred_real, d_pred_fake =dloss_calc(d_optimizer,                                    
                            real_images.float(), fake_data,Discriminator)
    
            # 2. Train Generator

        
            fake_data = Generator(noi)#noise(real_batch.size(0)))
            # Train G
            g_error = gloss_calc(g_optimizer, fake_data,Discriminator)
            
            
            
            # Display Progress
            if (batch_i) % 100 == 0:
                print("Discriminator_Error: ", d_error.item()," Generator_Error: ", g_error.item())
    
        t_end = time.time()
        duration_avg = (t_end - t_start) / (epoch + 1.0)
        print("Elapsed Time: ",duration_avg)
        torch.save(Generator,'Generator.h')
        torch.save(Discriminator,'Discriminator.h')
        eval_generate(Generator,8)





    
    