import torch
from torch import nn,optim
import itertools


def _real_data_target(size):
    data = torch.ones(size, 1, 8, 8)
    if torch.cuda.is_available(): return data.cuda()
    return data

def _fake_data_target(size):
    data = torch.zeros(size, 1, 8, 8)
    if torch.cuda.is_available(): return data.cuda()
    return data

def _real_feature_target(size):
    data = torch.ones(size,1)
    if torch.cuda.is_available(): return data.cuda()
    return data

def _fake_feature_target(size):
    data = torch.zeros(size,1)
    if torch.cuda.is_available(): return data.cuda()
    return data

def _reconstruction_loss(optimizer,real_data,Encoder,Decoder):
    reconstruction=Decoder(Encoder(real_data))
    loss = nn.L1Loss()
    optimizer.zero_grad()
    error_recons=loss(real_data,reconstruction)*0.9
    error_recons.backward()
    optimizer.step()
    return error_recons
def _adv_feature_loss(minimizing,real_feature,fake_feature,Discriminator,optimizer1,optimizer2):
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


def _adv_img_loss(minimizing,real_data,fake_data,Discriminator,optimizer1,optimizer2):
    loss = nn.BCELoss()
    def dloss_calc_adv(optimizer, real_data, fake_data,Discriminator):
        
        optimizer.zero_grad()
        prediction_real = Discriminator(real_data)
        error_real = loss(prediction_real, real_data_target(real_data.size(0)))
        error_real.backward()
        prediction_fake = Discriminator(fake_data)
        error_fake = loss(prediction_fake, fake_data_target(real_data.size(0)))
        error_fake.backward()
        optimizer.step()
        return error_real + error_fake
    def gloss_calc_adv(optimizer, real_data,fake_data,Discriminator):
        optimizer.zero_grad()
        prediction_fake=Discriminator(fake_data_target)
        error_fake = loss(prediction_fake, real_data_target(real_data.size(0)))
        error_fake.backward()
        optimizer.step()
       
    if minimizing:
        return dloss_calc_adv(optimizer1,real_data,fake_data,Discriminator)
    else:
        return gloss_calc_adv(optimizer2,real_data,fake_data,Discriminator)
    
    
def _gen_image_loss(minimizing,real_data, fake_data,Discriminator, optimizer1,optimizer2,weight=0.8):
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

def noise_vector(size):
    n = torch.randn(size,99)
    if torch.cuda.is_available(): return n.cuda() 
    return n

def eval_generate(Decoder,num_images):
    Decoder.eval()
    noi_input=noise_vector(num_images)
    output=Decoder(noi_input)
    output=output.detach().cpu()
    plt.figure(figsize=(16, 16))
    grid_border_size = 2
    grid = utils.make_grid(output)
    
    plt.imshow((grid.numpy().transpose((1, 2, 0))*0.5)+0.5)
    plt.axis('off')
    plt.ioff()
    plt.show()
    
def train(Encoder,Decoder,Discriminator_reconstruct,Discriminator_feature,train_load,num_epochs=30):
    
    dr_optimizer = optim.Adam(Discriminator_reconstruct.parameters(), lr=0.00005, betas=(0, 0.999))
    df_optimizer = optim.Adam(Discriminator_feature.parameters(), lr=0.00005, betas=(0, 0.999))
    e_optimizer = optim.Adam((itertools.chain(*[Decoder.parameters(), Generator.parameters()]), lr=0.00005, betas=(0, 0.999))
    d_optimizer = optim.Adam(Decoder.parameters(), lr=0.00005, betas=(0, 0.999))
    g_optimizer = optim.Adam(Generator.parameters(), lr=0.00005, betas=(0, 0.999))
    
    Encoder.train()
    Decoder.train()
    Discriminator_reconstruct.train()
    Discriminator_feature.train()
                                 
    for epoch in range(num_epochs):
        
        t_start = time.time()                    
                             
        print("Epoch:", epoch)
        for batch_i, (real_images, gender,glasses) in enumerate(train_load):        
 
            batch_size = real_images.size(0)
            real_images=real_images.to(device,dtype=torch.float)
            
            # Train Reconstruction
            
            recons_loss=reconstruction_loss(g_optimizer,real_images.float(),Encoder,Decoder)            
            
            # Train on Adversarial Feature Loss
                #.Discrminator
            noi = noise_vector(real_images.size(0))
            real_feature=Encoder(real_images.float()).detach()
            df_error_adv=adv_feature_loss(True,noi,real_feature,Discriminator_feature,df_optimizer,e_optimizer)
                #.Generator
            real_feature=Encoder(real_images.float())   
            en_error_adv=adv_feature_loss(False,noi,real_feature,Discriminator_feature,df_optimizer,e_optimizer)
            
            #Adversarial Image Loss
                #.Discriminator
            fake_data=Decoder(Encoder(real_images.float())).detach()
            d_error1=adv_img_loss(True,real_images.float(),fake_data, Discriminator_reconstruct,dr_optimizer,g_optimizer)
                #.Generator
            fake_data=Decoder(Encoder(real_images.float()))
            g_error2=adv_img_loss(True,real_images.float(),fake_data, Discriminator_reconstruct,dr_optimizer,g_optimizer)

             # Generative Image Loss
                #.Discriminator
            fake_data = Decoder(noi).detach()
            d_error, d_pred_real, d_pred_fake =gen_image_loss(True,real_images.float(),fake_data,Discriminator_reconstruct,
                                                              dr_optimizer,d_optimizer)
                #. Generator                   
            fake_data = Decoder(noi)#noise(real_batch.size(0)))
            g_error = gen_image_loss(False, real_images.float(), fake_data,Discriminator_reconstruct,
                                    dr_optimizer,d_optimizer)
            
            # Display Progress
            if (batch_i) % 300 == 0:
                print("Batch: ", batch_i)
                print("1:Discriminator_Error: ", d_error.item()," Generator_Error: ", g_error.item()," Recons_Error: ", recons_loss.item())
                print("2:Feature Discriminator Error: ",df_error_adv.item(),"Encoder Error: ", en_error_adv.item())
                print("3 Discriminator_adv_error", d_error1.item(), "Generator_error: ", g_error2.item())
        t_end = time.time()
        print("Elapsed Time: ",duration_avg)
        torch.save(Encoder,'Encoder_64batch.h')
        torch.save(Decoder,'Decoder_64batch.h')
        torch.save(Discriminator_feature,'Discriminator_feature_64batch.h')
        torch.save(Discriminator_reconstruct,'Discriminator_reconstruct_64batch.h')
        eval_generate(Decoder,8)
        