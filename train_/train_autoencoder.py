import torch
from torch import nn,optim
import itertools
from functional.functional import noise_vector
from eval.eval import eval_generate
import time

from .loss import reconstruction_loss_ae,adv_img_loss,gen_image_loss,adv_feature_loss,noise_vector


def train(AE_model,train_load,num_epochs=30,device=None):
    Encoder=AE_model[0]
    Decoder=AE_model[1]
    Discriminator_reconstruct=AE_model[2]
    Discriminator_feature=AE_model[3]
    
    
    dr_optimizer = optim.Adam(Discriminator_reconstruct.parameters(), lr=0.00005, betas=(0, 0.999))
    df_optimizer = optim.Adam(Discriminator_feature.parameters(), lr=0.00005, betas=(0, 0.999))
    g_optimizer = optim.Adam((itertools.chain(*[Encoder.parameters(), Decoder.parameters()])), lr=0.00005, betas=(0, 0.999))
    d_optimizer = optim.Adam(Decoder.parameters(), lr=0.00005, betas=(0, 0.999))
    e_optimizer = optim.Adam(Encoder.parameters(), lr=0.00005, betas=(0, 0.999))
    
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
            
            recons_loss=reconstruction_loss_ae(g_optimizer,real_images.float(),Encoder,Decoder)            
            
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
            if (batch_i) % 600 == 0:
                print("Batch: ", batch_i)
                print("1:Discriminator_Error: ", d_error.item()," Generator_Error: ", g_error.item()," Recons_Error: ",                                recons_loss.item())
                print("2:Feature Discriminator Error: ",df_error_adv.item(),"Encoder Error: ", en_error_adv.item())
                print("3 Discriminator_adv_error", d_error1.item(), "Generator_error: ", g_error2.item())
                eval_generate(Decoder,8)
        t_end = time.time()
        duration_avg = (t_end - t_start) / (epoch + 1.0)
        
        print("Elapsed Time: ",duration_avg)
        torch.save(Encoder,'Encoder.h')
        torch.save(Decoder,'Decoder.h')
        torch.save(Discriminator_feature,'Discriminator_f.h')
        torch.save(Discriminator_reconstruct,'Discriminator_r.h')
        