#model list=[Encoder,Discriminator_z, Discriminator_x,Generator,Phi,InvPhi]

import torch
from torch import nn,optim
import itertools

from functional.functional import noise_vector

from .loss import *


def joint_train(model_list,train_load,num_epochs=30,device=None):
    for epoch in range(num_epochs):
        t_start = time.time()                    

        print("Epoch:", epoch)
        #model_list=[Encoder,Decoder,D_reconstruct,D_feature,Phi,Invphi]
        Encoder=model_list[0]
        Decoder=model_list[1]
        D_reconstruct=model_list[2]
        D_feature=model_list[3]
        Phi=model_list[4]
        Invphi=model_list[5]
        
        dr_optimizer = optim.Adam(D_reconstruct.parameters(), lr=0.00005, betas=(0, 0.999))
        df_optimizer = optim.Adam(D_feature.parameters(), lr=0.00005, betas=(0, 0.999))
        g_optimizer = optim.Adam(itertools.chain(*[Encoder.parameters(),Decoder.parameters(),
                                                    Phi.parameters(),Invphi.parameters()]), lr=0.00005, betas=(0, 0.999))
        
        d_optimizer = optim.Adam(Decoder.parameters(), lr=0.00005, betas=(0, 0.999))
        e_optimizer = optim.Adam(Encoder.parameters(), lr=0.00005, betas=(0, 0.999))
        
        opt_phi = optim.Adam(Phi.parameters(), lr=0.0001, betas=(0.9, 0.999))
        opt_invphi = optim.Adam(Invphi.parameters(), lr=0.0001, betas=(0.9, 0.999))  
        params = [Phi.parameters(), Invphi.parameters()]
        opt_transform=optim.Adam(itertools.chain(*params),lr=0.001,betas=(0.9,0.999))
        
        for batch_i, (real_images, gender,glasses) in enumerate(train_load):
            batch_size = real_images.size(0)
            real_images=real_images.to(device,dtype=torch.float)
            
            noi = noise_vector(real_images.size(0))

            # 1. GAN Loss
            fake_data = Decoder(noi).detach()
            d_error, d_pred_real, d_pred_fake =gen_image_loss(True,real_images.float(), 
                                                              fake_data,D_reconstruct,
                                                              dr_optimizer,d_optimizer)
            fake_data = Decoder(noi)#noise(real_batch.size(0)))
            g_error = gen_image_loss(False, real_images.float(), fake_data,
                                    D_reconstruct,
                                    dr_optimizer,d_optimizer)

            
            #2 Adv.Feature Loss
            real_feature=Encoder(real_images.float()).detach()
            df_error_adv=adv_feature_loss(True,noi,real_feature,
                                          D_feature,
                                          df_optimizer,e_optimizer)
            
            
            real_feature=Encoder(real_images.float())
            en_error_adv=adv_feature_loss(False,noi,real_feature,
                                          D_feature,
                                          df_optimizer,e_optimizer)
            
            
            ######################################################
            latent_vector=Encoder(real_images).detach()
            glass_vector,gender_vector,remain=Phi(latent_vector)
            
            #3 Reconstruction Loss
            z_tilde=Invphi(torch.cat((glass_vector,gender_vector,remain),1))
            loss_reconstruction=reconstruction_loss_phi(latent_vector,z_tilde,opt_transform)
            
            
            #4 Task Loss
            opt_phi.zero_grad()       
            loss=TripleletLoss(glass_vector,glasses) + TripleletLoss(gender_vector,gender)  
            loss.backward(retain_graph=True)
            opt_phi.step()

            
            #5 Cyclic Loss
            loss_cycle=cyclic_loss(glass_vector,gender_vector,remain,glasses,gender,opt_transform,
                                   [Encoder,Decoder,Phi,Invphi])
            
            
            
            
            ##############################################################
            #6 Adv.Image Loss
            fake_data=Decoder(Invphi(torch.cat(Phi(Encoder(real_images.float())),1))).detach()
            d_error1=adv_img_loss(True,real_images.float(),fake_data,
                                  D_reconstruct,dr_optimizer,g_optimizer)

            fake_data=Decoder(Invphi(torch.cat(Phi(Encoder(real_images.float())),1)))
            g_error2=adv_img_loss(True,real_images.float(),fake_data,
                                  D_reconstruct,dr_optimizer,g_optimizer)
            
            # 7. Full_Reconstruction Loss
            
            reconstructed_data=fake_data
            recons_loss=full_reconstuction_loss(g_optimizer,real_images.float(),reconstructed_data)    
            
            if (batch_i) % 300 == 0:
                print("Batch: ", batch_i)
                print("1:Discriminator_Error: ", d_error.item()," Generator_Error: ", g_error.item()," Recons_Error: ",                                recons_loss.item())
                print("2:Feature Discriminator Error: ",df_error_adv.item(),"Encoder Error: ", en_error_adv.item())
                print("3 Discriminator_adv_error", d_error1.item(), "Generator_error: ", g_error2.item())
                print("4 Task Loss: ", loss.item())
                print("5 Reconstruction Loss: ",loss_reconstruction.item())
                print("6 Cyclic Loss: ",loss_cycle.item())
        t_end = time.time()
        duration_avg = (t_end - t_start) / (epoch + 1.0)
        print("Elapsed Time: ",duration_avg)
        torch.save(Encoder,'EncoderF.h')
        torch.save(Decoder,'DecoderF.h')
        torch.save(D_feature,'DiscriminatorfF.h')
        torch.save(D_reconstruct,'DiscriminatorrF.h')
        torch.save(Phi,'PhiF.h')
        torch.save(Invphi,'invphiF.h')
