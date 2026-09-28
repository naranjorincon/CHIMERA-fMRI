import sys
import numpy as np
import torch
from utils.helper_functions import * #incldues write_to_file
import warnings
import torch.nn as nn

def fcn_train(model, 
              train_loader, 
              output_prep_choice,
              output_average, 
              device: str="cpu", 
              optimizer=None,
              VAE_flag: bool = False):

    model.train()

    train_mse_list=[]
    train_mae_list=[]
    train_rho_list=[]
    train_rho_demean_list=[]
    train_loss = 0
    for i, data in enumerate(train_loader):
        # i is iteration, data is batch x dimensions, probably BxCxPxV = batch x channel x patch x verteces
        inputs, targets = data[0].to(device), data[1].to(device)
        optimizer.zero_grad(set_to_none=True) # True by default anyway
        if VAE_flag is True: #kl loss is -1/2 * sum(1+ logvar - mu**2 - var)
            pred, latent, latent_logvar = model(inputs) #latent = z_mu, latent_sigma=sigma_mu
            kld_loss = -0.5 * torch.sum(1 + latent_logvar - (latent ** 2) - latent_logvar.exp(), dim = 1) # dim1 bc each subject has own loss
            kld_loss = torch.mean(kld_loss, dim=0) #now average across subjecrs to get single loss for all in batch    
        else:
            pred = model(inputs)

        del inputs#, latent

        # Output Losses
        Lr_mse = torch.FloatTensor(torch.nn.MSELoss()(targets, pred)) # MSE should be low 
        train_mse_list.append(Lr_mse.detach().numpy())
        Lr_mae = torch.FloatTensor(torch.nn.L1Loss()(targets, pred)) # MAE should be low 
        train_mae_list.append(Lr_mae.detach().numpy())

        #make into numpy vars and to cpu
        pred = pred.detach().numpy()
        targets = targets.detach().numpy()
        targets = targets.reshape(targets.shape[0],-1)
        pred = pred.reshape(pred.shape[0],-1)
        output_average = output_average[np.newaxis]
        output_average = output_average.reshape(output_average.shape[0],-1)
        if "demean" in output_prep_choice or "norm" in output_prep_choice: # if doing any demeaning, then predictions are of original and we must demean here
            tr_corr_demean = np.corrcoef(targets, pred) #[subj*2 x subj*2] matrix where quadrant1 = target_target, quad2=target_pred, quad3=pred_target, quad4=pred_pred
            split_half_horizontal = np.split(tr_corr_demean, 2, axis = 0) # 0 is top rectangle, 1 is bottom rectangle
            top_right_quad = np.split(split_half_horizontal[0], 2, axis = 1)[1]
            train_rho_demean_list.append(np.diag(top_right_quad)) #target_i with prediction_i is diagonal
            #original in this case
            tr_corr_org = np.corrcoef((targets+output_average), (pred+output_average)) # going to be low-ish cause 256->mesh size sphere but curious
            split_half_horizontal = np.split(tr_corr_org, 2, axis = 0) # 0 is top rectangle, 1 is bottom rectangle
            top_right_quad = np.split(split_half_horizontal[0], 2, axis = 1)[1]
            train_rho_list.append(np.diag(top_right_quad))

        else:
            tr_corr_demean = np.corrcoef((targets-output_average), (pred-output_average))
            split_half_horizontal = np.split(tr_corr_demean, 2, axis = 0) # 0 is top rectangle, 1 is bottom rectangle
            top_right_quad = np.split(split_half_horizontal[0], 2, axis = 1)[1]
            train_rho_demean_list.append(np.diag(top_right_quad))

            tr_corr_org = np.corrcoef(targets, pred)# going to be low-ish cause 256->mesh size sphere but curious
            split_half_horizontal = np.split(tr_corr_org, 2, axis = 0) # 0 is top rectangle, 1 is bottom rectangle
            top_right_quad = np.split(split_half_horizontal[0], 2, axis = 1)[1]
            train_rho_list.append(np.diag(top_right_quad))

        if VAE_flag:
            loss = Lr_mse +  kld_loss # loss uses demean so add that
            torch.nn.utils.clip_grad_norm_(model.parameters(), 4.0)
        else:
            loss = Lr_mse # loss uses demean so add that
    
        loss.backward()
        train_loss += loss.item()

        optimizer.step()

    across_sub_mae_mean = np.mean(train_mae_list) # across all elements, so no axis ==> mean of flatten mat->vector, so across all subs and channels and patches and verteces
    # across_sub_mae_std = np.std(train_mae_list)
    across_sub_mse_mean = np.mean(train_mse_list)
    # across_sub_mse_std = np.std(train_mse_list)
    # # because of batching, some in the list are different size so make into whole array
    upto_n_minus1 = np.asarray(train_rho_demean_list[:-1]).squeeze() # all upto last item, do that seperate then concat
    upto_n_minus1 = upto_n_minus1.reshape(1, upto_n_minus1.shape[0]*upto_n_minus1.shape[1]) #vectorizes to 1xB*tril
    n_minus_1 = np.asarray(train_rho_demean_list[-1])[np.newaxis,:] 
    train_rho_demean_list = np.concatenate((upto_n_minus1,n_minus_1), axis=1) # add at end of col
    across_sub_corr_demean = np.mean(train_rho_demean_list)
    # across_sub_corr_demean_std = np.std(train_rho_demean_list)

    # same for original corr values
    upto_n_minus1 = np.asarray(train_rho_list[:-1]).squeeze() # all upto last item, do that seperate then concat
    upto_n_minus1 = upto_n_minus1.reshape(1, upto_n_minus1.shape[0]*upto_n_minus1.shape[1]) #vectorizes to 1xB*tril
    n_minus_1 = np.asarray(train_rho_list[-1])[np.newaxis,:] 
    train_rho_list = np.concatenate((upto_n_minus1,n_minus_1), axis=1) # add at end of col    across_sub_corr_org = np.mean(tr_corr_subs_org)
    across_sub_corr_org = np.mean(train_rho_list)
    # across_sub_corr_org_std = np.std(train_rho_list)
    
    return train_loss, across_sub_mae_mean, across_sub_mse_mean, across_sub_corr_demean, across_sub_corr_org


def train_fusion(model, train_loader, device="cpu", optimizer=None, annealer=None, scheduler_flag:bool=True, lr_schedule=None, recon_weights=None):
    '''
    Train function only using MSE but for multimodal VAEs. Ideally, should have a main train function that can
    adapt to architecture. For now, seperating them.
    '''
    model.train()
    epoch_loss, epoch_recon, epoch_kl, epoch_grad = [], [], [], []
    optimizer.zero_grad() #inits grads as None instead of 0s with certain mem advantages. Kinda part of the culture.

    #experts 
    for batch in train_loader:
        ICA15, PFM14, scf100, scf200, scf300, glss360 = [b.to(device) for b in batch]
    
        topomaps = {"ICA": ICA15, "PFM": PFM14}
        connectomes = {"schaefer100": scf100, "schaefer200": scf200, "schaefer300": scf300, "glasser360": glss360}
        out = model.compute_loss(
                    topomaps=topomaps,
                    connectomes=connectomes,
                    recon_weights=recon_weights, #should be a dict
                )
        loss = out["loss"]

        loss.backward()
        # Gradient clipping
        grad_norm = nn.utils.clip_grad_norm_(
            model.parameters(), 200
        ).item()

        optimizer.step()
        annealer.step()
        if scheduler_flag is True:
            lr_schedule.step()

        epoch_loss.append(out["loss"].item())
        epoch_recon.append(out["recon_loss"].item())
        epoch_kl.append(out["kl_loss"].item())
        epoch_grad.append(grad_norm)

    return epoch_loss, epoch_recon, epoch_kl, epoch_grad

