import sys
import numpy as np
import torch
from utils.helper_functions import *
# from utils.functions_kraken_loss import *


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


# def train_mvae(model, train_loader, mean_train_label, device, optimizer, netmat_prep_choice: str="demean", recon_weights: list=[1.0, 1.0]):
#     '''
#     Train function only using MSE but for multimodal VAEs. Ideally, should have a main train function that can
#     adapt to architecture. For now, seperating them.
#     '''
#     optimizer.zero_grad() #inits grads as None instead of 0s with certain mem advantages. Kinda part of the culture.
#     model.train()

#     tr_mae_subs = []
#     tr_mse_subs = []
#     tr_corr_subs_demean = []
#     tr_corr_subs_org = []
#     tr_corr_subs_surf = [] 
#     tr_epoch_loss = 0
#     tr_mae_subs_surf=[]
#     tr_mse_subs_surf=[]


#     for i, data in enumerate(train_loader): # for loop that goes over each batch in training loop
#         surface_inputs, connectome_inputs = data[0].to(device), data[1].to(device) # inputs = graph, output=tr_demean_mesh ico-n
#         # list each element seperate and for each encoder/expert seperately
#         outputs = model([surface_inputs, connectome_inputs])

#         # calc loss, need to vectorize connectome as input and output here, need to keep as tensor, though.
#         connectome_inputs_vectorized = torch.tril(connectome_inputs, diagonal=-1)
#         total_loss, recon_loss, kl_loss = model.elbo(
#             [surface_inputs, connectome_inputs_vectorized],  #mave inputs
#             [outputs["reconstructions"][0], ],
#             outputs["mu"],
#             outputs["log_var"],
#             recon_weights=recon_weights #how to weight the encoders for total MSE loss
#         )

#         # #check for possible instability
#         # if not torch.isfinite(total_loss):
#         #     # warnings.warn(f"Skipping batch {i} due to non-finite loss. resettign to none.")
#         #     optimizer.zero_grad()
#         #     total_loss = None
        
#         # continue the trains tep
#         total_loss.backward()
#         optimizer.step()

#         #outputs of current epoch for monitoring training
#         surface_recon, connectome_recon = outputs["reconstructions"][0], outputs["reconstructions"][1]
        
#         # netmat performance
#         batch_n = connectome_recon.shape[0] #same batch size for both
#         connectome_inputs = connectome_inputs.detach().numpy() #also symmetric matrix
#         connectome_recon = connectome_recon.detach().numpy() #should be a symmetric matrix NxN

#         # #surface performance
#         surface_inputs = surface_inputs.detach().numpy()
#         targets_surf = targets_surf.reshape(batch_n, surface_inputs.shape[1]*surface_inputs.shape[2]*surface_inputs.shape[3])

#         surface_recon = surface_recon.detach().numpy()
#         pred_surf = pred_surf.reshape(batch_n, surface_inputs.shape[1]*surface_inputs.shape[2]*surface_inputs.shape[3])
        
#         del outputs#, netmat_recon, connectome_inputs, surface_recon, surface_inputs

#         # MAE and MSE metrics per training iteration/batch but seperated for encoders
#         mae = np.mean( np.abs((targets - pred)), axis=0, keepdims=True) #LA.norm((targets - pred), ord=1, axis=0) #np.abs( (targets - pred) ) # |y - y_hat|
#         tr_mae_subs.append(mae)
#         mse = np.mean( (targets - pred)**2 , axis=0, keepdims=True) #(LA.norm((targets - pred), ord=2, axis=0)) ** 2 #needs to be **2 becasue norm-2 squareroots result. np.mean( (targets - pred)**2 ) #(y - y_hat)^2
#         tr_mse_subs.append(mse) # 1xCxPxV, MSE for this batch

#         #same for surfaces
#         mae_surf = np.mean( np.abs((targets_surf - pred_surf)), axis=0, keepdims=True) #LA.norm((targets - pred), ord=1, axis=0) #np.abs( (targets - pred) ) # |y - y_hat|
#         tr_mae_subs_surf.append(mae_surf)
#         mse_surf = np.mean( (targets_surf - pred_surf)**2 , axis=0, keepdims=True) #(LA.norm((targets - pred), ord=2, axis=0)) ** 2 #needs to be **2 becasue norm-2 squareroots result. np.mean( (targets - pred)**2 ) #(y - y_hat)^2
#         tr_mse_subs_surf.append(mse_surf) # 1xCxPxV, MSE for this batch

#         if "demean" in netmat_prep_choice: 
#             tr_corr_mat = np.corrcoef(targets, pred) #[subj*2 x subj*2] matrix where quadrant1 = target_target, quad2=target_pred, quad3=pred_target, quad4=pred_pred
#             top_right_quad = tr_corr_mat[batch_n:,:batch_n] #big corr matrix only need topright or bottom left quadrants
#             tr_corr_subs_demean.append(np.diag(top_right_quad))
#             tr_corr_mat = np.corrcoef((targets+mean_train_label), (pred+mean_train_label)) # going to be low-ish cause 256->mesh size sphere but curious
#             top_right_quad = tr_corr_mat[batch_n:,:batch_n]
#             tr_corr_subs_org.append(np.diag(top_right_quad))
#         else: # if data was preped by demeaning, then for original need to readd mean
#             tr_corr_mat = np.corrcoef((targets-mean_train_label), (pred-mean_train_label))
#             top_right_quad = tr_corr_mat[batch_n:,:batch_n]
#             tr_corr_subs_demean.append(np.diag(top_right_quad))
#             tr_corr_mat = np.corrcoef(targets, pred)# going to be low-ish cause 256->mesh size sphere but curious
#             top_right_quad = tr_corr_mat[batch_n:,:batch_n]
#             tr_corr_subs_org.append(np.diag(top_right_quad))

#         tr_corr_mat = np.corrcoef(targets_surf, pred_surf)
#         top_right_quad = tr_corr_mat[batch_n:,:batch_n]
#         tr_corr_subs_surf.append(np.diag(top_right_quad))
        
#         del tr_corr_mat, top_right_quad #big matrix, remove here to not add more to mem usage/storage

#         # optimizer.zero_grad()
#         total_loss.backward()
#         tr_epoch_loss += total_loss.item()

#         optimizer.step()

#     across_sub_mae_mean, across_sub_mae_std = np.mean(tr_mae_subs), np.std(tr_mae_subs)
#     across_sub_mse_mean, across_sub_mse_std = np.mean(tr_mse_subs), np.std(tr_mse_subs)
#     across_sub_mse_mean_surf, across_sub_mae_mean_surf = np.mean(tr_mse_subs_surf), np.mean(tr_mae_subs_surf)

#     upto_n_minus1 = np.asarray(tr_corr_subs_demean[:-1])#.squeeze() # all upto last item, do that seperate then concat
#     upto_n_minus1 = upto_n_minus1.reshape(1, upto_n_minus1.shape[0]*upto_n_minus1.shape[1]) #vectorizes to 1xB*tril
#     n_minus_1 = np.asarray(tr_corr_subs_demean[-1])[np.newaxis,:] 
#     tr_corr_subs_demean = np.concatenate((upto_n_minus1,n_minus_1), axis=1) # add at end of col
#     across_sub_corr_demean, across_sub_corr_demean_std = np.mean(tr_corr_subs_demean), np.std(tr_corr_subs_demean)

#     # same for original corr values
#     upto_n_minus1 = np.asarray(tr_corr_subs_org[:-1])#.squeeze() # all upto last item, do that seperate then concat
#     upto_n_minus1 = upto_n_minus1.reshape(1, upto_n_minus1.shape[0]*upto_n_minus1.shape[1]) #vectorizes to 1xB*tril
#     n_minus_1 = np.asarray(tr_corr_subs_org[-1])[np.newaxis,:] 
#     tr_corr_subs_org = np.concatenate((upto_n_minus1,n_minus_1), axis=1) # add at end of col    across_sub_corr_org = np.mean(tr_corr_subs_org)
#     across_sub_corr_org, across_sub_corr_org_std = np.mean(tr_corr_subs_org), np.std(tr_corr_subs_org)

#     #surface results
#     # same for original corr values
#     upto_n_minus1 = np.asarray(tr_corr_subs_surf[:-1])#.squeeze() # all upto last item, do that seperate then concat
#     upto_n_minus1 = upto_n_minus1.reshape(1, upto_n_minus1.shape[0]*upto_n_minus1.shape[1]) #vectorizes to 1xB*tril
#     n_minus_1 = np.asarray(tr_corr_subs_surf[-1])[np.newaxis,:] 
#     tr_corr_subs_surf = np.concatenate((upto_n_minus1,n_minus_1), axis=1) # add at end of col    across_sub_corr_org = np.mean(tr_corr_subs_org)
#     across_sub_corr_surf_mean, across_sub_corr_surf_std = np.mean(tr_corr_subs_surf), np.std(tr_corr_subs_surf)
    
#     return tr_epoch_loss, across_sub_mae_mean, across_sub_mae_mean_surf, across_sub_mse_mean, across_sub_mse_mean_surf, across_sub_corr_demean, across_sub_corr_demean_std, across_sub_corr_org, across_sub_corr_surf_mean


