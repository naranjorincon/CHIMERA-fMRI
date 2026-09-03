#import system and os modules and yaml etc
import os
import sys
sys.path.append('../')
sys.path.append('./')
sys.path.append('../../')
import torch
import argparse
import yaml
import copy

# import helper modules
import numpy as np
import pandas as pd
# from models import models
from models import experiment_topomap_recon
from utils.helper_functions import * #so id ont have to write ut.write_to_file everytime
from utils import functions_train
import torch.optim as optim 
import glob

def fcn_validate(model, val_loader, output_prep_choice,
              output_average, 
              device: str="cpu", 
              VAE_flag: bool = False):
    
    model.eval().to(device)
    mse_val_list = []
    mae_val_list = []
    val_rho_list=[]
    val_rho_demean_list=[]
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            inputs, targets = data[0].to(device), data[1].to(device)
            
            if VAE_flag:
                pred, latent, log_latent = model(inputs) # pred will be a iterable, so pred[0] is the outcome and pred[1] is the latent which we dont need
                del latent, inputs, log_latent
            else:
                pred = model(inputs) # pred will be a iterable, so pred[0] is the outcome and pred[1] is the latent which we dont need
                del inputs

            # Output Losses
            Lr_mse = torch.FloatTensor(torch.nn.MSELoss()(targets, pred)) # MSE should be low 
            mse_val_list.append(Lr_mse.detach().numpy())
            Lr_mae = torch.FloatTensor(torch.nn.L1Loss()(targets, pred)) # MAE should be low 
            mae_val_list.append(Lr_mae.detach().numpy())

            #make into numpy vars and to cpu
            pred = pred.detach().numpy()
            targets = targets.detach().numpy()
            targets = targets.reshape(targets.shape[0],-1)
            pred = pred.reshape(pred.shape[0],-1)
            output_average = output_average[np.newaxis]
            output_average = output_average.reshape(output_average.shape[0],-1)
            if "demean" in output_prep_choice or "norm" in output_prep_choice: # if doing any demeaning, then predictions are of original and we must demean here
                val_corr_demean = np.corrcoef(targets, pred) #[subj*2 x subj*2] matrix where quadrant1 = target_target, quad2=target_pred, quad3=pred_target, quad4=pred_pred
                split_half_horizontal = np.split(val_corr_demean, 2, axis = 0) # 0 is top rectangle, 1 is bottom rectangle
                top_right_quad = np.split(split_half_horizontal[0], 2, axis = 1)[1]
                val_rho_demean_list.append(np.diag(top_right_quad)) #target_i with prediction_i is diagonal
                #original in this case
                val_corr_org = np.corrcoef((targets+output_average), (pred+output_average)) # going to be low-ish cause 256->mesh size sphere but curious
                split_half_horizontal = np.split(val_corr_org, 2, axis = 0) # 0 is top rectangle, 1 is bottom rectangle
                top_right_quad = np.split(split_half_horizontal[0], 2, axis = 1)[1]
                val_rho_list.append(np.diag(top_right_quad))

            else:
                val_corr_demean = np.corrcoef((targets-output_average), (pred-output_average))
                split_half_horizontal = np.split(val_corr_demean, 2, axis = 0) # 0 is top rectangle, 1 is bottom rectangle
                top_right_quad = np.split(split_half_horizontal[0], 2, axis = 1)[1]
                val_rho_demean_list.append(np.diag(top_right_quad))

                val_corr_org = np.corrcoef(targets, pred)# going to be low-ish cause 256->mesh size sphere but curious
                split_half_horizontal = np.split(val_corr_org, 2, axis = 0) # 0 is top rectangle, 1 is bottom rectangle
                top_right_quad = np.split(split_half_horizontal[0], 2, axis = 1)[1]
                val_rho_list.append(np.diag(top_right_quad))
    
    across_sub_mae_mean = np.mean(mae_val_list) # mean across batches
    # across_sub_mae_std = np.std(mae_val_list)
    across_sub_mse_mean = np.mean(mse_val_list)
    # across_sub_mse_std = np.std(mse_val_list)
    # because of batching, some in the list are different size so make into whole array
    upto_n_minus1 = np.asarray(val_rho_demean_list[:-1]).squeeze() # all upto last item, do that seperate then concat
    upto_n_minus1 = upto_n_minus1.reshape(1, upto_n_minus1.shape[0]*upto_n_minus1.shape[1]) #vectorizes to 1xB*tril
    n_minus_1 = np.asarray(val_rho_demean_list[-1])[np.newaxis,:] 
    val_corr_demean_flat = np.concatenate((upto_n_minus1,n_minus_1), axis=1) # add at end of col
    across_sub_corr_demean = np.mean(val_corr_demean_flat)
    # across_sub_corr_demean_std = np.std(val_corr_demean_flat)

    # same for original corr values
    upto_n_minus1 = np.asarray(val_rho_list[:-1]).squeeze() # all upto last item, do that seperate then concat
    upto_n_minus1 = upto_n_minus1.reshape(1, upto_n_minus1.shape[0]*upto_n_minus1.shape[1]) #vectorizes to 1xB*tril
    n_minus_1 = np.asarray(val_rho_list[-1])[np.newaxis,:] 
    val_corr_org_flat = np.concatenate((upto_n_minus1,n_minus_1), axis=1) # add at end of col
    across_sub_corr_org = np.mean(val_corr_org_flat)
    # across_sub_corr_org_std = np.std(val_corr_org_flat)

    return across_sub_mae_mean, across_sub_mse_mean, across_sub_corr_demean, across_sub_corr_org


def whole_model_arch(config):

    #we will be either reconstructing or translating. Get this first based on input and output
    # input
    data_type_input   = config['data']['data_type_input']
    input_dim         = config['data']['input_dim']
    prep_type_input   = config['data']['prep_type_input']
    input_information = [data_type_input, input_dim]
    #output
    data_type_output   = config['data']['data_type_output']
    output_dim         = config['data']['output_dim']
    prep_type_output   = config['data']['prep_type_output']
    output_information = [data_type_output, output_dim]

    # Training configuration, depends on model being used
    icores = None if config['data']['icores'] == None else config['data']['icores']
    if not icores is None:
        model_config_topomap = {
        "dim": config['transformer']['dim'],
        "depth": config['transformer']['depth'],
        "heads": config['transformer']['heads'],
        "num_vertices": config['sub_ico_{}'.format(icores)]['num_vertices'],
        "num_channels": config['data']['input_dim'],
        "num_patches": config['sub_ico_{}'.format(icores)]['num_patches'],
        # "dim_head": config['transformer']['dim_head'],
        "dropout": config['transformer']['dropout'],
        "emb_dropout": config['transformer']['emb_dropout'],
        "VAE_flag": config['transformer']['VAE_flag'],
        "VAE_latent_dim": config['transformer']['vae_dim'],
        "latent_samples": config['transformer']['latent_samples'],
        "decoder_name": config['transformer']['decoder_name']
        }
        #choose which to use
        chosen_model_config=model_config_topomap
    else:
        model_config_connectome = {
            "connectome_features": int(0.5 * input_dim*(input_dim-1)),
            "dim": config['transformer']['dim'],
            # "depth": config['transformer']['depth'],
            # "heads": config['transformer']['heads'],
            "emb_dropout": config['transformer']['emb_dropout'], 
            # "dropout":config['transformer']['dropout'],
            "decoder_name": config['transformer']['decoder_name']
        }

        #choose which to use
        chosen_model_config=model_config_connectome

    #infer operation to be done
    if input_information == output_information:
        operation = "reconstruction"
    else:
        operation = "translation"

    #infer type of data being used
    topomap_representations = ["ICA", "PFM", "GRAD", "NMF"] #TODO expand this list as we do more brain reps or get more
    connectome_representations = ["schaefer", "glasser", "kong", "MSHBM", "TM", "IM"]
    assert data_type_input in topomap_representations+connectome_representations, "input is not in correct options of viable recons/translations."

    if data_type_input in topomap_representations:
        input_representation_type = "topomap"
    elif data_type_input in connectome_representations:
        input_representation_type = "connectome"

    if operation == 'reconstruction':
        output_representation_type = copy.deepcopy(input_representation_type)
    else:
        if data_type_output in topomap_representations:
            output_representation_type = "topomap"
        elif data_type_output in connectome_representations:
            output_representation_type = "connectome"

    #model_output path, and save model path
    model_output_path = config['logging']['model_output_path']
    model_save_path   = config['logging']['model_save_path']

    #training model details
    fcn_train = getattr(functions_train, config['training']['fcn_train'])  
    fcn_model_module = getattr(experiment_topomap_recon, config['training']['fcn_model_to_use']) 
    dataset_choice = config['training']['dataset_choice']
    overfit_condition = config['training']['overfit_condition']
    overfit_condition_sub_range = config['training']['overfit_condition_sub_range'] if overfit_condition is False else 0 #subset of subjects to debug on
    train_batch_sz = config['training']['bs'] if overfit_condition is False else 8
    LR = config['training']['LR']
    val_epoch = config['training']['val_epoch']
    train_epoch_range = config['training']['epochs']
    bilateral_condition = config['training']['bilateral_condition'] #Bool    
    model_type = config['data']['model_type']
    hemi_cond = config['training']['hemi_cond']
    left_or_right = "L" if hemi_cond == "1L" else "R"
    sub_ids_path = config['data']['sub_ids_path']    
    model_details = config['transformer']['model_details']
    decoder_name = config['transformer']['decoder_name']
    write_fpath = config['logging']['live_logfile'].format(model_type, operation, data_type_input, input_dim, data_type_output, output_dim, decoder_name) + '.print'
    write_to_file(f"Using ico-{icores} surf data.", filepath=write_fpath)
    write_to_file(f"Model details are: {model_details}\n", filepath=write_fpath)

    # init model validation vals and test flag
    device = "cpu"
    best_mae = 1e+9
    best_mse = 1e+9
    best_demean_rho = -1
    TEST_FLAG = config['testing']['immediate_test_flag']
    # te_batch_size = 1 #config['testing']['bs_test']
    folder_to_save_model = f'{model_save_path}/{dataset_choice}/{model_type}/{operation}'
    folder_to_save_losses = f'{model_output_path}/{dataset_choice}/{model_type}/{operation}/{model_details}'

    # make necessary folders
    if not os.path.exists(folder_to_save_model):
        os.makedirs(folder_to_save_model)
        
    if not os.path.exists(folder_to_save_losses):
        os.makedirs(folder_to_save_losses)

    # my directory is where i go to data_root_path then brain_reps/ABCD_Netmats as necessary. 
    chosen_test_model = config['testing']['chosen_test_model'] #MSE, MAE, or RHO
    folder_to_save_test=f'{folder_to_save_losses}/{chosen_test_model}'
    if not os.path.exists(folder_to_save_test):
        os.makedirs(folder_to_save_test) # Create the directory

    ############################################# LOAD IN DATA NETMATS AND/OR SURFACE MESHES #############################################
    main_brainrep_data_path_root = "/ceph/chpc/shared/janine_bijsterbosch_group/naranjorincon_scratch/NeuroTranslate"
    sub_ids_path= "/ceph/chpc/shared/janine_bijsterbosch_group/naranjorincon_scratch/NeuroTranslate/CHIMERA-fMRI/utils/subj_ids/ABCDv6/ABCD_train_val_test_split.csv"
    train_val_test_csv = pd.read_csv(sub_ids_path)
    get_sub_ids = train_val_test_csv["subID"] #add sub- for later use
    get_sub_ids = np.asarray(['sub-'+get_sub_ids[iii] for iii in range(len(get_sub_ids))])
    df_with_sub_prefix = pd.DataFrame({
        "index": np.arange(len(get_sub_ids)),
        "subID": get_sub_ids
    })

    input_files, output_files = fcn_get_file_lists_and_sort(main_brainrep_data_path_root, dataset_choice, 
                                data_type_input, data_type_output, 
                                input_dim, output_dim, icores=icores, 
                                input_representation_type=input_representation_type, output_representation_type=output_representation_type,
                                get_sub_ids=get_sub_ids, operation=operation,
                                left_or_right=left_or_right, df_with_sub_prefix=df_with_sub_prefix)

    #get data as a torch dataset                
    train_loader, val_loader, test_loader, output_average = fcn_get_torch_loaders(input_files, output_files, 
                                                    train_batch_sz, prep_type_input, 
                                                    prep_type_output, 
                                                    overfit_condition_sub_range=overfit_condition_sub_range
                                                    )
    
    write_to_file(f"Loaded in data. Tunning on dataset: {dataset_choice}", filepath=write_fpath)
    
    # Initialize model, optimizer, etc.
    model = fcn_model_module(**chosen_model_config).to(device)
    
    # initialize optimizer / loss
    scheduler = False #default is false, unless otherwise specified by the yml configuration file
    if config['optimisation']['optimiser']=='Adam':
        write_to_file('using Adam optimiser',  filepath=write_fpath)
        optimizer = optim.Adam(model.parameters(),
                               lr=LR,
                               weight_decay=config['Adam']['weight_decay'])
        if config['Adam']['use_scheduler']:
            scheduler = True
            lr_schedule = optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                                T_max = config['CosineDecay']['T_max'],
                                                                eta_min= config['CosineDecay']['eta_min']
                                                                )

    elif config['optimisation']['optimiser']=='SGD':
        write_to_file('using SGD optimiser',  filepath=write_fpath)
        optimizer = optim.SGD(model.parameters(), lr=LR, 
                                                weight_decay=config['SGD']['weight_decay'],
                                                momentum=config['SGD']['momentum'],
                                                nesterov=config['SGD']['nesterov'])
    elif config['optimisation']['optimiser']=='AdamW':
        write_to_file('using AdamW optimiser',  filepath=write_fpath)
        optimizer = optim.AdamW(model.parameters(),
                                lr=LR,
                                weight_decay=config['AdamW']['weight_decay'])
        if config['AdamW']['use_scheduler']:
            scheduler = True
            if config['AdamW']['scheduler']=='CosineDecay': # TODO currrently only set for CosineDecay bc that is what was used in swMSSiT paper from dahan
                lr_schedule = optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                        T_max = config['CosineDecay']['T_max'],
                                                        eta_min= config['CosineDecay']['eta_min'],
                                                        last_epoch=-1
                                                        )


    # Find number of parameters
    model_params = sum(p.numel() for p in model.parameters())
    write_to_file(f"Model params: {model_params}", filepath=write_fpath)

    #if using existing model
    finetune=config['training']['finetune']
    if finetune is True:
        path_to_model=folder_to_save_model
        write_to_file(f'\nUsing existing model. {path_to_model}\nDetails:*{model_details}_{chosen_test_model}.pt', filepath=write_fpath)
        model_path = sorted(glob.glob(f"{path_to_model}/*{model_details}_{chosen_test_model}.pt")) # look at training script for details, but all models saves as type_details_chosen: ex-kBGTLN_d6h5_demeanL2_skewloss_RHO.pt
        chosen_model = model_path[0]
        write_to_file(f'\n\nmodel loaded is {chosen_model}', filepath=write_fpath)
        model.load_state_dict(torch.load(chosen_model)) # most recent model
    
    model.to(device)
    running_train_loss = 0
    df_train = pd.DataFrame(columns=['train_mae', 'train_mae_sigma', 'train_mse', 'train_mse_sigma', 'train_loss', 'train_demean_corr', 'train_demean_corr_sigma', 'train_orig_corr', 'train_orig_corr_sigma'])
    df_val = pd.DataFrame(columns=['val_mae', 'val_mae_sigma', 'val_mse', 'val_mse_sigma', 'val_loss', 'val_demean_corr', 'val_demean_corr_sgima', 'val_orig_corr', 'val_orig_corr_sigma'])

    write_to_file("Training has begun.", filepath=write_fpath)
    lr_list=[]
    for epoch in range(train_epoch_range):
        
        [tr_epoch_loss, 
         across_sub_mae_mean, 
        #  across_sub_mae_std, 
         across_sub_mse_mean, 
        #  across_sub_mse_std, 
         across_sub_corr_demean, 
        #  across_sub_corr_demean_std, 
         across_sub_corr_org, 
        #  across_sub_corr_org_std
         ] = fcn_train(model, train_loader, prep_type_output, output_average, device, optimizer, chosen_model_config["VAE_flag"])
        
        if scheduler: # if you are using a scheduler, this should be TRUE o.w. FALSE so no need to do the "step" to change LR
            lr_schedule.step() #after each epoch        
            curr_lr = optimizer.param_groups[0]['lr'] #lr_schedule.get_last_lr()[-1]
        else:
            curr_lr = LR

        # Convert tensors to floats
        train_loss_value = float(tr_epoch_loss)
        running_train_loss += train_loss_value
        lr_list.append(curr_lr)
        
        write_to_file('| Training | Epoch - {} | LR - {:.4f}| Loss - {:.4f} | MAE - {:.4f} | MSE = {:.4f} | demeanCorr {:.4f}'.format(epoch, curr_lr, running_train_loss, across_sub_mae_mean, across_sub_mse_mean, across_sub_corr_demean), filepath=write_fpath)

        new_row = pd.DataFrame({'train_mae': [across_sub_mae_mean], 'train_mse': [across_sub_mse_mean],
                                'train_loss': [train_loss_value], 'train_demean_corr': [across_sub_corr_demean], 'train_orig_corr': [across_sub_corr_org]
                                })
        
        df_train = pd.concat([df_train, new_row], ignore_index=True)
        df_train.to_csv(os.path.join(folder_to_save_losses, 'train_losses_patch.csv'))
        if epoch%val_epoch == 0:
            [grpavg_val_mae,
            #  grpstd_val_mae,
             grpavg_val_mse,
            #  grpstd_val_mse,
             val_deman_corr,
            #  val_deman_corr_std, 
             val_orig_corr, 
            #  val_orig_corr_std
             ] = fcn_validate(model, val_loader, prep_type_output, output_average, device, chosen_model_config["VAE_flag"])
                                                                                                                                                        
            write_to_file(f'| Validation | Epoch - {epoch} | MAE - {grpavg_val_mae:.4f} | MSE = {grpavg_val_mse:.4f} | demeanCorr {val_deman_corr:.4f}', filepath=write_fpath)

            # save model with best MSE - gives leeway to values around 0 so maybe betetr for correlation values?
            curr_val_mse = grpavg_val_mse
            if curr_val_mse < best_mse:
                best_mse = curr_val_mse
                write_to_file('saving MSE model...', filepath=write_fpath)
                torch.save(model.state_dict(), os.path.join(folder_to_save_model,f'{model_type}_{model_details}_MSE.pt'))
            # save model with best MAE - forces values closer to 0
            curr_val_mae = grpavg_val_mae
            if curr_val_mae < best_mae:
                best_mae = curr_val_mae
                write_to_file('saving MAE model...', filepath=write_fpath)
                torch.save(model.state_dict(), os.path.join(folder_to_save_model,f'{model_type}_{model_details}_MAE.pt'))
            # save model with best RHO_demean
            curr_val_demean_rho = val_deman_corr # prioritize model with best demean correlation performance with validation set
            if curr_val_demean_rho > best_demean_rho:
                best_demean_rho = curr_val_demean_rho
                write_to_file('saving RHO model...', filepath=write_fpath)
                torch.save(model.state_dict(), os.path.join(folder_to_save_model,f'{model_type}_{model_details}_RHO.pt'))

            new_row = pd.DataFrame({'val_mae': [grpavg_val_mae], 'val_mse': [grpavg_val_mse],
                                    'val_demean_corr': [val_deman_corr], 'val_orig_corr': [val_orig_corr]
                                    })
            
            df_val = pd.concat([df_val, new_row], ignore_index=True)
            df_val.to_csv(os.path.join(folder_to_save_losses, 'val_losses_patch.csv'))


        write_to_file('saving LAST model...', filepath=write_fpath)
        torch.save(model.state_dict(), os.path.join(folder_to_save_model,f'{model_type}_{model_details}_LAST.pt'))

    df_version_lr_list = pd.DataFrame(lr_list)
    df_version_lr_list.to_csv(os.path.join(folder_to_save_test, 'model_lr_list.csv'))
    
    # TESTING #
    if TEST_FLAG:
        write_to_file('TEST FLAG ON. TESTING.', filepath=write_fpath)
        # see all models
        model_path = sorted(glob.glob(f"{folder_to_save_model}/*{model_details}_{chosen_test_model}.pt")) # look at training script for details, but all models saves as type_details_chosen: ex-kBGTLN_d6h5_demeanL2_skewloss_RHO.pt
        chosen_model = model_path[0]
        write_to_file(f'\n\nmodel loaded is {chosen_model}', filepath=write_fpath)
        model.load_state_dict(torch.load(chosen_model)) # most recent model

        # Find number of parameters
        model_params = sum(p.numel() for p in model.parameters())
        write_to_file(f"\n\nModel params: {model_params}", filepath=write_fpath)

        # Testing below
        model.eval().to(device)
    
        # lists to keep track
        mse_train_list = []
        mae_train_list = []
        mse_test_list = []
        mae_test_list = []
        te_ground_truth = []
        te_pred = []
        tr_ground_truth = []
        tr_pred = []
        with torch.no_grad():
            for i, data in enumerate(test_loader):
                inputs, targets = data[0].to(device), data[1].to(device)#.squeeze()
                if chosen_model_config["VAE_flag"] is True:
                    pred, latent, log_latent = model(inputs) # pred will be a iterable, so pred[0] is the outcome and pred[1] is the latent which we dont need
                    del latent, inputs, log_latent
                else:
                    pred = model(inputs) # pred will be a iterable, so pred[0] is the outcome and pred[1] is the latent which we dont need
                    del inputs
                
                # just having some output to see while testing, otherwise terminal is silent. Nice to see progress IMO
                if i % 100 == 0:
                    write_to_file(f"checkpoint. Running test subject: {i}", filepath=write_fpath)

                mse = torch.FloatTensor(torch.nn.MSELoss()(targets, pred)) # MSE should be low 
                mse_test_list.append(mse.detach().numpy())
                mae = torch.FloatTensor(torch.nn.L1Loss()(targets, pred)) # MAE should be low 
                mae_test_list.append(mae.detach().numpy())

                #make into numpy vars and to cpu
                pred = pred.detach().numpy()
                targets = targets.detach().numpy()
                targets = targets.reshape(targets.shape[0],-1).squeeze()
                pred = pred.reshape(pred.shape[0],-1).squeeze()

                te_ground_truth.append(targets)
                te_pred.append(pred)

            write_to_file("Done with TESTING loop.", filepath=write_fpath)

            # to optimize testing and data saving, will only get best, mid, and lowest corr
            te_ground_truth = np.asarray(te_ground_truth)
            te_pred = np.asarray(te_pred)
            across_sub_rho = np.corrcoef(te_ground_truth, te_pred) # gives sub_dim*2 x sub_dim*2 and will likely be two square clusters truth and pred
            write_to_file(f"SZ of bigg matrix: {across_sub_rho.shape}", filepath=write_fpath)
            np.save(f"{folder_to_save_test}/te_big_corr_matrix.npy", across_sub_rho) # save for viz later

            import itertools
            # Take only the first 5 batches from the dataloader
            for batch_idx, (images, labels) in enumerate(itertools.islice(train_loader, 5)):
                # write_to_file(f"Batch {batch_idx}: {images.shape} <--> {labels.shape}", filepath=write_fpath)
                inputs, targets = images, labels

                if chosen_model_config["VAE_flag"] is True:
                    pred, latent, log_latent = model(inputs) # pred will be a iterable, so pred[0] is the outcome and pred[1] is the latent which we dont need
                    del latent, inputs, log_latent
                else:
                    pred = model(inputs) # pred will be a iterable, so pred[0] is the outcome and pred[1] is the latent which we dont need
                    del inputs
                
                # just having some output to see while testing, otherwise terminal is silent. Nice to see progress IMO
                if i % 100 == 0:
                    write_to_file(f"checkpoint. Running TRAIN subject: {i}", filepath=write_fpath)

                mse = torch.FloatTensor(torch.nn.MSELoss()(targets, pred)) # MSE should be low 
                mse_train_list.append(mse.detach().numpy())
                mae = torch.FloatTensor(torch.nn.L1Loss()(targets, pred)) # MAE should be low 
                mae_train_list.append(mae.detach().numpy())

                #make into numpy vars and to cpu
                pred = pred.detach().numpy()
                targets = targets.detach().numpy()
                targets = targets.reshape(targets.shape[0],-1).squeeze() #BatchxFlatten_Array
                #split batchxaray into 1xarray to make subjects be separate entries on the list for later correlation matrix
                targets = np.split(targets,len(targets), axis=0) #now its a list, each with 1xFlatten_Array
                for ii in targets:
                    tr_ground_truth.append(ii.squeeze())

                pred = pred.reshape(pred.shape[0],-1).squeeze()
                pred = np.split(pred,len(pred), axis=0) #list
                for ii in pred:
                    tr_pred.append(ii.squeeze())

            write_to_file(f"Done with TRAINING loop.", filepath=write_fpath)

            # to optimize testing and data saving, will only get best, mid, and lowest corr
            tr_ground_truth = np.asarray(tr_ground_truth)
            tr_pred = np.asarray(tr_pred)
            across_sub_rho = np.corrcoef(tr_ground_truth, tr_pred) # gives sub_dim*2 x sub_dim*2 and will likely be two square clusters truth and pred
            np.save(f"{folder_to_save_test}/tr_big_corr_matrix.npy", across_sub_rho) # save for viz later
        
        # save training losses
        df_version_mae = pd.DataFrame(mae_train_list)
        df_version_mae.to_csv(os.path.join(folder_to_save_test, 'mae_train_model.csv'))
        df_version_mse = pd.DataFrame(mse_train_list)
        df_version_mse.to_csv(os.path.join(folder_to_save_test, 'mse_train_model.csv'))

        # save test losses
        df_version_mae = pd.DataFrame(mae_test_list)
        df_version_mae.to_csv(os.path.join(folder_to_save_test, 'mae_test_model.csv'))
        df_version_mse = pd.DataFrame(mse_test_list)
        df_version_mse.to_csv(os.path.join(folder_to_save_test, 'mse_test_model.csv'))

        #save subjects that were kept i.e. had good data and were not scrubbed during cleaning
        # train_subjects_to_keep.to_csv(os.path.join(folder_to_save_test, 'train_subjects_to_keep.csv'))
        # validation_subjects_to_keep.to_csv(os.path.join(folder_to_save_test, 'validation_subjects_to_keep.csv'))
        # test_subjects_to_keep.to_csv(os.path.join(folder_to_save_test, 'test_subjects_to_keep.csv'))

        write_to_file("TRAIN Mean MAE:", filepath=write_fpath)
        write_to_file(np.nanmean(mae_train_list), filepath=write_fpath)
        write_to_file("TEST Mean MAE:", filepath=write_fpath)
        write_to_file(np.nanmean(mae_test_list), filepath=write_fpath)

        write_to_file("TRAIN Mean MSE:", filepath=write_fpath)
        write_to_file(np.nanmean(mse_train_list), filepath=write_fpath)
        write_to_file("TEST Mean MSE:", filepath=write_fpath)
        write_to_file(np.nanmean(mse_test_list), filepath=write_fpath)

        np.save(f"{folder_to_save_test}/train_ground_truth.npy", tr_ground_truth)
        np.save(f"{folder_to_save_test}/train_pred.npy", tr_pred)
        np.save(f"{folder_to_save_test}/test_ground_truth.npy", te_ground_truth)
        np.save(f"{folder_to_save_test}/test_pred.npy", te_pred)

        write_to_file(f"TRAIN AND TEST COMPLETE for model:\n{model_details}", filepath=write_fpath)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')

    parser.add_argument(
                        'config',
                        type=str,
                        default='',
                        help='args from yaml file')
    
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Call training
    whole_model_arch(config)
