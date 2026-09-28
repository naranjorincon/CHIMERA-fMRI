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
import itertools
import numpy as np
import pandas as pd
# from models import models
from models import models, experiment_topomap_recon, experiment_connectome_recon, experts, moe_model, poe_model
from utils.helper_functions import * #so id ont have to write ut.write_to_file everytime
from utils import functions_train
import torch.optim as optim 
import glob
import torch.nn.functional as F
import warnings
import torch.nn as nn

@torch.no_grad()
def evaluate(model, val_loader, device, conn_keys, topo_keys) -> dict:
    """
    Evaluate all translation directions on the validation set.
    Returns dict of mean Pearson r and MSE for each direction.
    """
    model.eval()

    metrics = {
        "topo_ICA→topo_ICA":    [],   # autoencoding ICA surface
        # "topo_PFM→topo_PFM":    [],   # autoencoding PFM surface
        "topo_ICA→conn_schaefer100":    [],   # cross-modal
        "conn_schaefer100→topo_ICA":    [],
        "conn_schaefer100→conn_schaefer100":    [],   # autoencoding connectome
        "joint→topo_ICA":       [],   # joint (both inputs → surface)
        "joint→conn_schaefer100":       [],
    }
    mse = {k: [] for k in metrics}

    for batch in val_loader:
        ICA15, PFM14, scf100, scf200, scf300, glss360 = [b.to(device) for b in batch]
        topomaps = {"ICA": ICA15, "PFM": PFM14}
        connectomes = {"schaefer100": scf100, "schaefer200": scf200, "schaefer300": scf300, "glasser360": glss360}

        # topo 15 only → all outputs
        out_s = model.translate(topomaps={"ICA": ICA15})
        metrics["topo_ICA→topo_ICA"].append(batch_pearson(out_s["topo_ICA"], ICA15))
        metrics["topo_ICA→conn_schaefer100"].append(batch_pearson(out_s["conn_schaefer100"], scf100))
        mse["topo_ICA→topo_ICA"].append(F.mse_loss(out_s["topo_ICA"], ICA15).item())
        mse["topo_ICA→conn_schaefer100"].append(F.mse_loss(out_s["conn_schaefer100"], scf100).item())

        # # topo 14 only → all outputs
        # out_s = model.translate(surface={"PFM": PFM14})
        # metrics["topo_PFM→topo_PFM"].append(batch_pearson(out_s["topo_PFM"], PFM14))
        # metrics["topo_PFM→conn_100"].append(batch_pearson(out_s["conn_100"], scf100))
        # mse["topo_PFM→topo_PFM"].append(F.mse_loss(out_s["topo_PFM"], PFM14).item())
        # mse["topo_PFM→conn_100"].append(F.mse_loss(out_s["conn_100"], scf100).item())

        # Connectome 100 only → all outputs
        out_c = model.translate(connectomes={"schaefer100": scf100})
        metrics["conn_schaefer100→topo_ICA"].append(batch_pearson(out_c["topo_ICA"], ICA15))
        metrics["conn_schaefer100→conn_schaefer100"].append(batch_pearson(out_c["conn_schaefer100"], scf100))
        mse["conn_schaefer100→topo_ICA"].append(F.mse_loss(out_c["topo_ICA"], ICA15).item())
        mse["conn_schaefer100→conn_schaefer100"].append(F.mse_loss(out_c["conn_schaefer100"], scf100).item())

        # Joint → all outputs
        out_j = model.translate(topomaps=topomaps, connectomes=connectomes)
        metrics["joint→topo_ICA"].append(batch_pearson(out_j["topo_ICA"], ICA15))
        metrics["joint→conn_schaefer100"].append(batch_pearson(out_j["conn_schaefer100"], scf100))
        mse["joint→topo_ICA"].append(F.mse_loss(out_j["topo_ICA"], ICA15).item())
        mse["joint→conn_schaefer100"].append(F.mse_loss(out_j["conn_schaefer100"], scf100).item())

    results = {}
    for k in metrics:
        results[f"corr/{k}"] = float(np.nanmean(metrics[k]))
        results[f"mse/{k}"]  = float(np.mean(mse[k]))
    return results

@torch.no_grad()
def test_inference(model, train_loader, test_loader, device, conn_keys, topo_keys) -> dict:
    """
    Evaluate all translation directions on the validation set.
    Returns dict of mean Pearson r and MSE for each direction.
    """
    model.eval()

    metrics = {
        "topo_ICA→topo_ICA":    [],   # autoencoding ICA surface
        # "topo_PFM→topo_PFM":    [],   # autoencoding PFM surface
        "topo_ICA→conn_schaefer100":    [],   # cross-modal
        "conn_schaefer100→topo_ICA":    [],
        "conn_schaefer100→conn_schaefer100":    [],   # autoencoding connectome
        "joint→topo_ICA":       [],   # joint (both inputs → surface)
        "joint→conn_schaefer100":       [],
    }
    mse = {k: [] for k in metrics}
    test_output_list=[]
    for batch in test_loader:
        ICA15, PFM14, scf100, scf200, scf300, glss360 = [b.to(device) for b in batch]
        topomaps = {"ICA": ICA15, "PFM": PFM14}
        connectomes = {"schaefer100": scf100, "schaefer200": scf200, "schaefer300": scf300, "glasser360": glss360}

        # # topo 15 only → all outputs
        # out_s = model.translate(topomaps={"ICA": ICA15})

        # # Connectome 100 only → all outputs
        # out_c = model.translate(connectomes={"schaefer100": scf100})

        # Joint → all outputs
        out_j = model.translate(topomaps=topomaps, connectomes=connectomes)
        metrics["joint→topo_ICA"].append(batch_pearson(out_j["topo_ICA"], ICA15))
        metrics["joint→conn_schaefer100"].append(batch_pearson(out_j["conn_schaefer100"], scf100))
        mse["joint→topo_ICA"].append(F.mse_loss(out_j["topo_ICA"], ICA15).item())
        mse["joint→conn_schaefer100"].append(F.mse_loss(out_j["conn_schaefer100"], scf100).item())

        PRED_ICA15=out_j["topo_ICA"]
        PRED_PFM14=out_j["topo_PFM"]
        PRED_SCHF100=out_j["conn_schaefer100"]
        PRED_SCHF200=out_j["conn_schaefer200"]
        PRED_SCHF300=out_j["conn_schaefer300"]
        PRED_GLSS360=out_j["conn_glasser360"]

        test_output_list.append([PRED_ICA15, ICA15,
        PRED_PFM14, PFM14,
        PRED_SCHF100, scf100,
        PRED_SCHF200, scf200,
        PRED_SCHF300, scf300,
        PRED_GLSS360, glss360])

    mse_test = mse
    metrics_test = metrics
    
    # TRAIN ONLY NOW
    metrics = {
        "topo_ICA→topo_ICA":    [],   # autoencoding ICA surface
        # "topo_PFM→topo_PFM":    [],   # autoencoding PFM surface
        "topo_ICA→conn_schaefer100":    [],   # cross-modal
        "conn_schaefer100→topo_ICA":    [],
        "conn_schaefer100→conn_schaefer100":    [],   # autoencoding connectome
        "joint→topo_ICA":       [],   # joint (both inputs → surface)
        "joint→conn_schaefer100":       [],
    }
    mse = {k: [] for k in metrics}
    train_output_list=[]    
    first_five = list(itertools.islice(train_loader, 5))
    for batch in first_five:
        ICA15, PFM14, scf100, scf200, scf300, glss360 = [b.to(device) for b in batch]
        topomaps = {"ICA": ICA15, "PFM": PFM14}
        connectomes = {"schaefer100": scf100, "schaefer200": scf200, "schaefer300": scf300, "glasser360": glss360}

        # # topo 15 only → all outputs
        # out_s = model.translate(topomaps={"ICA": ICA15})

        # # Connectome 100 only → all outputs
        # out_c = model.translate(connectomes={"schaefer100": scf100})

        # Joint → all outputs
        out_j = model.translate(topomaps=topomaps, connectomes=connectomes)
        metrics["joint→topo_ICA"].append(batch_pearson(out_j["topo_ICA"], ICA15))
        metrics["joint→conn_schaefer100"].append(batch_pearson(out_j["conn_schaefer100"], scf100))
        mse["joint→topo_ICA"].append(F.mse_loss(out_j["topo_ICA"], ICA15).item())
        mse["joint→conn_schaefer100"].append(F.mse_loss(out_j["conn_schaefer100"], scf100).item())
        
        PRED_ICA15=out_j["topo_ICA"]
        PRED_PFM14=out_j["topo_PFM"]
        PRED_SCHF100=out_j["conn_schaefer100"]
        PRED_SCHF200=out_j["conn_schaefer200"]
        PRED_SCHF300=out_j["conn_schaefer300"]
        PRED_GLSS360=out_j["conn_glasser360"]

        train_output_list.append([PRED_ICA15, ICA15,
        PRED_PFM14, PFM14,
        PRED_SCHF100, scf100,
        PRED_SCHF200, scf200,
        PRED_SCHF300, scf300,
        PRED_GLSS360, glss360])

    mse_train = mse
    metrics_train = metrics

    results_test = {}
    for k in metrics_test:
        results_test[f"corr/{k}"] = float(np.nanmean(metrics_test[k]))
        results_test[f"mse/{k}"]  = float(np.mean(mse_test[k]))

    results_train = {}
    for k in metrics_train:
        results_train[f"corr/{k}"] = float(np.nanmean(metrics_train[k]))
        results_train[f"mse/{k}"]  = float(np.mean(mse_train[k]))

    test_output_list = [np.concatenate(x, axis=0) for x in zip(*test_output_list)]
    train_output_list = [np.concatenate(x, axis=0) for x in zip(*train_output_list)]
    return results_test, results_train, test_output_list, train_output_list

def whole_model_arch(config):
    #infer type of data being used
    # topomap_representations = ["ICA", "PFM", "GRAD", "NMF"] #TODO expand this list as we do more brain reps or get more
    # connectome_representations = ["schaefer", "glasser", "kong", "MSHBM", "TM", "IM"]

    #model_output path, and save model path
    model_output_path = config['logging']['model_output_path']
    model_save_path   = config['logging']['model_save_path']
    device = "cpu"
    icores=2 #hard coded for now cause all topomaps do ico-02 anyway
    model_config_ICA = {
        "dim": config['topomap_encoders']['dim'],
        "depth": config['topomap_encoders']['depth'],
        "heads": config['topomap_encoders']['heads'],
        "num_vertices": config[f'sub_ico_{icores}']['num_vertices'],
        "num_channels": config['data']['topo_dims'][0],
        "num_patches": config[f'sub_ico_{icores}']['num_patches'],
        "dropout": config['topomap_encoders']['dropout'],
        "emb_dropout": config['topomap_encoders']['emb_dropout'],
        "specific_model": "2026_0831_d6h3_tiny_adamW_recon_ICAd15_ICAd15_ico02_LinearDecoder"
    }
    model_config_PFM = {
        "dim": config['topomap_encoders']['dim'],
        "depth": config['topomap_encoders']['depth'],
        "heads": config['topomap_encoders']['heads'],
        "num_vertices": config[f'sub_ico_{icores}']['num_vertices'],
        "num_channels": config['data']['topo_dims'][1],
        "num_patches": config[f'sub_ico_{icores}']['num_patches'],
        "dropout": config['topomap_encoders']['dropout'],
        "emb_dropout": config['topomap_encoders']['emb_dropout'],
        "specific_model": "2026_0903_d6h3_tiny_adamW_recon_PFMd14_PFMd14_ico02_LinearDecoder"
    }
    # connectome details
    input_dim=100
    model_config_schaefer100 = {
        "connectome_features": int(0.5 * input_dim*(input_dim-1)),
        "dim": config['connectome_encoders']['dim'],
        "emb_dropout": config['connectome_encoders']['emb_dropout'],
        "specific_model": "2026_0909_adamW_recon_demean_demean_schaeferd100_schaeferd100_LinearDecoder"
    }
    input_dim=200
    model_config_schaefer200 = {
        "connectome_features": int(0.5 * input_dim*(input_dim-1)),
        "dim": config['connectome_encoders']['dim'],
        "emb_dropout": config['connectome_encoders']['emb_dropout'],
        "specific_model": "2026_0909_adamW_recon_demean_demean_schaeferd200_schaeferd200_LinearDecoder"
    }
    input_dim=300
    model_config_schaefer300 = {
        "connectome_features": int(0.5 * input_dim*(input_dim-1)),
        "dim": config['connectome_encoders']['dim'],
        "emb_dropout": config['connectome_encoders']['emb_dropout'],
        "specific_model": "2026_0909_adamW_recon_demean_demean_schaeferd300_schaeferd300_LinearDecoder"
    }
    input_dim=360
    model_config_glasser360 = {
        "connectome_features": int(0.5 * input_dim*(input_dim-1)),
        "dim": config['connectome_encoders']['dim'],
        "emb_dropout": config['connectome_encoders']['emb_dropout'],
        "specific_model": "2026_0909_adamW_recon_demean_demean_glasserd360_glasserd360_LinearDecoder"
    }

    #choose which to use
    topomap_encoder = getattr(experiment_topomap_recon, config['topomap_encoders']['expert_name']) 
    connectome_encoder = getattr(experiment_connectome_recon, config['connectome_encoders']['expert_name']) 
    dataset_choice = config['training']['dataset_choice']
    model_saved_main_path=f"{model_save_path}/{dataset_choice}"
    # topomap experts
    ICA_encoder = topomap_encoder(**model_config_ICA).to(device).eval()
    specific_model=model_config_ICA["specific_model"]
    ICA_encoder.load_state_dict(torch.load(f"{model_saved_main_path}/SiT/reconstruction/SiT_{specific_model}_MSE.pt", weights_only=True))
    PFM_encoder = topomap_encoder(**model_config_PFM).to(device).eval()
    specific_model=model_config_PFM["specific_model"]
    PFM_encoder.load_state_dict(torch.load(f"{model_saved_main_path}/SiT/reconstruction/SiT_{specific_model}_MSE.pt", weights_only=True))

    # connectome experts
    schaefer100_encoder = connectome_encoder(**model_config_schaefer100).to(device).eval()
    specific_model=model_config_schaefer100["specific_model"]
    schaefer100_encoder.load_state_dict(torch.load(f"{model_saved_main_path}/linear/reconstruction/linear_{specific_model}_MSE.pt", weights_only=True))
    schaefer200_encoder = connectome_encoder(**model_config_schaefer200).to(device).eval()
    specific_model=model_config_schaefer200["specific_model"]
    schaefer200_encoder.load_state_dict(torch.load(f"{model_saved_main_path}/linear/reconstruction/linear_{specific_model}_MSE.pt", weights_only=True))
    schaefer300_encoder = connectome_encoder(**model_config_schaefer300).to(device).eval()
    specific_model=model_config_schaefer300["specific_model"]
    schaefer300_encoder.load_state_dict(torch.load(f"{model_saved_main_path}/linear/reconstruction/linear_{specific_model}_MSE.pt", weights_only=True))
    glasser360_encoder = connectome_encoder(**model_config_glasser360).to(device).eval()
    specific_model=model_config_glasser360["specific_model"]
    glasser360_encoder.load_state_dict(torch.load(f"{model_saved_main_path}/linear/reconstruction/linear_{specific_model}_MSE.pt", weights_only=True))

    #training model details
    fcn_train = getattr(functions_train, config['training']['fcn_train'])  
    # fcn_model_module = archictecture
    dataset_choice = config['training']['dataset_choice']
    overfit_condition = config['training']['overfit_condition']
    # overfit_condition_sub_range = config['training']['overfit_condition_sub_range'] if overfit_condition is True else 0 #subset of subjects to debug on
    train_batch_sz = config['training']['bs'] if overfit_condition is False else 3
    LR = config['training']['LR']
    val_epoch = config['training']['val_epoch']
    train_epoch_range = config['training']['epochs']
    # bilateral_condition = config['training']['bilateral_condition'] #Bool    
    model_type = config['data']['model_type']
    # hemi_cond = config['training']['hemi_cond']
    # left_or_right = "L" if hemi_cond == "1L" else "R"
    # sub_ids_path = config['data']['sub_ids_path']    
    model_details = config['fusion_params']['model_details']
    write_fpath = config['logging']['live_logfile'].format(model_type) + '.print' #.format(model_type, operation, data_type_input, input_dim, data_type_output, output_dim, decoder_name) + '.print'
    write_to_file(f"Model details are: {model_details}\n", filepath=write_fpath)

    # init model validation vals and test flag
    device = "cpu"
    TEST_FLAG = config['testing']['immediate_test_flag']
    folder_to_save_model = f'{model_save_path}/{dataset_choice}/{model_type}'
    folder_to_save_losses = f'{model_output_path}/{dataset_choice}/{model_type}/{model_details}'

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
    if overfit_condition is False:
        sub_ids_path             = config['data']['sub_ids_path']
    else:
        sub_ids_path             = '/ceph/chpc/shared/janine_bijsterbosch_group/naranjorincon_scratch/NeuroTranslate/CHIMERA-fMRI/utils/subj_ids/ABCDv6/ABCD_train_val_test_split_smallversion.csv'

    train_val_test_csv           = pd.read_csv(sub_ids_path)
    get_sub_ids                  = train_val_test_csv["subID"] #add sub- for later use
    get_sub_ids                  = np.asarray(['sub-'+get_sub_ids[iii] for iii in range(len(get_sub_ids))])
    df_with_sub_prefix           = pd.DataFrame({"index": np.arange(len(get_sub_ids)), "subID": get_sub_ids})
    
    topomaps        = config['data']['topomaps'] #: ["ICA", "PFM"]
    topo_dims       = config['data']['topo_dims'] #: [15, 14]
    topo_prep       = config['data']['topo_prep'] #: ["norm", "norm"]
    connectomes     = config['data']['connectomes'] #: ["schaefer", "schaefer", "schaefer", "glasser"]
    connectome_dims = config['data']['connectome_dims'] #: [100, 200, 300, 360]
    connectome_prep = config['data']['connectome_prep'] #: ["demean", "demean", "demean", "demean"]

    all_data_type_input           = topomaps+connectomes # ["ICA", "PFM"]+["schaefer", "schaefer", "schaefer", "glasser"]
    all_input_dim                 =  topo_dims+connectome_dims
    all_input_representation_type = ["topomap", "topomap", "connectome", "connectome", "connectome", "connectome"]
    all_prep_type_input           = topo_prep+connectome_prep
    write_to_file(all_prep_type_input,  filepath=write_fpath)
    write_to_file(f"check check check: {all_data_type_input}, {type(all_data_type_input)}",  filepath=write_fpath)
    input_files=[]
    for chosen_ii, id in enumerate(all_data_type_input):
        input_files_curr, _ = fcn_get_file_lists_and_sort(
                                    main_brainrep_data_path_root=main_brainrep_data_path_root,
                                    dataset_choice=dataset_choice, 
                                    data_type_input=all_data_type_input[chosen_ii],
                                    input_dim=all_input_dim[chosen_ii], 
                                    icores='2', 
                                    input_representation_type=all_input_representation_type[chosen_ii],
                                    get_sub_ids=get_sub_ids, 
                                    operation='reconstruction', 
                                    left_or_right="L",
                                    df_with_sub_prefix=df_with_sub_prefix
                                    )
        input_files.append(input_files_curr)

    write_to_file(f"len {len(input_files)} {len(input_files[0])} {len(input_files[-1])}",  filepath=write_fpath)
    #get data as a torch dataset                
    train_loader, val_loader, test_loader, output_average_list = fcn_get_torch_loaders(input_files=input_files, 
                                                    train_batch_sz=train_batch_sz, 
                                                    prep_choice_input=all_prep_type_input,
                                                    regular_reconstruction=False,
                                                    train_val_test_csv=train_val_test_csv
                                                    )


    # def train(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    write_to_file(f"Device: {device}",  filepath=write_fpath)

    sit_encs, topo_networks, conn_encs, conn_latents, conn_n_tri = build_encoders(ICA_encoder, PFM_encoder, schaefer100_encoder, schaefer200_encoder, schaefer300_encoder, glasser360_encoder, device)
    
    model_kwargs = dict(
        sit_encoders   = sit_encs,
        topo_networks = topo_networks,
        conn_encoders = conn_encs,
        conn_latents  = conn_latents,
        conn_n_tri    = conn_n_tri,
        fusion_dim    = config['fusion_params']['fusion_dim'], #args.fusion_dim,
        sit_latent    = config['topomap_encoders']['dim'],
        pool_patches  = config['fusion_params']['pool_patches'], #args.pool_patches,
        beta          = 0.0,            # start at 0, anneal up
        free_bits     = 0.1,
        conn_hidden   = config['fusion_params']['conn_hidden'], #args.conn_hidden,
    )

    if model_type == "PoE":
        model = poe_model.PoEMultimodalModel(**model_kwargs).to(device)
    elif model_type == "MoE":
        model = moe_model.MoEMultimodalModel(**model_kwargs, n_iwae_samples=config['fusion_params']['n_iwae_samples']).to(device)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    write_to_file(f"\nModel: {model_type.upper()}, Strategy: {'A (pooled)' if config['fusion_params']['pool_patches'] else 'B (patch-wise)'}", filepath=write_fpath)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen    = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    write_to_file(f"Trainable params : {trainable:,}",  filepath=write_fpath)
    write_to_file(f"Frozen params    : {frozen:,}\n",  filepath=write_fpath)

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

    annealer = models.KLAnnealer(
        model,
        beta_start   = 0.0,
        beta_end     = config['fusion_params']['beta'],
        anneal_steps = len(train_loader) * config['fusion_params']['warmup_epochs'],
    )

    # weight inits for each expert. Might need to adjust if surfaces >> connectomes.
    recon_weights = {"topo_ICA": 1.0, "topo_PFM": 1.0,
                     "conn_schaefer100": 1.0, "schaefer200": 1.0,
                     "conn_schaefer300": 1.0, "glasser360": 1.0}

    # ── training ──────────────────────────────────────────────────────────────
    topo_keys = list(sit_encs.keys())
    conn_keys = list(conn_encs.keys())
    best_val_corr = -1.0
    best_val_mse = 10e100
    df_train = pd.DataFrame(columns=['train_mse', 'train_loss', 'train_corr', 'train_kl'])
    df_val = pd.DataFrame(columns=['val_mse', 'val_corr'])
    df_test = pd.DataFrame(columns=['test_mse', 'test_corr'])
    df_train_inference = pd.DataFrame(columns=['train_mse', 'train_corr'])
    mean = lambda lst: float(np.mean(lst)) if lst else float("nan")
    skipped = 0
    for epoch in range(train_epoch_range):
        epoch_loss, epoch_recon, epoch_kl, epoch_grad = fcn_train(model, train_loader, device=device,
                    optimizer=optimizer, annealer=annealer, scheduler_flag=scheduler,
                    lr_schedule=lr_schedule, recon_weights=recon_weights)

        # Skip NaN batches
        if np.isinf(mean(epoch_loss)) or np.isnan(mean(epoch_loss)):
            warnings.warn("[Epoch Non-finite loss, skipping batch. :( )]")
            optimizer.zero_grad()
            skipped += 1
            continue

        write_to_file(f'| Training | Epoch - {epoch:03d}/{train_epoch_range} | LR - {lr_schedule.get_last_lr()[0]:.2e}| kl_beta={annealer.current_beta:.3f}| kl={mean(epoch_kl):.4f} | Loss - {mean(epoch_loss):.4f} | MSE = {mean(epoch_recon):.4f}', filepath=write_fpath)
        new_row = pd.DataFrame({'train_mse': [mean(epoch_recon)],
                                'train_loss': [mean(epoch_loss)], 
                                'train_kl': [mean(epoch_kl)]})
        
        df_train = pd.concat([df_train, new_row], ignore_index=True)
        df_train.to_csv(os.path.join(folder_to_save_losses, 'train_losses_patch.csv'))
        # ── validation (every N epochs) ───────────────────────────────────────
        if epoch % val_epoch == 0:
            val_metrics = evaluate(model, val_loader, device, conn_keys, topo_keys)

            write_to_file("  Validation:", filepath=write_fpath)
            for k, v in val_metrics.items():
                write_to_file(f"    {k}: {v:.4f}", filepath=write_fpath)

            joint_corr = val_metrics.get("corr/joint→conn_schaefer100", -1) #best model with corr joint->schf100
            joint_mse = val_metrics.get("mse/joint→conn_schaefer100", -1)
            new_row = pd.DataFrame({'val_mse': [joint_mse], 'val_corr': [joint_corr]})
            df_val = pd.concat([df_val, new_row], ignore_index=True)
            df_val.to_csv(os.path.join(folder_to_save_losses, 'val_losses_patch.csv'))
            if joint_corr > best_val_corr:
                best_val_corr = joint_corr
                torch.save(model.state_dict(), f"{folder_to_save_model}/{model_type}_RHO_all2schaefer100.pt")
                write_to_file(f"  ✓ New best saved (corr/joint→conn_schaefer100={best_val_corr:.4f})", filepath=write_fpath)

            if joint_mse < best_val_mse:
                best_val_corr = joint_mse
                torch.save(model.state_dict(), f"{folder_to_save_model}/{model_type}_MSE_all2schaefer100.pt")
                write_to_file(f"  ✓ New best saved (mse/joint→conn_schaefer100={best_val_corr:.4f})", filepath=write_fpath)


    write_to_file(f"\nTraining complete. skipped={skipped} Best corr/joint→conn_schaefer100 = {best_val_corr:.4f}", filepath=write_fpath)
    write_to_file(f"\nTraining complete. Best mse/joint→conn_schaefer100 = {best_val_corr:.4f}", filepath=write_fpath)

    # TESTING #
    if TEST_FLAG:
        results_test, results_train, test_output_list, train_output_list = test_inference(model=model,
                                                     train_loader=train_loader,
                                                     test_loader=test_loader,
                                                     device=device,
                                                     conn_keys=conn_keys, topo_keys=topo_keys
                                                    )
        write_to_file('TEST FLAG ON. TESTING.', filepath=write_fpath)

        joint_corr = results_test.get("corr/joint→conn_schaefer100", -1) #best model with corr joint->schf100
        joint_mse = results_test.get("mse/joint→conn_schaefer100", -1)
        new_row = pd.DataFrame({'test_mse': [joint_mse], 'tests_corr': [joint_corr]})
        df_test = pd.concat([df_test, new_row], ignore_index=True)
        df_test.to_csv(os.path.join(folder_to_save_losses, 'inference_test.csv'))

        joint_corr = results_train.get("corr/joint→conn_schaefer100", -1) #best model with corr joint->schf100
        joint_mse = results_train.get("mse/joint→conn_schaefer100", -1)
        new_row = pd.DataFrame({'train_mse': [joint_mse], 'train_corr': [joint_corr]})
        df_train_inference = pd.concat([df_train_inference, new_row], ignore_index=True)
        df_train_inference.to_csv(os.path.join(folder_to_save_losses, 'inference_train.csv'))

        # see all models  
        model_path = sorted(glob.glob(f"{folder_to_save_model}/*{model_type}_{chosen_test_model}_all2schaefer100.pt")) # look at training script for details, but all models saves as type_details_chosen: ex-kBGTLN_d6h5_demeanL2_skewloss_RHO.pt
        chosen_model = model_path[0]
        write_to_file(f'\n\nmodel loaded is {chosen_model}', filepath=write_fpath)
        model.load_state_dict(torch.load(chosen_model)) # most recent model

        # Find number of parameters
        model_params = sum(p.numel() for p in model.parameters())
        write_to_file(f"\n\nModel params: {model_params}", filepath=write_fpath)

    names_all = ["PRED_ICA15", "ICA15", "PRED_PFM14", "PFM14",
            "PRED_SCHF100", "scf100", "PRED_SCHF200", "scf200",
            "PRED_SCHF300", "scf300", "PRED_GLSS360", "glss360"]
    for tii in range(len(names_all)):
        write_to_file(f"LIST SHAPE: {test_output_list[tii].shape}", filepath=write_fpath)
        np.save(f"{folder_to_save_test}/test_{model_type}_{names_all[tii]}.npy", test_output_list[tii])
        np.save(f"{folder_to_save_test}/train_{model_type}_{names_all[tii]}.npy", train_output_list[tii])
        
    write_to_file(f"TEST complete for model:\n{model_details}", filepath=write_fpath)

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
    # train(config)
