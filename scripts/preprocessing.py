# -*- coding: utf-8 -*-
# @Author: Samuel Naranjo Rincon
# @Last Modified time: 2026-04-08 

'''
This file is used to preprocess data surface metrics into triangular patches
filling the entire surface. 
inputs: (M,C) - M mesh vertices; C channels
outputs: (N,L,V,C) - N subjects; L sequence lenght; V number of vertices per patch; C channels
'''

import pandas as pd
import nibabel as nb
import numpy as np
import yaml
import os
import argparse
import sys
sys.path.append('../')
sys.path.append('./')
sys.path.append('../../')
# from utils.utils import write_to_file

def main(config,options):
    def write_to_file(content):
        # task = config['data']['task']
        with open(f"/ceph/chpc/shared/janine_bijsterbosch_group/naranjorincon_scratch/NeuroTranslate/surf2netmat/batch/SiT_prep.print", 'a') as file:
            file.write(str(content) + '\n')

    write_to_file('')
    write_to_file('#'*30)
    write_to_file('Starting: preprocessing script')
    write_to_file('#'*30)

    #### PARAMETERS #####
    ico = config['resolution']['ico'] # 6
    sub_ico = config['resolution']['sub_ico'] # 2

    # configuration = config['data']['configuration']  # template #template #native
    num_channels = config['data']['channels'] # 4
    split = options.type #config['data']['split'] # train #validation # or test
    translation = config['data']['translation'] # 
    output_folder = config['output']['folder'] # ../data/{surf2mat}/{template}/
    output_folder_netmat = config['output']['output_folder_netmat']
    dataset = config['data']['dataset'] 
    path_to_data = config['data']['data_path'] #  /data/
    nm_fs_data = config['data']['fs_data_path'] # d15_fs_LR
    sub_ids_path = config['data']['sub_ids_path'] # ../labels/HCPdb/
    parcellation_type=config['data']['parcellation_type']
    parcellation_name = config['data']['parcellation_name']
    num_vertices = config['sub_ico_{}'.format(sub_ico)]['num_vertices'] # sub_ico_2
    num_patches = config['sub_ico_{}'.format(sub_ico)]['num_patches'] # sub_ico_2
    chosen_hemi = config['data']['hemisphere'] #1L or 1R or 2 for both

    ####
    # regardless of below, we wanna use same mesh anyway
    if dataset == "HCPYA":
        ids = pd.read_csv(os.path.join(sub_ids_path, '{}/{}_upt.csv'.format(translation,split)))['ids'].to_numpy().reshape(-1)
    elif dataset == "HCPYA_ABCDdr":
        ids = pd.read_csv(f"{sub_ids_path}/{split}_upt.csv")['ids'].to_numpy().reshape(-1)
        print(ids)
        # ids = pd.read_csv(f"{sub_ids_path}/{split}_upt.csv", header=None).iloc[0,:].to_numpy().reshape(-1)
    elif dataset == "ABCD":
        if parcellation_type == "full":
            ids = pd.read_csv(f"{sub_ids_path}/{split}_netmat.csv", header=None).iloc[0,:].to_numpy().reshape(-1)
        elif parcellation_type == "partial":
            ids = pd.read_csv(f"{sub_ids_path}/{split}_partialnetmat.csv", header=None).iloc[0,:].to_numpy().reshape(-1)
    
    # labels
    if dataset == "HCPYA":
        if split == 'train':
            # labels = pd.read_csv(os.path.join(label_path, '{}/{}.csv'.format(task,'vecmats'))).to_numpy()
            labels = pd.read_csv(os.path.join(sub_ids_path, '{}/{}.csv'.format(translation,'vecmats_train_upt'))).to_numpy()
        elif split == 'validation':
            # labels = pd.read_csv(os.path.join(label_path, '{}/{}.csv'.format(task,'vecmats_validation'))).to_numpy() 
            labels = pd.read_csv(os.path.join(sub_ids_path, '{}/{}.csv'.format(translation,'vecmats_validation_upt'))).to_numpy()  
        elif split == 'test':
            # labels = pd.read_csv(os.path.join(label_path, '{}/{}.csv'.format(task,'vecmats_test'))).to_numpy()
            labels = pd.read_csv(os.path.join(sub_ids_path, '{}/{}.csv'.format(translation,'vecmats_test_upt'))).to_numpy()    
    elif dataset == "HCPYA_ABCDdr":
            assert parcellation_type == "full", write_to_file(f"Currently, there is nothing for -- {parcellation_type} netmats.") 
            if parcellation_type == "full":
                labels = pd.read_csv(f"{sub_ids_path}/vecmats_{split}_upt.csv").to_numpy()
              
    elif dataset == "ABCD":
            if parcellation_name == "schaefer":
                if parcellation_type == "full":
                    labels = pd.read_csv(f"{sub_ids_path}/{split}_netmat.csv").to_numpy()
                if parcellation_type == "partial":
                    labels = pd.read_csv(f"{sub_ids_path}/{split}_partialnetmat.csv").to_numpy()
            
                labels = labels.T #transpose to make subjXvector
                labels_check_nans = np.isnan(labels).sum() #check if has nans and how many
                write_to_file(f"Label NaN Count: {labels_check_nans}")
            
            elif parcellation_name == "glasser":
                if parcellation_type == "full":
                    labels = pd.read_csv(f"{sub_ids_path}/{split}_netmat.csv")#.to_numpy()
                if parcellation_type == "partial":
                    labels = pd.read_csv(f"{sub_ids_path}/{split}_partialnetmat.csv")#.to_numpy()
                labels_check_nans = 1 # not zero so it assumes there are NaNs in glasser TODO is update this to be more general cause hard coded right now
                # dont transpose to check NaNs across columns==Subjects
    
    
    '''
    Below is meant to get the NaNs from vecnetmats and remove them from ids. This shoudl workd fine later because 
    when getting resamp_surf*.gii it goes off of ids so if we update ids here, it should fix the more downstream process.
    However, there ARE also NaNs in surf data so this might need more work later.
    '''
    if labels_check_nans !=0: # if there are nans in vecnetmats
        if parcellation_name == "schaefer": #scahefer we know the NaN subjects a priori, glasser we need to clean as the script goes
            # when need to run local - add this to start: '/Users/snaranjo/Desktop/neurotranslate/mount_point'
            read_nan_subj_netmats = pd.read_csv('/ceph/chpc/shared/janine_bijsterbosch_group/naranjorincon_scratch/NeuroTranslate/surf2netmat/utils/subjects_with_nan.txt',sep=" ", header=None)
            read_nan_subj_netmats = read_nan_subj_netmats[0].values #makes df into numpy, not sure how differnt if at all from df.to_numpy()
            write_to_file(f"Subs VEC_NETMATs to remove - {read_nan_subj_netmats}")
            nan_mask = ~np.isin(ids, read_nan_subj_netmats) #ids is NDA* and read_nan_subj_netmats are those needed removed, so nan_mask is NOT is in, so ids excluding the ones we want to remove
            new_ids = ids[nan_mask] # here we set that as idx and get new_ids. If 3 subs to remove then new_ids = count(ids) - 3
            write_to_file(f"Full IDs no vecnetmat NaN -- {new_ids.shape}") # make sure this makes sense based on above print "read_nan_subj_netmats"

            nan_indices = np.where(np.isin(ids, read_nan_subj_netmats))[0]
            write_to_file(f"NaN indeces: {nan_indices}")

            mask = np.ones(labels.shape[0], dtype=bool) # make mask of size of labels
            mask[nan_indices] = False # make where we found the NaNs ie subjects removed into false (all tohers are truw cause ones)
            labels_nan_clean = labels[mask] # update accordingly
        elif parcellation_name == "glasser":
            nanii = labels.isna().any() # look for NaNs across columns
            write_to_file(f"GLASSER Label NaN Count: {nanii.sum()}") # num of subjects with NaNs in their netmats
            read_nan_subj_netmats =  ids[nanii] # subject IDs to remove 
            write_to_file(f"Subs VEC_NETMATs to remove - {read_nan_subj_netmats}")

            nan_mask = ~np.isin(ids, read_nan_subj_netmats) #ids is NDA* and read_nan_subj_netmats are those needed removed, so nan_mask is NOT is in, so ids excluding the ones we want to remove
            new_ids = ids[nan_mask] # here we set that as idx and get new_ids. If 3 subs to remove then new_ids = count(ids) - 3
            write_to_file(f"Full IDs no vecnetmat NaN -- {new_ids.shape}")

            nan_indices = np.where(nanii)
            write_to_file(f"NaN indeces: {nan_indices}")

            labels = labels.to_numpy()
            labels = labels.T # here is where we can transpose them
            mask = np.ones(labels.shape[0], dtype=bool) # make mask of size of labels
            mask[nan_indices] = False # make where we found the NaNs ie subjects removed into false (all tohers are truw cause ones)
            labels_nan_clean = labels[mask] # update accordingly to only True values?
            write_to_file(f"NaN indeces: {labels_nan_clean} \n {type(labels_nan_clean)}")
        else:
            assert 1==2, write_to_file(f"Unknown parcelaltion chosen. Quitting.")

        if chosen_hemi == '2LR':
            labels_nan_clean = np.concatenate((labels_nan_clean,labels_nan_clean))

        write_to_file(f"Some subjects had nans(N={len(nan_indices)}). Removed, new ids and labels shape ==> {labels_nan_clean.shape}; original:{labels.shape}")
    else:
        labels_nan_clean = labels # same labels without cleaning because no NaNs found in them
        if chosen_hemi == '2LR':
            labels_nan_clean = np.concatenate((labels_nan_clean,labels_nan_clean))
            
        new_ids = ids # jsut leave it be but next lines expect new_ids so this is one way around that
    
    assert np.isnan(labels_nan_clean).sum() == 0, "Cleaning labels did not work."

    write_to_file("")
    write_to_file("split choice:"+split)
    
    # below, goal is to also clean surfaces so that we also remove subjects with NaNs
    ids = new_ids #ids with removed labels
    num_subjects = ids.shape[0]
    write_to_file('Num of subjects:'+str(num_subjects))
    write_to_file('')

    data = [] # list of numpy arrays each is a numpy array version of the shape.gii info
    nan_surf_subject_count = 0
    sub_ids_nan_vals = []
    write_to_file(f'Dataset is {dataset}')
    if chosen_hemi == '1L':
        write_to_file('Left hemisphere was chosen.')
        for i, id in enumerate(ids): # reads in actual id num with 'id' inside the pandas column from the read csv, see above ids variable

            if dataset == "ABCD":
                filename = os.path.join(f'{path_to_data}/resamp_{id}.L.shape.gii')
            elif dataset == "HCPYA_ABCDdr":
                filename = os.path.join(f'{path_to_data}/resamp_{id}.L.shape.gii')
            else:
                filename = os.path.join(path_to_data,'ICA_fs_LR_32',nm_fs_data,'resamp_{}.L.shape.gii'.format(id)) # shape.gii is a metric file for values at each vertex, in our case the ICA values
            
            # check if real, ow skip to next
            check_exists = os.path.exists(filename)
            if check_exists is False:
                write_to_file('This path did not exists, skipping:{}'.format(filename))
                continue

            # data is then an array where each element is a shape.gii image
            # agg based on channels, bc if you look below at print, data is [15,49k] so resamp gray ordinates 
            # for each ICA!! I think this then aggregates based on channels? See below if changes
            # from nibabel website, agg_data() is "Aggregate GIFTI data arrays into an ndarray or tuple of ndarray", assuming that makes conversion to numpy easy
            data.append(np.array(nb.load(filename).agg_data())[:num_channels,:])

            if np.isnan(data[i]).sum() != 0: # so if there are nans
                data_check_nans = np.isnan(data[i]).sum()
                nan_surf_subject_count += 1
                sub_ids_nan_vals.append(id)
                write_to_file(f"DATA SUBJ ID:{id}/ iter:{i} NaN Count - {data_check_nans}")

            if i%300==0:
                # check_load = nb.load(filename)
                # check_agg_data = nb.load(filename).agg_data()
                write_to_file('\nLoading GIFTI for subject: {}'.format(i))
                # write_to_file('\nChecking data metrics. \n MRI data has loaded number of data arrays: {} \n After Using agg_data(), looking into first tuple: {}'.format(len(check_load.darrays),len(check_agg_data[0])))
                write_to_file('\nActual stored data value, presumably aggregated through channels=inputdim:{}'.format(len(data[i])))
            # from sanity checks, I see now that our data values are dim C x TS, our in our case inputdim x TS for each vertex in the sphere
    elif chosen_hemi == '1R':
        write_to_file('Right hemisphere was chosen.')
        for i, id in enumerate(ids):
        
            if dataset == "ABCD":
                filename = os.path.join(f'{path_to_data}/resamp_{id}.L.shape.gii')
            elif dataset == "HCPYA_ABCDdr":
                filename = os.path.join(f'{path_to_data}/resamp_{id}.L.shape.gii')
            else:
                filename = os.path.join(path_to_data,'ICA_fs_LR_32',nm_fs_data,'resamp_{}.L.shape.gii'.format(id)) # shape.gii is a metric file for values at each vertex, in our case the ICA values
            
            # check if real, ow skip to next
            check_exists = os.path.exists(filename)
            if check_exists is False:
                write_to_file('This path did not exists, skipping:{}'.format(filename))
                continue

            data.append(np.array(nb.load(filename).agg_data())[:num_channels,:])

            if np.isnan(data[i]).sum() != 0: # so if there are nans
                data_check_nans = np.isnan(data[i]).sum()
                nan_surf_subject_count += 1
                sub_ids_nan_vals.append(id)
                write_to_file(f"DATA SUBJ ID:{id}/ iter:{i} NaN Count - {data_check_nans}")

            if i%300==0:
                write_to_file('\nLoading GIFTI for subject: {}'.format(i))

    elif chosen_hemi == '2LR':
        left_nan_list = []
        # left_nan_idx = []
        right_nan_list = []
        right_nan_idx = []

        clean_subj_ids = []
        write_to_file('Both hemispheres were chosen. Starting with Left.')
        for i,id in enumerate(ids):
            # data here becomes [iiL,iiR, ii+1L,ii+1R .... NL,NR] so shape is list of num_subs*2?
            if dataset == "ABCD":
                filename = os.path.join(f'{path_to_data}/resamp_{id}.L.shape.gii')
            elif dataset == "HCPYA_ABCDdr":
                filename = os.path.join(f'{path_to_data}/resamp_{id}.L.shape.gii')
            else:
                filename = os.path.join(path_to_data,'ICA_fs_LR_32',nm_fs_data,'resamp_{}.L.shape.gii'.format(id)) # shape.gii is a metric file for values at each vertex, in our case the ICA values
            
            # check if real, ow skip to next
            check_exists = os.path.exists(filename)
            if check_exists is False:
                write_to_file('This path did not exists, skipping:{}'.format(filename))
                continue
            
            curr_surf_data = np.array(nb.load(filename).agg_data())[:num_channels,:]
            if np.isnan(curr_surf_data).sum() != 0: # so if there are nans in currentt subj
                data_check_nans = np.isnan(curr_surf_data).sum()
                nan_surf_subject_count += 1
                # left_nan_list.append(id)
                # left_nan_idx.append(i)
                write_to_file(f"DATA SUBJ ID:{id}/ iter:{i} NaN Count - {data_check_nans}")
                continue # skip this and dont add to data list since NaNs anyway. Pre-emptive cleaning ig.

            data.append(curr_surf_data)
            
            clean_subj_ids.append(id) #subj id that is CLEAN and itll be LLLLL...RRRRR

            # write_to_file('Both hemispheres were chosen. Now doing with Right.')
        # l_order=0 #only load R hemis that had a clean L
        # for i,id in enumerate(ids): #ids > clean_subj_ids, so using that to idx with i - maybe not smartest cause assumes ids > clean_subj_ids to work
        #     # load rigth hemisphere
        #     if id != clean_subj_ids[l_order]: #check that for an id a left hemi has been saved (so clean L hemi)
        #         # write_to_file(f"Left hemi ID:{clean_subj_ids[l_order]}, idx:{l_order}")
        #         continue

            if dataset == "ABCD": #load the ID cause passed L hemi clean check
                filename = os.path.join(f'{path_to_data}/resamp_{id}.R.shape.gii')
            else:
                filename = os.path.join(path_to_data,'ICA_fs_LR_32',nm_fs_data,'resamp_{}.R.shape.gii'.format(id))

            curr_surf_data = np.array(nb.load(filename).agg_data())[:num_channels,:]
            # see if for some reason the L was fine but R has NaNs. 
            if np.isnan(curr_surf_data).sum() != 0: # so if there are nans in currentt subj
                data_check_nans = np.isnan(curr_surf_data).sum()
                nan_surf_subject_count += 1
                # right_nan_list.append(id)
                # right_nan_idx.append(i)
                write_to_file(f"DATA SUBJ ID:{id}/ iter:{i} NaN Count - {data_check_nans}")
                
                # in case somehow L is good but R has NaNs, delete L and skip this R to keep it clean for 2LR condition training. 
                del data[-1], clean_subj_ids[-1] # delete most recent which is L hemisphere for this subj
                # l_order += 1 
                continue
            
            data.append(curr_surf_data)
            clean_subj_ids.append(id) #subj id that is CLEAN and itll be LRLR..LR
            # l_order += 1 
            if i%300==0:
                write_to_file('\nLoading GIFTI for subject: {}'.format(i))

        write_to_file(f"After RIGHT data loading. clean_subj_ids: {len(clean_subj_ids)}")
        # write_to_file(f"After LEFT data loading. clean_subj_ids: {len(clean_subj_ids)}")

        assert len(data) == len(clean_subj_ids), "data and clean_subj_List don't match. Something wrong with NaN exclusion."

        ids = np.sort(np.concatenate((ids,ids))) # when doing 2LR, need to double IDS
        write_to_file(f"check sort ids to make sure irder is therefire LRLRLR..LR {ids[0:2]} {ids[-1:-3]}")
        write_to_file(f"ids shape: {ids.shape}")
        write_to_file(f"ids unique: {len(np.unique(ids))}")

        clean_subj_ids = np.asarray(clean_subj_ids) # no need to concat or sort because already LRLRLR..LR
        write_to_file(f"check sort clean_subj_ids to make sure irder is therefire LRLRLR...LR {clean_subj_ids}")
        write_to_file(f"clean_subj_ids shape: {clean_subj_ids.shape}")
        write_to_file(f"clean_subj_ids unique: {len(np.unique(clean_subj_ids))}")

        clean_ids = np.isin(ids, clean_subj_ids) #ids that ARE in clean_subj_ids
        write_to_file(f"check sort clean_ids to make sure order is therefore LRLR..LR {ids[0:2]} {ids[-1]}")
        write_to_file(f"clean_ids shape: {ids[clean_ids].shape}")
        write_to_file(f"clean_ids unique: {len(np.unique(ids[clean_ids]))}")
        df_version = pd.DataFrame(ids[clean_ids])
        df_version.to_csv(f'{sub_ids_path}/{split}_subj_IDs_clean_BOTH_{dataset}.csv')
        write_to_file(f"Clean subj count BOTH: - {ids[clean_ids].shape}.")

        data = np.asarray(data) # for 2LR data is already cleaned cause Subj with NaNs get skipped
        labels = labels_nan_clean[clean_ids] # boolean ind and it wokred on ids==shape==labels_nan_clean.
        write_to_file(f"2LR merged labels shape: {labels.shape}")
        write_to_file(f"2LR merged data shape: {data.shape}")

        ids = ids[clean_ids] #now make IDs bve the clean ones
        write_to_file(f"2LR merged IDs shape: {ids.shape}")

        num_subjects = len(np.unique(ids))
        write_to_file(f"Num of subjects is: {num_subjects}")

    if chosen_hemi == '1L' or chosen_hemi == '1R':
        if sub_ids_nan_vals: # if True, then list is not empty == there are subjects with NaNs
            write_to_file(f"1L/1R Surf NaNs found, so cleaning - {len(sub_ids_nan_vals)}.")
            'Below, creates DF of subject IDS that are clean.'

            clean_subs = ~np.isin(ids, sub_ids_nan_vals)
            df_version = pd.DataFrame(ids[clean_subs])
            df_version.to_csv(f'{sub_ids_path}/{split}_subj_IDs_clean_{dataset}.csv')
            write_to_file(f"Clean subj count: - {clean_subs.shape}.")

            'Identifying NaN indeces'
            old_data = data
            data_nan_indices = np.where(np.isin(ids, sub_ids_nan_vals))[0]
            write_to_file(f"surf_nan_idx - {data_nan_indices}") # subjects with surf data that also have NaNs and need removal

            # merge data and labels cause both have different subjects with nans
            mask_data = np.ones(len(old_data), dtype=bool) # make mask of length of data == N, 
            write_to_file(f"Data mask shape: {mask_data.shape}")
            mask_data[data_nan_indices] = False # make where we found the NaNs ie subjects removed into false (all tohers are truw cause ones)
            data_np = np.asarray(old_data) # make list to np array
            data_nan_clean = data_np[mask_data] # that also allows us to index with the mask
            write_to_file(f"Clean data shape, should be N - surf_nan_idx: {data_nan_clean.shape}")

            mask_surfaces = np.ones(labels_nan_clean.shape[0], dtype=bool)
            mask_surfaces[data_nan_indices] = False # where the surf idx are for NaN make that false in label variable
            merged_labels_with_surf = labels_nan_clean[mask_surfaces] # now that those are false, index w mask and we have new labels which is clean of own NaNs and surf NaNs
            data = data_nan_clean
            labels = merged_labels_with_surf

            ids = ids[mask_surfaces]
            num_subjects = data.shape[0]
            write_to_file(f"Num of subjects is: {num_subjects}")
        else:
            write_to_file("NO NaNs found in Surf, so left alone.")
            clean_subs = ~np.isin(ids, sub_ids_nan_vals)
            df_version = pd.DataFrame(ids[clean_subs])
            df_version.to_csv(f'{sub_ids_path}/{split}_subj_IDs_clean_{dataset}.csv')
            data = np.asarray(data)

    # if chosen_hemi == '2LR':
    #     labels = np.concatenate((labels,labels))
    
    write_to_file(f"Cleaned surfaces and vec_netmats, make sure are same size and merged correctly: \nSURFACES:{data.shape} \nVEC_NETMATS:{labels.shape}")
    write_to_file(f"Count of subjects that needed to be removed based on SURF NaNs: {len(sub_ids_nan_vals)}")

    assert data.shape[0] == labels.shape[0], "surface IDs and netmat IDs are not equal"

    # lets also make sure this whole cleanign process even worked, check for NaNs again
    assert np.isnan(data).sum() == 0 and np.isnan(labels).sum() == 0, "Somehow cleaning failed try again."
    
    ## data normalisation done across 4 channels: myelination, cortical thickness, sulca depth, curvature
    norm_flag = config['data']['norm_flag']
    write_to_file('raw resamp loaded data, order is L1R1,L2R2,...,LNRN, where N = subject. Size of data should be 2*length(subjects) if bilateral, o.w. len(subs). It is length:{}'.format(len(data)))        
    write_to_file('looking at ONE of the values:\n Has shape: {}'.format(data[10].shape))
    # now, data has all feat_dim spheres and their respective L and R hemisphere data
    write_to_file('\n Data shape is now this: {} \n make sure it makes sense based off of hemisphere(s) chosen. Normalized data will be the same size and thsi variable will change.'.format(data.shape))
    if norm_flag:
        means_ch = np.mean(data, axis=1) # shape is subjxica_channels x allvertex
        stds_ch = np.std(data, axis=1)
        # if dataset == "ABCD":
        #     means = np.load(os.path.join(config['data']['sub_ids_path'],'/{}_means.npy'.format(chosen_hemi)))
        #     stds = np.load(os.path.join(config['data']['sub_ids_path'],'/{}_stds.npy'.format(chosen_hemi)))
        # elif dataset == "HCPYA":
        #     means = np.load(os.path.join(config['data']['label_path'],'{}/{}_means.npy'.format(translation,chosen_hemi)))
        #     stds = np.load(os.path.join(config['data']['label_path'],'{}/{}_stds.npy'.format(translation,chosen_hemi)))

        write_to_file(f'{means_ch} \n{stds_ch}')
        normalised_data = (data - means_ch)/stds_ch
        write_to_file('normalizing ICA maps')
        # write_to_file(f'data size: {normalised_data.shape}')
    else:
        normalised_data = data
        write_to_file('RAW ICA maps, NOT normalization')
        # write_to_file(f'data size: {normalised_data.shape}')

    write_to_file('Checking that normalization is applied to whole channel vector, not just single subtraction. \n Original data: {}, Normalized data: {}'.format(data.shape, normalised_data.shape))

    #shape of the data is nbr_subjects * 2 (cause L and R I think),channels, ICA dim, tiangles/pathces * nbr_vertices_per_triangle
    unet_flag = config['data']['unet_arch']
    patch_indeces_path = config['data']['patches_path']
    if unet_flag:
        write_to_file(f'\n UNET FLAG TRUE. Starting at at ICO-06 and going to resample to {sub_ico}')
        indices_mesh_triangles = pd.read_csv('{}/triangle_indices_ico_{}_sub_ico_{}.csv'.format(patch_indeces_path,ico,sub_ico))
        
        if chosen_hemi == '2LR':
            write_to_file('\n Because both hemispheres chosen each subject has one sphere per hemisphere, so data is num_subj*2 C P V')
            data = np.zeros((num_subjects*2, num_channels, num_patches, num_vertices))
            for i,id in enumerate(np.unique(ids)): # subjects?
                if i%400==0:
                    write_to_file('Preping patches for sub: {}'.format(id))
                for j in range(num_patches): # for each columns
                    indices_to_extract = indices_mesh_triangles[str(j)].to_numpy()
                    data[i,:,j,:] = normalised_data[2*i][:,indices_to_extract] # the 2*i is so that in iterating, so when doing the i+1 it doesnt overlap with prev data.
                    data[i+num_subjects,:,j,:] = normalised_data[2*i+1][:,indices_to_extract]
        else:
            write_to_file('\nBecause one hemisphere chosen, data is num_subj C P V')
            data = np.zeros((num_subjects, num_channels, num_patches, num_vertices))
            for i,id in enumerate(ids): # subjects?
                if i%400==0:
                    write_to_file('Preping patches for sub: {}'.format(id))
                for j in range(num_patches): # for each columns
                    indices_to_extract = indices_mesh_triangles[str(j)].to_numpy()
                    data[i,:,j,:] = normalised_data[i][:,indices_to_extract]
    else:
        # the csv below is a verteces x patches matrix for icoN where N is the ico you chose in tha yml file
        indices_mesh_triangles = pd.read_csv('{}/triangle_indices_ico_{}_sub_ico_{}.csv'.format(patch_indeces_path,ico,sub_ico))

        if chosen_hemi == '2LR':
            data = np.zeros((num_subjects*2, num_channels, num_patches, num_vertices))
            write_to_file(f'ids: {ids.shape}, {ids[0:10]}')
            write_to_file(f'data: {data.shape}, patches: {num_patches}')
            for i,id in enumerate(np.unique(ids)): # subjects?
                if i%300==0:
                    write_to_file('Preping patches for sub: {}'.format(id))
                for j in range(num_patches): # for each columns
                    indices_to_extract = indices_mesh_triangles[str(j)].to_numpy()
                    # write_to_file(f'IDX to extract {indices_to_extract.shape}')
                    data[i,:,j,:] = normalised_data[2*i][:,indices_to_extract] # the 2*i is so that in iterating, so when doing the i+1 it doesnt overlap with prev data.
                    # below shows that normalized_data is in order sub 1, sub 2 ... sub N but the new data list is all L spheres first then all R, hence the data[i] then data[i+num_subs]
                    data[i+num_subjects,:,j,:] = normalised_data[2*i+1][:,indices_to_extract]
        else:
            write_to_file('\nBecause one hemisphere chosen, data is num_subj C P V')
            data = np.zeros((num_subjects, num_channels, num_patches, num_vertices))
            write_to_file(f'ICO-{sub_ico} dummy data shape: {data.shape}')
            write_to_file(f'Double check data ids: {ids.shape}')
            for i,id in enumerate(ids): # subjects?
                if i%300==0:
                    write_to_file('Preping patches for sub: {}'.format(id))
                for j in range(num_patches): # for each columns
                    indices_to_extract = indices_mesh_triangles[str(j)].to_numpy()
                    data[i,:,j,:] = normalised_data[i][:,indices_to_extract]
                    #data[i+num_subjects,:,:,j,:] = normalised_data[2*i+1][:,indices_to_extract]   

    write_to_file('')
    write_to_file('#'*30)
    write_to_file(f'#Saving: {split} data')
    write_to_file('#'*30)
    write_to_file('')

    try:
        os.makedirs(output_folder,exist_ok=False)
        write_to_file('Creating folder: {}'.format(output_folder))
    except OSError:
        write_to_file('folder already exist: {}'.format(output_folder))
    
    if unet_flag:
        # data = data.squeeze() # meant to make into-> batchXchannlXverteces, but that was for netmat2surf so ico6 was the prediction, this is different cause input not pred
        if norm_flag:
            if chosen_hemi == '2LR':
                filename = os.path.join(output_folder,'{}_{}_surf_ico5.npy'.format(chosen_hemi,split))
                np.save(filename,data)
                filename = os.path.join(output_folder_netmat,'{}_{}_netmat_clean.npy'.format(chosen_hemi,split))
                # labels = np.concatenate((labels,labels)) 
                np.save(filename,labels)
            else:
                filename = os.path.join(output_folder,'{}_{}_surf_norm_ico5.npy'.format(chosen_hemi,split))
                np.save(filename,data)
                filename = os.path.join(output_folder_netmat,'{}_{}_netmat_clean.npy'.format(chosen_hemi,split))
                labels = labels
                # np.save(filename,labels)
        else:
            if chosen_hemi == '2LR':
                filename = os.path.join(output_folder,'{}_{}_surf_ico5.npy'.format(chosen_hemi,split))
                np.save(filename,data)
                filename = os.path.join(output_folder_netmat,'{}_{}_netmat_clean.npy'.format(chosen_hemi,split))
                # labels = np.concatenate((labels,labels))
                np.save(filename,labels)
            else:
                filename = os.path.join(output_folder,'{}_{}_surf_ico5.npy'.format(chosen_hemi,split))
                np.save(filename,data)
                filename = os.path.join(output_folder_netmat,'{}_{}_netmat_clean.npy'.format(chosen_hemi,split))
                labels = labels
                # np.save(filename,labels)
    else:
        if norm_flag:
            if chosen_hemi == '2LR':
                filename = os.path.join(output_folder,'{}_{}_surf.npy'.format(chosen_hemi,split))
                np.save(filename,data)
                filename = os.path.join(output_folder_netmat,'{}_{}_netmat_clean.npy'.format(chosen_hemi,split))
                # labels = np.concatenate((labels,labels))
                # np.save(filename,labels)
            else:
                filename = os.path.join(output_folder,'{}_{}_surf_norm.npy'.format(chosen_hemi,split))
                np.save(filename,data)
                filename = os.path.join(output_folder_netmat, '{}_{}_netmat_clean.npy'.format(chosen_hemi,split))
                labels = labels
                # np.save(filename,labels)
        else:
            if chosen_hemi == '2LR':
                filename = os.path.join(output_folder,'{}_{}_surf.npy'.format(chosen_hemi,split))
                np.save(filename,data)
                filename = os.path.join(output_folder_netmat,'{}_{}_netmat_clean.npy'.format(chosen_hemi,split))
                # labels = np.concatenate((labels,labels))
                # np.save(filename,labels)
            else:
                filename = os.path.join(output_folder,'{}_{}_surf.npy'.format(chosen_hemi,split))
                np.save(filename,data)
                filename = os.path.join(output_folder_netmat,'{}_{}_netmat_clean.npy'.format(chosen_hemi,split))
                labels = labels
                # np.save(filename,labels)

    write_to_file('')
    write_to_file('Data_prep_shape: {} \n Data_prep_type: {}'.format(data.shape, type(data)))
    write_to_file('labels: {}'.format(labels.shape))


if __name__ == '__main__':

    # Set up argument parser
        
    parser = argparse.ArgumentParser(description='preprocessing cortical sheet maps for patching.')
    
    parser.add_argument('--config_path',
                        type=str,
                        default='./config/hparams_surf2mat.yml',
                        help='Path to YAML file containing parameter information.')
    parser.add_argument('--type',
                        default='train',
                        help='Preprocessing type: train, validation, test.')
    
    options = parser.parse_args()

    with open(options.config_path) as f:
        config = yaml.safe_load(f)

    # Call training
    main(config, options)