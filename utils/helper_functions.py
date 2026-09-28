import torch
import numpy as np
import pandas as pd
import nibabel as nib
import os
import subprocess
import glob
import copy
# =========================================== WHAT WE ACTUALLY USE =========================================== #
def write_to_file(content, filepath="", also_print=True):
    with open(filepath, 'a') as file:
        file.write(str(content) + '\n')
    if also_print:
        print(content)
    # return

def resample_to_native(metric_in=None,
                       ico06_sphere=None,
                       subject_sphere=None, 
                       method="BARYCENTRIC",
                       metric_out=None,
    ):
    cmd = [
        "wb_command", "-metric-resample",
        metric_in, 
        ico06_sphere,
        subject_sphere,
        method,
        metric_out] #TODO "-metric", "CORTEX_LEFT"

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"wb command failed. See below:\n{result.stderr}")

    print(result.stdout)
    return metric_out

def ico_matrix_to_native_mesh(input_mat, tri_indices_ico6subico2_fpath=None,
                              ico06_sphere=None, subject_sphere=None,
                              out_fpath=None):
    if input_mat.ndim > 3:
        input_mat = input_mat.squeeze()

    batch_sz = 1
    num_channels, num_patches, num_vert = input_mat.shape
    print(f"C:{num_channels}, P:{num_patches} V:{num_vert}")

    main_root="/ceph/chpc/shared/janine_bijsterbosch_group/naranjorincon_scratch//NeuroTranslate/CHIMERA-fMRI"
    main_ico_path=f"{main_root}/patch_extraction"
    if tri_indices_ico6subico2_fpath is None:
        #give default and infer based on patches and vertices        
        if num_patches == 80:
            low_ico_res = 1
        elif num_patches == 320:
            low_ico_res = 2
        elif num_patches == 1280:
            low_ico_res = 3
        elif num_patches == 5120:
            low_ico_res = 4
        elif num_patches == 20480:
            low_ico_res = 5

        tri_indices_ico6subico2_fpath=f"{main_ico_path}/triangle_indices_ico_6_sub_ico_{low_ico_res}.csv"

    indices_mesh_triangles = pd.read_csv(tri_indices_ico6subico2_fpath)
    mesh_ico6 = np.zeros([batch_sz, num_channels, 40962]) # forced to be 40962 to make ico6 verteces

    for b in range(batch_sz):
        for i in range(num_channels):
            for j in range(num_patches):
                indices_to_insert = indices_mesh_triangles[str(j)].to_numpy()
                mesh_ico6[b, i, indices_to_insert] = input_mat[i, j, :] #orig is input_mat[b, i, j, :]

    out = nib.GiftiImage()
    for b in range(batch_sz):
        for i in range(num_channels):
            out.add_gifti_data_array(nib.gifti.GiftiDataArray(mesh_ico6[b, i, :].astype("float32")))

    #turn to native space
    if ico06_sphere is None:
        ico06_sphere = f"{main_root}/surfaces/ico-6.L.surf.gii"
        subject_sphere = f"{main_root}/surfaces/naranjo_ico.L.surf.gii"   

    ext = os.path.splitext(os.path.basename(out_fpath))[1]
    out_fpath = out_fpath.replace(ext, '.gii')
    out.to_filename(out_fpath)
    resample_to_native(
        metric_in=out_fpath, ico06_sphere=ico06_sphere,
        subject_sphere=subject_sphere, metric_out=out_fpath)
    
    return out_fpath

def make_nemat_allsubj(data, num_nodes):
    '''
    Takes a numpy array of size [num_subj, size_vectorized_netmat] and reshapes to [num_subj, num_nodes, num_nodes]
    '''
    out = np.zeros([data.shape[0], num_nodes, num_nodes]) # init with ones of same size as output
    for i in range(data.shape[0]): # for each sub
        out[i, :, :] = make_netmat(data[i], num_nodes)
    return out

def make_netmat(data, num_nodes=100):
    '''
    Makes netmat from upper triangle in numpy
    '''
    out_mat_init = np.zeros((num_nodes,num_nodes))#.reshape(netmat_dim,netmat_dim)
    indeces = np.triu_indices_from(out_mat_init,k=1) # k=1 means no diagonal?
    out_mat_init[indeces] = data #presumably, data is from upper triangle from matlab but needs index lower trinagle to have good visuals not sure why...
    out_mat_init = out_mat_init + out_mat_init.T
    np.fill_diagonal(out_mat_init, 1)
    return out_mat_init

def fcn_extract_good_subjects(first_data, second_data, first_IDs_removal, second_IDs_removal):
    unifying_train_list_to_remove = (first_IDs_removal+second_IDs_removal).astype(bool)
    first_data = np.delete(first_data, unifying_train_list_to_remove, axis=0)
    second_data = np.delete(second_data, unifying_train_list_to_remove, axis=0)
    return first_data, second_data

def fcn_to_load_data_correctly(file_path):
    #infer type, surfaces are *.npy and connectomes are parcellationXparcellation format and *.csv
    extension_type = file_path.split('.')[-1]
    if extension_type == "npy":
        subject_file_data = np.load(file_path) #this way I can use arguments!! and for TOPOMAP data
        # subject_file_data = subject_file_data[0] #single channel
    elif extension_type == "csv":
        subject_file_data = pd.read_csv(file_path, header=None).to_numpy()
        subject_file_data = get_upper_tris(subject_file_data)
    return subject_file_data.squeeze()
    
def get_upper_tris(mat):
    trius = []
    if mat.ndim == 3: # then BatchxROWxCOL
        for i in range(mat.shape[0]):
            triu = mat[i, :, :][np.triu_indices_from(mat[i, :, :], k=1)]
            trius.append(triu)
    elif mat.ndim == 2: # so a single matrix
        triu = mat[np.triu_indices_from(mat, k=1)] #second one cause its a tuple 
        trius.append(triu)
    return np.array(trius)

def fcn_prepare_data(data, train_mu, train_sigma, prep_choice=None):
    if prep_choice is None or prep_choice == "raw":
        return data #do nothing and return it as is

    if prep_choice == "norm":
        data = (data - train_mu) / (train_sigma + 1e-99)
    elif prep_choice == "demean":
        data = (data - train_mu)
    return data

def fisher_z_transform(correlation_values):
    """
    Apply Fisher Z-transform to correlation values.
    
    Args:
    - Correlation_values vectorized upper/lower triangle
    
    Returns:
    - Fisher Z-transformed values.
    """

    z_output = 0.5 * np.log((1 + correlation_values) / (1 - correlation_values))
    return z_output
        
def fcn_get_clean_data(data=None, sample_IDs=None, indeces=None):
    #flatten data but keep subjects so make 2D
    get_original_shape = list(data.shape) #list so tuple becomes editable
    data = data.reshape(data.shape[0],-1) #flatten data
    index_range = np.ones(data.shape[0]) #1d of 1s

    no_nan_condition=False
    no_inf_condition=False #assume there are inf/nans
    #nans or infs
    data_nan = np.isnan(data); data_inf = np.isinf(data)
    if np.sum(data_nan.sum()) == 0:
        no_nan_condition = True

    if np.sum(data_inf.sum()) == 0:
        no_inf_condition = True

    #get subjects for outputting which subjects we keep
    if no_nan_condition is True and no_inf_condition is True:
        print("data is clean")
        subjects_to_remove = index_range*0 #all stay as ones #np.asarray([]) #empty list
        data = data.reshape(get_original_shape) #backt to tuple so reshape can happen
        return data, subjects_to_remove.astype(bool)
    else:
        print("Found NaNs or INFs. Identifying for cleaning,.")
        #infer if topomap or connectome
        find_inf_mask  = np.sum(data_inf,axis=-1).astype(bool)
        find_nan_mask  = np.sum(data_nan,axis=-1).astype(bool) #add backwards to get subject sums of NaNs
        subjects_to_remove = (find_inf_mask+find_nan_mask)
        subjects_to_remove = index_range*(1*subjects_to_remove) # subjects_to_remove #sample_IDs[subjects_to_remove]
        data = data.reshape(get_original_shape)
        return data, subjects_to_remove.astype(bool)

def fcn_load_clean_prep_data(brain_rep_files=None, 
                             train_val_test_csv=None):
    '''Function takes in a subject list, loads in those subjects and their 
    input/output data. Then preps them from raw input --> toch datasets for model.
    '''
    #params for function 
    brain_rep_files = np.asarray(brain_rep_files) #only do a subset
    if train_val_test_csv is None: #default
        train_val_test_csv = pd.read_csv("/ceph/chpc/shared/janine_bijsterbosch_group/naranjorincon_scratch/NeuroTranslate/CHIMERA-fMRI/utils/subj_ids/ABCDv6/ABCD_train_val_test_split.csv")

    get_train_iix = train_val_test_csv[train_val_test_csv['sample_split'] == "train"]["index"].to_numpy()#[:100]
    get_validation_iix = train_val_test_csv[train_val_test_csv['sample_split'] == "validation"]["index"].to_numpy()#[:100]
    get_test_iix = train_val_test_csv[train_val_test_csv['sample_split'] == "test"]["index"].to_numpy()
    print(f"Train Subjects:{len(get_train_iix)}\nValidation Subjects: {len(get_validation_iix)}\nTest Subjects: {len(get_test_iix)}")

    get_train_IDs = train_val_test_csv[train_val_test_csv['sample_split'] == "train"]["subID"].to_numpy()#[:100]
    get_validation_IDs = train_val_test_csv[train_val_test_csv['sample_split'] == "validation"]["subID"].to_numpy()#[:100]
    get_test_IDs = train_val_test_csv[train_val_test_csv['sample_split'] == "test"]["subID"].to_numpy()

    # split into sample types
    train_brain_rep_files      = brain_rep_files[get_train_iix]
    validation_brain_rep_files = brain_rep_files[get_validation_iix]
    test_brain_rep_files       = brain_rep_files[get_test_iix]
    
    data_train = []
    for file in train_brain_rep_files:
        data_train.append(fcn_to_load_data_correctly(file)) #loads into this list
    
    data_validation = []
    for file in validation_brain_rep_files:
        data_validation.append(fcn_to_load_data_correctly(file)) #loads into this list

    data_test = []
    for file in test_brain_rep_files:
        data_test.append(fcn_to_load_data_correctly(file)) #loads into this list

    # get which subjects to remove from train/val/test if any
    data_train = np.asarray(data_train)
    data_train, remove_train_IDs = fcn_get_clean_data(data_train, get_train_IDs)
    #do validation and test now
    data_validation = np.asarray(data_validation)
    data_validation, remove_validation_IDs = fcn_get_clean_data(data_validation, get_validation_IDs)
    
    data_test = np.asarray(data_test)
    data_test, remove_test_IDs = fcn_get_clean_data(data_test, get_test_IDs)

    print(f"SUBJECT TEST TO REMOVE BC PFM MISSING: {get_test_IDs[remove_test_IDs]}")
    
    return data_train, data_validation, data_test, remove_train_IDs, remove_validation_IDs, remove_test_IDs

def fcn_get_final_train_val_test_split(input_files, 
                                  output_files,
                                  prep_choice_input: str="norm", prep_choice_output:str="norm", train_batch_sz: int=32, train_val_test_csv=None):
    
    [data_train,
     data_validation,
     data_test, 
     remove_train_IDs, 
     remove_validation_IDs, 
     remove_test_IDs] = fcn_load_clean_prep_data(brain_rep_files=input_files, train_val_test_csv=train_val_test_csv)
    
    [data_train_label,
     data_validation_label,
     data_test_label,
     remove_train_IDs_label,
     remove_validation_IDs_label,
     remove_test_IDs_label] = fcn_load_clean_prep_data(brain_rep_files=output_files, train_val_test_csv=train_val_test_csv)
    
    #not feed this into the fcn for removing correct subjects
    data_train, data_train_label = fcn_extract_good_subjects(data_train, data_train_label, remove_train_IDs, remove_train_IDs_label)
    data_validation, data_validation_label = fcn_extract_good_subjects(data_validation, data_validation_label, remove_validation_IDs, remove_validation_IDs_label)
    data_test, data_test_label = fcn_extract_good_subjects(data_test, data_test_label, remove_test_IDs, remove_test_IDs_label)
    #normalize as needed or transform as required
    train_mu = np.mean(data_train,axis=0)
    train_sigma = np.std(data_train, axis=0)
    print(f"TRAIN MEAN SHAPE: {train_mu.shape}")
    print(f"TRAIN SIGMA SHAPE: {train_sigma.shape}")
    # main_root="/ceph/chpc/shared/janine_bijsterbosch_group/naranjorincon_scratch/NeuroTranslate/CHIMERA-fMRI"
    # topomap_or_connectome = "connectome"
    # if topomap_or_connectome == "connectome":
    #     # if data_train.shape[1] == 4950:
    #     #     parcel_size = 100
    #     # elif data_train.shape[1] == 44850:
    #     #     parcel_size = 300
    #     # elif data_train.shape[1] == 64620:
    #     #     parcel_size = 360

    #     # train_mu = make_netmat(train_mu, parcel_size)
    #     outpath=f"{main_root}/train_AVERAGE_connectome_{prep_choice_input}_{prep_choice_output}.npy"
    #     np.save(outpath,train_mu)
    # elif topomap_or_connectome == "topomap":
    #     tri_indices_ico6subico2_fpath=f"{main_root}/patch_extraction/triangle_indices_ico_6_sub_ico_2.csv"
    #     ico06_sphere=f"{main_root}/surfaces/ico-6.L.surf.gii"
    #     subject_sphere=f"{main_root}/surfaces/naranjo_ico.L.surf.gii" 
    #     outpath=f"{main_root}/test_surface_check_AVERAGE_{prep_choice_input}_{prep_choice_output}.npy"
    #     ico_matrix_to_native_mesh(train_mu, tri_indices_ico6subico2_fpath, ico06_sphere, subject_sphere, outpath)
            
    # prep for outputs
    train_mu_label = np.mean(data_train_label,axis=0)
    train_sigma_label = np.std(data_train_label, axis=0)
    print(f"TRAIN MEAN SHAPE: {train_mu_label.shape}")
    print(f"TRAIN SIGMA SHAPE: {train_sigma_label.shape}")
    print(f"PREPARING DATA. Choices are input:{prep_choice_input} and output:{prep_choice_output}")
    data_train = fcn_prepare_data(data_train, train_mu, train_sigma, prep_choice_input)
    data_validation = fcn_prepare_data(data_validation, train_mu, train_sigma, prep_choice_input)
    data_test = fcn_prepare_data(data_test, train_mu, train_sigma, prep_choice_input)
    data_train_label = fcn_prepare_data(data_train_label, train_mu_label, train_sigma_label, prep_choice_output)
    data_validation_label = fcn_prepare_data(data_validation_label, train_mu_label, train_sigma_label, prep_choice_output)
    data_test_label = fcn_prepare_data(data_test_label, train_mu_label, train_sigma_label, prep_choice_output)
    #for connectome, expects parcle_size by parcel_size connectome, so converting it as necessary
    ########## BELOW IS MEANT TO GO FROM VECTORIZED_UPPER TRIANGLE TO ORIGINAL CONNECTOME SHAPE, WAS SUPPOSED TO BE FOR BNT FRAMEWORK
    # if topomap_or_connectome == "connectome": #only needs to make input be the connectome, output should still be the upper triangle 
    #     data_train = make_nemat_allsubj(data_train, parcel_size)
    #     data_validation = make_nemat_allsubj(data_validation, parcel_size)
    #     data_test = make_nemat_allsubj(data_test, parcel_size)

    #     data_train_label = make_nemat_allsubj(data_train_label, parcel_size)
    #     data_validation_label = make_nemat_allsubj(data_validation_label, parcel_size)
    #     data_test_label = make_nemat_allsubj(data_test_label, parcel_size)
        
    #### MODEL DATALOADERS
    train_dataset = torch.utils.data.TensorDataset(torch.from_numpy(data_train).float(), torch.from_numpy(data_train_label).float())
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = train_batch_sz, shuffle=True, num_workers=10)
    val_dataset = torch.utils.data.TensorDataset(torch.from_numpy(data_validation).float(), torch.from_numpy(data_validation_label).float())
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size = train_batch_sz, shuffle=False, num_workers=10)
    test_dataset = torch.utils.data.TensorDataset(torch.from_numpy(data_test).float(), torch.from_numpy(data_test_label).float())
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = 1, shuffle=False, num_workers=10)

    return train_loader, val_loader, test_loader, train_mu_label 

def fcn_get_final_train_val_test_split_fusioncase(input_files,  
                                  prep_choice_input: str="norm", train_val_test_csv=None):
    
    [data_train,
     data_validation,
     data_test, 
     remove_train_IDs, 
     remove_validation_IDs, 
     remove_test_IDs] = fcn_load_clean_prep_data(brain_rep_files=input_files, train_val_test_csv=train_val_test_csv)
    
    if remove_train_IDs.sum() > 0:
        data_train = np.delete(data_train, remove_train_IDs, axis=0)
    if remove_validation_IDs.sum() > 0:
        data_validation = np.delete(data_validation, remove_validation_IDs, axis=0)
    if remove_test_IDs.sum() > 0:
        data_test = np.delete(data_test, remove_test_IDs, axis=0)
    
    #normalize as needed or transform as required
    train_mu = np.nanmean(data_train,axis=0)
    train_sigma = np.nanstd(data_train, axis=0)
    print(f"TRAIN MEAN SHAPE: {train_mu.shape}")
    print(f"TRAIN SIGMA SHAPE: {train_sigma.shape}")
    
    print(f"PREPARING DATA. Choices are input:{prep_choice_input}")
    data_train = fcn_prepare_data(data_train, train_mu, train_sigma, prep_choice_input)
    data_validation = fcn_prepare_data(data_validation, train_mu, train_sigma, prep_choice_input)
    data_test = fcn_prepare_data(data_test, train_mu, train_sigma, prep_choice_input)
        
    return data_train, data_validation, data_test, train_mu

def fcn_get_torch_loaders(input_files=None, output_files=None, 
                          train_batch_sz=None, prep_choice_input=None, 
                          prep_choice_output=None,
                          regular_reconstruction: bool=True, train_val_test_csv=None
    ):

    if regular_reconstruction is True:
        train_loader, val_loader, test_loader, train_mu = fcn_get_final_train_val_test_split(input_files, output_files, 
                                    prep_choice_input=prep_choice_input, prep_choice_output=prep_choice_output, train_batch_sz=train_batch_sz, train_val_test_csv=train_val_test_csv)

    else: #so not regular and using fusion insteed
        #here this should be a list of input files
        assert isinstance(input_files, list) and isinstance(prep_choice_input, list), "For fusion case, this should be a list so we can iterate though file type and save them all for PoE input model."

        all_train_data_list      = []
        all_validation_data_list = []
        all_test_data_list       = []
        train_mu                 = []
        for i, (curr_input_files) in enumerate(input_files):
            curr_prep_choice = prep_choice_input[i]
            # print(len(curr_input_files))
            data_train, data_validation, data_test, curr_train_mu = fcn_get_final_train_val_test_split_fusioncase(input_files=curr_input_files,
                                        prep_choice_input=curr_prep_choice, train_val_test_csv=train_val_test_csv)
            
            # store them here as list in order, its any order but for us it should be as ICA,PFM,sch100,sch200,sch300,glasser360
            all_train_data_list.append(torch.from_numpy(data_train).float())
            all_validation_data_list.append(torch.from_numpy(data_validation).float())
            all_test_data_list.append(torch.from_numpy(data_test).float())
            train_mu.append(curr_train_mu)

        train_ds       = torch.utils.data.TensorDataset(all_train_data_list[0], all_train_data_list[1], all_train_data_list[2], all_train_data_list[3], all_train_data_list[4], all_train_data_list[5])
        validation_ds  = torch.utils.data.TensorDataset(all_validation_data_list[0], all_validation_data_list[1], all_validation_data_list[2], all_validation_data_list[3], all_validation_data_list[4], all_validation_data_list[5])
        test_ds        = torch.utils.data.TensorDataset(all_test_data_list[0], all_test_data_list[1], all_test_data_list[2], all_test_data_list[3], all_test_data_list[4], all_test_data_list[5])

        train_loader        = torch.utils.data.DataLoader(train_ds, batch_size=train_batch_sz, shuffle=True,  drop_last=True)
        val_loader          = torch.utils.data.DataLoader(validation_ds, batch_size=train_batch_sz, shuffle=False, drop_last=False)
        test_loader         = torch.utils.data.DataLoader(test_ds, batch_size=1, shuffle=False, drop_last=False)

        #### MODEL DATALOADERS
        # train_loader = torch.utils.data.DataLoader(all_train_data_list, batch_size = train_batch_sz, shuffle=True, num_workers=10)
        # val_loader = torch.utils.data.DataLoader(all_validation_data_list, batch_size = train_batch_sz, shuffle=False, num_workers=10)
        # test_loader = torch.utils.data.DataLoader(all_test_data_list, batch_size = 1, shuffle=False, num_workers=10)
        
    return train_loader, val_loader, test_loader, train_mu

def fcn_get_file_lists_and_sort(main_brainrep_data_path_root: str="", dataset_choice: str="", 
                                data_type_input=None, data_type_output=None, 
                                input_dim=None, output_dim=None, icores=None, 
                                input_representation_type=None, output_representation_type=None,
                                get_sub_ids=None, operation: str='reconstruction', left_or_right: str="L",
                                df_with_sub_prefix=None):
    
    input_files_reference = []
    if input_representation_type == "topomap":
        input_files = glob.glob(f"{main_brainrep_data_path_root}/brain_reps_datasets/{dataset_choice}/{data_type_input}_maps/{data_type_input}_d{input_dim}_ico0{icores}/*sub*{left_or_right}*")
        [input_files_reference.append((input_files[i].split('/')[-1]).split('_')[1]) for i in range(len(input_files))]
        input_ids = np.asarray(input_files_reference)
    elif input_representation_type == "connectome":
        input_files = glob.glob(f"{main_brainrep_data_path_root}/ABCD_NetMats/{dataset_choice}/{data_type_input}_d{input_dim}/netmats/*sub*")
        [input_files_reference.append((input_files[i].split('/')[-1]).split('.')[0]) for i in range(len(input_files))]
        input_ids = np.asarray(input_files_reference)

    #only use input files found through glob that are part of our actual subject list, "get_sub_ids"
    found_files_vs_actual_input_subjects = np.isin(input_ids, get_sub_ids) #ideally equal i guess
    input_ids = input_ids[found_files_vs_actual_input_subjects]
    input_files = np.asarray(input_files)[found_files_vs_actual_input_subjects]

    # original list is for N subjects, but it can happen where input files are less than. So once all done, need to update this list of subjects with new N_1 < N and with the right train/val/test split
    order = dict(zip(df_with_sub_prefix["subID"], df_with_sub_prefix["index"])) #uses or CSV to keep order of files found by glob, VERY IMPORTANT
    if input_representation_type == "topomap":
        input_files = sorted(input_files,key=lambda x: order[os.path.basename(x).split(".")[0].split('_')[1]])
    elif input_representation_type == "connectome":
        input_files = sorted(input_files,key=lambda x: order[os.path.basename(x).split(".")[0]])

    if operation == "reconstruction":
        output_files = copy.deepcopy(input_files) #make perfect copy and also already an array and matches subjIDs and everything
    else:
        output_files_reference = []
        if output_representation_type == "topomap":
            output_files = glob.glob(f"{main_brainrep_data_path_root}/brain_reps_datasets/{dataset_choice}/{data_type_output}_maps/{data_type_output}_d{output_dim}_ico0{icores}/*sub*{left_or_right}*")
            [output_files_reference.append((output_files[i].split('/')[-1]).split('_')[1]) for i in range(len(output_files))]
            output_ids = np.asarray(output_files_reference)
        elif output_representation_type == "connectome":
            output_files = glob.glob(f"{main_brainrep_data_path_root}/ABCD_NetMats/{dataset_choice}/{data_type_output}_d{output_dim}/netmats/*sub*")
            [output_files_reference.append((output_files[i].split('/')[-1]).split('.')[0]) for i in range(len(output_files))]
            output_ids = np.asarray(output_files_reference)

        #make into array
        print(f"len of output files retrieved that match input: {len(output_files)}")
        output_files = np.asarray(output_files)

        #ensure output files match input files cause we might have more/less topomaps VS connectomes or of one connectome VS another connectome
        if len(input_files) != len(output_files): #different files, so picking only output files that also have an input
            who_in_output_has_input = np.isin(output_ids, input_ids)
            output_files = output_files[who_in_output_has_input] #update
        
        #then sort
        if output_representation_type == "topomap":
            output_files = sorted(output_files,key=lambda x: order[os.path.basename(x).split(".")[0].split('_')[1]])
        elif output_representation_type == "connectome":
            output_files = sorted(output_files,key=lambda x: order[os.path.basename(x).split(".")[0]])

    return input_files, output_files

def build_encoders(ICA_encoder, PFM_encoder, schaefer100_encoder, schaefer200_encoder, schaefer300_encoder, glasser360_encoder, device='cpu'):
    """
    Load pretrained SiT encoder and connectome encoders.
    """
    sit_encoders = {
        "ICA": ICA_encoder,
        "PFM": PFM_encoder
    }

    conn_encoders = {
        "schaefer100": schaefer100_encoder,
        "schaefer200": schaefer200_encoder,
        "schaefer300": schaefer300_encoder,
        "glasser360": glasser360_encoder
    }
    # conn_encoders["100"].load_state_dict(torch.load("conn_enc_100.pt", weights_only=True))
    # conn_encoders["200"].load_state_dict(torch.load("conn_enc_200.pt", weights_only=True))
    topo_networks = {"ICA": 15, "PFM": 14}
    conn_latents  = {"schaefer100": 256, "schaefer200": 256, "schaefer300": 256, "glasser360": 256}
    conn_n_tri    = {"schaefer100": int(0.5*100*(100-1)), "schaefer200": int(0.5*200*(200-1)),
                        "schaefer300": int(0.5*300*(300-1)), "glasser360": int(0.5*360*(360-1))
                        } #upper triangle elements are equal to parcellation N --> 0.5 * N*(N-1)

    return sit_encoders, topo_networks, conn_encoders, conn_latents, conn_n_tri

def batch_pearson(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    Mean Pearson correlation across subjects (batch dim).
    Both tensors are flattened per subject before correlation.
    Returns a scalar float.
    """
    # pred   = pred.detach().cpu().float()
    # target = target.detach().cpu().float()
    pred = pred.detach().cpu().float().numpy()
    target = target.detach().cpu().float().numpy()

    target = target.reshape(target.shape[0],-1)
    pred = pred.reshape(pred.shape[0],-1)

    B = pred.shape[0]
    # pred   = pred.view(B, -1).numpy()
    # target = target.view(B, -1).numpy()

    from scipy.stats import pearsonr
    corrs = []
    for i in range(B):
        r, _ = pearsonr(pred[i], target[i])
        if np.isfinite(r):
            corrs.append(r)
    return float(np.mean(corrs)) if corrs else float("nan")