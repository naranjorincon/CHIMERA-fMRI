

import pandas as pd
import nibabel as nb
import numpy as np
import yaml
import os
import argparse
# import sys

def main(config):
    if "/Users/snaranjo" in os.getcwd():
        local_pc_flag = "/Users/snaranjo/Desktop/neurotranslate/mount_point"
    else:
        local_pc_flag=""

    print('#'*30)
    print('Starting: Preprocessing Script')
    print('#'*30)
    root=f"{local_pc_flag}/ceph/chpc/shared/janine_bijsterbosch_group/naranjorincon_scratch/NeuroTranslate" #format is sub-***

    ico = config['resolution']['ico']
    all_sub_ico = config['resolution']['all_sub_ico'] #[1, 2, 3]
    dataset = config['data']['dataset']
    all_maps = config['data']['all_maps'] #['ICA', 'PFM']
    all_maps_dims = config['data']['all_maps_dims'] #[15, 14]
    overwrite_sample_split = config['data']['overwrite_sample_split']
    sub_ids_path = config['data']['sub_ids_path']
    patch_indeces_path = config['data']['patch_indeces_path'].format(root) # f"{root}/CHIMERA-fMRI/patch_extraction"
    chosen_hemi = config['data']['hemisphere'] #1L or 1R or 2 for both
    for map_ii, mID in enumerate(all_maps): #0=ICA and same for all_maps_dims so leave as a single iix to match them
        for ico_ii in all_sub_ico:
            num_vertices = config['sub_ico_{}'.format(ico_ii)]['num_vertices']
            num_patches = config['sub_ico_{}'.format(ico_ii)]['num_patches']
            # path2map_data should already exist, this should be your ico-06 spheres, output from step_1*.py script
            path2map_data=f"{root}/brain_reps_datasets/{dataset}/{all_maps[map_ii]}_maps/{all_maps[map_ii]}_d{all_maps_dims[map_ii]}_ico06"
            output_path = f"{root}/brain_reps_datasets/{dataset}/{all_maps[map_ii]}_maps/{all_maps[map_ii]}_d{all_maps_dims[map_ii]}_ico0{ico_ii}"
            if not os.path.isdir(output_path):
                os.makedirs(output_path) #goes to the same location as ico06 but saves each ico as separate folder with ico01..N format
                
            train_validation_test_path = config['data']['train_validation_test_path'].format(root, dataset) #f'{root}/CHIMERA-fMRI/utils/subj_ids/{dataset}'
            if not os.path.isdir(train_validation_test_path):
                os.makedirs(train_validation_test_path)

            if not os.path.isfile(f"{train_validation_test_path}/ABCD_train_val_test_split.csv") or overwrite_sample_split is True:
                #path to directoriy with singletons, siblings, and twins; Actually, by splitting by site we should circumvent this!!
                # need to be thorough and actually check though. For now, easy fix.
                ABCD_sites = pd.read_csv(f"{root}/CHIMERA-fMRI/utils/subj_ids/ABCD_full_ID_site.csv")["site"].to_numpy()
                ABCD_subID = pd.read_csv(f"{root}/CHIMERA-fMRI/utils/subj_ids/ABCD_full_ID_site.csv")["Session"].to_list()
                print(f"Unique sites: {np.unique(ABCD_sites)}.\nLen of sites{len(ABCD_sites)} should match len of subIDs{len(ABCD_subID)}")
                #this list i am using for subIDs is old so need to remove the NDA** part to compare with my other list.
                print(f"Fixing from this {ABCD_subID[0]} to this {ABCD_subID[0][7:]}!!")#needs to be 7, unfortunately for now it has to be a list comprehension
                ABCD_subID = np.asarray([iix[7:] for iix in ABCD_subID]) #verified it works! nice

                '''now, cross reference this with 3001 subject list we will be using. 
                Get subjects who are in both lists so that all subjects have site info. 
                Ideally, the list is full overlap and we have site infor for all 3001 subjects'''
                #verified!
                wapiaw26_subject_list = pd.read_csv(sub_ids_path, header=None)[0].to_list()
                wapiaw26_subject_list = np.asarray([iix[4:] for iix in wapiaw26_subject_list]) #has to be 4 to reomve sub-

                #now check overlap and make a mask, the ABCD_subID is longer list so we will use that.
                check_overlap_between_lists = np.isin(ABCD_subID, wapiaw26_subject_list) #how many of the 8k are in the 3050 list. Ideally all are true. If not, only get subjects that are true, i.e. in the 3050 AND we have site information
                # output list will be same len and ABCD_subID but boolean. Use as mask on ABCD_subID.
                subjects_with_site_info = ABCD_subID[check_overlap_between_lists] #for now, looks like overlap is only 2352/3050 SAD
                print(f"Overlap between site info and WAPIAW26 subject list is: {len(subjects_with_site_info)}/{len(wapiaw26_subject_list)}")

                #now that we have overlap, get full subject list of all resutling subjects and thier correspoinding site information
                corresponding_sites_for_overlap_subjects = ABCD_sites[check_overlap_between_lists]
                print(f"Unique sites are: {np.unique(corresponding_sites_for_overlap_subjects)}. See if we still have all sites of some are missing.")
                # create CSV from that and save

                df_subID_sites=pd.DataFrame({
                    "subID": subjects_with_site_info,
                    "site": corresponding_sites_for_overlap_subjects
                })
                df_subID_sites.to_csv(f"{train_validation_test_path}/ABCD_subsample_with_siteIDs.csv")

                define_validation = [15, 5, 7]
                define_test = [18, 8, 12]
                site_nums = np.arange(1,21) #1-20
                mask = np.isin(site_nums, [define_validation, define_test])
                define_train = site_nums[~mask] # 1-20 not in either validation or test
                train_num = df_subID_sites[df_subID_sites['site'].isin(define_train)]["subID"].to_list()
                val_num = df_subID_sites[df_subID_sites['site'].isin(define_validation)]["subID"].to_list()
                test_num = df_subID_sites[df_subID_sites['site'].isin(define_test)]["subID"].to_list()
                print(f"TRAIN: {len(train_num)} \nVALIDATION: {len(val_num)} \nTEST:{len(test_num)}")
                if overwrite_sample_split is True:
                    df_subID_sites_final_split=pd.DataFrame({
                        "index": np.arange(len(df_subID_sites)),
                        "subID": train_num+val_num+test_num,
                        "sample_split": ["train"]*len(train_num) + ["validation"]*len(val_num) + ["test"]*len(test_num)
                    })
                    df_subID_sites_final_split.to_csv(f"{train_validation_test_path}/ABCD_train_val_test_split.csv", index=None)
            else:
                df_subID_sites_final_split = pd.read_csv(f"{train_validation_test_path}/ABCD_train_val_test_split.csv")

            # %%
            ids = df_subID_sites_final_split["subID"].tolist()
            print(f"Number of subjects for prep is {len(ids)}")

            print('#'*30)
            print(f'Topography Maps: Preprocessing for Dataset: {dataset}')
            print('#'*30)

            if chosen_hemi == '2LR':
                print("Not ready yet. Needs fixing. Will raise error")
                raise ValueError('Error from chosen hemisphere.')
            elif chosen_hemi == '1L':
                print('LEFT hemisphere was chosen.')
                hemisphere_chosen='L'
            elif chosen_hemi == '1R':
                print('RIGHT hemisphere was chosen.')
                hemisphere_chosen='R'

            data = [] 
            sub_skipped =[]
            for i, id in enumerate(ids):
                filename=f'{path2map_data}/resamp_sub-{id}.{hemisphere_chosen}.shape.gii'
                if not os.path.isfile(filename):
                    print(f"sub {id}({i}) does not have mesh file.")
                    sub_skipped.append(i)
                    get_mesh_data=np.zeros((all_maps_dims[map_ii],40962)) #all will be dim x 40962 cause ico06
                else:
                    get_mesh_data=nb.load(filename).agg_data()

                # save data
                data.append(np.array(get_mesh_data))
                if i%250==0: #every 250 subjects
                    print(f'Done with loading mesh for: {i}')

            data=np.asarray(data)
            if len(sub_skipped) > 0:
                subID_df = df_subID_sites_final_split["subID"]
                subiix = np.asarray(ids)
                print(f"Count is {len(sub_skipped)} subjects to skip cause no data.")
                data = np.delete(data, sub_skipped, axis=0) #subjects with no mesh data
                ids = np.delete(subiix, sub_skipped, axis=0) #remove ids that have no data amke new ids
                subjects_skipped_topomap = pd.DataFrame({"subIDs_remove": subID_df.iloc[sub_skipped]})
                subjects_skipped_topomap.to_csv(f"{train_validation_test_path}/subjects_notopomap.csv")

            indices_mesh_triangles=pd.read_csv(f'{patch_indeces_path}/triangle_indices_ico_{ico}_sub_ico_{ico_ii}.csv')
            num_subjects, num_channels = data.shape[0], data.shape[1]
            if chosen_hemi == '2LR':
                print("Not ready yet.")
            else:
                print('\nBecause one hemisphere chosen, data is num_subj C P V')
                data_ico_lowres = np.zeros((num_channels, num_patches, num_vertices))
                print(f'ICO-{ico_ii} data shape: {data_ico_lowres.shape}')
                for i, id in enumerate(ids): # subjects?
                    if i%250==0:
                        print(f'Preping patches for sub: {i}')

                    for j in range(num_patches): # for each columns
                        indices_to_extract = indices_mesh_triangles[str(j)].to_numpy()
                        data_ico_lowres[:,j,:] = data[i][:,indices_to_extract] #will be subXmapsX320X153 for ico2

                    #save that subject
                    np.save(f"{output_path}/resamp_sub-{id}_{hemisphere_chosen}.npy",data_ico_lowres)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='preprocessing cortical sheet maps for patching.')
    parser.add_argument('config',
                        type=str,
                        default='',
                        help='Path to YAML file containing parameter information.')

    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Call training
    main(config)

    