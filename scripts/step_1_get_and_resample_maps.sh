#!/bin/bash
#SBATCH -J resamping_meshes
#SBATCH -o /ceph/chpc/shared/janine_bijsterbosch_group/naranjorincon_scratch/NeuroTranslate/mmfMRI/batch/resamping_meshes.out%j
#SBATCH -e /ceph/chpc/shared/janine_bijsterbosch_group/naranjorincon_scratch/NeuroTranslate/mmfMRI/batch/resamping_meshes.err%j
#SBATCH --partition=tier2_cpu
#SBATCH --account=janine_bijsterbosch 
#SBATCH --mem-per-cpu 5G
#SBATCH --cpus-per-task 2
#SBATCH -t 0-06:00:00 # it really takes like 12h


# Adapted 05.29.2025 by Samuel Naranjo Rincon - orginally made for HCP but adapted to be for either HCP-YA or ABCD
#
# to use the transformer pipeline as in Dahan et al 2021 (https://arxiv.org/abs/2203.16414), I need to first resample our brain_maps 
# into a tessellated sphere with batches of equal size. Our brain maps are contained within the cifti file dtseries.nii
# which is a 'dense time series' cifti file - don't be confused by the nii, its NOT a volume file in the way regular nii or nii.gz
# files are. This script is meant ot be used to convert brain maps of ICA contained in the brain_representation folder
# into left and right spheres that represent left and right cortex, respectively. 
# info can be found in:
# 1. https://manpages.debian.org/testing/connectome-workbench/wb_command.1.en.html#Maps
# 2. https://www.humanconnectome.org/software/workbench-command/-surface-create-sphere
# 3. https://www.humanconnectome.org/software/workbench-command/-metric-resample
#

module load workbench  # /1.5.0, we don't have 1.5.0 anymore I think we now have the more updated one. Lets try and use the updated one to see if it still works.

dataset="ABCDv6" #ABCD or HCPYA or HCPYA_ABCDdr

# step 1: make a folder for where we will put our brain data
scratch_dir="/ceph/chpc/shared/janine_bijsterbosch_group/naranjorincon_scratch" #technically not scratch, but no need to change variable name
surface_root="${scratch_dir}/NeuroTranslate/brain_reps_datasets/${dataset}"
mkdir -p "${surface_root}" #-p is option to make dir and anything along the way if it don't exist yet
# step 2: extract the L and R cortex data from the dense time series from each subj and move to the dir we made
if [ "${dataset}" == "HCPYA" ]; then
    mapdata="/ceph/chpc/shared/janine_bijsterbosch_group/tyoeasley/brain_representations/ICA_reps/3T_HCP1200_MSMAll_d25_ts2_Z/component_maps"

    # comments to know what's happening
    echo "getting data from $mapdata and transferring to $surface_root"
    cd "$mapdata"
    count=0
    for file in *.dtseries*; do
        count=$[count +1 ];
        echo "Inside subject: ${count}"
        echo "extracting cortex from: $file"
        id_num=("${file[@]%.*.*}")

        wb_command -cifti-separate "$mapdata/$file" COLUMN -metric CORTEX_LEFT "$surface_root/${id_num}_subj_L_cortex.shape.gii"; 
        wb_command -cifti-separate "$mapdata/$file" COLUMN -metric CORTEX_RIGHT "$surface_root/${id_num}_subj_R_cortex.shape.gii"; 

    done

    # step 3: go back to the surface folder where the data was extracted to and create a sphere to project the cortex
    # data onto
    cd "${surface_root}"

    # now, lets make a sphere for each subject with 32492 verteces. Once we have this sphere, its easier to down sample or upsample
    # as is the case in the Dahan et al 2021 pipeline
    if test -e "./naranjo_ico.L.surf.gii"; then # if it DOES exist
        echo "Already made the ico6 template spheres"
    else
        wb_command -surface-create-sphere 32492 naranjo_ico.R.surf.gii
        wb_command -surface-flip-lr naranjo_ico.R.surf.gii naranjo_ico.L.surf.gii
        wb_command -set-structure naranjo_ico.R.surf.gii CORTEX_RIGHT
        wb_command -set-structure naranjo_ico.L.surf.gii CORTEX_LEFT
    fi

    # highly reccomend that you view 5 or 8 or 10 random subjects to make sure viz is good, you can do this 
    # with wb_view command on terminal and choose the surface file to project onto
    # and the data files to load in seperated by L and R hemispheres. I checked, and the 10 local subjects I have all ran correctly

    # now we have an R sphere and an L sphere of verteces num 32492, and we only need the one casue they all are in that surface space

    # step 4: map the brain_rep onto the 32k sphere and then upsample to the Dahan 2021 ico-6 sphere of 40962 verteces (ico-6)
    for file in *L_cortex.shape*; do
        id_num=("${file[@]%_*_*_*.*.*}")
        echo "Resampling from og sphere to ico-6: sub-${id_num}"

        wb_command -metric-resample "${id_num}_subj_L_cortex.shape.gii" naranjo_ico.L.surf.gii "${template_ico6_spehre_path}/ico-${ico_res}.L.surf.gii" BARYCENTRIC "resamp_${id_num}.L.shape.gii"
        wb_command -metric-resample "${id_num}_subj_R_cortex.shape.gii" naranjo_ico.R.surf.gii "${template_ico6_spehre_path}/ico-${ico_res}.R.surf.gii" BARYCENTRIC "resamp_${id_num}.R.shape.gii"

    done
    mkdir ./original_space
    mv *cortex* ./original_space

elif [ "${dataset}" == "ABCD" ]; then
    mapdata="${scratch_dir}/NeuroTranslate/generate_ICA/ABCD_ICA/ICAd15/groupICA15.dr/dr_output/" #dr_{SUBJID}/surf.dscalar.nii

    # comments to know what's happening
    echo "getting data from ${mapdata} and transferring to ${surface_root}"
    cd "${mapdata}"
    # for all subjects makes their L and R data
    count=0
    for dd in dr*; do
        count=$[count +1 ];
        echo "Inside subject: ${count}"
        
        # cd ${dd} # inside subject
        dr_file="surf.dscalar.nii"
        id_num=("${dd#*_}") # subject file to transform
        wb_command -cifti-separate "${dd}/${dr_file}" COLUMN -metric CORTEX_LEFT "${surface_root}/${id_num}_subj_L_cortex.shape.gii"; 
        wb_command -cifti-separate "${dd}/${dr_file}" COLUMN -metric CORTEX_RIGHT "${surface_root}/${id_num}_subj_R_cortex.shape.gii"; 

    done

    # now we reformat from that space to ico6
    cd "${surface_root}"
    if test -e "./naranjo_ico.L.surf.gii"; then # if it DOES exist
        echo "Already made the ico6 template spheres"
    else
        wb_command -surface-create-sphere 32492 naranjo_ico.R.surf.gii
        wb_command -surface-flip-lr naranjo_ico.R.surf.gii naranjo_ico.L.surf.gii
        wb_command -set-structure naranjo_ico.R.surf.gii CORTEX_RIGHT
        wb_command -set-structure naranjo_ico.L.surf.gii CORTEX_LEFT
    fi

    # step 4: map the brain_rep onto the 32k sphere and then upsample to the Dahan 2021 ico-6 sphere of 40962 verteces (ico-6)
    for file in *L_cortex.shape*; do
        id_num="${file%%_*}"
        echo "Resampling from og sphere to ico-6: sub-${id_num}"

        wb_command -metric-resample "${id_num}_subj_L_cortex.shape.gii" naranjo_ico.L.surf.gii "${template_ico6_spehre_path}/ico-${ico_res}.L.surf.gii" BARYCENTRIC "resamp_${id_num}.L.shape.gii"
        wb_command -metric-resample "${id_num}_subj_R_cortex.shape.gii" naranjo_ico.R.surf.gii "${template_ico6_spehre_path}/ico-${ico_res}.R.surf.gii" BARYCENTRIC "resamp_${id_num}.R.shape.gii"

    done
    mkdir ./original_space
    mv *cortex* ./original_space

elif [ "${dataset}" == "ABCDv6" ]; then
    #init
    topography_type="PFM"
    subjID_fpath="/ceph/chpc/shared/janine_bijsterbosch_group/WAPIAW_2026/qc/individual_subjects/individual_subject_list.txt"
    subjID_list=$(cat ${subjID_fpath}) # to test things out use head instead of cat
    ico_res="6"

    # now we reformat from that space to ico6
    template_ico6_spehre_path="${scratch_dir}/NeuroTranslate/mmfMRI/surfaces"
    if test -e "${template_ico6_spehre_path}/naranjo_ico.L.surf.gii"; then # if it DOES exist
        echo "Already made the ico6 template spheres"
    else #if it does not then make one
        wb_command -surface-create-sphere 32492 naranjo_ico.R.surf.gii
        wb_command -surface-flip-lr naranjo_ico.R.surf.gii naranjo_ico.L.surf.gii
        wb_command -set-structure naranjo_ico.L.surf.gii CORTEX_LEFT
        wb_command -set-structure naranjo_ico.R.surf.gii CORTEX_RIGHT
    fi

    echo "+++++++++++++++++++++++ CHOSE ${topography_type}! +++++++++++++++++++++++"
    if [[ ${topography_type} == "ICA" ]]; then
        topo_dim="15"
        map_dimension_icores="${topography_type}_d${topo_dim}_ico06"
        output_path_resampled_spheres="${surface_root}/${topography_type}_maps/${map_dimension_icores}"
        mkdir -p "${output_path_resampled_spheres}"
        mapdata="/ceph/chpc/shared/janine_bijsterbosch_group/WAPIAW_2026/ICA/ABCD_ICAd${topo_dim}_4mm/ABCD_ICAd${topo_dim}_ses-all_C.dr/dr_output" #path to surface data
        find ${mapdata} -type f -name "surf.dscalar.nii" > "${output_path_resampled_spheres}/tmp_subject_data_file.txt" #create txt file with path to each subject's surface scalar data
    elif [[ ${topography_type} == "PFM" ]]; then
        topo_dim="14"
        map_dimension_icores="${topography_type}_d${topo_dim}_ico06"
        output_path_resampled_spheres="${surface_root}/${topography_type}_maps/${map_dimension_icores}"
        mkdir -p "${output_path_resampled_spheres}"
        mapdata="/ceph/chpc/shared/janine_bijsterbosch_group/WAPIAW_2026/profumo/individual_subject_maps/"
        find ${mapdata} -type f -name "*.dscalar.nii" > "${output_path_resampled_spheres}/tmp_subject_data_file.txt"
    fi
    
    echo "getting data from ${mapdata} and transferring to ${output_path_resampled_spheres}"
    #Subject list and tmp_subject_data_file.txt correspond in order 1-to-1 so the following is possible
    sub_surf_txt="${output_path_resampled_spheres}/tmp_subject_data_file.txt"
    # for all subjects makes their L and R data
    count=0
    for subjID in ${subjID_list}; do
        count=$[count +1 ];
        if ((count % 500 == 0)); then
            echo "Inside subject: ${count}"
        fi

        if [ -f "${output_path_resampled_spheres}/original_${subjID}_subj_R_cortex.shape.gii" ]; then #right hemi cause done second, so if they ahve that they have the left one
            echo "Original sphere file already exists for ${subjID}. Skipping."
            continue
        fi 
        specific_input_file=$(sed ''${count}'q;d' "${sub_surf_txt}")
        wb_command -cifti-separate "${specific_input_file}" COLUMN -metric CORTEX_LEFT "${output_path_resampled_spheres}/original_${subjID}_L_cortex.shape.gii"; 
        wb_command -cifti-separate "${specific_input_file}" COLUMN -metric CORTEX_RIGHT "${output_path_resampled_spheres}/original_${subjID}_R_cortex.shape.gii"; 

        if [ -f "${output_path_resampled_spheres}/resamp_${subjID}.R.shape.gii" ]; then #right hemi cause done second, so if they ahve that they have the left one
            echo "Resampled sphere file already exists for ${subjID}. Skipping."
            continue
        fi 
        wb_command -metric-resample "${output_path_resampled_spheres}/original_${subjID}_L_cortex.shape.gii"  ${template_ico6_spehre_path}/naranjo_ico.L.surf.gii "${template_ico6_spehre_path}/ico-${ico_res}.L.surf.gii" BARYCENTRIC "${output_path_resampled_spheres}/resamp_${subjID}.L.shape.gii"
        wb_command -metric-resample "${output_path_resampled_spheres}/original_${subjID}_R_cortex.shape.gii"  ${template_ico6_spehre_path}/naranjo_ico.R.surf.gii "${template_ico6_spehre_path}/ico-${ico_res}.R.surf.gii" BARYCENTRIC "${output_path_resampled_spheres}/resamp_${subjID}.R.shape.gii"

    done

    #to not over load and have many copies per subject, lets reset and remove the original copies s.t. we only have the ico-version we created
    rm -r "${output_path_resampled_spheres}/original*"
    chmod 751 -R "${output_path_resampled_spheres}"

elif [ "${dataset}" == "HCPYA_ABCDdr" ]; then
    mapdata="${scratch_dir}/NeuroTranslate/generate_ICA/HCPYA_ICA/ICAd15_dr/dr_output/" #dr_{SUBJID}/surf.dscalar.nii

    # comments to know what's happening
    echo "getting data from $mapdata and transferring to $surface_root for dataset $dataset"
    cd "$mapdata"
    # for all subjects makes their L and R data
    count=0
    for dd in dr*; do
        count=$[count +1 ];
        echo "Inside subject: ${count}"
        
        # cd ${dd} # inside subject
        dr_file="surf.dscalar.nii"
        id_num=("${dd#*_}") # subject file to transform
        if test -e "$surface_root/${id_num}_subj_R_cortex.shape.gii"; then #left is done first, so if they have R then both. if not, redo.
            echo "${id_num} already has extracted surface. L and R."
        else
            wb_command -cifti-separate "${dd}/${dr_file}" COLUMN -metric CORTEX_LEFT "$surface_root/${id_num}_subj_L_cortex.shape.gii"; 
            wb_command -cifti-separate "${dd}/${dr_file}" COLUMN -metric CORTEX_RIGHT "$surface_root/${id_num}_subj_R_cortex.shape.gii"; 
        fi

    done

    # now we reformat from that space to ico6
    cd "$surface_root"
    if test -e "./naranjo_ico.L.surf.gii"; then # if it DOES exist
        echo "Already made the ico6 template spheres"
    else
        wb_command -surface-create-sphere 32492 naranjo_ico.R.surf.gii
        wb_command -surface-flip-lr naranjo_ico.R.surf.gii naranjo_ico.L.surf.gii
        wb_command -set-structure naranjo_ico.R.surf.gii CORTEX_RIGHT
        wb_command -set-structure naranjo_ico.L.surf.gii CORTEX_LEFT
    fi

    # step 4: map the brain_rep onto the 32k sphere and then upsample to the Dahan 2021 ico-6 sphere of 40962 verteces (ico-6)
    for file in *L_cortex.shape*; do
        id_num="${file%%_*}"
        echo "Resampling from og sphere to ico-6: sub-${id_num}"

        if test -e "./resamp_${id_num}.R.shape.gii"; then
            echo "${id_num} already has resampled spheres."       
        else
            wb_command -metric-resample "${id_num}_subj_L_cortex.shape.gii" naranjo_ico.L.surf.gii "${template_ico6_spehre_path}/ico-${ico_res}.L.surf.gii" BARYCENTRIC "resamp_${id_num}.L.shape.gii"
            wb_command -metric-resample "${id_num}_subj_R_cortex.shape.gii" naranjo_ico.R.surf.gii "${template_ico6_spehre_path}/ico-${ico_res}.R.surf.gii" BARYCENTRIC "resamp_${id_num}.R.shape.gii"
        fi

    done

    mkdir ./original_space
    mv *cortex* ./original_space    

elif [ "${dataset}" == "infomap_prior_ABCDdr" ]; then
    mapdata="/ceph/chpc/shared/janine_bijsterbosch_group/naranjorincon_scratch/NeuroTranslate/brain_reps_datasets/infomap_prior_ABCDdr/dr_infomap_priors/ABCD_infomap20_no_smooth/spatial" #path to surface data
    map_dimension_icores="infomap_d20_ico06"
    output_path_resampled_spheres="${surface_root}/infomap_dr_maps/${map_dimension_icores}"
    mkdir -p "${output_path_resampled_spheres}"

    # comments to know what's happening
    echo "getting data from ${mapdata} and transferring to ${surface_root}"
    cd "${mapdata}"
    # for all subjects makes their L and R data
    count=0
    for dd in dr*; do    
        count=$[count +1 ];
        echo "Inside subject: ${count}"
        dr_file="surf.dscalar.nii"
        id_num=("${dd#*_}") # subject file to transform, come with sub-XXX

        #check if already ran
        FILE_left="${output_path_resampled_spheres}/original_${id_num}_subj_L_cortex.shape.gii"
        FILE_right="${output_path_resampled_spheres}/original_${id_num}_subj_R_cortex.shape.gii"
        if [ -f "${FILE_left}" ] && [ -f "${FILE_right}" ]; then
            echo "${id_num} exists."
            continue
        fi

        wb_command -cifti-separate "${dd}/${dr_file}" COLUMN -metric CORTEX_LEFT "${output_path_resampled_spheres}/original_${id_num}_subj_L_cortex.shape.gii"; 
        wb_command -cifti-separate "${dd}/${dr_file}" COLUMN -metric CORTEX_RIGHT "${output_path_resampled_spheres}/original_${id_num}_subj_R_cortex.shape.gii"; 
    done

    # now we reformat from that space to ico6
    template_ico6_spehre_path="${scratch_dir}/NeuroTranslate/mmfMRI/surfaces"
    if test -e "${template_ico6_spehre_path}/naranjo_ico.L.surf.gii"; then # if it DOES exist
        echo "Already made the ico6 template spheres"
    else #if it does not then make one
        wb_command -surface-create-sphere 32492 ${template_ico6_spehre_path}/naranjo_ico.R.surf.gii
        wb_command -surface-flip-lr ${template_ico6_spehre_path}/naranjo_ico.R.surf.gii ${template_ico6_spehre_path}/naranjo_ico.L.surf.gii
        wb_command -set-structure ${template_ico6_spehre_path}/naranjo_ico.L.surf.gii CORTEX_LEFT
        wb_command -set-structure ${template_ico6_spehre_path}/naranjo_ico.R.surf.gii CORTEX_RIGHT
    fi

    cd "${output_path_resampled_spheres}"
    # step 4: map the brain_rep onto the 32k sphere and then upsample to the Dahan 2021 ico-6 sphere of 40962 verteces (ico-6)
    ico_res="6"
    for file in *L_cortex.shape*; do #for each left file, there is a right one so this should be good enough
        id_num=${file#*sub-}; id_num=${id_num%%_subj*};
        echo "Resampling from og sphere to ico-${ico_res}: sub-${id_num}"

        FILE_left="${output_path_resampled_spheres}/resamp_sub-${id_num}.L.shape.gii"
        FILE_right="${output_path_resampled_spheres}/resamp_sub-${id_num}.R.shape.gii"
        if [ -f "${FILE_left}" ] && [ -f "${FILE_right}" ]; then
            echo "${id_num} exists."
            continue
        fi

        wb_command -metric-resample "${output_path_resampled_spheres}/original_sub-${id_num}_subj_L_cortex.shape.gii" "${template_ico6_spehre_path}/naranjo_ico.L.surf.gii" "${template_ico6_spehre_path}/ico-${ico_res}.L.surf.gii" BARYCENTRIC "${output_path_resampled_spheres}/resamp_sub-${id_num}.L.shape.gii"
        wb_command -metric-resample "${output_path_resampled_spheres}/original_sub-${id_num}_subj_R_cortex.shape.gii" "${template_ico6_spehre_path}/naranjo_ico.R.surf.gii" "${template_ico6_spehre_path}/ico-${ico_res}.R.surf.gii" BARYCENTRIC "${output_path_resampled_spheres}/resamp_sub-${id_num}.R.shape.gii"

    done

    #to not over load and have many copies per subject, lets reset and remove the original copies s.t. we only have the ico-version we created
    chmod 771 ${output_path_resampled_spheres}
    rm -rf "${output_path_resampled_spheres}/original*" #not needed cann be recomputed if necessary
else
    echo "That dataset is invalid/does not exit"
    # unknown_dataset_glaf=T
fi

# mkdir ./original_space
# mv *cortex* ./original_space
chmod -R 771 "${surface_root}"
