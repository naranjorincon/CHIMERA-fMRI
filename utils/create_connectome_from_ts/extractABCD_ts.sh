#!/bin/bash
#SBATCH -J ExtractABCD_Timeseries
#SBATCH -o /ceph/chpc/shared/janine_bijsterbosch_group/naranjorincon_scratch/NeuroTranslate/CHIMERA-fMRI/batch/ExtractABCD_Timeseries_Schaefer300.out%j
#SBATCH -e /ceph/chpc/shared/janine_bijsterbosch_group/naranjorincon_scratch/NeuroTranslate/CHIMERA-fMRI/batch/ExtractABCD_Timeseries_Schaefer300.err%j
#SBATCH --partition=tier2_cpu 
#SBATCH --account=janine_bijsterbosch
#SBATCH -t 0-02:00:00 

module load workbench

# need subj ID file
subjID_fpath="/ceph/chpc/shared/janine_bijsterbosch_group/WAPIAW_2026/qc/individual_subjects/individual_subject_list_highNSI_3001.txt"
# subjID_fpath="/ceph/chpc/shared/janine_bijsterbosch_group/naranjorincon_scratch/wapiaw26/subject_list_only5.txt"

# Schaefer100
# parcel_file="/ceph/chpc/shared/janine_bijsterbosch_group/naranjorincon_scratch/NeuroTranslate/ABCD_NetMats/schaefer100/Schaefer2018_100Parcels_17Networks_order.dlabel.nii"
# dir_path="/ceph/chpc/shared/janine_bijsterbosch_group/naranjorincon_scratch/NeuroTranslate/ABCD_NetMats/ABCDv6/schaefer_d100/untranspose"

# Schaefer200
# parcel_file="/ceph/chpc/shared/janine_bijsterbosch_group/naranjorincon_scratch/NeuroTranslate/ABCD_NetMats/schaefer200/Schaefer2018_200Parcels_17Networks_order.dlabel.nii"
# dir_path="/ceph/chpc/shared/janine_bijsterbosch_group/naranjorincon_scratch/NeuroTranslate/ABCD_NetMats/ABCDv6/schaefer_d200/untranspose"

# Schaefer300
parcel_file="/ceph/chpc/shared/janine_bijsterbosch_group/naranjorincon_scratch/NeuroTranslate/ABCD_NetMats/schaefer300/Schaefer2018_300Parcels_17Networks_order.dlabel.nii"
dir_path="/ceph/chpc/shared/janine_bijsterbosch_group/naranjorincon_scratch/NeuroTranslate/ABCD_NetMats/ABCDv6/schaefer_d300/untranspose"

# # Glasser360
# parcel_file="/ceph/chpc/shared/janine_bijsterbosch_group/naranjorincon_scratch/NeuroTranslate/ABCD_NetMats/glasser360/Q1-Q6_RelatedParcellation210.CorticalAreas_dil_Colors_210P_Orig.32k_fs_LR.dlabel.nii"
# dir_path="/ceph/chpc/shared/janine_bijsterbosch_group/naranjorincon_scratch/NeuroTranslate/ABCD_NetMats/ABCDv6/glasser_d360/untranspose"

# loop
mkdir -p ${dir_path}
subjID_list=$(cat ${subjID_fpath})
for subjID in ${subjID_list};
do
    echo "Extracting Timeseries For: ${subjID}"

    wb_command -cifti-parcellate \
        /ceph/chpc/shared/janine_bijsterbosch_group/WAPIAW_2026/cortex_only_data/ses-all/4mm_smooth/${subjID}_cortex_only_demean_smooth_4mm.dtseries.nii \
        ${parcel_file} \
        COLUMN \
        ${dir_path}/${subjID}.ptseries.nii

    wb_command -cifti-convert -to-text \
        ${dir_path}/${subjID}.ptseries.nii \
        ${dir_path}/untranspose_${subjID}.txt

done #< "${subjID_fpath}"

chmod 771 ${dir_path}/*
