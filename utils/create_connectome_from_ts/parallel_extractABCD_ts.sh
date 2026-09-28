subject_list="/ceph/chpc/shared/janine_bijsterbosch_group/WAPIAW_2026/qc/individual_subjects/individual_subject_list_highNSI_3001.txt"
# subject_list="/ceph/chpc/shared/janine_bijsterbosch_group/WAPIAW_2026/qc/individual_subjects/individual_subject_list_N100_SAMUEL_experiment.txt"
subjID_list=$(cat ${subject_list})

log_dir="/ceph/chpc/shared/janine_bijsterbosch_group/naranjorincon_scratch/NeuroTranslate/CHIMERA-fMRI/batch/create_connectome_from_ts"
script_dir="/ceph/chpc/shared/janine_bijsterbosch_group/naranjorincon_scratch/NeuroTranslate/CHIMERA-fMRI/utils/create_connectome_from_ts"
curr_script_file="${script_dir}/parallel_extract_ABCD_ts_template.sh"
mkdir -p "$log_dir" #"$job_dir"

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

for subject_ID in ${subjID_list};
do
    ## if subejct exists already skip, kind of like the overwrite flag but here skips the job so better/faster
    # EXPECTED_OUT="/ceph/chpc/shared/janine_bijsterbosch_group/CHIMERA-fMRI/btwn_method_comparisons/subject_results/cross_method_corr_${subject_ID}.csv"
    # if [[ -f "$EXPECTED_OUT" ]]; then
    #     echo "Skipping ${subject_ID} — output already exists"
    #     continue
    # fi
    
    echo "\
\
#!/bin/bash
#SBATCH -J ExtractABCD_Timeseries
#SBATCH -o ${log_dir}/parll_extractABCD_ts.o%j
#SBATCH -e ${log_dir}/parll_extractABCD_ts.e%j
#SBATCH --account=janine_bijsterbosch
#SBATCH --partition=tier2_cpu
#SBATCH --mem=7G
#SBATCH -t 0-0:10:00

source activate neurotranslate
echo "Subject: ${subject_ID}"

module load workbench
# need subj ID file
subjID_fpath=${subject_list}

mkdir -p ${dir_path}
echo "Extracting Timeseries For: ${subject_ID}"

wb_command -cifti-parcellate \
    /ceph/chpc/shared/janine_bijsterbosch_group/WAPIAW_2026/cortex_only_data/ses-all/4mm_smooth/${subject_ID}_cortex_only_demean_smooth_4mm.dtseries.nii \
    ${parcel_file} \
    COLUMN \
    ${dir_path}/${subject_ID}.ptseries.nii

wb_command -cifti-convert -to-text \
    ${dir_path}/${subject_ID}.ptseries.nii \
    ${dir_path}/untranspose_${subject_ID}.txt


\
" > "${curr_script_file}"  # Overwrite submission script
    # Make script executable
    chmod +x "${curr_script_file}" || { echo "Error changing the script permission!"; exit 1; }

    # Submit script
    sbatch "${curr_script_file}" || { echo "Error submitting jobs!"; exit 1; }
done