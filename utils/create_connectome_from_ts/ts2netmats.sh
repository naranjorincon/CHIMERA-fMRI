#!/bin/bash
#SBATCH -J ts2netmats
#SBATCH -o /ceph/chpc/shared/janine_bijsterbosch_group/naranjorincon_scratch/NeuroTranslate/CHIMERA-fMRI/batch/ts2netmats.out%j
#SBATCH -e /ceph/chpc/shared/janine_bijsterbosch_group/naranjorincon_scratch/NeuroTranslate/CHIMERA-fMRI/batch/ts2netmats.err%j
#SBATCH --partition=tier2_cpu 
#SBATCH --account=janine_bijsterbosch 
#SBATCH --mem=40GB #30GB is enough for schaefer100,300 but glasser needs more apprx: 40GB
#SBATCH -t 0-04:00:00 

module load fsl

script_path="/ceph/chpc/shared/janine_bijsterbosch_group/naranjorincon_scratch/NeuroTranslate/ABCD_NetMats/utils/ts2netmats.py"
subjects_list_path="/ceph/chpc/shared/janine_bijsterbosch_group/WAPIAW_2026/qc/individual_subjects/individual_subject_list_highNSI_3001.txt"

parcellation_type="schaefer_d200" #schaefer_d100, schaefer_d200, schaefer_d300, glasser_d360

dir_path="/ceph/chpc/shared/janine_bijsterbosch_group/naranjorincon_scratch/NeuroTranslate/ABCD_NetMats/ABCDv6/${parcellation_type}/transpose"
Fnetmats_output="/ceph/chpc/shared/janine_bijsterbosch_group/naranjorincon_scratch/NeuroTranslate/ABCD_NetMats/ABCDv6/${parcellation_type}/netmats"
Pnetmats_output="/ceph/chpc/shared/janine_bijsterbosch_group/naranjorincon_scratch/NeuroTranslate/ABCD_NetMats/ABCDv6/${parcellation_type}/partial_netmats"
mkdir -p ${Fnetmats_output}
mkdir -p ${Pnetmats_output}

fslipython ${script_path} ${dir_path} ${Fnetmats_output} ${Pnetmats_output} ${subjects_list_path}

chmod -R 771 /ceph/chpc/shared/janine_bijsterbosch_group/naranjorincon_scratch/NeuroTranslate/ABCD_NetMats/ABCDv6

#Visualize some people to qa check that this went well
source activate neurotranslate
script_path_python="/ceph/chpc/shared/janine_bijsterbosch_group/naranjorincon_scratch/NeuroTranslate/CHIMERA-fMRI/utils/create_connectome_from_ts"
python ${script_path_python}/qa_netmats.py

#remove unused parts, the untransposed og and the ptseries, just keep the output
# rm_old_tmp_data="/ceph/chpc/shared/janine_bijsterbosch_group/naranjorincon_scratch/NeuroTranslate/ABCD_NetMats/ABCDv6/${parcellation_type}/untranspose/*.*"
# rm -rf ${rm_old_tmp_data}