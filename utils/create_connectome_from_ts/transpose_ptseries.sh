#!/bin/bash
#SBATCH -J TransposeABCD_tseries
#SBATCH -o /ceph/chpc/shared/janine_bijsterbosch_group/naranjorincon_scratch/NeuroTranslate/CHIMERA-fMRI/batch/TransposeABCD_tseries.out%j
#SBATCH -e /ceph/chpc/shared/janine_bijsterbosch_group/naranjorincon_scratch/NeuroTranslate/CHIMERA-fMRI/batch/TransposeABCD_tseries.err%j
#SBATCH --partition=tier2_cpu 
#SBATCH --account=janine_bijsterbosch
#SBATCH -t 0-05:00:00 

source activate neurotranslate

parcellation_type="schaefer_d200" #schaefer_d100, schaefer_d200, schaefer_d300, glasser_d360

script_path="/ceph/chpc/shared/janine_bijsterbosch_group/naranjorincon_scratch/NeuroTranslate/ABCD_NetMats/utils/transpose_ptseries.py"
# subjects_list_path="/ceph/chpc/shared/janine_bijsterbosch_group/naranjorincon_scratch/wapiaw26/subject_list_only5.txt"
subjects_list_path="/ceph/chpc/shared/janine_bijsterbosch_group/WAPIAW_2026/qc/individual_subjects/individual_subject_list_highNSI_3001.txt"
untranspose_ts_path="/ceph/chpc/shared/janine_bijsterbosch_group/naranjorincon_scratch/NeuroTranslate/ABCD_NetMats/ABCDv6/${parcellation_type}/untranspose"
save_path="/ceph/chpc/shared/janine_bijsterbosch_group/naranjorincon_scratch/NeuroTranslate/ABCD_NetMats/ABCDv6/${parcellation_type}/transpose"
mkdir -p ${save_path}

python ${script_path} ${subjects_list_path} ${untranspose_ts_path} ${save_path}

chmod -R 771 /ceph/chpc/shared/janine_bijsterbosch_group/naranjorincon_scratch/NeuroTranslate/ABCD_NetMats/ABCDv6