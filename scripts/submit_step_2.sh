#!/bin/bash
#SBATCH -J making_maps_netmats_for_translation
#SBATCH -o /ceph/chpc/shared/janine_bijsterbosch_group/naranjorincon_scratch/NeuroTranslate/CHIMERA-fMRI/batch/preprocessing/prep.out%j
#SBATCH -e /ceph/chpc/shared/janine_bijsterbosch_group/naranjorincon_scratch/NeuroTranslate/CHIMERA-fMRI/batch/preprocessing/prep.err%j
#SBATCH --partition=tier2_cpu
#SBATCH --account=janine_bijsterbosch 
#SBATCH --mem=12G
#SBATCH -t 0-04:00:00 #this 12G and 2h is based on my own trial for doing all prep on N=2310 subjects and for ICA,PFM for ico-01,2,3

source activate neurotranslate
root="/ceph/chpc/shared/janine_bijsterbosch_group/naranjorincon_scratch/NeuroTranslate/CHIMERA-fMRI"
script_path="${root}/scripts"
config_file_path="${root}/config"

#run code
python3 "${script_path}/step_2_prep_maps_different_ico.py" "${config_file_path}/preprocessing.yml"