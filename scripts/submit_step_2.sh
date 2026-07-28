#!/bin/bash
#SBATCH -J making_maps_netmats_for_translation
#SBATCH -o /ceph/chpc/shared/janine_bijsterbosch_group/naranjorincon_scratch/NeuroTranslate/surf2netmat/batch/making_maps_netmats_prep.out%j
#SBATCH -e /ceph/chpc/shared/janine_bijsterbosch_group/naranjorincon_scratch/NeuroTranslate/surf2netmat/batch/making_maps_netmats_prep.err%j
#SBATCH --partition=tier2_cpu
#SBATCH --account=janine_bijsterbosch 
#SBATCH --mem-per-cpu 30G
#SBATCH --cpus-per-task 10
#SBATCH -t 0-01:00:00 # it really takes like 12h

source activate neurotranslate
root="/ceph/chpc/shared/janine_bijsterbosch_group/naranjorincon_scratch/NeuroTranslate/surf2netmat"
script_path="${root}/tools"
config_file_path="${root}/config/resampling_sphere_prep"

#run code
# python3 "${script_path}/step_2_prep_maps_and_netmats.py" --config_path "${config_file_path}/INFOMAPd20_netmats.yml"
python3 "${script_path}/step_2_prep_maps_and_netmats.py" --config_path "${config_file_path}/ICAd15_netmats.yml"