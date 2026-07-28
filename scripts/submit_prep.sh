#!/bin/bash
#SBATCH -J prepSiT
#SBATCH -o /ceph/chpc/shared/janine_bijsterbosch_group/naranjorincon_scratch/NeuroTranslate/surf2netmat/batch/prepSiT.out%j
#SBATCH -e /ceph/chpc/shared/janine_bijsterbosch_group/naranjorincon_scratch/NeuroTranslate/surf2netmat/batch/prepSiT.err%j
#SBATCH --partition=tier2_cpu
#SBATCH --account=janine_bijsterbosch 
#SBATCH --mem-per-cpu 16G
#SBATCH --cpus-per-task 10
#SBATCH -t 0-06:00:00  # might depend on epoch, approx 50epoch = 24 hours

source activate neurotranslate
cd /ceph/chpc/shared/janine_bijsterbosch_group/naranjorincon_scratch/NeuroTranslate/surf2netmat/tools

# set location variables
yaml_loc="/ceph/chpc/shared/janine_bijsterbosch_group/naranjorincon_scratch/NeuroTranslate/surf2netmat/config/resampling_sphere_prep"

# Go to prep folder and for each settings file, run the prep for it
for file in $(find "$yaml_loc" -type f -name "ICAd15_*100.yml" -print); do
    echo Runnign This preprocessing settings file: "$file"
    python3 ./preprocessing.py "$file"
    
done

# bash chpc_resample_sphere_fs_space.sh # if you are running this, itll take like 7 hours for ABCD (N>8,000). but only about 1GB max.
