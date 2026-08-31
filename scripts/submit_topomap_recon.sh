#!/bin/bash
#SBATCH -J topomap_recon
#SBATCH -o /ceph/chpc/shared/janine_bijsterbosch_group/naranjorincon_scratch/NeuroTranslate/CHIMERA-fMRI/batch/train_topomap_recon.out%j
#SBATCH -e /ceph/chpc/shared/janine_bijsterbosch_group/naranjorincon_scratch/NeuroTranslate/CHIMERA-fMRI/batch/train_topomap_recon.err%j
#SBATCH --partition=tier2_cpu
#SBATCH --account=janine_bijsterbosch
#SBATCH --mem-per-cpu 20G 
#SBATCH --cpus-per-task 10
#SBATCH -t 0-03:00:00  # might depend on epoch, approx 50epoch = 24 hours

source activate neurotranslate
module load workbench
echo Activated environment with name: $CONDA_DEFAULT_ENV

scratch_path=/ceph/chpc/shared/janine_bijsterbosch_group/naranjorincon_scratch
working_dir_path=${scratch_path}/NeuroTranslate/CHIMERA-fMRI

# where the config files are
yaml_loc="${working_dir_path}/config"
# cd ${yaml_loc} #should be in config/../ path

condition="hparams_topo_recon.yml"
chosen_param_config=$(find "$yaml_loc" -type f -name "$condition")
echo chosen param file is: ${chosen_param_config}

config_model_name=$(grep 'unique_ID' ${chosen_param_config} | awk '{ print $2 }'); #find model details, get the second column
echo Model name and config file to freeze: ${config_model_name}

# date_time_stamp=$(date +"%Y%m%d_%Hh_%Mm_%Ss")
model_type="SiT_LN"
mkdir -p ${working_dir_path}/tmp_files/${model_type}
touch ${working_dir_path}/tmp_files/${model_type}/config_${config_model_name}.yml
cp ${chosen_param_config} ${working_dir_path}/tmp_files/${model_type}/config_${config_model_name}.yml

# config_model_name="070426_translation_norm_ICAd15_demean_glasserd360_fullcorr_1L"
echo "param file created, copied, and saved at tmp path! If you want to submit another job, go ahead."
echo "Using --> ${working_dir_path}/tmp_files/${model_type}/config_${config_model_name}.yml" # ${chosen_param_config}
python3 ${working_dir_path}/scripts/train.py ${working_dir_path}/tmp_files/${model_type}/config_${config_model_name}.yml #${chosen_param_config}

## after training and test, visualize it
# echo "using ${config_model_name}"
# python3 ${working_dir_path}/utils/viz_top2conn_outputs_EXAMmodels.py ${working_dir_path}/tmp_files/${model_type}/config_${config_model_name}.yml

# # then look at downstream analyses
# python3 ${working_dir_path}/utils/downstream_analyses.py ${working_dir_path}/tmp_files/${model_type}/config_${config_model_name}.yml

chmod -R 741 ${working_dir_path}/tmp_files/

echo DONE
