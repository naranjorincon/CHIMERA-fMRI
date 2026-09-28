#!/bin/bash
#SBATCH -J ExtractABCD_Timeseries
#SBATCH -o /ceph/chpc/shared/janine_bijsterbosch_group/naranjorincon_scratch/NeuroTranslate/CHIMERA-fMRI/batch/create_connectome_from_ts/parll_extractABCD_ts.o%j
#SBATCH -e /ceph/chpc/shared/janine_bijsterbosch_group/naranjorincon_scratch/NeuroTranslate/CHIMERA-fMRI/batch/create_connectome_from_ts/parll_extractABCD_ts.e%j
#SBATCH --account=janine_bijsterbosch
#SBATCH --partition=tier2_cpu
#SBATCH --mem=7G
#SBATCH -t 0-0:10:00

source activate neurotranslate
echo Subject: sub-ZZNX6W2P

module load workbench
# need subj ID file
subjID_fpath=/ceph/chpc/shared/janine_bijsterbosch_group/WAPIAW_2026/qc/individual_subjects/individual_subject_list_highNSI_3001.txt

mkdir -p /ceph/chpc/shared/janine_bijsterbosch_group/naranjorincon_scratch/NeuroTranslate/ABCD_NetMats/ABCDv6/schaefer_d300/untranspose
echo Extracting Timeseries For: sub-ZZNX6W2P

wb_command -cifti-parcellate     /ceph/chpc/shared/janine_bijsterbosch_group/WAPIAW_2026/cortex_only_data/ses-all/4mm_smooth/sub-ZZNX6W2P_cortex_only_demean_smooth_4mm.dtseries.nii     /ceph/chpc/shared/janine_bijsterbosch_group/naranjorincon_scratch/NeuroTranslate/ABCD_NetMats/schaefer300/Schaefer2018_300Parcels_17Networks_order.dlabel.nii     COLUMN     /ceph/chpc/shared/janine_bijsterbosch_group/naranjorincon_scratch/NeuroTranslate/ABCD_NetMats/ABCDv6/schaefer_d300/untranspose/sub-ZZNX6W2P.ptseries.nii

wb_command -cifti-convert -to-text     /ceph/chpc/shared/janine_bijsterbosch_group/naranjorincon_scratch/NeuroTranslate/ABCD_NetMats/ABCDv6/schaefer_d300/untranspose/sub-ZZNX6W2P.ptseries.nii     /ceph/chpc/shared/janine_bijsterbosch_group/naranjorincon_scratch/NeuroTranslate/ABCD_NetMats/ABCDv6/schaefer_d300/untranspose/untranspose_sub-ZZNX6W2P.txt



