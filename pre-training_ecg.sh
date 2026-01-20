#!/bin/bash
#SBATCH --job-name=pre-train_ecg   # Job name
#SBATCH --mem=48G                  # Increase memory to better match A40 node capabilities
#SBATCH --gres=gpu:a40:4            # Request 4 A40 GPUs
#SBATCH --time=16:00:00             # Maximize available wall time under `normal` QoS
#SBATCH -c 32                       # Increase CPU cores for better data loading efficiency
#SBATCH --qos=normal
#SBATCH --output=preprocess-%j.out  # Standard output log
#SBATCH --error=preprocess-%j.err   # Standard error log

# Activate your virtual environment (assuming you have one)
source /fs01/home/nooshinm/anaconda3/etc/profile.d/conda.sh
conda activate ecg_fm
export HYDRA_FULL_ERROR=1

FAIRSEQ_SIGNALS_ROOT="fairseq-signals"
MANIFEST_DIR="/scratch/ssd004/scratch/nooshinm/preprocessing/data/manifests/cmsc"
OUTPUT_DIR="/scratch/ssd004/scratch/nooshinm/output"


# point to the *npy list*, not the .tsv manifest
NPY_ROOT="/scratch/ssd004/scratch/nooshinm/preprocessing/data/physioCLR_npy"
TRAIN_LIST="$NPY_ROOT/train/filelist.txt"

fairseq-hydra-train \
	common.user_dir=/fs01/home/nooshinm/PhysioCLR \
    task.data=$TRAIN_LIST \
    dataset.valid_subset=valid \
    dataset.batch_size=64 \
    dataset.num_workers=8 \
    dataset.disable_validation=false \
    distributed_training.distributed_world_size=4 \
    optimization.update_freq=[4] \
    checkpoint.save_dir=$OUTPUT_DIR \
    checkpoint.save_interval=1 \
    checkpoint.keep_last_epochs=1 \
    common.log_format=csv \
    --config-dir $FAIRSEQ_SIGNALS_ROOT/examples/physioclr/config/pretraining \
    --config-name physioCLR_ecg

# fairseq-hydra-train \
#     task.data=$MANIFEST_DIR \
#     dataset.valid_subset=valid \
#     dataset.batch_size=64 \
#     dataset.num_workers=8 \
#     dataset.disable_validation=false \
#     distributed_training.distributed_world_size=4 \
#     optimization.update_freq=[4] \
#     checkpoint.save_dir=$OUTPUT_DIR \
#     checkpoint.save_interval=1 \
#     checkpoint.keep_last_epochs=1 \
#     common.log_format=csv \
#     --config-dir $FAIRSEQ_SIGNALS_ROOT/examples/w2v_cmsc/config/pretraining \
#     --config-name mimic_iv_ecg_physionet_pretrained