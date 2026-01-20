#!/bin/bash
#SBATCH --job-name=finetune_ecg   # Job name
#SBATCH --mem=48G                  # Increase memory to better match A40 node capabilities
#SBATCH --gres=gpu:a40:1            # Request 1 A40 GPUs
#SBATCH --time=16:00:00             # Maximize available wall time under `normal` QoS
#SBATCH -c 32                       # Increase CPU cores for better data loading efficiency
#SBATCH --qos=normal
#SBATCH --output=finetune-%j.out  # Standard output log
#SBATCH --error=finetune-%j.err   # Standard error log
# Activate your virtual environment (assuming you have one)
source /fs01/home/nooshinm/anaconda3/etc/profile.d/conda.sh
conda activate ecg_fm

PRETRAINED_MODEL="/scratch/ssd004/scratch/nooshinm/output/checkpoints/checkpoint_best.pt"
FAIRSEQ_SIGNALS_ROOT="fairseq-signals"
MANIFEST_DIR="/scratch/ssd004/scratch/nooshinm/preprocessing/data/manifests"
OUTPUT_DIR="/scratch/ssd004/scratch/nooshinm/output/test"
LABEL_DIR="/scratch/ssd004/scratch/nooshinm/preprocessing/data/physionet2021/labels/"
POS_WEIGHT=$(cat $LABEL_DIR/pos_weight.txt)

fairseq-hydra-inference \
    task.data=$MANIFEST_DIR \
    common_eval.path=/scratch/ssd004/scratch/nooshinm/output/finetuning/physionet_finetuned.pt \
    common_eval.results_path=$OUTPUT_DIR  \
    common_eval.save_outputs=True \
    model.num_labels=26 \
    dataset.valid_subset=test \
    dataset.batch_size=64 \
    dataset.num_workers=3 \
    dataset.disable_validation=false \
    distributed_training.distributed_world_size=1 \
    distributed_training.find_unused_parameters=True \
    common.log_format=csv \
    +common.log_file=$OUTPUT_DIR/log_test.log \
    +task.label_file=$LABEL_DIR/y.npy \
    --config-dir $FAIRSEQ_SIGNALS_ROOT/examples/w2v_cmsc/config/finetuning/ecg_transformer \
    --config-name physionet_finetuned