#!/bin/bash
#SBATCH --job-name=finetune_ecg   # Job name
#SBATCH --mem=48G                  # Increase memory to better match A40 node capabilities
#SBATCH --gres=gpu:a40:4            # Request 4 A40 GPUs
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
OUTPUT_DIR="/scratch/ssd004/scratch/nooshinm/output/finetuning"
LABEL_DIR="/scratch/ssd004/scratch/nooshinm/preprocessing/data/physionet2021/labels/"
NUM_LABELS=$(($(wc -l < "$LABEL_DIR/label_def.csv") - 1))
POS_WEIGHT=$(cat $LABEL_DIR/pos_weight.txt)
fairseq-hydra-train \
	common.user_dir=/path/to/PhysioCLR \
    task.data=$MANIFEST_DIR \
    model.model_path=$PRETRAINED_MODEL \
    model.num_labels=$NUM_LABELS \
    optimization.lr=[1e-06] \
    optimization.max_epoch=100 \
    dataset.batch_size=64 \
    dataset.num_workers=5 \
    dataset.disable_validation=true \
    distributed_training.distributed_world_size=1 \
    distributed_training.find_unused_parameters=True \
    checkpoint.save_dir=$OUTPUT_DIR \
    checkpoint.save_interval=1 \
    checkpoint.keep_last_epochs=0 \
    common.log_format=csv \
    +task.label_file=$LABEL_DIR/y.npy \
    +criterion.pos_weight=$POS_WEIGHT \
    --config-dir $FAIRSEQ_SIGNALS_ROOT/examples/physioclr/config/finetuning/ecg_transformer_physioCLR \
    --config-name physionet_finetuned


    # --config-dir $FAIRSEQ_SIGNALS_ROOT/examples/w2v_cmsc/config/finetuning/ecg_transformer \
    # --config-name physionet_finetuned