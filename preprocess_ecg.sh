#!/bin/bash
#SBATCH --job-name=preprocess_ecg   # Job name
#SBATCH --mem=16G
#SBATCH --gres=gpu:a40:1
#SBATCH --time 8:00:00
#SBATCH -c 16 
#SBATCH --qos=normal
#SBATCH --output=preprocess-%j.out  # Standard output log
#SBATCH --error=preprocess-%j.err   # Standard error log

# Activate your virtual environment (assuming you have one)
source /fs01/home/nooshinm/anaconda3/etc/profile.d/conda.sh
conda activate ecg_fm

# Set dataset directories
PROCESSED_ROOT="/scratch/ssd004/scratch/$USER/preprocessing/data"
PHYSIONET_ROOT="/fs01/home/nooshinm/physionet.org/training"  # Update with your actual path
MIMIC_IV_ECG_ROOT="/fs01/datasets/MIMIC-IV-ECG/physionet.org/files/mimic-iv-ecg/1.0/"         # Update with your actual path
EVALUATION_ROOT="/fs01/home/nooshinm/evaluation-2021"        # For PhysioNet evaluation weights


# Navigate to fairseq-signals preprocessing scripts
cd fairseq-signals/scripts/preprocess/ecg

# --- PhysioCLR output for handcrafted features --------------------------
PHYCLR_NPY_ROOT="/scratch/ssd004/scratch/$USER/preprocessing/data/physioCLR_npy"
SAMPLING_RATE=500  

# --- Preprocess PhysioNet 2021 ---
echo "Preprocessing PhysioNet 2021 dataset"
python physionet2021_records.py \
    --processed_root "$PROCESSED_ROOT/physionet2021" \
    --raw_root "$PHYSIONET_ROOT"

python physionet2021_signals.py \
    --processed_root "$PROCESSED_ROOT/physionet2021" \
    --raw_root "$PHYSIONET_ROOT" \
    --manifest_file "$PROCESSED_ROOT/manifest.csv"

# echo "splits.py"
 python ../splits.py \
     --strategy "custom_splits" \
     --split_column "dataset" \
     --processed_root "$PROCESSED_ROOT/physionet2021" \
     --filter_cols "nan_any,constant_leads_any" \
     --dataset_subset "cpsc_2018,cpsc_2018_extra,georgia,ptb-xl,chapman_shaoxing,ningbo"

 mkdir -p $PROCESSED_ROOT/physionet2021/labels
 echo "physionet2021_labels.py"
 python physionet2021_labels.py \
     --processed_root "$PROCESSED_ROOT/physionet2021" \
     --weights_path "$EVALUATION_ROOT/weights.csv" \
     --weight_abbrev_path "$EVALUATION_ROOT/weights_abbreviations.csv"

 echo "prepare_clf_labels.py"
 python ../prepare_clf_labels.py \
     --output_dir "$PROCESSED_ROOT/physionet2021/labels" \
     --labels "$PROCESSED_ROOT/physionet2021/labels/labels.csv" \
     --meta_splits "$PROCESSED_ROOT/physionet2021/meta_split.csv"

# --- Preprocess MIMIC-IV-ECG ---
echo "Preprocessing MIMIC-IV-ECG dataset"
python mimic_iv_ecg_records.py \
    --processed_root "$PROCESSED_ROOT/mimic_iv_ecg" \
    --raw_root "$MIMIC_IV_ECG_ROOT" 

python mimic_iv_ecg_signals.py \
    --processed_root "$PROCESSED_ROOT/mimic_iv_ecg" \
    --raw_root "$MIMIC_IV_ECG_ROOT" \
    --manifest_file "$PROCESSED_ROOT/manifest.csv"

python ../splits.py \
    --strategy "grouped" \
    --processed_root "$PROCESSED_ROOT/mimic_iv_ecg" \
    --group_col "subject_id" \
    --filter_cols "nan_any,constant_leads_any"

#--- Create Manifests for Combined Datasets ---
echo "Creating combined manifests for PhysioNet and MIMIC-IV-ECG"
MANIFEST_DIR="$PROCESSED_ROOT/manifests"
mkdir -p "$MANIFEST_DIR"

cd /fs01/home/nooshinm/fairseq-signals/scripts/preprocess

python manifests.py \
   --split_file_paths "$PROCESSED_ROOT/physionet2021/segmented_split.csv,$PROCESSED_ROOT/mimic_iv_ecg/segmented_split.csv" \
   --save_dir "$MANIFEST_DIR"

cd /fs01/home/nooshinm/fairseq-signals/fairseq_signals/data/ecg/preprocess
# If training a CMSC model, convert the manifest accordingly
python convert_to_cmsc_manifest.py \
   "$MANIFEST_DIR" \
   --dest "$MANIFEST_DIR"
    
echo "Creating combined manifests for PhysioNet"
MANIFEST_DIR="$PROCESSED_ROOT/manifests"
mkdir -p "$MANIFEST_DIR"

cd /fs01/home/nooshinm/fairseq-signals/scripts/preprocess

python manifests.py \
    --split_file_paths "$PROCESSED_ROOT/physionet2021/segmented_split.csv" \
    --save_dir "$MANIFEST_DIR"

cd /fs01/home/nooshinm/fairseq-signals/fairseq_signals/data/ecg/preprocess
# If training a CMSC model, convert the manifest accordingly
python convert_to_cmsc_manifest.py \
    "$MANIFEST_DIR" \
    --dest "$MANIFEST_DIR"


# ------------------------------------------------------------------------
#  STEP  —  Build .npy files + filelist.txt  (PhysioCLR)
# ------------------------------------------------------------------------
echo "Extracting handcrafted features for PhysioCLR"

mkdir -p "$PHYCLR_NPY_ROOT/train" "$PHYCLR_NPY_ROOT/valid" "$PHYCLR_NPY_ROOT/test"

# helper function --------------------------------------------------------
extract_split () {
  SPLIT=$1                                  # train / valid / test
  TSV="$MANIFEST_DIR/${SPLIT}.tsv"
  DEST="$PHYCLR_NPY_ROOT/${SPLIT}"
  if [ -f "$TSV" ]; then
      python scripts/extract_features_physioCLR.py \
            --manifest "$TSV" \
            --dest     "$DEST" \
            --sr       $SAMPLING_RATE
  else
      echo "$TSV not found, skipping $SPLIT split"
  fi
}
extract_split train
extract_split valid
extract_split test