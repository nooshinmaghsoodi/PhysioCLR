"""
Create .npy files that ECGFeatDataset expects.

Example
-------
python extract_features_physioCLR.py \
       --manifest  /scratch/.../manifests/cmsc/train.tsv \
       --dest      /scratch/.../physioCLR_npy \
       --sr        500
"""
import argparse, os, numpy as np, pandas as pd, neurokit2 as nk
from pathlib import Path
from extract_ecg_features import extract_ecg_features, features_to_vector  

def main(a):
    df = pd.read_csv(a.manifest, sep="\t")
    dest = Path(a.dest); dest.mkdir(parents=True, exist_ok=True)
    outlist = dest / "filelist.txt"
    with open(outlist, "w") as w:
        for row in df.itertuples():
            ecg = np.load(row.ecg_path)                     # or load mat, etc.
            feats = extract_ecg_features(ecg, sampling_rate=a.sr, average_leads=True)
            vec  = features_to_vector(feats)
            r_pk = feats["rpeaks"]       # crude example

            outdict = {"id": row.id,
                       "segment": ecg.astype("float32"),
                       "feat_vec": vec.astype("float32"),
                       "r_idx": r_pk.astype("int64")}
            npy_path = dest / f"{row.id}.npy"
            np.save(npy_path, outdict)
            w.write(str(npy_path) + "\n")
    print("Saved", len(df), "npy files and", outlist)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--dest", required=True)
    p.add_argument("--sr", type=int, default=500)
    main(p.parse_args())
