"""
Usage
-----
python scripts/infer_physioCLR.py \
       --ckpt checkpoints/physioCLR.pt \
       --test-list lists/kgh_test.txt
"""
import argparse, json, numpy as np, torch
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from physioclr_ext.models.wav2vec2_physioCLR import Wav2Vec2PhysioCLR
from physioclr_ext.data.ecg_feature_dataset import ECGFeatDataset
from torch.utils.data import DataLoader

def load_model(path):
    cp = torch.load(path, map_location="cpu")
    model = Wav2Vec2PhysioCLR.build_model(cp["cfg"]["model"])
    model.load_state_dict(cp["model"])
    model.eval()
    return model

def main(args):
    ds = ECGFeatDataset([x.strip() for x in open(args.test_list)])
    loader = DataLoader(ds, batch_size=64, shuffle=False,
                        collate_fn=ds.collater, num_workers=4)

    model = load_model(args.ckpt).cuda()
    y_true, y_pred = [], []

    with torch.no_grad():
        for batch in loader:
            logits = model(batch["source"].cuda(), features_only=True)[1]["features"][:,0]
            y_pred.append(torch.sigmoid(logits).cpu().numpy())
            y_true.append(batch["label"].numpy())

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)

    print("AUROC :", roc_auc_score(y_true, y_pred))
    print("F1     :", f1_score(y_true, y_pred>0.5))
    print("Recall :", recall_score(y_true, y_pred>0.5))
    print("Prec.  :", precision_score(y_true, y_pred>0.5))

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--test-list", required=True)
    main(p.parse_args())
