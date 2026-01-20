from dataclasses import dataclass, field
import torch, torch.nn.functional as F
from fairseq_signals import metrics
from fairseq_signals.criterions import register_criterion, BaseCriterion
from physioclr_ext.augmentations.heartbeat_shuffle import shuffle_beats


@dataclass
class PhysioCLRCfg:
    temperature: float = field(default=0.1, metadata={"help": "softmax τ"})
    sim_threshold: float = field(default=0.25, metadata={"help": "cos-sim ≥ δ ⇒ positive"})
    alpha: float = field(default=0.2,  metadata={"help": "global-MSE weight"})
    beta:  float = field(default=0.1,  metadata={"help": "peak-MSE weight"})

@register_criterion("physioclr", dataclass=PhysioCLRCfg)
class PhysioCLRCriterion(BaseCriterion):
    """
    Hybrid contrastive + reconstruction loss described in the TBME paper:
        L = L_contrastive + α·L_global + β·L_peak
    """

    def __init__(self, cfg, task):
        super().__init__(task)
        self.tau, self.delta = cfg.temperature, cfg.sim_threshold
        self.alpha, self.beta = cfg.alpha, cfg.beta

    # -------- helper ----------------------------------------------------

    @staticmethod
    def cosine_batch(z1, z2):
        z1 = F.normalize(z1, dim=-1)
        z2 = F.normalize(z2, dim=-1)
        return torch.mm(z1, z2.t())        # (B,B)

    # -------- main forward ----------------------------------------------

    def forward(self, model, sample, reduce=True):
        """
        sample keys (added by the new Dataset below)
        --------------------------------------------
        ├─ 'source'          float32 (B, C, T)
        ├─ 'features'        float32 (B, F)  ← PCA-reduced handcrafted vec
        ├─ 'r_peaks'         int64   (B, maxN)
        └─ 'id'              str
        """
        src = sample["source"]
        feats = sample["features"]

        # 1) encode original
        rep, extra = model(src, features_only=True)         # (B,T',D)
        h_anchor = extra["features"][:, 0]                  # CLS-token

        # 2) heartbeat-shuffle view
        
        x_shuffle = shuffle_beats(src.clone(), sample["r_peaks"])
        h_shuffle = model(x_shuffle, features_only=True)[1]["features"][:, 0]

        # 3) temporal-adjacent positive (CMSC-style)
        #    ─ assume dataset gives neighbour in batch right after anchor
        idx_pos_adj = (torch.arange(src.size(0)) + 1) % src.size(0)
        h_adj = h_anchor[idx_pos_adj]

        # 4) feature-similar positives / negatives
        sim_mat = self.cosine_batch(feats, feats)
        mask_pos = sim_mat >= self.delta
        mask_neg = sim_mat <  self.delta

        # contrastive numerator and denominator
        logits = torch.mm(F.normalize(h_anchor, dim=-1),
                          F.normalize(torch.cat([h_shuffle, h_adj, h_anchor], 0), dim=-1).t()
                         ) / self.tau
        b = src.size(0)
        # positive logits are (anchor, shuffle) & (anchor, adj) & feature-based ones
        pos_mask = torch.zeros_like(logits, dtype=torch.bool)
        pos_mask[:,                :b] = torch.eye(b)                # shuffle
        pos_mask[:,               b:2*b] = torch.eye(b)              # adjacent
        pos_mask[:, 2*b:3*b] = mask_pos                              # feature

        neg_mask = torch.zeros_like(logits, dtype=torch.bool)
        neg_mask[:, 2*b:3*b] = mask_neg

        loss_pos = -torch.logsumexp(logits.masked_fill(~pos_mask, -1e4), dim=-1)
        loss_neg =  torch.logsumexp(logits.masked_fill(~neg_mask, -1e4), dim=-1)
        l_con = (loss_pos + loss_neg).mean()

        # 5) reconstruction losses (decoder output in extra)
        x_hat = extra["reconstruction"]                       # (B,C,T)
        l_global = F.mse_loss(x_hat, src)

        # peak-aware
        peaks_anchor = extra["peaks"]           # provided by model
        peaks_hat    = extra["peaks_hat"]
        l_peaks = F.mse_loss(peaks_hat, peaks_anchor)

        total = l_con + self.alpha * l_global + self.beta * l_peaks
        logging_output = {"loss": total, "ntokens": src.numel(), "nsentences": b}
        return total, logging_output
