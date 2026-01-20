import torch
import random
from torch import Tensor

def shuffle_beats(x: Tensor, r_peaks: Tensor) -> Tensor:
    """
    ECG-specific augmentation that shuffles whole heartbeats.

    Args
    ----
    x        : (B, C, T)   raw segment
    r_peaks  : (B, N) long – sample indices of R-peaks for every batch element

    Returns
    -------
    Tensor   same shape as x with beats permuted
    """
    b, c, t = x.size()
    x_out = torch.zeros_like(x)

    for i in range(b):
        # build heartbeat index list [ (start, end), … ]
        idx = r_peaks[i][r_peaks[i] > 0]          # strip padding
        idx = torch.cat([idx, torch.tensor([t])])
        beats = [(idx[j], idx[j + 1]) for j in range(len(idx) - 1)]

        random.shuffle(beats)                     # in-place permutation
        cursor = 0
        for s, e in beats:
            length = e - s
            x_out[i, :, cursor:cursor + length] = x[i, :, s:e]
            cursor += length                      # keep final length = T
    return x_out