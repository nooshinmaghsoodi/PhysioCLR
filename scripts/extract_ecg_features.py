import numpy as np
import neurokit2 as nk

# ---------------------------
# Helpers
# ---------------------------

def _to_1d_lead(ecg: np.ndarray, *, average_leads: bool = True, lead_index: int = 1) -> np.ndarray:
    """
    Accepts ecg as (T,), (C,T), or (T,C). Returns 1D signal.
    """
    ecg = np.asarray(ecg)

    if ecg.ndim == 1:
        x = ecg
    elif ecg.ndim == 2:
        # assume (C,T) if first dim small
        if ecg.shape[0] <= 16 and ecg.shape[1] > ecg.shape[0]:
            x = ecg.mean(axis=0) if average_leads else ecg[min(lead_index, ecg.shape[0]-1)]
        else:
            # (T,C)
            x = ecg.mean(axis=1) if average_leads else ecg[:, min(lead_index, ecg.shape[1]-1)]
    else:
        raise ValueError(f"ECG must be 1D or 2D, got shape {ecg.shape}")

    x = x.astype(np.float64, copy=False)
    # robust centering
    x = x - np.median(x)
    return x


def _pad_trunc(v: np.ndarray, n: int) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    if v.size >= n:
        return v[:n]
    out = np.zeros(n, dtype=np.float32)
    out[: v.size] = v
    return out


def _safe_diff(x: np.ndarray, fs: int) -> np.ndarray:
    # derivative in mV/ms if input is mV
    # dx/dt where dt = 1/fs seconds = 1000/fs ms
    return np.diff(x) * (fs / 1000.0)


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a); b = np.asarray(b)
    if a.size < 5 or b.size < 5:
        return 0.0
    a = a - a.mean()
    b = b - b.mean()
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom < 1e-12:
        return 0.0
    return float(np.dot(a, b) / denom)


# ---------------------------
# Main feature extraction
# ---------------------------

def extract_ecg_features(
    ecg: np.ndarray,
    *,
    sampling_rate: int = 500,
    average_leads: bool = True,
    lead_index: int = 1,
    beat_cap: int = 10,
    window_seconds: float = 5.0,
) -> dict:
    """
    Extract physiology-inspired ECG features (beat-level + segment-level).

    Returns a dict with:
      - rpeaks (indices)
      - beat-level arrays (length beat_cap, padded)
      - segment-level scalars
    """
    fs = int(sampling_rate)
    x = _to_1d_lead(ecg, average_leads=average_leads, lead_index=lead_index)

    # Clean & R-peaks
    try:
        x_clean = nk.ecg_clean(x, sampling_rate=fs, method="neurokit")
        signals, info = nk.ecg_peaks(x_clean, sampling_rate=fs)
        rpeaks = np.asarray(info["ECG_R_Peaks"], dtype=np.int64)
    except Exception:
        x_clean = x
        rpeaks = np.array([], dtype=np.int64)

    # Delineation (P/QRS/T onsets/offsets)
    # NeuroKit delineate may fail in noisy cases; we degrade gracefully.
    p_on = p_off = qrs_on = qrs_off = t_on = t_off = np.array([], dtype=np.int64)
    try:
        _, waves = nk.ecg_delineate(
            x_clean, rpeaks, sampling_rate=fs, method="dwt", show=False
        )
        p_on  = np.asarray(waves.get("ECG_P_Onsets", []), dtype=np.int64)
        p_off = np.asarray(waves.get("ECG_P_Offsets", []), dtype=np.int64)
        qrs_on  = np.asarray(waves.get("ECG_QRS_Onsets", []), dtype=np.int64)
        qrs_off = np.asarray(waves.get("ECG_QRS_Offsets", []), dtype=np.int64)
        t_on  = np.asarray(waves.get("ECG_T_Onsets", []), dtype=np.int64)
        t_off = np.asarray(waves.get("ECG_T_Offsets", []), dtype=np.int64)
    except Exception:
        pass

    # Build per-beat indices aligned to rpeaks
    # We will align by position: kth onset/offset corresponds to kth rpeak when available.
    nbeats = int(min(len(rpeaks), beat_cap))

    # RR per beat: RR_i = r[i] - r[i-1] (ms), pad first as 0
    rr = np.zeros(nbeats, dtype=np.float32)
    if len(rpeaks) >= 2:
        rr[1:nbeats] = (rpeaks[1:nbeats] - rpeaks[0:nbeats-1]) * (1000.0 / fs)

    # Heart rate approx from RR
    hr = np.zeros(nbeats, dtype=np.float32)
    with np.errstate(divide="ignore", invalid="ignore"):
        hr = np.where(rr > 1e-3, 60000.0 / rr, 0.0).astype(np.float32)

    # Helper to convert onset/offset arrays to durations (ms)
    def dur_ms(on, off):
        out = np.zeros(nbeats, dtype=np.float32)
        m = min(nbeats, len(on), len(off))
        if m > 0:
            out[:m] = (off[:m] - on[:m]) * (1000.0 / fs)
        return out

    p_dur  = dur_ms(p_on, p_off)
    qrs_dur = dur_ms(qrs_on, qrs_off)
    t_dur  = dur_ms(t_on, t_off)

    # PR: from P onset to QRS onset
    pr = np.zeros(nbeats, dtype=np.float32)
    m = min(nbeats, len(p_on), len(qrs_on))
    if m > 0:
        pr[:m] = (qrs_on[:m] - p_on[:m]) * (1000.0 / fs)

    # QT: from QRS onset to T offset (then QTc via Bazett)
    qt = np.zeros(nbeats, dtype=np.float32)
    m = min(nbeats, len(qrs_on), len(t_off))
    if m > 0:
        qt[:m] = (t_off[:m] - qrs_on[:m]) * (1000.0 / fs)
    # Bazett QTc: QT / sqrt(RR_seconds)
    qtc = np.zeros(nbeats, dtype=np.float32)
    rr_sec = rr / 1000.0
    with np.errstate(divide="ignore", invalid="ignore"):
        qtc = np.where(rr_sec > 1e-6, qt / np.sqrt(rr_sec), 0.0).astype(np.float32)

    # Amplitudes relative to baseline: use median of a quiet pre-QRS window if possible
    def baseline_for_beat(i: int) -> float:
        if len(qrs_on) > i and qrs_on[i] > int(0.08 * fs):
            s = max(0, qrs_on[i] - int(0.08 * fs))
            e = qrs_on[i]
            return float(np.median(x_clean[s:e]))
        # fallback to global median
        return float(np.median(x_clean))

    r_amp = np.zeros(nbeats, dtype=np.float32)
    p_amp = np.zeros(nbeats, dtype=np.float32)
    t_amp = np.zeros(nbeats, dtype=np.float32)
    st_j60 = np.zeros(nbeats, dtype=np.float32)
    qrs_slope = np.zeros(nbeats, dtype=np.float32)

    dx = _safe_diff(x_clean, fs)  # mV/ms

    for i in range(nbeats):
        b = baseline_for_beat(i)

        # R amplitude
        r_idx = int(rpeaks[i])
        if 0 <= r_idx < len(x_clean):
            r_amp[i] = float(x_clean[r_idx] - b)

        # P amplitude: max in [P_on, P_off]
        if len(p_on) > i and len(p_off) > i and p_off[i] > p_on[i]:
            s, e = int(p_on[i]), int(p_off[i])
            if 0 <= s < e <= len(x_clean):
                p_amp[i] = float(np.max(x_clean[s:e]) - b)

        # T amplitude: max in [T_on, T_off]
        if len(t_on) > i and len(t_off) > i and t_off[i] > t_on[i]:
            s, e = int(t_on[i]), int(t_off[i])
            if 0 <= s < e <= len(x_clean):
                t_amp[i] = float(np.max(x_clean[s:e]) - b)

        # ST at J+60ms: J-point approximated as QRS offset
        if len(qrs_off) > i:
            j = int(qrs_off[i]) + int(0.06 * fs)
            if 0 <= j < len(x_clean):
                st_j60[i] = float(x_clean[j] - b)

        # QRS upstroke slope: max |dV/dt| in [QRS_on, R_peak]
        if len(qrs_on) > i:
            s = int(qrs_on[i])
            e = int(rpeaks[i])
            if 1 <= s < e <= len(dx):
                qrs_slope[i] = float(np.max(np.abs(dx[s:e])))

    # Beat morphology stability: adjacent beat correlation
    # define beat windows between midpoints of R-peaks
    adj_corr = []
    if len(rpeaks) >= 3:
        mids = ((rpeaks[:-1] + rpeaks[1:]) // 2).astype(int)
        # beats are [mids[i-1], mids[i]) roughly around rpeak i
        for i in range(1, min(len(mids), beat_cap)):
            s1, e1 = mids[i-1], mids[i]
            s2, e2 = mids[i], mids[min(i+1, len(mids)-1)]
            if 0 <= s1 < e1 <= len(x_clean) and 0 <= s2 < e2 <= len(x_clean):
                a = x_clean[s1:e1]
                b = x_clean[s2:e2]
                # resample to same length for correlation
                L = min(len(a), len(b))
                if L >= 30:
                    adj_corr.append(_pearson(a[:L], b[:L]))
    adj_corr = np.asarray(adj_corr, dtype=np.float32)
    if adj_corr.size == 0:
        adj_corr = np.zeros(1, dtype=np.float32)

    # Segment-level aggregates (computed over valid beats only)
    def agg_stats(v):
        v = np.asarray(v, dtype=np.float32)
        v = v[np.isfinite(v)]
        v = v[v != 0]  # treat zeros as missing from padding
        if v.size == 0:
            return dict(mean=0.0, std=0.0, median=0.0, iqr=0.0)
        q1, q3 = np.percentile(v, [25, 75])
        return dict(
            mean=float(v.mean()),
            std=float(v.std(ddof=0)),
            median=float(np.median(v)),
            iqr=float(q3 - q1),
        )

    # HRV-like features from RR
    rr_valid = rr[rr > 0]
    rmssd = 0.0
    pnn50 = 0.0
    cvrr = 0.0
    pct_short_rr = 0.0
    if rr_valid.size >= 2:
        diffs = np.diff(rr_valid)
        rmssd = float(np.sqrt(np.mean(diffs**2)))
        pnn50 = float(np.mean(np.abs(diffs) > 50.0)) * 100.0
        cvrr = float(rr_valid.std(ddof=0) / (rr_valid.mean() + 1e-6))
        med = float(np.median(rr_valid))
        pct_short_rr = float(np.mean(rr_valid < 0.8 * med)) * 100.0

    # Abnormal proportions
    pct_pr_gt_200 = float(np.mean(pr[pr > 0] > 200.0)) * 100.0 if np.any(pr > 0) else 0.0
    pct_qrs_ge_120 = float(np.mean(qrs_dur[qrs_dur > 0] >= 120.0)) * 100.0 if np.any(qrs_dur > 0) else 0.0
    pct_qrs_100_119 = float(np.mean((qrs_dur >= 100.0) & (qrs_dur < 120.0))) * 100.0 if np.any(qrs_dur > 0) else 0.0
    pct_qtc_ge_500 = float(np.mean(qtc[qtc > 0] >= 500.0)) * 100.0 if np.any(qtc > 0) else 0.0

    # Absent P proxy (very low / missing)
    pct_absent_p = float(np.mean(p_amp == 0.0)) * 100.0 if nbeats > 0 else 0.0
    pct_inverted_t = float(np.mean(t_amp < -0.05)) * 100.0 if nbeats > 0 else 0.0  # heuristic

    # ST abnormal |ST| >= 0.1 mV
    pct_abn_st = float(np.mean(np.abs(st_j60) >= 0.1)) * 100.0 if nbeats > 0 else 0.0
    st_mad = float(np.median(np.abs(st_j60 - np.median(st_j60)))) if nbeats > 0 else 0.0

    # Package results
    feats = {
        "rpeaks": rpeaks,
        # per-beat (padded/truncated by features_to_vector)
        "RR": rr,
        "PR": pr,
        "QRS_dur": qrs_dur,
        "QTc": qtc,
        "P_dur": p_dur,
        "T_dur": t_dur,
        "R_amp": r_amp,
        "T_amp": t_amp,
        "P_amp": p_amp,
        "QRS_slope": qrs_slope,
        "ST_J60": st_j60,
        "adjacent_beat_corr": adj_corr,
        # segment aggregates
        "RR_stats": agg_stats(rr_valid),
        "PR_stats": agg_stats(pr),
        "QRS_stats": agg_stats(qrs_dur),
        "QTc_stats": agg_stats(qtc),
        "Pdur_stats": agg_stats(p_dur),
        "Tdur_stats": agg_stats(t_dur),
        "Ramp_stats": agg_stats(r_amp),
        "Pamp_stats": agg_stats(p_amp),
        "Tamp_stats": agg_stats(t_amp),
        "QRSslope_stats": agg_stats(qrs_slope),
        "ST_stats": agg_stats(st_j60),
        "rmssd": rmssd,
        "pnn50": pnn50,
        "cvrr": cvrr,
        "pct_short_rr": pct_short_rr,
        "pct_pr_gt_200": pct_pr_gt_200,
        "pct_qrs_ge_120": pct_qrs_ge_120,
        "pct_qrs_100_119": pct_qrs_100_119,
        "pct_qtc_ge_500": pct_qtc_ge_500,
        "pct_absent_p": pct_absent_p,
        "pct_inverted_t": pct_inverted_t,
        "pct_abn_st": pct_abn_st,
        "st_mad": st_mad,
        "adj_corr_mean": float(np.mean(adj_corr)),
        "adj_corr_std": float(np.std(adj_corr)),
        "adj_corr_min": float(np.min(adj_corr)),
    }
    return feats


def features_to_vector(feats: dict, *, beat_cap: int = 10) -> np.ndarray:
    """
    Produce a fixed-length vector used for similarity-based pair selection.

    NOTE:
    Your paper says you start from a fixed 150-d feature vector then reduce to 50-d via PCA.
    Here we output a stable fixed vector; do PCA outside this function if desired.
    """
    # ----- per-beat block (10 features × beat_cap) = 100 dims -----
    per_beat = [
        _pad_trunc(feats["RR"], beat_cap),
        _pad_trunc(feats["PR"], beat_cap),
        _pad_trunc(feats["QRS_dur"], beat_cap),
        _pad_trunc(feats["QTc"], beat_cap),
        _pad_trunc(feats["P_dur"], beat_cap),
        _pad_trunc(feats["T_dur"], beat_cap),
        _pad_trunc(feats["R_amp"], beat_cap),
        _pad_trunc(feats["T_amp"], beat_cap),
        _pad_trunc(feats["P_amp"], beat_cap),
        _pad_trunc(feats["QRS_slope"], beat_cap),
    ]
    per_beat_vec = np.concatenate(per_beat, axis=0).astype(np.float32)  # 10*beat_cap

    # ----- segment-level block: target ~50 dims -----
    # For each stats dict: mean/std/median/iqr (4 dims)
    def pack_stats(d):
        return np.array([d["mean"], d["std"], d["median"], d["iqr"]], dtype=np.float32)

    seg = []
    seg += [pack_stats(feats["RR_stats"])]
    seg += [pack_stats(feats["PR_stats"])]
    seg += [pack_stats(feats["QRS_stats"])]
    seg += [pack_stats(feats["QTc_stats"])]
    seg += [pack_stats(feats["Pdur_stats"])]
    seg += [pack_stats(feats["Tdur_stats"])]
    seg += [pack_stats(feats["Ramp_stats"])]
    seg += [pack_stats(feats["Pamp_stats"])]
    seg += [pack_stats(feats["Tamp_stats"])]
    seg += [pack_stats(feats["QRSslope_stats"])]
    seg += [pack_stats(feats["ST_stats"])]

    # HRV + proportions + morphology stability
    seg += [np.array([
        feats["rmssd"],
        feats["pnn50"],
        feats["cvrr"],
        feats["pct_short_rr"],
        feats["pct_pr_gt_200"],
        feats["pct_qrs_ge_120"],
        feats["pct_qrs_100_119"],
        feats["pct_qtc_ge_500"],
        feats["pct_absent_p"],
        feats["pct_inverted_t"],
        feats["pct_abn_st"],
        feats["st_mad"],
        feats["adj_corr_mean"],
        feats["adj_corr_std"],
        feats["adj_corr_min"],
    ], dtype=np.float32)]

    seg_vec = np.concatenate(seg, axis=0).astype(np.float32)

    vec = np.concatenate([per_beat_vec, seg_vec], axis=0).astype(np.float32)

    return vec
