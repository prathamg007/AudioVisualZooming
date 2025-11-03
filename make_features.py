# make_features.py
import os
import json
import math
import argparse
import numpy as np
import soundfile as sf
from tqdm import tqdm
from scipy.signal import stft, get_window

# --------------------- Config (Task-1 compliant) ---------------------
FS = 16000
NFFT = 512
HOP = 256
WINDOW = get_window("hann", NFFT, fftbins=True)
MIC_SPACING_M = 0.08   # 8 cm
C_SOUND = 343.0
ANGLE_GRID_DEG = np.arange(0, 181, 10)  # 0..180 step 10 => K=19

# --------------------- Helpers ---------------------
def load_meta(meta_path):
    with open(meta_path, "r") as f:
        m = json.load(f)
    # Fallbacks if not present in meta (but your meta should have these)
    fc = float(m.get("fov_center", 90.0))
    fw = float(m.get("fov_width", 10.0))
    return fc, fw

def expected_phi_table(freqs_hz):
    """
    phi[f,k] = expected inter-mic phase (radians) for frequency f and look-angle θ_k
    phi(f,θ) = 2π * f * (d * cosθ / c)
    Shapes: freqs_hz: [F], returns phi: [F,K]
    """
    thetas = np.deg2rad(ANGLE_GRID_DEG)           # [K]
    tau = (MIC_SPACING_M * np.cos(thetas)) / C_SOUND  # [K] seconds
    phi = 2 * np.pi * freqs_hz[:, None] * tau[None, :]  # [F,K]
    return phi

def compute_features(mix_stereo, fov_center_deg, fov_width_deg):
    """
    mix_stereo: [T,2] float32 (mic1, mic2)
    Returns dict of arrays [F,T] for LPS, cosIPD, sinIPD, DFin, DFout
    """
    # STFT (one-sided; scipy returns F = NFFT//2+1)
    f, t, Y1 = stft(
        mix_stereo[:, 0],
        fs=FS, window=WINDOW, nperseg=NFFT, noverlap=NFFT - HOP,
        nfft=NFFT, detrend=False, return_onesided=True, boundary=None, padded=False,
    )
    _, _, Y2 = stft(
        mix_stereo[:, 1],
        fs=FS, window=WINDOW, nperseg=NFFT, noverlap=NFFT - HOP,
        nfft=NFFT, detrend=False, return_onesided=True, boundary=None, padded=False,
    )
    # Y1, Y2 shapes: [F, T], complex

    # LPS = log(|Y1|^2)
    mag2 = np.maximum(np.abs(Y1) ** 2, 1e-8)
    LPS = np.log(mag2)  # [F,T]

    # IPD = angle(Y1 * conj(Y2))  -> then cos/sin
    cross = Y1 * np.conj(Y2)      # [F,T]
    IPD = np.angle(cross)         # [-pi, pi], [F,T]
    cosIPD = np.cos(IPD)
    sinIPD = np.sin(IPD)

    # Directional similarity d_theta(f,t,k) = cos( IPD(f,t) - phi(f,k) )
    phi = expected_phi_table(f)   # [F,K]
    # Broadcast: IPD [F,T] -> [F,T,1]; phi [F,K] -> [F,1,K]
    dtheta = np.cos(IPD[..., None] - phi[:, None, :])  # [F,T,K]

    # Build inside/outside FOV masks over K
    fov_center = float(np.clip(fov_center_deg, 0.0, 180.0))
    half = float(fov_width_deg) / 2.0
    lo = max(0.0, fov_center - half)
    hi = min(180.0, fov_center + half)

    in_mask = (ANGLE_GRID_DEG >= lo) & (ANGLE_GRID_DEG <= hi)  # [K]
    if not np.any(in_mask):
        # ensure at least one angle is inside
        nearest_idx = int(np.argmin(np.abs(ANGLE_GRID_DEG - fov_center)))
        in_mask = np.zeros_like(ANGLE_GRID_DEG, dtype=bool)
        in_mask[nearest_idx] = True
    out_mask = ~in_mask

    # Max over K with masks (use -inf to ignore)
    neg_inf = -1e9
    d_in  = np.where(in_mask[None, None, :], dtheta, neg_inf)   # [F,T,K]
    d_out = np.where(out_mask[None, None, :], dtheta, neg_inf)  # [F,T,K]
    DFin  = np.max(d_in,  axis=-1)  # [F,T]
    DFout = np.max(d_out, axis=-1)  # [F,T]

    return {
        "LPS": LPS.astype(np.float32),
        "cosIPD": cosIPD.astype(np.float32),
        "sinIPD": sinIPD.astype(np.float32),
        "DFin": DFin.astype(np.float32),
        "DFout": DFout.astype(np.float32),
    }

# --------------------- Main ---------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, required=True, help="Path to dataset (contains data_xxxxxx and splits.json)")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing features_XXXXXX.npz")
    args = ap.parse_args()

    root = args.root
    if not os.path.isdir(root):
        raise FileNotFoundError(f"Dataset root not found: {root}")

    data_dirs = sorted([d for d in os.listdir(root) if d.startswith("data_") and os.path.isdir(os.path.join(root, d))])
    if not data_dirs:
        raise RuntimeError(f"No 'data_XXXXXX' folders found under: {root}")

    for folder in tqdm(data_dirs, desc="Feature extraction"):
        sid = folder.split("_")[1]
        sub = os.path.join(root, folder)

        mix_path  = os.path.join(sub, f"mixture_{sid}.wav")
        meta_path = os.path.join(sub, f"meta_{sid}.json")
        feat_path = os.path.join(sub, f"features_{sid}.npz")

        # Skip if already exists (unless --overwrite)
        if (not args.overwrite) and os.path.exists(feat_path):
            continue

        # Basic file checks
        if not os.path.isfile(mix_path):
            print(f"Skipping {folder}: missing {os.path.basename(mix_path)}")
            continue
        if not os.path.isfile(meta_path):
            print(f"Skipping {folder}: missing {os.path.basename(meta_path)}")
            continue

        # Load audio (T,2) and meta
        mix, fs = sf.read(mix_path)   # shape [T,2]
        if fs != FS:
            raise ValueError(f"{mix_path} sampling rate {fs} != expected {FS}")
        if mix.ndim != 2 or mix.shape[1] != 2:
            raise ValueError(f"{mix_path} must be 2-channel; got shape {mix.shape}")

        fov_center_deg, fov_width_deg = load_meta(meta_path)

        # Compute features
        feats = compute_features(mix.astype(np.float32), fov_center_deg, fov_width_deg)

        # Save features beside audio
        np.savez_compressed(
            feat_path,
            LPS=feats["LPS"],
            cosIPD=feats["cosIPD"],
            sinIPD=feats["sinIPD"],
            DFin=feats["DFin"],
            DFout=feats["DFout"],
            fov_center=float(fov_center_deg),
            fov_width=float(fov_width_deg),
            fs=int(FS),
            mic_spacing_m=float(MIC_SPACING_M),
            angle_grid_deg=ANGLE_GRID_DEG.astype(np.int16),
            nfft=int(NFFT),
            hop=int(HOP),
        )

    print("✔ Feature extraction complete.")

if __name__ == "__main__":
    main()
