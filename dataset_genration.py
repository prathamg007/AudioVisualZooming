import os
import json
import random
import numpy as np
import soundfile as sf
from tqdm import tqdm
from scipy.ndimage import shift

# ---------------- CONFIG ----------------
root_dir = r"D:\Main_SP_Cup\matlab_implementation\dataset"         # folder with data_000001 ... data_060059
fs = 16000
speed_of_sound = 343.0          # m/s
mic_spacing = 0.08              # 8 cm
sir_db = 0                      # fixed
snr_db = 5                      # fixed
min_angle_sep = 20              # deg
beamwidth_range = [5, 15]       # deg
duration_s = 6.0                # seconds (target/interference length)

# ---------------- UTILITIES ----------------
def normalize_audio(x):
    return x / (np.max(np.abs(x)) + 1e-8)

def rms(x):
    return np.sqrt(np.mean(x**2) + 1e-12)

def apply_delay(signal, delay_s, fs):
    """Apply fractional delay using scipy shift (safe for sub-sample delays)."""
    delay_samples = delay_s * fs
    return shift(signal, delay_samples, mode='nearest')

def mix_signals(x_t1, x_t2, x_i1, x_i2, sir_db, snr_db):
    """Mix target + interference at fixed SIR and SNR."""
    x_t1, x_t2 = normalize_audio(x_t1), normalize_audio(x_t2)
    x_i1, x_i2 = normalize_audio(x_i1), normalize_audio(x_i2)

    # Scale interference for exact SIR
    p_t, p_i = rms(x_t1), rms(x_i1)
    scale_i = p_t / (10**(sir_db/20) * p_i + 1e-12)
    x_i1 *= scale_i
    x_i2 *= scale_i

    # Combine
    mix1 = x_t1 + x_i1
    mix2 = x_t2 + x_i2

    # Add noise for exact SNR
    mix_power = rms(mix1)
    noise_power = mix_power / (10**(snr_db/20))
    noise = np.random.randn(len(mix1))
    noise = noise / rms(noise) * noise_power
    mix1 += noise
    mix2 += noise
    return mix1, mix2

def fix_length(x, L):
    if len(x) > L:
        return x[:L]
    elif len(x) < L:
        return np.pad(x, (0, L - len(x)))
    return x

# ---------------- MAIN GENERATION ----------------
data_folders = sorted([d for d in os.listdir(root_dir) if d.startswith("data_")])
L = int(fs * duration_s)

for folder in tqdm(data_folders, desc="Generating mixtures"):
    sample_id = folder.split('_')[1]
    subdir = os.path.join(root_dir, folder)

    target_path = os.path.join(subdir, f"target_{sample_id}.flac")
    interf_path = os.path.join(subdir, f"interference_{sample_id}.flac")
    mix_path    = os.path.join(subdir, f"mixture_{sample_id}.wav")
    meta_path   = os.path.join(subdir, f"meta_{sample_id}.json")

    # Skip if target or interference missing
    if not (os.path.exists(target_path) and os.path.exists(interf_path)):
        print(f"⚠️ Skipping {folder} — missing files.")
        continue

    # Skip if mixture already exists (so it can resume safely)
    if os.path.exists(mix_path) and os.path.exists(meta_path):
        continue

    # ---- Load audio ----
    t_audio, _ = sf.read(target_path)
    i_audio, _ = sf.read(interf_path)
    t_audio, i_audio = fix_length(t_audio, L), fix_length(i_audio, L)

    # ---- Random geometry ----
    theta_t = random.uniform(0, 180)
    while True:
        theta_i = random.uniform(0, 180)
        if abs(theta_t - theta_i) >= min_angle_sep:
            break
    fov_center = theta_t
    fov_width = random.uniform(beamwidth_range[0], beamwidth_range[1])

    # ---- Compute delays ----
    tau_t = (mic_spacing * np.cos(np.deg2rad(theta_t))) / speed_of_sound
    tau_i = (mic_spacing * np.cos(np.deg2rad(theta_i))) / speed_of_sound

    # ---- Apply delays ----
    t_mic1 = t_audio
    t_mic2 = apply_delay(t_audio, tau_t, fs)
    i_mic1 = i_audio
    i_mic2 = apply_delay(i_audio, tau_i, fs)

    # ---- Mix ----
    mix1, mix2 = mix_signals(t_mic1, t_mic2, i_mic1, i_mic2, sir_db, snr_db)
    mixture = np.stack([mix1, mix2], axis=1)

    # ---- Save mixture and metadata ----
    sf.write(mix_path, mixture, fs)

    meta = {
        "theta_target": round(theta_t, 2),
        "theta_interf": round(theta_i, 2),
        "fov_center": round(fov_center, 2),
        "fov_width": round(fov_width, 2),
        "sir_db": sir_db,
        "snr_db": snr_db,
        "fs": fs,
        "mic_spacing_m": mic_spacing
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

print("✔ Mixture and metadata generation complete.")

# ---------------- RANDOM TRAIN/VAL/TEST SPLIT ----------------
all_samples = sorted([d for d in os.listdir(root_dir) if d.startswith("data_")])
random.shuffle(all_samples)  # RANDOM split each time you run
n = len(all_samples)
train_end, val_end = int(0.8*n), int(0.9*n)

splits = {
    "train": all_samples[:train_end],
    "val":   all_samples[train_end:val_end],
    "test":  all_samples[val_end:]
}

with open(os.path.join(root_dir, "splits.json"), "w") as f:
    json.dump(splits, f, indent=2)

print(f"✔ Random split complete: train={len(splits['train'])}, "
      f"val={len(splits['val'])}, test={len(splits['test'])}")
print("✔ All done.")
