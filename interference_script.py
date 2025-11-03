import os
import librosa
import soundfile as sf
import numpy as np
from tqdm import tqdm

# ---------- CONFIG ----------
musan_root = r"D:\Main_SP_Cup\matlab_implementation\musan"
dataset_root = r"D:\Main_SP_Cup\matlab_implementation\dataset"
sr_target = 16000
chunk_sec = 6
chunk_len = sr_target * chunk_sec

# ---------- COLLECT ALL WAV FILES ----------
wav_files = []
for root, _, files in os.walk(musan_root):
    for f in files:
        if f.lower().endswith(".wav"):
            wav_files.append(os.path.join(root, f))

print(f"Found {len(wav_files)} .wav interference files.\n")

# ---------- LOAD AND CHUNK ----------
interference_chunks = []

for wav_path in tqdm(wav_files, desc="Processing interference audios"):
    try:
        y, sr = librosa.load(wav_path, sr=sr_target, mono=True)
        if len(y) < chunk_len:
            continue
        total_chunks = len(y) // chunk_len
        for i in range(total_chunks):
            start = i * chunk_len
            end = start + chunk_len
            chunk = y[start:end]
            interference_chunks.append(chunk)
    except Exception as e:
        print(f"⚠️ Error reading {wav_path}: {e}")

print(f"\n✅ Generated {len(interference_chunks)} interference chunks.\n")

# ---------- SAVE TO DATASET ----------
data_folders = sorted([
    f for f in os.listdir(dataset_root)
    if f.startswith("data_") and os.path.isdir(os.path.join(dataset_root, f))
])

print(f"Found {len(data_folders)} target folders to fill.\n")

pairs = min(len(data_folders), len(interference_chunks))
print(f"➡️ Filling {pairs} folders with interference files.\n")

for i in tqdm(range(pairs), desc="Saving interference chunks"):
    folder_name = data_folders[i]
    folder_path = os.path.join(dataset_root, folder_name)
    idx = int(folder_name.split("_")[-1])  # Extract number (e.g. 1 from data_000001)

    out_name = f"interference_{idx:06d}.flac"
    out_path = os.path.join(folder_path, out_name)

    sf.write(out_path, interference_chunks[i], sr_target, format="FLAC")

print(f"\n✅ Done! Interference audio added to {pairs} folders.")
print(f"🔹 Remaining target folders ({len(data_folders) - pairs}) left empty (not enough interference).")
