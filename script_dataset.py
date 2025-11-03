import os
import librosa
import soundfile as sf
import numpy as np

# ---------- CONFIG ----------
root_dir = r"D:\Main_SP_Cup\matlab_implementation\LibriSpeech\train-clean-100"
output_dir = r"D:\Main_SP_Cup\matlab_implementation\dataset"
sr_target = 16000          # Target sample rate
chunk_sec = 6              # Chunk length in seconds
chunk_len = sr_target * chunk_sec

# ---------- SETUP ----------
os.makedirs(output_dir, exist_ok=True)

# Get all category folders (e.g., 83, 241, 510, ...)
category_folders = [
    f for f in os.listdir(root_dir)
    if os.path.isdir(os.path.join(root_dir, f))
]

global_counter = 1

# ---------- PROCESS ----------
for cat_idx, category in enumerate(sorted(category_folders), start=1):
    category_path = os.path.join(root_dir, category)
    speaker_folders = [
        f for f in os.listdir(category_path)
        if os.path.isdir(os.path.join(category_path, f))
    ]

    print(f"[{cat_idx}/{len(category_folders)}] Category: {category} "
          f"({len(speaker_folders)} speaker folders)")

    # Process each speaker folder under this category
    for speaker in speaker_folders:
        speaker_path = os.path.join(category_path, speaker)

        # Collect only .flac files
        flac_files = sorted([
            f for f in os.listdir(speaker_path)
            if f.lower().endswith('.flac')
        ])
        if not flac_files:
            continue

        full_audio = np.array([], dtype=np.float32)

        for file in flac_files:
            path = os.path.join(speaker_path, file)
            try:
                y, sr = librosa.load(path, sr=sr_target, mono=True)
                full_audio = np.concatenate((full_audio, y))
            except Exception as e:
                print(f"  ⚠️ Skipping {path}: {e}")

        total_samples = len(full_audio)
        total_chunks = total_samples // chunk_len
        if total_chunks == 0:
            continue

        for i in range(total_chunks):
            start = i * chunk_len
            end = start + chunk_len
            chunk = full_audio[start:end]

            out_name = f"data_{global_counter:06d}.flac"
            out_path = os.path.join(output_dir, out_name)
            sf.write(out_path, chunk, sr_target, format='FLAC')
            global_counter += 1

    print(f"  ✅ Finished category {category}\n")

print(f"✅ All done! Total saved chunks: {global_counter - 1}")
