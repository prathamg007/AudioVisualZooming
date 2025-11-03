# evaluate_pipeline_pt_compare.py
import os, torch, numpy as np, soundfile as sf, json
import torch.nn as nn
from tqdm import tqdm
from train import DualMaskGRU, NeuralBeamformer, FS, NFFT, HOP
from pystoi import stoi
from pesq import pesq
from scipy.signal import get_window
import time

# ---------------- STFT + ISTFT ----------------
def stft(x):
    WIN = torch.tensor(get_window("hann", NFFT, fftbins=True), dtype=torch.float32)
    return torch.stft(x, n_fft=NFFT, hop_length=HOP, window=WIN.to(x.device), return_complex=True)

def istft(X, length):
    WIN = torch.tensor(get_window("hann", NFFT, fftbins=True), dtype=torch.float32)
    return torch.istft(X, n_fft=NFFT, hop_length=HOP, window=WIN.to(X.device), length=length)

# ---------------- SI-SDR ----------------
def si_sdr_loss(estimate, target, eps=1e-8):
    t_energy = torch.sum(target**2) + eps
    scale = torch.sum(estimate * target) / t_energy
    s_target = scale * target
    e_noise = estimate - s_target
    num = torch.sum(s_target**2) + eps
    den = torch.sum(e_noise**2) + eps
    return -10 * torch.log10(num / den + eps)

# ---------------- Evaluation ----------------
def evaluate_single(test_dir, model_dir, test_name):
    sid = test_name.split("_")[1]
    subdir = os.path.join(test_dir, test_name)
    mix_path = os.path.join(subdir, f"mixture_{sid}.wav")
    feat_path = os.path.join(subdir, f"features_{sid}.npz")
    target_path = os.path.join(subdir, f"target_{sid}.flac")
    out_path = os.path.join(subdir, f"processed_signal.wav")

    # ---------- Load mixture / target ----------
    mix, fs = sf.read(mix_path)
    tgt, fs2 = sf.read(target_path)
    assert fs == FS, f"Sample rate mismatch: {fs} != {FS}"

    mix_t = torch.from_numpy(mix).unsqueeze(0).float().to(device)
    Y1c, Y2c = stft(mix_t[:,:,0]), stft(mix_t[:,:,1])
    Y = torch.stack([Y1c, Y2c], dim=-1)
    F, Tstft = Y1c.shape[1], Y1c.shape[2]

    # ---------- Load pre-computed features ----------
    with np.load(feat_path) as Z:
        feats = np.stack([Z["LPS"], Z["cosIPD"], Z["sinIPD"], Z["DFin"], Z["DFout"]], axis=-1)
    feats_t = torch.from_numpy(feats).unsqueeze(0).float().to(device)
    if feats_t.shape[2] != Tstft:
        feats_t = nn.functional.interpolate(
            feats_t.permute(0,3,1,2), size=(F,Tstft), mode="nearest"
        ).permute(0,2,3,1)

    # ---------- Stage A ----------
    with torch.no_grad():
        Min, Mout = netA(feats_t)

    # ---------- Stage B ----------
    Yin, Yout = Y * Min.unsqueeze(-1), Y * Mout.unsqueeze(-1)
    with torch.no_grad():
        W = netB(Yin, Yout)
    Wc = W[...,:2] + 1j * W[...,2:]
    Yhat = torch.sum(torch.conj(Wc) * Y, dim=-1)

    # ---------- ISTFT ----------
    x_hat = istft(Yhat, length=mix.shape[0])
    sf.write(out_path, x_hat.squeeze().cpu().numpy(), FS)

    # ---------- Metrics ----------
    x_hat_np = x_hat.cpu().numpy().squeeze()
    tgt_np = tgt[:len(x_hat_np)]
    mix_mono = np.mean(mix, axis=1)[:len(x_hat_np)]

    # -- model output vs target --
    sdr_out = -si_sdr_loss(torch.tensor(x_hat_np), torch.tensor(tgt_np)).item()
    pesq_out = pesq(FS, tgt_np, x_hat_np, 'wb')
    stoi_out = stoi(tgt_np, x_hat_np, FS, extended=False)

    # -- mixture (input) vs target --
    sdr_in = -si_sdr_loss(torch.tensor(mix_mono), torch.tensor(tgt_np)).item()
    pesq_in = pesq(FS, tgt_np, mix_mono, 'wb')
    stoi_in = stoi(tgt_np, mix_mono, FS, extended=False)

    print(f"\n {test_name}")
    print(f"🔹 Input (Mixture):   SI-SDR={sdr_in:.2f} dB | PESQ={pesq_in:.2f} | STOI={stoi_in:.2f}")
    print(f"🔸 Output (Enhanced): SI-SDR={sdr_out:.2f} dB | PESQ={pesq_out:.2f} | STOI={stoi_out:.2f}")

    return sdr_in, pesq_in, stoi_in, sdr_out, pesq_out, stoi_out

# ---------------- Entry ----------------
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Evaluate SP-Cup pipeline with input/output comparison")
    ap.add_argument("--model_dir", type=str, required=True)
    ap.add_argument("--test_dir", type=str, required=True)
    ap.add_argument("--test_name", type=str, required=True)
    args = ap.parse_args()

    start_time = time.time()

    device = "cpu"   # run on CPU
    print(f"\n Evaluation on device: {device}")

    # Load models
    ckpt = torch.load(os.path.join(args.model_dir, "best.pt"), map_location=device)
    netA = DualMaskGRU().to(device)
    netB = NeuralBeamformer().to(device)
    netA.load_state_dict(ckpt["stageA"])
    netB.load_state_dict(ckpt["stageB"])
    netA.eval(); netB.eval()
    print(f" Loaded model from {os.path.join(args.model_dir, 'best.pt')}")

    # Evaluate single folder
    sdr_in, pesq_in, stoi_in, sdr_out, pesq_out, stoi_out = evaluate_single(
        args.test_dir, args.model_dir, args.test_name
    )

    end_time = time.time()  # end timer
    elapsed = end_time - start_time

    print("\n Summary:")
    print(f"Input  — SI-SDR={sdr_in:.2f} dB | PESQ={pesq_in:.2f} | STOI={stoi_in:.2f}")
    print(f"Output — SI-SDR={sdr_out:.2f} dB | PESQ={pesq_out:.2f} | STOI={stoi_out:.2f}")
    print(f"\n⏱️ Total runtime: {elapsed:.2f} seconds")