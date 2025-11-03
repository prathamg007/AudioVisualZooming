# train_fov_end2end_fast.py
import os, json, random, gc
import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# --------------------------- constants ---------------------------
FS   = 16000
NFFT = 512
HOP  = 256
WIN  = None
EPS  = 1e-8

# --------------------------- DSP helpers ---------------------------
def stft(x: torch.Tensor) -> torch.Tensor:
    return torch.stft(x, n_fft=NFFT, hop_length=HOP, window=WIN.to(x.device),
                      return_complex=True)

def istft(Xc: torch.Tensor, length: int) -> torch.Tensor:
    return torch.istft(Xc, n_fft=NFFT, hop_length=HOP, window=WIN.to(Xc.device),
                       length=length)

def si_sdr_loss(estimate: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    t_energy = torch.sum(target**2, dim=1, keepdim=True) + eps
    scale = torch.sum(estimate * target, dim=1, keepdim=True) / t_energy
    s_target = scale * target
    e_noise  = estimate - s_target
    num = torch.sum(s_target**2, dim=1) + eps
    den = torch.sum(e_noise**2,  dim=1) + eps
    return -10.0 * torch.log10(num / den + eps).mean()

# --------------------------- dataset ---------------------------
class FOVFeatDataset(Dataset):
    def __init__(self, root: str, split: str):
        super().__init__()
        self.root = root
        with open(os.path.join(root, "splits.json"), "r") as f:
            sp = json.load(f)
        self.ids = sp[split]

    def __len__(self): return len(self.ids)

    def __getitem__(self, idx: int):
        folder = self.ids[idx]
        sid = folder.split('_')[1]
        sub = os.path.join(self.root, folder)
        feat_path = os.path.join(sub, f"features_{sid}.npz")
        mix_path  = os.path.join(sub, f"mixture_{sid}.wav")
        targ_path = os.path.join(sub, f"target_{sid}.flac")

        if not (os.path.isfile(feat_path) and os.path.isfile(mix_path) and os.path.isfile(targ_path)):
            return None

        try:
            with np.load(feat_path, mmap_mode="r", allow_pickle=False) as Z:
                feats = np.stack([
                    np.asarray(Z["LPS"], dtype=np.float32),
                    np.asarray(Z["cosIPD"], dtype=np.float32),
                    np.asarray(Z["sinIPD"], dtype=np.float32),
                    np.asarray(Z["DFin"], dtype=np.float32),
                    np.asarray(Z["DFout"], dtype=np.float32),
                ], axis=-1)  # [F,T,5]

            mix, fs  = sf.read(mix_path,  dtype="float32", always_2d=True)
            tgt, fs2 = sf.read(targ_path, dtype="float32")
            if fs != FS or fs2 != FS or mix.ndim != 2 or mix.shape[1] != 2:
                return None
        except Exception:
            return None

        L = min(len(mix), len(tgt))
        mix_t  = torch.from_numpy(mix[:L])
        tgt_t  = torch.from_numpy(tgt[:L])
        feats_t= torch.from_numpy(feats[:, : (L//HOP + 1), :])
        return {"feats": feats_t, "mix": mix_t, "target": tgt_t, "length": L}

def collate(batch):
    batch = [b for b in batch if b is not None]
    if not batch: return None
    Lmax = max(b["length"] for b in batch)
    mixs, targs, feats = [], [], []
    for b in batch:
        pad = Lmax - b["length"]
        if pad > 0:
            z2 = torch.zeros(pad, 2, dtype=b["mix"].dtype)
            z1 = torch.zeros(pad,    dtype=b["target"].dtype)
            mixs.append(torch.cat([b["mix"], z2], dim=0))
            targs.append(torch.cat([b["target"], z1], dim=0))
        else:
            mixs.append(b["mix"]); targs.append(b["target"])
        feats.append(b["feats"])
    return {"mix": torch.stack(mixs,0), "target": torch.stack(targs,0), "feats": torch.stack(feats,0)}

# --------------------------- models ---------------------------
class DualMaskGRU(nn.Module):
    def __init__(self, in_ch=5, hid=128):
        super().__init__()
        self.inp   = nn.Conv2d(in_ch, hid, 1)
        self.gru1  = nn.GRU(hid, hid, batch_first=True, bidirectional=True)
        self.gru2  = nn.GRU(2*hid, hid, batch_first=True, bidirectional=True)
        self.head_in  = nn.Sequential(nn.Linear(2*hid, hid), nn.ReLU(), nn.Linear(hid, 1), nn.Sigmoid())
        self.head_out = nn.Sequential(nn.Linear(2*hid, hid), nn.ReLU(), nn.Linear(hid, 1), nn.Sigmoid())

    def forward(self, feats):
        B,F,T,C = feats.shape
        x = self.inp(feats.permute(0,3,1,2))   # [B,H,F,T]
        x = x.permute(0,2,3,1).reshape(B*F, T, -1)
        y,_ = self.gru1(x)
        y,_ = self.gru2(y)
        m_in  = self.head_in(y).reshape(B,F,T)
        m_out = self.head_out(y).reshape(B,F,T)
        return m_in, m_out

class NeuralBeamformer(nn.Module):
    def __init__(self, M=2, hid=128):
        super().__init__()
        self.M = M
        in_ch = 4*M
        self.fc1 = nn.Linear(in_ch, hid)
        self.gru = nn.GRU(hid, hid, batch_first=True, bidirectional=True)
        self.ln  = nn.LayerNorm(2*hid)
        self.act = nn.LeakyReLU(0.1)
        self.fc2 = nn.Linear(2*hid, 2*M)

    def forward(self, Yin, Yout):
        M = self.M
        Yin_r, Yin_i   = Yin.real, Yin.imag
        Yout_r, Yout_i = Yout.real, Yout.imag
        phi = torch.cat([Yin_r, Yin_i, Yout_r, Yout_i], dim=-1)
        B,F,T,_ = phi.shape
        phi = phi.permute(0,2,1,3).reshape(B*T, F, 4*M)
        x = self.fc1(phi)
        y,_ = self.gru(x)
        y = self.act(self.ln(y))
        w = self.fc2(y).reshape(B, T, F, 2*M).permute(0,2,1,3)
        return w

# --------------------------- training ---------------------------
def train(args):
    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    print(f"\n Training on device: {device}")
    if device.type == "cuda":
        torch.cuda.set_device(0)
        print(f"GPU: {torch.cuda.get_device_name(0)} | CUDA runtime: {torch.version.cuda}")

    global WIN
    WIN = torch.hann_window(NFFT).to(device)

    tr_ds, va_ds = FOVFeatDataset(args.root,"train"), FOVFeatDataset(args.root,"val")
    tr_dl = DataLoader(tr_ds, batch_size=args.batch, shuffle=True,  num_workers=2, pin_memory=True,  collate_fn=collate)
    va_dl = DataLoader(va_ds, batch_size=args.batch, shuffle=False, num_workers=2, pin_memory=True,  collate_fn=collate)

    netA, netB = DualMaskGRU().to(device), NeuralBeamformer().to(device)
    params = list(netA.parameters()) + list(netB.parameters())

    opt = torch.optim.Adam(params, lr=args.lr, weight_decay=1e-6)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=3)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type=="cuda"))

    os.makedirs(args.out, exist_ok=True)
    ckpt_path = os.path.join(args.out, "last.pt")
    start_ep, best_val, early_bad = 1, float('inf'), 0
    if os.path.exists(ckpt_path) and not args.no_resume:
        ck = torch.load(ckpt_path, map_location="cpu")
        netA.load_state_dict(ck["stageA"]); netB.load_state_dict(ck["stageB"])
        opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
        start_ep = ck["epoch"]+1; best_val = ck.get("best_val_loss", best_val)
        print(f"Resumed from {ckpt_path} @ epoch {start_ep-1}")

    for ep in range(start_ep, args.epochs+1):
        netA.train(); netB.train()
        tr_loss, n_batches = 0.0, 0
        pbar = tqdm(tr_dl, desc=f"Epoch {ep}/{args.epochs} [train]", dynamic_ncols=True)
        for batch in pbar:
            if batch is None: continue
            mix = batch["mix"].to(device)
            target = batch["target"].to(device)
            feats = batch["feats"].to(device)
            B,T,_ = mix.shape

            with torch.cuda.amp.autocast(dtype=torch.float16):
                Y1c, Y2c = stft(mix[:,:,0]), stft(mix[:,:,1])
                Y = torch.stack([Y1c,Y2c], dim=-1)
                F,Tstft = Y1c.shape[1], Y1c.shape[2]
                featsT = feats if feats.shape[2]==Tstft else nn.functional.interpolate(
                    feats.permute(0,3,1,2), size=(F,Tstft), mode="nearest").permute(0,2,3,1)
                Min, Mout = netA(featsT)
                Yin, Yout = Y*Min.unsqueeze(-1), Y*Mout.unsqueeze(-1)
                W = netB(Yin, Yout)
                Wc = W[...,:2] + 1j*W[...,2:]
                Yhat = torch.sum(torch.conj(Wc)*Y, dim=-1)
                x_hat = istft(Yhat, length=T)
                loss = si_sdr_loss(x_hat, target)

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(params, 10.0)
            scaler.step(opt)
            scaler.update()

            tr_loss += float(loss.detach().cpu())
            n_batches += 1
            pbar.set_postfix({"loss": f"{(tr_loss/n_batches):.4f}"})

            del mix,target,feats,featsT,Y1c,Y2c,Y,Min,Mout,Yin,Yout,W,Wc,Yhat,x_hat,loss
            torch.cuda.empty_cache(); gc.collect()

        tr_loss /= max(1,n_batches)

        # validation
        netA.eval(); netB.eval()
        va_loss = 0.0
        with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.float16):
            for batch in tqdm(va_dl, desc=f"Epoch {ep}/{args.epochs} [val]", dynamic_ncols=True):
                if batch is None: continue
                mix = batch["mix"].to(device); target = batch["target"].to(device); feats = batch["feats"].to(device)
                B,T,_ = mix.shape
                Y1c, Y2c = stft(mix[:,:,0]), stft(mix[:,:,1])
                Y = torch.stack([Y1c,Y2c], dim=-1)
                F,Tstft = Y1c.shape[1], Y1c.shape[2]
                featsT = feats if feats.shape[2]==Tstft else nn.functional.interpolate(
                    feats.permute(0,3,1,2), size=(F,Tstft), mode="nearest").permute(0,2,3,1)
                Min, Mout = netA(featsT)
                Yin, Yout = Y*Min.unsqueeze(-1), Y*Mout.unsqueeze(-1)
                W = netB(Yin, Yout)
                Wc = W[...,:2] + 1j*W[...,2:]
                Yhat = torch.sum(torch.conj(Wc)*Y, dim=-1)
                x_hat = istft(Yhat, length=T)
                loss = si_sdr_loss(x_hat, target)
                va_loss += float(loss.detach().cpu())

                del mix,target,feats,featsT,Y1c,Y2c,Y,Min,Mout,Yin,Yout,W,Wc,Yhat,x_hat,loss
                torch.cuda.empty_cache(); gc.collect()

        va_loss /= max(1,len(va_dl))
        print(f"Epoch {ep}: train {tr_loss:.4f} | val {va_loss:.4f} | lr {opt.param_groups[0]['lr']:.2e}")

        sched.step(va_loss)
        save_d = {"epoch":ep,"stageA":netA.state_dict(),"stageB":netB.state_dict(),
                  "best_val_loss":best_val,"opt":opt.state_dict(),"sched":sched.state_dict()}
        torch.save(save_d, os.path.join(args.out,"last.pt"))
        if va_loss < best_val - 1e-4:
            best_val, early_bad = va_loss, 0
            torch.save(save_d, os.path.join(args.out,"best.pt"))
            print(f"✔ Saved best (val={best_val:.4f})")
        else: early_bad += 1
        if args.early_stop and early_bad >= args.patience:
            print(f" Early stopping at epoch {ep}")
            break

# --------------------------- entry ---------------------------
if __name__ == "__main__":
    import argparse
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, required=True)
    ap.add_argument("--out", type=str, default="ckpts_fov_fast")
    ap.add_argument("--epochs", type=int, default=15)   # enough for 5k dataset
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--no_resume", action="store_true")
    ap.add_argument("--early_stop", action="store_true")
    ap.add_argument("--patience", type=int, default=3)
    args = ap.parse_args()
    train(args)
