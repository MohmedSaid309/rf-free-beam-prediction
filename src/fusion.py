# ============================================================
# Fusion experiments: ConvNeXt-B (vision) + GPS-Linear, concat + attention blocks (standalone)
# Same protocol/epochs/hyperparameters as before. Speed-ups:
#   1) all unique frames are resized to 224x224 ONCE and cached as uint8 (no JPEG decode per epoch)
#   2) normalization + MixUp done on the GPU
#   3) mixed precision (AMP) + channels_last
# ============================================================
import os, json, copy, time, random, pickle
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import timm
from tqdm.auto import tqdm

ROOT = "/content"
TRAIN_SCENARIOS = ['Scenario1', 'Scenario2', 'Scenario3', 'Scenario4']
TEST_SCENARIOS = ['Scenario5']
RESULTS_DIR = "/content/results"
os.makedirs(RESULTS_DIR, exist_ok=True)

NAME = "V4_ConvNeXt_B"
BACKBONE = "convnext_base"
EPOCHS, BATCH, MAX_LR, MIXUP_P, PATIENCE, WD = 15, 32, 1e-3, 0.3, 6, 1e-2


def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


# ---------------- 1) windows (same logic as your original code) ----------------
def build_samples(scenario_list, root=ROOT, stride=1):
    cache = f"{root}/cache_{'_'.join(scenario_list)}.pkl"
    if os.path.exists(cache):
        with open(cache, 'rb') as f:
            return pickle.load(f)
    samples = []
    for scenario in scenario_list:
        scenario_dir = f"{root}/{scenario}/{scenario}"
        csv_file = f"{scenario_dir}/{scenario.lower()}.csv"
        if not os.path.exists(csv_file):
            continue
        df = pd.read_csv(csv_file).sort_values('index').reset_index(drop=True)
        W = 5
        for i in range(0, len(df) - W + 1, stride):
            window = df.iloc[i:i + W]
            img_paths = [f"{scenario_dir}/{p.strip('./')}" for p in window['unit1_rgb'].values]
            gps_col = 'unit2_loc' if 'unit2_loc' in window.columns else 'unit1_loc'
            gps_paths = [f"{scenario_dir}/{window.iloc[j][gps_col].strip('./')}" for j in range(2)]
            try:
                coords = []
                for p in gps_paths:
                    with open(p, 'r') as f:
                        lines = f.readlines()
                    coords.append((float(lines[0].strip()), float(lines[1].strip())))
                d_lat = coords[1][0] - coords[0][0]
                d_lon = coords[1][1] - coords[0][1]
                gps_vector = [coords[0][0], coords[0][1], coords[1][0], coords[1][1],
                              d_lat, d_lon, float(np.sqrt(d_lat**2 + d_lon**2)),
                              float(np.arctan2(d_lon, d_lat))]
            except Exception:
                continue
            last_row = window.iloc[-1]
            beam_col = [c for c in df.columns if 'beam_index' in c.lower()]
            label = int(last_row[beam_col[0]]) - 1 if beam_col else 0
            if 0 <= label < 64:
                samples.append((img_paths, gps_vector, label))
    with open(cache, 'wb') as f:
        pickle.dump(samples, f)
    return samples


# ---------------- 2) one-time uint8 frame cache ----------------
FRAMES_NPY = f"{ROOT}/frames224_uint8.npy"
FRAMES_IDX = f"{ROOT}/frames224_index.pkl"
FRAMES_OK = f"{ROOT}/frames224.done"


def build_frame_cache(all_samples):
    paths = sorted({p for s in all_samples for (imgs, _, _) in s for p in imgs})
    if os.path.exists(FRAMES_OK) and os.path.exists(FRAMES_IDX):
        with open(FRAMES_IDX, 'rb') as f:
            if pickle.load(f) == paths:
                print(f"frame cache found ({len(paths)} frames)")
                return {p: i for i, p in enumerate(paths)}
    print(f"building frame cache: {len(paths)} frames (one time only)...")
    arr = np.lib.format.open_memmap(FRAMES_NPY, mode='w+', dtype=np.uint8,
                                    shape=(len(paths), 224, 224, 3))

    def load(i):
        img = Image.open(paths[i]).convert('RGB').resize((224, 224), Image.BILINEAR)
        arr[i] = np.asarray(img)

    with ThreadPoolExecutor(8) as ex:
        list(tqdm(ex.map(load, range(len(paths))), total=len(paths)))
    arr.flush(); del arr
    with open(FRAMES_IDX, 'wb') as f:
        pickle.dump(paths, f)
    open(FRAMES_OK, 'w').close()
    return {p: i for i, p in enumerate(paths)}


class FastBeamDataset(Dataset):
    def __init__(self, samples, path2idx):
        self.items = [([path2idx[p] for p in imgs], gps, y) for imgs, gps, y in samples]
        self.arr = None

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        if self.arr is None:                       # open memmap lazily inside each worker
            self.arr = np.load(FRAMES_NPY, mmap_mode='r')
        idxs, gps, y = self.items[i]
        frames = torch.from_numpy(np.stack([self.arr[j] for j in idxs]))   # [5,224,224,3] uint8
        return frames, torch.tensor(gps, dtype=torch.float32), torch.tensor(y, dtype=torch.long)


MEAN = torch.tensor([0.485, 0.456, 0.406] * 5).view(1, 15, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225] * 5).view(1, 15, 1, 1)


def prep(frames_u8, device):
    x = frames_u8.to(device, non_blocking=True)                    # [B,5,H,W,3]
    B = x.size(0)
    x = x.permute(0, 1, 4, 2, 3).reshape(B, 15, 224, 224).float().div_(255)
    x = (x - MEAN.to(device)) / STD.to(device)
    return x.contiguous(memory_format=torch.channels_last)


# ---------------- 3) model pieces ----------------
import sys, math


class GPSBranch(nn.Module):
    """G1 (best GPS-only model): BN -> Linear, as 32-d embedding"""
    def __init__(self, out=32):
        super().__init__()
        self.bn = nn.BatchNorm1d(8)
        self.fc = nn.Linear(8, out)

    def forward(self, g):
        return self.fc(self.bn(g))


class SEGate(nn.Module):                 # proposed: FC bottleneck (r=4) + sigmoid
    def __init__(self, d, r=4):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, d // r), nn.ReLU(), nn.Linear(d // r, d), nn.Sigmoid())

    def forward(self, z):
        return z * self.net(z)


class UFOGate(nn.Module):                # UFO-style: softmax q/k -> scalar context scales value proj -> sigmoid
    def __init__(self, d, r=4):
        super().__init__()
        self.q = nn.Linear(d, d // r); self.k = nn.Linear(d, d // r); self.v = nn.Linear(d, d)

    def forward(self, z):
        q = torch.softmax(self.q(z), dim=-1)
        k = torch.softmax(self.k(z), dim=-1)
        ctx = (q * k).sum(-1, keepdim=True) * q.size(-1)            # scalar context per sample
        return z * torch.sigmoid(ctx * self.v(z))


class SKGate1D(nn.Module):               # SK (1-D): conv k=3 / k=5 over channels, softmax selection from FC bottleneck, residual
    def __init__(self, d, r=8):
        super().__init__()
        self.c3 = nn.Conv1d(1, 1, 3, padding=1); self.c5 = nn.Conv1d(1, 1, 5, padding=2)
        self.fc = nn.Sequential(nn.Linear(d, d // r), nn.ReLU())
        self.a1 = nn.Linear(d // r, d); self.a2 = nn.Linear(d // r, d)

    def forward(self, z):
        x = z.unsqueeze(1)
        u1, u2 = self.c3(x).squeeze(1), self.c5(x).squeeze(1)
        s = self.fc(u1 + u2)
        w = torch.softmax(torch.stack([self.a1(s), self.a2(s)], 0), dim=0)
        return z + w[0] * u1 + w[1] * u2


class ECAGate(nn.Module):                # ECA: one 1-D conv across channels (adaptive k), sigmoid
    def __init__(self, d, gamma=2, b=1):
        super().__init__()
        k = int(abs(math.log2(d) / gamma + b / gamma)); k = k if k % 2 else k + 1
        self.conv = nn.Conv1d(1, 1, k, padding=k // 2, bias=False)

    def forward(self, z):
        return z * torch.sigmoid(self.conv(z.unsqueeze(1)).squeeze(1))


GATES = {"C0": None, "A1_SE": SEGate, "A2_UFO": UFOGate, "A3_SK": SKGate1D, "A4_ECA": ECAGate}


class FusionNet(nn.Module):
    def __init__(self, gate=None, num_classes=64):
        super().__init__()
        self.backbone = timm.create_model(BACKBONE, pretrained=True, in_chans=15, num_classes=0)
        self.gps = GPSBranch(32)
        d = self.backbone.num_features + 32                          # 1056
        self.gate = gate(d) if gate is not None else nn.Identity()
        self.head = nn.Sequential(nn.Linear(d, 512), nn.BatchNorm1d(512), nn.ReLU(),
                                  nn.Dropout(0.3), nn.Linear(512, num_classes))

    def forward(self, images, gps):
        z = torch.cat([self.backbone(images), self.gps(gps)], dim=1)
        return self.head(self.gate(z))


def topk_correct(output, target, ks=(1, 3, 5)):
    with torch.no_grad():
        _, pred = output.topk(max(ks), 1, True, True)
        correct = pred.t().eq(target.view(1, -1).expand_as(pred.t()))
        return [correct[:k].reshape(-1).float().sum().item() for k in ks]


@torch.no_grad()
def evaluate(model, loader, criterion, device, amp_on, amp_dtype):
    model.eval()
    loss, c, n = 0.0, np.zeros(3), 0
    for frames, gps, y in loader:
        x, gps, y = prep(frames, device), gps.to(device), y.to(device)
        with torch.autocast('cuda', dtype=amp_dtype, enabled=amp_on):
            out = model(x, gps)
        loss += criterion(out.float(), y).item() * y.size(0)
        c += np.array(topk_correct(out.float(), y)); n += y.size(0)
    return loss / n, *(100 * c / n)


# ---------------- 4) main ----------------
if __name__ == "__main__":
    exp = sys.argv[1] if len(sys.argv) > 1 else "C0"
    assert exp in GATES, f"choose one of {list(GATES)}"
    NAME = f"F_{exp}"
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_cuda = device.type == "cuda"
    torch.backends.cudnn.benchmark = True
    cc = torch.cuda.get_device_capability()[0] if use_cuda else 0
    amp_dtype = torch.bfloat16 if cc >= 8 else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=(use_cuda and amp_dtype == torch.float16))
    print(f"[{NAME}] device={device} | AMP {amp_dtype}")

    tr_s = build_samples(TRAIN_SCENARIOS); te_s = build_samples(TEST_SCENARIOS)
    path2idx = build_frame_cache([tr_s, te_s])
    full = FastBeamDataset(tr_s, path2idx)
    n_tr = int(0.9 * len(full)); n_val = len(full) - n_tr
    tr_sub, val_sub = random_split(full, [n_tr, n_val], generator=torch.Generator().manual_seed(42))
    test_ds = FastBeamDataset(te_s, path2idx)
    kw = dict(num_workers=2, pin_memory=True, persistent_workers=True, prefetch_factor=4)
    train_loader = DataLoader(tr_sub, batch_size=BATCH, shuffle=True, **kw)
    val_loader = DataLoader(val_sub, batch_size=BATCH, shuffle=False, **kw)
    test_loader = DataLoader(test_ds, batch_size=BATCH, shuffle=False, **kw)

    model = FusionNet(GATES[exp]).to(device).to(memory_format=torch.channels_last)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    gate_params = sum(p.numel() for p in model.gate.parameters())
    print(f"[{NAME}] params = {n_params:.3f} M | gate params = {gate_params:,}")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=MAX_LR / 25, weight_decay=WD)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=MAX_LR,
                                              steps_per_epoch=len(train_loader), epochs=EPOCHS)
    best_val, best_wts, bad, ran = float('inf'), copy.deepcopy(model.state_dict()), 0, 0
    t0 = time.time()

    for ep in range(EPOCHS):
        model.train(); tl, tc, tn = 0.0, 0.0, 0
        te0 = time.time()
        for frames, gps, y in tqdm(train_loader, desc=f"ep {ep+1}/{EPOCHS}", leave=False):
            x, gps, y = prep(frames, device), gps.to(device, non_blocking=True), y.to(device, non_blocking=True)
            if np.random.rand() < MIXUP_P:                          # MixUp on images only (as in paper)
                lam = np.random.beta(1.0, 1.0)
                idx = torch.randperm(x.size(0), device=device)
                x = lam * x + (1 - lam) * x[idx]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=amp_dtype, enabled=use_cuda):
                out = model(x, gps)
            loss = criterion(out.float(), y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer); scaler.update(); scheduler.step()
            tl += loss.item() * y.size(0); tc += topk_correct(out.float(), y, (1,))[0]; tn += y.size(0)

        vl, v1, v3, v5 = evaluate(model, val_loader, criterion, device, use_cuda, amp_dtype)
        ran = ep + 1
        print(f"[{NAME}] ep {ep+1:02d}/{EPOCHS} | {(time.time()-te0)/60:.1f} min | train loss {tl/tn:.4f} "
              f"T1 {100*tc/tn:.2f}% | val loss {vl:.4f} | val T1/T3/T5 {v1:.2f}/{v3:.2f}/{v5:.2f}")
        if vl < best_val:
            best_val, best_wts, bad = vl, copy.deepcopy(model.state_dict()), 0
            torch.save(best_wts, f"{RESULTS_DIR}/{NAME}_best.pth")
        else:
            bad += 1
            if bad >= PATIENCE:
                print("early stopping"); break

    model.load_state_dict(best_wts)
    _, t1, t3, t5 = evaluate(model, test_loader, criterion, device, use_cuda, amp_dtype)  # test ONCE
    res = dict(name=NAME, mode='fusion', params_M=round(n_params, 3), gate_params=gate_params,
               top1=round(t1, 2), top3=round(t3, 2), top5=round(t5, 2),
               best_val_loss=round(best_val, 4), epochs_run=ran, max_lr=MAX_LR, mixup=MIXUP_P,
               minutes=round((time.time() - t0) / 60, 1))
    with open(f"{RESULTS_DIR}/{NAME}.json", "w") as f:
        json.dump(res, f, indent=2)
    print(f"\n=== {NAME} | Scenario 5 | Top-1 {t1:.2f}% | Top-3 {t3:.2f}% | Top-5 {t5:.2f}% ===")
