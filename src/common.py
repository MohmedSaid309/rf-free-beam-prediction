# ============================================================
# common.py  -  shared code for all 8 experiments
# (same data pipeline / split / training protocol as the paper)
# ============================================================
import os, json, copy, time, random, pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import timm

ROOT = "/content"
TRAIN_SCENARIOS = ['Scenario1', 'Scenario2', 'Scenario3', 'Scenario4']
TEST_SCENARIOS = ['Scenario5']
RESULTS_DIR = "/content/results"
os.makedirs(RESULTS_DIR, exist_ok=True)

tfm = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


# ------------------------------------------------------------
# Data: build the list of windows once (cached) - identical logic to your original code
# ------------------------------------------------------------
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


class BeamDataset(Dataset):
    """mode = 'vision' -> loads 5 frames (GPS vector is returned but ignored by the model)
       mode = 'gps'    -> never opens images (fast), images = dummy tensor"""
    def __init__(self, samples, mode='vision', transform=tfm):
        self.samples, self.mode, self.transform = samples, mode, transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_paths, gps_vector, label = self.samples[idx]
        gps = torch.tensor(gps_vector, dtype=torch.float32)
        y = torch.tensor(label, dtype=torch.long)
        if self.mode == 'gps':
            return torch.zeros(1), gps, y
        imgs = [self.transform(Image.open(p).convert('RGB')) for p in img_paths]
        return torch.cat(imgs, dim=0), gps, y


def get_loaders(mode, batch_size, num_workers):
    tr = build_samples(TRAIN_SCENARIOS)
    te = build_samples(TEST_SCENARIOS)
    full = BeamDataset(tr, mode)
    n_tr = int(0.9 * len(full)); n_val = len(full) - n_tr
    tr_sub, val_sub = random_split(full, [n_tr, n_val],
                                   generator=torch.Generator().manual_seed(42))
    test = BeamDataset(te, mode)
    kw = dict(num_workers=num_workers, pin_memory=True,
              persistent_workers=num_workers > 0)
    return (DataLoader(tr_sub, batch_size=batch_size, shuffle=True, **kw),
            DataLoader(val_sub, batch_size=batch_size, shuffle=False, **kw),
            DataLoader(test, batch_size=batch_size, shuffle=False, **kw))


# ------------------------------------------------------------
# Models
# ------------------------------------------------------------
class VisionOnly(nn.Module):
    """timm backbone (15-ch stem, pretrained weights tiled+rescaled by timm) + same head as the paper"""
    def __init__(self, backbone_name, num_classes=64):
        super().__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=True,
                                          in_chans=15, num_classes=0)
        f = self.backbone.num_features
        self.head = nn.Sequential(nn.Linear(f, 512), nn.BatchNorm1d(512), nn.ReLU(),
                                  nn.Dropout(0.3), nn.Linear(512, num_classes))

    def forward(self, images, gps):
        return self.head(self.backbone(images))


class GPSLinear(nn.Module):                       # G1: linear (softmax regression) on 8 features
    def __init__(self, num_classes=64):
        super().__init__()
        self.bn = nn.BatchNorm1d(8)
        self.fc = nn.Linear(8, num_classes)

    def forward(self, images, gps):
        return self.fc(self.bn(gps))


class GPSMLP(nn.Module):                          # G2: 2-hidden-layer MLP (the paper's GPS branch + head)
    def __init__(self, num_classes=64):
        super().__init__()
        self.bn = nn.BatchNorm1d(8)
        self.net = nn.Sequential(
            nn.Linear(8, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, num_classes))

    def forward(self, images, gps):
        return self.net(self.bn(gps))


class _ResBlock(nn.Module):
    def __init__(self, d, p=0.1):
        super().__init__()
        self.f = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d * 2), nn.GELU(),
                               nn.Dropout(p), nn.Linear(d * 2, d))

    def forward(self, x):
        return x + self.f(x)


class GPSResMLP(nn.Module):                       # G3: deeper residual MLP
    def __init__(self, num_classes=64, d=256, blocks=4):
        super().__init__()
        self.bn = nn.BatchNorm1d(8)
        self.inp = nn.Linear(8, d)
        self.blocks = nn.Sequential(*[_ResBlock(d) for _ in range(blocks)])
        self.out = nn.Sequential(nn.LayerNorm(d), nn.Dropout(0.2), nn.Linear(d, num_classes))

    def forward(self, images, gps):
        return self.out(self.blocks(self.inp(self.bn(gps))))


class GPSFTTransformer(nn.Module):                # G4: feature-tokenizer transformer (each of the 8 features = 1 token)
    def __init__(self, num_classes=64, d=64, layers=3, heads=4):
        super().__init__()
        self.bn = nn.BatchNorm1d(8)
        self.w = nn.Parameter(torch.randn(8, d) * 0.02)
        self.b = nn.Parameter(torch.zeros(8, d))
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        layer = nn.TransformerEncoderLayer(d, heads, dim_feedforward=d * 2, dropout=0.1,
                                           activation='gelu', batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, layers)
        self.norm = nn.LayerNorm(d)
        self.out = nn.Linear(d, num_classes)

    def forward(self, images, gps):
        x = self.bn(gps).unsqueeze(-1) * self.w + self.b          # [B, 8, d]
        x = torch.cat([self.cls.expand(x.size(0), -1, -1), x], 1)  # [B, 9, d]
        return self.out(self.norm(self.enc(x)[:, 0]))


# ------------------------------------------------------------
# Training / evaluation
# ------------------------------------------------------------
def topk_correct(output, target, ks=(1, 3, 5)):
    with torch.no_grad():
        _, pred = output.topk(max(ks), 1, True, True)
        correct = pred.t().eq(target.view(1, -1).expand_as(pred.t()))
        return [correct[:k].reshape(-1).float().sum().item() for k in ks]


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    loss, c, n = 0.0, np.zeros(3), 0
    for images, gps, y in loader:
        images, gps, y = images.to(device), gps.to(device), y.to(device)
        out = model(images, gps)
        loss += criterion(out, y).item() * y.size(0)
        c += np.array(topk_correct(out, y)); n += y.size(0)
    return loss / n, *(100 * c / n)


def run_experiment(name, model, mode, epochs=15, batch_size=32, max_lr=1e-3,
                   mixup_p=0.0, patience=6, num_workers=2, weight_decay=1e-2):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, test_loader = get_loaders(mode, batch_size, num_workers)
    print(f"[{name}] train={len(train_loader.dataset)} val={len(val_loader.dataset)} "
          f"test={len(test_loader.dataset)}")

    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[{name}] params = {n_params:.3f} M | mode={mode} | max_lr={max_lr} | mixup={mixup_p}")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=max_lr / 25, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=max_lr,
                                              steps_per_epoch=len(train_loader), epochs=epochs)
    best_val, best_wts, bad, ran = float('inf'), copy.deepcopy(model.state_dict()), 0, 0
    t0 = time.time()

    for ep in range(epochs):
        model.train(); tl, tc, tn = 0.0, 0.0, 0
        for images, gps, y in train_loader:
            images, gps, y = images.to(device), gps.to(device), y.to(device)
            if mode == 'vision' and mixup_p > 0 and np.random.rand() < mixup_p:
                lam = np.random.beta(1.0, 1.0)
                idx = torch.randperm(images.size(0), device=device)
                images = lam * images + (1 - lam) * images[idx]
            optimizer.zero_grad()
            out = model(images, gps)
            loss = criterion(out, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step(); scheduler.step()
            tl += loss.item() * y.size(0); tc += topk_correct(out, y, (1,))[0]; tn += y.size(0)

        vl, v1, v3, v5 = evaluate(model, val_loader, criterion, device)
        ran = ep + 1
        print(f"[{name}] ep {ep+1:02d}/{epochs} | train loss {tl/tn:.4f} T1 {100*tc/tn:.2f}% | "
              f"val loss {vl:.4f} | val T1/T3/T5 {v1:.2f}/{v3:.2f}/{v5:.2f}")
        if vl < best_val:
            best_val, best_wts, bad = vl, copy.deepcopy(model.state_dict()), 0
            torch.save(best_wts, f"{RESULTS_DIR}/{name}_best.pth")
        else:
            bad += 1
            if bad >= patience:
                print(f"[{name}] early stopping"); break

    model.load_state_dict(best_wts)
    _, t1, t3, t5 = evaluate(model, test_loader, criterion, device)   # test evaluated ONCE
    res = dict(name=name, mode=mode, params_M=round(n_params, 3), top1=round(t1, 2),
               top3=round(t3, 2), top5=round(t5, 2), best_val_loss=round(best_val, 4),
               epochs_run=ran, max_lr=max_lr, mixup=mixup_p, minutes=round((time.time()-t0)/60, 1))
    with open(f"{RESULTS_DIR}/{name}.json", "w") as f:
        json.dump(res, f, indent=2)
    print(f"\n=== {name} | Scenario 5 | Top-1 {t1:.2f}% | Top-3 {t3:.2f}% | Top-5 {t5:.2f}% ===")
    return res
