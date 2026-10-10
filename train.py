"""Train / evaluate one experiment variant on DeepSense 6G (Scenarios 1-4 -> Scenario 5).

Example:
    python train.py --variant se --data_root /path/to/data --seed 42
"""
import argparse
import copy
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from src.dataset import DeepSenseMultiModalDataset, data_transforms
from src.models import VARIANTS, MultiModalNet


def accuracy_topk(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        return [correct[:k].reshape(-1).float().sum(0, keepdim=True) for k in topk]


def evaluate(model, loader, criterion, device):
    model.eval()
    loss_sum, c1, c3, c5, n = 0.0, 0, 0, 0, 0
    rows = []
    with torch.no_grad():
        for images, gps, labels in loader:
            images, gps, labels = images.to(device), gps.to(device), labels.to(device)
            out = model(images, gps)
            loss_sum += criterion(out, labels).item() * images.size(0)
            a1, a3, a5 = accuracy_topk(out, labels, topk=(1, 3, 5))
            c1, c3, c5, n = c1 + a1.item(), c3 + a3.item(), c5 + a5.item(), n + images.size(0)
            top3 = out.topk(3, 1, True, True)[1]
            rows += [{"top-1 beam": t[0].item() + 1, "top-2 beam": t[1].item() + 1,
                      "top-3 beam": t[2].item() + 1} for t in top3]
    return loss_sum / n, 100 * c1 / n, 100 * c3 / n, 100 * c5 / n, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=list(VARIANTS), default="se")
    ap.add_argument("--data_root", required=True, help="folder containing Scenario1..Scenario5")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42, help="training seed (the 90/10 split always uses seed 42)")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out_dir", default="results")
    args = ap.parse_args()

    cfg = VARIANTS[args.variant]
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- data -------------------------------------------------------------
    train_all = DeepSenseMultiModalDataset(["Scenario1", "Scenario2", "Scenario3", "Scenario4"],
                                           args.data_root, transform=data_transforms)
    n_train = int(0.9 * len(train_all))
    train_set, val_set = random_split(train_all, [n_train, len(train_all) - n_train],
                                      generator=torch.Generator().manual_seed(42))
    test_set = DeepSenseMultiModalDataset(["Scenario5"], args.data_root, transform=data_transforms)
    print(f"train={len(train_set)} val={len(val_set)} test={len(test_set)}")

    kw = dict(batch_size=args.batch_size, num_workers=args.workers, pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **kw)
    val_loader = DataLoader(val_set, shuffle=False, **kw)
    test_loader = DataLoader(test_set, shuffle=False, **kw)

    # ---- model / optimisation --------------------------------------------
    model = MultiModalNet(args.variant).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=1e-2)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=cfg["max_lr"],
                                              steps_per_epoch=len(train_loader), epochs=args.epochs)

    best_val, best_wts, patience, bad = float("inf"), copy.deepcopy(model.state_dict()), 6, 0
    ckpt = os.path.join(args.out_dir, f"best_{args.variant}.pth")

    for epoch in range(args.epochs):
        model.train()
        for images, gps, labels in train_loader:
            images, gps, labels = images.to(device), gps.to(device), labels.to(device)
            if np.random.rand() < cfg["mixup"]:  # MixUp on images only, labels unchanged
                lam = np.random.beta(1.0, 1.0)
                idx = torch.randperm(images.size(0)).to(device)
                images = lam * images + (1 - lam) * images[idx, :]
            optimizer.zero_grad()
            loss = criterion(model(images, gps), labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            scheduler.step()

        v_loss, v1, v3, v5, _ = evaluate(model, val_loader, criterion, device)
        print(f"[{epoch + 1}/{args.epochs}] val loss {v_loss:.4f} | top-1 {v1:.2f} top-3 {v3:.2f} top-5 {v5:.2f}")
        if v_loss < best_val:
            best_val, best_wts, bad = v_loss, copy.deepcopy(model.state_dict()), 0
            torch.save(best_wts, ckpt)
        else:
            bad += 1
            if bad >= patience:
                print("Early stopping")
                break

    # ---- test on Scenario 5 ----------------------------------------------
    model.load_state_dict(best_wts)
    _, t1, t3, t5, rows = evaluate(model, test_loader, criterion, device)
    pd.DataFrame(rows).to_csv(os.path.join(args.out_dir, f"submission_scenario5_{args.variant}_seed{args.seed}.csv"),
                              index_label="sample_index")
    print(f"\nScenario 5 [{args.variant}, seed {args.seed}]  Top-1 {t1:.2f}  Top-3 {t3:.2f}  Top-5 {t5:.2f}")


if __name__ == "__main__":
    main()
