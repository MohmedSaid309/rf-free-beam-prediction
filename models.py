import math

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------------
# Attention blocks (all operate on the fused [B, C] vector)
# ----------------------------------------------------------------------------
class SEFusionAttention(nn.Module):
    """Proposed SE-style feature-wise gate (reduction 4)."""

    def __init__(self, dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(dim, dim // 4), nn.ReLU(),
            nn.Linear(dim // 4, dim), nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.attention(x)


class UFOAttention(nn.Module):
    """'UFO-style' gate (not a full UFO-ViT implementation)."""

    def __init__(self, dim, reduction=4):
        super().__init__()
        hidden_dim = max(dim // reduction, 16)
        self.q_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim)
        self.gate = nn.Sigmoid()

    def forward(self, x):
        q = F.softmax(self.q_proj(x), dim=-1)
        k = F.softmax(self.k_proj(x), dim=-1)
        v = self.v_proj(x)
        kv_context = k * q
        attn_weights = self.gate(self.out_proj(v * kv_context.sum(dim=-1, keepdim=True)))
        return x * attn_weights


class CoordAttention1D(nn.Module):
    def __init__(self, in_dim, reduction=16):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 1, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(1)
        self.act = nn.SiLU()
        self.conv_h = nn.Conv1d(1, 1, kernel_size=3, padding=1)
        self.conv_w = nn.Conv1d(1, 1, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        identity = x
        x_in = x.unsqueeze(1)
        x_h = F.adaptive_avg_pool1d(x_in, x_in.size(-1))
        out = self.act(self.bn1(self.conv1(x_h)))
        a_h = self.sigmoid(self.conv_h(out)).squeeze(1)
        a_w = self.sigmoid(self.conv_w(out)).squeeze(1)
        return identity + identity * a_h * a_w


class SKAttention1D(nn.Module):
    def __init__(self, dim, reduction=8, M=2):
        super().__init__()
        self.M = M
        d = max(int(dim / reduction), 32)
        self.convs = nn.ModuleList([
            nn.Sequential(nn.Conv1d(1, 1, kernel_size=3, padding=1, bias=False),
                          nn.BatchNorm1d(1), nn.SiLU()),
            nn.Sequential(nn.Conv1d(1, 1, kernel_size=5, padding=2, bias=False),
                          nn.BatchNorm1d(1), nn.SiLU()),
        ])
        self.fc = nn.Linear(dim, d)
        self.fcs = nn.ModuleList([nn.Linear(d, dim) for _ in range(M)])
        nn.init.constant_(self.fc.weight, 0)
        nn.init.constant_(self.fc.bias, 0)

    def forward(self, x):
        identity = x
        x_in = x.unsqueeze(1)
        feats = [conv(x_in).squeeze(1) for conv in self.convs]
        U = sum(feats)
        Z = self.fc(U)
        weights = torch.stack([fc(Z) for fc in self.fcs], dim=1)
        weights = F.softmax(weights, dim=1)
        V = sum(weights[:, i, :] * feats[i] for i in range(self.M))
        return identity + V


class ECA1DBlock(nn.Module):
    def __init__(self, channels, gamma=2, b=1):
        super().__init__()
        k = int(abs((math.log(channels, 2) + b) / gamma))
        k = k if k % 2 != 0 else k + 1
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.conv(x.unsqueeze(1))
        return x * self.sigmoid(y).squeeze(1)


# ----------------------------------------------------------------------------
# Experiment configurations (Table I of the paper)
# ----------------------------------------------------------------------------
VARIANTS = {
    # name: backbone, attention, activation, gps_dropout, gps_out, head_dropout,
    #       optimizer lr, OneCycle peak lr, mixup prob
    "baseline": dict(backbone="convnext_base", attn=None, act="relu", gps_drop=0.2, gps_out=32,
                     head_drop=0.3, lr=1e-4, max_lr=1e-3, mixup=0.3),
    "coord":    dict(backbone="convnext_base", attn="coord", act="silu", gps_drop=0.0, gps_out=32,
                     head_drop=0.25, lr=1e-4, max_lr=8e-4, mixup=0.2),
    "sk":       dict(backbone="convnext_base", attn="sk", act="silu", gps_drop=0.0, gps_out=32,
                     head_drop=0.25, lr=1e-4, max_lr=8e-4, mixup=0.2),
    "eca":      dict(backbone="efficientnetv2_rw_m", attn="eca", act="silu", gps_drop=0.0, gps_out=64,
                     head_drop=0.2, lr=1e-3, max_lr=2e-3, mixup=0.2),
    "ufo":      dict(backbone="convnext_base", attn="ufo", act="relu", gps_drop=0.2, gps_out=32,
                     head_drop=0.3, lr=1e-4, max_lr=1e-3, mixup=0.3),
    "se":       dict(backbone="convnext_base", attn="se", act="relu", gps_drop=0.2, gps_out=32,
                     head_drop=0.3, lr=1e-4, max_lr=1e-3, mixup=0.3),  # proposed
}


class MultiModalNet(nn.Module):
    def __init__(self, variant="se", num_classes=64):
        super().__init__()
        cfg = VARIANTS[variant]
        Act = nn.ReLU if cfg["act"] == "relu" else nn.SiLU

        self.vision_backbone = timm.create_model(cfg["backbone"], pretrained=True,
                                                 in_chans=15, num_classes=0)
        with torch.no_grad():
            was_training = self.vision_backbone.training
            self.vision_backbone.eval()
            v_dim = self.vision_backbone(torch.randn(2, 15, 224, 224)).shape[1]
            self.vision_backbone.train(was_training)

        gps_layers = [nn.BatchNorm1d(8), nn.Linear(8, 64), Act()]
        if cfg["gps_drop"] > 0:
            gps_layers.append(nn.Dropout(cfg["gps_drop"]))
        gps_layers += [nn.Linear(64, cfg["gps_out"]), Act()]
        self.gps_mlp = nn.Sequential(*gps_layers)

        fused_dim = v_dim + cfg["gps_out"]  # 1024 + 32 = 1056 for ConvNeXt-Base
        attn = cfg["attn"]
        self.attention = {
            None: nn.Identity,
            "se": lambda: SEFusionAttention(fused_dim),
            "ufo": lambda: UFOAttention(fused_dim, reduction=4),
            "coord": lambda: CoordAttention1D(fused_dim, reduction=16),
            "sk": lambda: SKAttention1D(fused_dim, reduction=8),
            "eca": lambda: ECA1DBlock(fused_dim),
        }[attn]()

        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, 512), nn.BatchNorm1d(512), Act(),
            nn.Dropout(cfg["head_drop"]), nn.Linear(512, num_classes),
        )

    def forward(self, images, gps):
        fused = torch.cat([self.vision_backbone(images), self.gps_mlp(gps)], dim=1)
        return self.classifier(self.attention(fused))
