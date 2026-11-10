"""Evaluate a released checkpoint on Scenario 5 and print Top-1/3/5.

    python evaluate_checkpoint.py F_A3_SK_best.pth        --model sk_attention
    python evaluate_checkpoint.py F_C0_s42_best.pth       --model concatenation
    python evaluate_checkpoint.py V4_ConvNeXt_B_best.pth  --model vision_convnext

The data must be arranged as described in the README (ROOT is set at the top of the scripts).
sk_attention uses fusion.py, because that script produced the released SK checkpoint.
"""
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

ap = argparse.ArgumentParser()
ap.add_argument("ckpt")
ap.add_argument("--model", required=True,
                choices=["sk_attention", "concatenation", "vision_convnext"])
args = ap.parse_args()

if args.model == "vision_convnext":
    import v4_convnext_base_fast as M
    build = lambda: M.VisionOnly(M.BACKBONE)
elif args.model == "sk_attention":
    import fusion as M
    build = lambda: M.FusionNet(M.GATES["A3_SK"])
else:
    import fusion_v2 as M
    gate, use_gps = M.EXPS["C0"]
    build = lambda: M.FusionNet(gate, use_gps)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_cuda = device.type == "cuda"
cc = torch.cuda.get_device_capability()[0] if use_cuda else 0
amp_dtype = torch.bfloat16 if cc >= 8 else torch.float16

tr_s = M.build_samples(M.TRAIN_SCENARIOS)
te_s = M.build_samples(M.TEST_SCENARIOS)
path2idx = M.build_frame_cache([tr_s, te_s])
loader = DataLoader(M.FastBeamDataset(te_s, path2idx), batch_size=32, shuffle=False,
                    num_workers=2, pin_memory=True)

model = build().to(device).to(memory_format=torch.channels_last)
model.load_state_dict(torch.load(args.ckpt, map_location=device))
crit = nn.CrossEntropyLoss(label_smoothing=0.1)
_, t1, t3, t5 = M.evaluate(model, loader, crit, device, use_cuda, amp_dtype)
print(f"{args.model} | Scenario 5 | Top-1 {t1:.2f}% | Top-3 {t3:.2f}% | Top-5 {t5:.2f}%")
