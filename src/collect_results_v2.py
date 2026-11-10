# groups runs by experiment (all seeds) and prints mean +- std; old files without "_sXX" count as seed 42
import json, glob, re
import numpy as np
from collections import defaultdict
g = defaultdict(list)
for f in sorted(glob.glob('/content/results/*.json')):
    r = json.load(open(f))
    g[re.sub(r'_s\d+$', '', r['name'])].append(r)
print(f"{'Experiment':20s} {'n':>2s} {'Params(M)':>9s} {'Top-1':>14s} {'Top-3':>14s} {'Top-5':>14s}")
for k, rs in g.items():
    def ms(m):
        v = np.array([r[m] for r in rs]); sd = v.std(ddof=1) if len(v) > 1 else 0.0
        return f"{v.mean():6.2f} ± {sd:4.2f}"
    print(f"{k:20s} {len(rs):2d} {rs[0]['params_M']:9.3f} {ms('top1'):>14s} {ms('top3'):>14s} {ms('top5'):>14s}")
print("\nper-run (check best_val_loss when choosing between runs, NOT test):")
for k, rs in g.items():
    for r in rs:
        print(f"  {r['name']:26s} val_loss {r['best_val_loss']:.4f}  T1/T3/T5 {r['top1']:.2f}/{r['top3']:.2f}/{r['top5']:.2f}")
