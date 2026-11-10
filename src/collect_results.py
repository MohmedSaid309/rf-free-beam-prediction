import json, glob
rows = [json.load(open(f)) for f in sorted(glob.glob('/content/results/*.json'))]
print(f"{'Exp':22s} {'Params(M)':>9s} {'Top-1':>7s} {'Top-3':>7s} {'Top-5':>7s}")
for r in rows:
    print(f"{r['name']:22s} {r['params_M']:9.3f} {r['top1']:7.2f} {r['top3']:7.2f} {r['top5']:7.2f}")
print("\nLaTeX rows:")
for r in rows:
    print(f"{r['name'].replace('_',' ')} & {r['top1']:.2f} & {r['top3']:.2f} & {r['top5']:.2f} \\\\")
