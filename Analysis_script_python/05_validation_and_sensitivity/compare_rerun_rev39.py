"""Compare a full rerun from the repository with the stored outputs: every JSON number, every CSV (numeric columns with relative tolerance 1e-9, other columns exactly)
and every figure bitmap (pixel by pixel). Usage: compare_rerun_rev39.py <rerun_dir> <stored_analysis_dir> <package_figure_dir> <rev39_fig1_dir>"""
import sys, os, json, numpy as np, pandas as pd
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
RR, ST, PF, F1 = sys.argv[1:5]
MAP = {"out25": "out25", "checks_out": "checks_out", "robustness_out": "robustness_out", "framework": "framework_out", "blocks28": "blocks28", "synthetic": "synthetic_out", "novelty": "novelty_out", "hsu": "hsu_out"}
def flat(o, p=""):
    if isinstance(o, dict):
        for k, v in o.items(): yield from flat(v, f"{p}.{k}")
    elif isinstance(o, list):
        for i, v in enumerate(o): yield from flat(v, f"{p}[{i}]")
    elif isinstance(o, bool) or o is None or isinstance(o, str): yield p, o
    elif isinstance(o, (int, float)): yield p, float(o)
eq = lambda a, b: (isinstance(a, float) and isinstance(b, float) and ((a != a and b != b) or abs(a - b) <= 1e-9 * max(1.0, abs(b)))) or a == b
nj = nn = nc = 0; diffs = []; missing = []
for sub, sd in MAP.items():
    if not os.path.isdir(f"{RR}/{sub}"): continue
    for f in sorted(os.listdir(f"{RR}/{sub}")):
        a, b = f"{RR}/{sub}/{f}", f"{ST}/{sd}/{f}"
        if not f.endswith((".json", ".csv")): continue
        if not os.path.exists(b): missing.append(f"{sub}/{f}"); continue
        if f.endswith(".json"):
            x, y = dict(flat(json.load(open(a)))), dict(flat(json.load(open(b)))); nj += 1; nn += sum(isinstance(v, float) for v in x.values())
            bad = [k for k in x if k in y and not eq(x[k], y[k])] + sorted(set(x) ^ set(y))
            if bad: diffs.append((f"{sub}/{f}", len(bad), bad[:3]))
        else:
            p, q = pd.read_csv(a), pd.read_csv(b); nc += 1
            if p.shape != q.shape or list(p.columns) != list(q.columns): diffs.append((f"{sub}/{f}", "shape/columns", (p.shape, q.shape))); continue
            num = p.select_dtypes("number").columns; oth = [c for c in p.columns if c not in num]
            dn = float(((p[num] - q[num]).abs() / q[num].abs().clip(lower=1)).max().max()) if len(num) else 0.0
            if dn > 1e-9 or not p[oth].astype(str).equals(q[oth].astype(str)): diffs.append((f"{sub}/{f}", dn, ""))
print(f"JSON files {nj} ({nn} numbers), CSV files {nc}; files differing: {len(diffs)}; rerun files without a stored counterpart: {len(missing)}")
for d_ in diffs: print("DIFF", d_)
if missing: print("not stored:", missing)
figs = []
for f in sorted(os.listdir(f"{RR}/figures")):
    if not f.endswith(".tif"): continue
    ref = f"{F1}/{f}" if f in ("Figure_1.tif", "Graphical_abstract.tif") else f"{PF}/{f}"
    if not os.path.exists(ref): figs.append((f, "no packaged counterpart")); continue
    a, b = np.asarray(Image.open(f"{RR}/figures/{f}")), np.asarray(Image.open(ref))
    figs.append((f, "identical" if a.shape == b.shape and np.array_equal(a, b) else f"differs: shape {a.shape} vs {b.shape}, share of pixels changed {float((a != b).any(-1).mean()) if a.shape == b.shape else float('nan'):.4f}"))
for f in figs: print("FIG", f)
