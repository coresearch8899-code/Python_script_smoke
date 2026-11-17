"""rev36 figures: Fig. 5 identifiability map, Fig. 6 exposure versus loss shares with EIDI and the 288-specification ensemble, Fig. 7 with a regret panel.
The rev29 scenario plot becomes Supplementary Fig. S1. Usage: figures_rev36.py <out25> <checks_out> <novelty_out> <framework_out> <blocks28_dir> <fig_dir> <figures_rev29.py> <synthetic_csv>"""
import sys, json, subprocess, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
O25, CK, NV, FW, B28, FD, F29, SYN = sys.argv[1:9]
subprocess.run([sys.executable, F29, O25, CK, SYN, FD, FW, f"{B28}/joint_bootstrap_total_loss.json"], check=True)   # Fig. 1, old Fig. 5, Fig. 6, Fig. 7, abstract
import shutil
for e in ("pdf", "png", "tif"): shutil.move(f"{FD}/Figure_5.{e}", f"{FD}/Figure_S1.{e}")
B, Or, Aq, Y, V, INK, MUT, GR = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#6250d6", "#0b0b0b", "#52514e", "#c3c2b7"
plt.rcParams.update({"font.size": 8.5, "font.family": "DejaVu Sans", "axes.edgecolor": GR, "xtick.color": MUT, "ytick.color": MUT, "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False})
def save(fig, name):
    dpi = max(1000, int(np.ceil(7500 / fig.get_size_inches()[0]))); fig.savefig(f"{FD}/{name}.pdf"); fig.savefig(f"{FD}/{name}.png", dpi=dpi); plt.close(fig)
    Image.open(f"{FD}/{name}.png").convert("RGB").save(f"{FD}/{name}.tif", compression="tiff_lzw", dpi=(dpi, dpi))
lab = lambda a, k: a.set_title(f"({k})", loc="center", fontsize=9.5, pad=6)
# ---- Fig. 5 identifiability map ----
M = pd.read_csv(f"{NV}/identifiability_map.csv"); conds = ["Correct model", "True rain rule 5 mm", "True rain rule 2 mm reset", "March-April PM2.5 bias x 2", "AOD noise SD 0.10"]
names = ["Correct model", "True rain rule:\ncomplete at 5 mm", "True rain rule:\nfull reset at 2 mm", "PM$_{2.5}$ doubled in\nMarch–April", "AOD noise\n(SD 0.10)"]; rhos = [0.0, 0.3, 0.5, 0.7, 0.9]
code = {"split": 2, "total with component range": 1, "no estimate": 0}; Z = np.array([[code[M[(M.condition == c) & (M.rho_target == r)].report.iloc[0]] for r in rhos] for c in conds])
fig, ax = plt.subplots(figsize=(7.2, 3.6)); cmap = ListedColormap(["#f2d0c4", "#fbe7b5", "#cfe9dc"]); ax.imshow(Z, cmap=cmap, vmin=-0.5, vmax=2.5, aspect="auto")
for i, c in enumerate(conds):
    for j, r in enumerate(rhos):
        q = M[(M.condition == c) & (M.rho_target == r)].iloc[0]
        fz = lambda v: "0" if abs(v) < 0.5 else f"{v:+.0f}".replace("-", "−")
        ax.text(j, i, f"dep {fz(q.relbias_dep)} %\ntotal {fz(q.relbias_tot)} %", ha="center", va="center", fontsize=6.6, color=INK)
ax.set_xticks(range(5)); ax.set_xticklabels([f"{r:.1f}" for r in rhos]); ax.set_yticks(range(5)); ax.set_yticklabels(names, fontsize=7.2); ax.set_xlabel("Correlation of soiling dose and AOD on clear days (observed: 0.42)")
for s_ in ax.spines.values(): s_.set_visible(False)
ax.set_xticks(np.arange(-.5, 5), minor=True); ax.set_yticks(np.arange(-.5, 5), minor=True); ax.grid(which="minor", color="white", lw=2); ax.tick_params(which="minor", length=0)
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color="#cfe9dc", label="Report the split"), Patch(color="#fbe7b5", label="Report the total with a component range"), Patch(color="#f2d0c4", label="No reliable estimate")], loc="upper center", bbox_to_anchor=(0.45, 1.16), ncol=3, frameon=False, fontsize=7)
fig.subplots_adjust(left=0.2, right=0.98, top=0.86, bottom=0.14); save(fig, "Figure_5")
# ---- Fig. 6 exposure versus loss shares and ensemble ----
C = ["THA", "MMR", "LAO", "OTH"]; labs = ["Thailand", "Myanmar", "Laos", "Other"]
RR = json.load(open(f"{O25}/R_revision_results.json")); ex = json.load(open(f"{CK}/exposure_common.json"))["p850"]["loss_days"]; im = RR["R3_shapley"]["p850"]["shapley_share"]; ci = json.load(open(f"{B28}/checks_rev25b.json"))["A_CI_total_share"]
EN = pd.read_csv(f"{NV}/attribution_ensemble.csv")
fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.0)); x = np.arange(4); w = 0.38; a = ax[0]
a.bar(x - w / 2, [ex[c] for c in C], w, color=B, label="Fire exposure"); a.bar(x + w / 2, [im[c] for c in C], w, color=Or, label="Fire-attributable PV loss", yerr=np.array([[im[c] - ci[c][0] for c in C], [ci[c][1] - im[c] for c in C]]), capsize=2, error_kw=dict(lw=0.8))
for i, c in enumerate(C): a.text(i, max(ex[c], ci[c][1]) + 4, f"{im[c] - ex[c]:+.0f}".replace("-", "−"), ha="center", fontsize=7, color=INK)
a.axhline(0, color=MUT, lw=0.6); a.set_xticks(x); a.set_xticklabels(labs); a.set_ylabel("Share (%)"); a.set_ylim(-35, 112); a.legend(fontsize=6.5, frameon=False, loc="upper right"); lab(a, "a"); a.grid(axis="y", color="#e6e5e0", lw=0.6); a.set_axisbelow(True)
a = ax[1]; tr = [("p850", "850 hPa"), ("3d850", "3-D"), ("p700", "700 hPa")]; rng = np.random.default_rng(1)
for k, (t, tl) in enumerate(tr):
    g = EN[EN.transport == t]
    for c, col, off in (("THA", Or, -0.17), ("MMR", B, 0.17)):
        a.scatter(k + off + rng.uniform(-0.07, 0.07, len(g)), g[f"L_{c}"], s=4, color=col, alpha=0.45, lw=0)
        a.scatter([k + off], [g[f"E_{c}"].mean()], marker="_", s=180, color=INK, lw=1.4)
a.set_xticks(range(3)); a.set_xticklabels([t[1] for t in tr]); a.set_ylabel("Share of fire-attributable loss (%)"); a.set_ylim(0, 100)
a.scatter([], [], s=12, color=Or, label="Thailand, 96 specifications each"); a.scatter([], [], s=12, color=B, label="Myanmar"); a.scatter([], [], marker="_", s=120, color=INK, label="Exposure share"); a.legend(fontsize=6.3, frameon=False, loc="upper right")
lab(a, "b"); a.grid(axis="y", color="#e6e5e0", lw=0.6); a.set_axisbelow(True); fig.tight_layout(); save(fig, "Figure_6")
# ---- Fig. 7: add regret panel ----
f6 = pd.read_csv(f"{O25}/R3_fig6a_components.csv", index_col=0); W = pd.read_csv(f"{FW}/washing_out_of_sample.csv"); W = W[W.beta == "training"]; RG = json.load(open(f"{NV}/washing_regret.json"))["summary"]
fig, ax = plt.subplots(1, 3, figsize=(7.4, 2.8)); a = ax[0]; bot = np.zeros(len(f6))
for col_, c_, lb in (("thai_atm_MWh", Or, "Atm.: Thai fires"), ("abroad_atm_MWh", Y, "Atm.: fires abroad"), ("nonfire_atm_MWh", B, "Atm.: non-fire"), ("dep_JanMay", V, "Deposition")):
    a.bar(f6.index.astype(str), f6[col_].values, bottom=bot, color=c_, width=0.62, label=lb, edgecolor="white", lw=1); bot += f6[col_].values
a.set_ylabel("PV loss, Jan–May (MWh)"); a.set_ylim(0, 215); a.legend(fontsize=5.2, frameon=False, ncol=2, loc="upper right", columnspacing=0.6, handlelength=1); a.set_xticks(range(len(f6))); a.set_xticklabels(["2023*", "2024", "2025", "2026"]); lab(a, "a")
a = ax[1]
for e, c_, st, lb in ((1.0, Or, "o-", "η 100 %"), (0.9, Y, "s-", "90 %"), (0.75, Aq, "^-", "75 %"), (0.5, B, "v-", "50 %")):
    q = W[W.eta == e].sort_values("cost_THB"); a.plot(q.cost_THB / 1000, q.test_saving_THB / 1000, st, color=c_, ms=2.2, lw=1.0, label=lb)
a.axhline(0, color=MUT, lw=0.8); a.set_xlabel("Cost per wash (kTHB)"); a.set_ylabel("Modelled saving, 2025–26 (kTHB)"); a.legend(fontsize=5.8, frameon=False, loc="upper right"); lab(a, "b")
a = ax[2]; cs = list(RG); xx = np.arange(len(cs))
a.bar(xx - 0.2, [RG[c]["test_max_regret_best"] for c in cs], 0.4, color=GR, label="Best-estimate policy"); a.bar(xx + 0.2, [RG[c]["test_max_regret_minimax"] for c in cs], 0.4, color=Aq, label="Minimax-regret policy")
a.set_xticks(xx); a.set_xticklabels(cs); a.set_xlabel("Cost per wash (kTHB)"); a.set_ylabel("Worst-case regret, 2025–26 (kTHB)"); a.legend(fontsize=5.8, frameon=False, loc="upper left"); lab(a, "c")
fig.tight_layout(); save(fig, "Figure_7"); print("figures done")
