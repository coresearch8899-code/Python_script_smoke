"""rev29 figures: Fig. 1 (inputs -> modules -> validation -> outputs), Fig. 5 (synthetic recovery, larger type), Fig. 7 (panel b now the held-out replay), graphical abstract.
Same style, sizes and bitmap rules as figures_rev25.py. Usage: figures_rev29.py <out25> <checks_out> <synthetic_csv> <fig_dir> <framework_out> <joint28_json>"""
import sys, json, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
O25, CK, SYN, FD, FW, J28 = sys.argv[1:7]; DPI = 1000; MINPX = 7500
B, Or, Aq, Y, V, INK, MUT, GR = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#6250d6", "#0b0b0b", "#52514e", "#c3c2b7"
plt.rcParams.update({"font.size": 8.5, "font.family": "DejaVu Sans", "axes.edgecolor": GR, "xtick.color": MUT, "ytick.color": MUT, "pdf.fonttype": 42,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.titlesize": 9.5, "axes.titlelocation": "center"})
def save(fig, name):
    dpi = max(DPI, int(np.ceil(MINPX / fig.get_size_inches()[0])))
    fig.savefig(f"{FD}/{name}.pdf"); fig.savefig(f"{FD}/{name}.png", dpi=dpi); plt.close(fig)
    Image.open(f"{FD}/{name}.png").convert("RGB").save(f"{FD}/{name}.tif", compression="tiff_lzw", dpi=(dpi, dpi))
# ---------- Fig 1 ----------
fig, ax = plt.subplots(figsize=(7.4, 4.0)); ax.axis("off"); ax.set_xlim(0, 104); ax.set_ylim(0, 60)
def box(x, y, w, h, text, col):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2", fc=col, ec="none", alpha=0.18))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=6.3, color=INK)
cols = [("Inputs", B, ["Operator: metered PV\nproduction, capacity,\nwash dates", "Open: AOD, PM$_{2.5}$,\nrainfall, irradiance", "Open: reanalysis winds,\nsatellite fires,\nboundaries", "Costs: tariff, wash\ncost, wash efficiency"]),
        ("Modules", Y, ["1 Reference and\nsensor screening", "2 Aerosol and soiling\nloss terms", "3 Exposure, haze\nand model-based\nattribution", "4 Cleaning decision"]),
        ("Validation", Aq, ["Multi-source and\ndrift checks", "Out-of-sample tests,\nsynthetic recovery", "Placebo, counterfactual\nsupport, transport\nset-ups", "Train/test replay,\nwash efficiency"]),
        ("Outputs", Or, ["Usable reference,\nrejected sensors", "Total loss with a\nrange for its\ncomponents", "Attribution range\nby region", "Cost range for\nthe decision"])]
for i, (title, col, items) in enumerate(cols):
    x = 2 + i * 26; ax.text(x + 10.5, 57, title, ha="center", fontsize=9, fontweight="bold", color=INK); hh = 52 / len(items)
    for j, it in enumerate(items): box(x, 53 - (j + 1) * hh + 1.2, 21, hh - 3.2, it, col)
    if i < 3:
        for j in range(4):
            yy = 53 - (j + 0.5) * hh; ax.annotate("", xy=(x + 25.3, yy), xytext=(x + 22.6, yy), arrowprops=dict(arrowstyle="-|>", color=MUT, lw=0.9))
save(fig, "Figure_1")
# ---------- Fig 5 synthetic recovery ----------
S = pd.read_csv(SYN)
L = {"A1": "Correct model", "A2": "True β$_A$ halved", "A3": "True β$_S$ halved", "H1": "Dose–AOD r = 0.2", "B1": "Dose–AOD r = 0.7", "B2": "Dose–AOD r = 0.9", "C1": "Noisy rain", "C2": "Rain one day late",
     "H4": "Noisy PM$_{2.5}$", "H5": "Noisy irradiance", "G2": "Noisy AOD", "F1": "Linear drift", "F2": "Step of −3 %", "F3": "Seasonal reference bias", "E1": "March–April PM$_{2.5}$ × 2",
     "H2": "Rain removes at most 70 %", "H3": "Exponential rain removal", "D1": "True rain rule 5 mm", "D2": "True rain rule 2 mm reset", "G1": "True hinge response", "H6": "Combined errors"}
order = list(L); T = S.set_index(S.scenario.str[:2]).loc[order]; y = np.arange(len(order))[::-1]
groups = {B: order[:6], Aq: order[6:11], Y: order[11:15], Or: order[15:20], V: order[20:]}
col = {k: c for c, v in groups.items() for k in v}; cc = [col[k] for k in order]
fig, axs = plt.subplots(1, 4, figsize=(7.4, 4.9), sharey=True, gridspec_kw=dict(wspace=0.12))
for a, (k, ttl, xl) in zip(axs, [("atm", "(a) Atmospheric", "Error (%)"), ("dep", "(b) Deposition", "Error (%)"), ("tot", "(c) Total", "Error (%)"), ("frac", "(d) Atmospheric fraction", "Error (points)")]):
    pct = k != "frac"; e = (T[f"bias_{k}"] / T[f"true_{k}"] * 100 if pct else T[f"bias_{k}"]).values; r = (T[f"rmse_{k}"] / T[f"true_{k}"] * 100 if pct else T[f"rmse_{k}"]).values
    a.axvline(0, color=MUT, lw=0.7); a.grid(axis="x", color="#e6e5e0", lw=0.5); a.set_axisbelow(True)
    for yi, ei, ri, ci in zip(y, e, r, cc):
        a.plot([ei - ri, ei + ri], [yi, yi], color=ci, lw=1.1, alpha=0.55, solid_capstyle="butt"); a.plot(ei, yi, "o", ms=3, color=ci)
    a.set_title(ttl, fontsize=8, pad=4); a.set_xlabel(xl, fontsize=7.5); a.tick_params(labelsize=7.2)
axs[0].set_yticks(y); axs[0].set_yticklabels([L[k] for k in order], fontsize=7.2); axs[0].set_ylim(-0.7, len(order) - 0.3)
fig.subplots_adjust(left=0.225, right=0.985, top=0.95, bottom=0.09); save(fig, "Figure_5")
# ---------- Fig 6 attribution, nominal intervals with 28-day blocks ----------
C = ["THA", "MMR", "LAO", "OTH"]; labs = ["Thailand", "Myanmar", "Laos", "Other"]; RR0 = json.load(open(f"{O25}/R_revision_results.json")); S_ = RR0["R3_shapley"]
ex0 = json.load(open(f"{CK}/exposure_common.json"))["p850"]["loss_days"]; im0 = S_["p850"]["shapley_share"]; ci = json.load(open(J28.replace("joint_bootstrap_total_loss.json", "checks_rev25b.json")))["A_CI_total_share"]
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.9)); x = np.arange(4); w = 0.38; a = ax[0]
a.bar(x - w / 2, [ex0[c] for c in C], w, color=B, label="Fire exposure")
a.bar(x + w / 2, [im0[c] for c in C], w, color=Or, label="Fire-attributable PV loss", yerr=np.array([[im0[c] - ci[c][0] for c in C], [ci[c][1] - im0[c] for c in C]]), capsize=2, error_kw=dict(lw=0.8))
a.axhline(0, color=MUT, lw=0.6); a.set_xticks(x); a.set_xticklabels(labs); a.set_ylabel("Share (%)"); a.set_ylim(-35, 105); a.legend(fontsize=6.5, frameon=False, loc="upper right")
a.set_title("(a)", loc="center", fontsize=9.5, pad=6); a.grid(axis="y", color="#e6e5e0", lw=0.6); a.set_axisbelow(True)
a = ax[1]; st = {"850 hPa": S_["p850"]["shapley_share"], "700 hPa": S_["p700"]["shapley_share"], "3-D from 850 hPa": S_["3d850"]["shapley_share"]}; ord_ = ["3-D from 850 hPa", "700 hPa", "850 hPa"]; bot = np.zeros(3)
for c, col_, lb in zip(C, (Or, B, Aq, GR), labs):
    v = np.array([max(st[k][c], 0) for k in ord_]); a.barh(ord_, v, left=bot, color=col_, label=lb, edgecolor="white", lw=1); bot += v
for i, k in enumerate(ord_):
    a.text(st[k]["THA"] / 2, i, f"{st[k]['THA']:.0f}%", ha="center", va="center", color="white", fontsize=7)
    if st[k]["MMR"] > 10: a.text(st[k]["THA"] + st[k]["MMR"] / 2, i, f"{st[k]['MMR']:.0f}%", ha="center", va="center", color="white", fontsize=7)
a.set_xlim(0, 100); a.set_xlabel("Share of fire-attributable PV loss (%)"); a.legend(fontsize=6.5, frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.45, -0.22))
a.set_title("(b)", loc="center", fontsize=9.5, pad=6); fig.tight_layout(); save(fig, "Figure_6")
# ---------- Fig 7: Jan-May decomposition and held-out washing replay ----------
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.9)); grid = lambda a_: (a_.grid(axis="y", color="#e6e5e0", lw=0.6), a_.set_axisbelow(True))
label = lambda a_, k: a_.set_title(f"({k})", loc="center", fontsize=9.5, pad=6)
f6 = pd.read_csv(f"{O25}/R3_fig6a_components.csv", index_col=0); a = ax[0]; bot = np.zeros(len(f6))
for col_, c_, lab in (("thai_atm_MWh", Or, "Atmospheric: Thai fires"), ("abroad_atm_MWh", Y, "Atmospheric: fires abroad"), ("nonfire_atm_MWh", B, "Atmospheric: non-fire aerosol"), ("dep_JanMay", V, "Deposition (soiling)")):
    a.bar(f6.index.astype(str), f6[col_].values, bottom=bot, color=c_, width=0.62, label=lab, edgecolor="white", lw=1); bot += f6[col_].values
for i, t_ in enumerate(bot): a.text(i, t_ + 2, f"{t_:.0f}", ha="center", fontsize=7)
a.set_ylabel("PV energy loss, Jan–May (MWh)"); a.set_ylim(0, 215); a.set_yticks([0, 50, 100, 150]); a.legend(fontsize=5.6, frameon=False, ncol=2, loc="upper right", columnspacing=0.8, handlelength=1.2)
a.set_xticks(range(len(f6))); a.set_xticklabels(["2023*", "2024", "2025", "2026"]); label(a, "a"); grid(a)
W = pd.read_csv(f"{FW}/washing_out_of_sample.csv"); a = ax[1]; W = W[W.beta == "training"]
for e, c_, st, lab in ((1.0, Or, "o-", "Wash efficiency 100 %"), (0.9, Y, "s-", "90 %"), (0.75, Aq, "^-", "75 %"), (0.5, B, "v-", "50 %")):
    q = W[W.eta == e].sort_values("cost_THB"); a.plot(q.cost_THB / 1000, q.test_saving_THB / 1000, st, color=c_, ms=2.8, lw=1.1, label=lab)
a.axhline(0, color=MUT, lw=0.8); a.set_xlabel("Cost per fleet wash (kTHB)"); a.set_ylabel("Saving vs no washing, 2025–2026 (kTHB)"); a.legend(fontsize=6.2, frameon=False, loc="upper right"); label(a, "b"); grid(a)
fig.tight_layout(); save(fig, "Figure_7")
# ---------- graphical abstract ----------
J = json.load(open(J28)); RR = json.load(open(f"{O25}/R_revision_results.json"))
ex = json.load(open(f"{CK}/exposure_common.json"))["p850"]["loss_days"]; imp = RR["R3_shapley"]["p850"]["shapley_share"]
fig = plt.figure(figsize=(5.12, 2.6)); g = fig.add_gridspec(1, 3, width_ratios=[1.1, 1.05, 0.9], wspace=0.25); K = ["OTH", "LAO", "MMR", "THA"]; N = ["Other", "Laos", "Myanmar", "Thailand"]
a = fig.add_subplot(g[0]); a.barh(N, [ex[c] for c in K], color=[GR, Aq, B, Or]); a.set_xlim(0, 100); a.set_title("Upwind fire exposure", fontsize=8); a.set_xlabel("%", fontsize=7); a.tick_params(labelsize=7)
for i, c in enumerate(K): a.text(ex[c] + 2, i, f"{ex[c]:.0f}", va="center", fontsize=7)
b = fig.add_subplot(g[1]); b.axis("off"); b.set_xlim(0, 1); b.set_ylim(0, 1)
b.text(0.5, 0.90, "Operational PV +\nopen environmental data", ha="center", va="center", fontsize=7, fontweight="bold")
b.text(0.5, 0.62, f"Demonstration, 2.51 MWp fleet\n{J['point_hinge']['tot']:.0f}–{J['point_linear']['tot']:.0f} MWh yr$^{{-1}}$\n(nominal 95 % {J['CI95_pooled_tot'][0]:.0f}–{J['CI95_pooled_tot'][1]:.0f})", ha="center", va="center", fontsize=6.6)
b.annotate("", xy=(0.98, 0.38), xytext=(0.02, 0.38), xycoords="axes fraction", arrowprops=dict(arrowstyle="-|>", color=MUT, lw=1.2))
b.text(0.5, 0.16, "Atmospheric share 57–76 %\nacross specifications", ha="center", va="center", fontsize=6.2, color=MUT)
c = fig.add_subplot(g[2]); c.barh(N, [imp[k] for k in K], color=[GR, Aq, B, Or]); c.set_xlim(0, 100); c.set_title("Attributed PV loss", fontsize=8); c.set_xlabel("%", fontsize=7); c.tick_params(labelsize=7); c.set_yticklabels([])
for i, k in enumerate(K): c.text(imp[k] + 2, i, f"{imp[k]:.0f}", va="center", fontsize=7)
fig.suptitle("Decomposing and attributing haze-related PV losses", fontsize=9.5, fontweight="bold", y=1.05)
fig.savefig(f"{FD}/Graphical_abstract.pdf", bbox_inches="tight"); fig.savefig(f"{FD}/Graphical_abstract.png", dpi=DPI, bbox_inches="tight"); plt.close(fig)
Image.open(f"{FD}/Graphical_abstract.png").convert("RGB").save(f"{FD}/Graphical_abstract.tif", compression="tiff_lzw", dpi=(DPI, DPI)); print("figures done")
