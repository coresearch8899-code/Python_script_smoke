"""rev37 Fig. 1: contribution-driven framework, and the graphical abstract with the transport-height qualification. Run after figures_rev36.py; overwrites Figure_1 and
Graphical_abstract in <fig_dir>. Usage: figures_rev37.py <fig_dir> <figures_rev29.py> <out25> <checks_out> <joint28_json>"""
import sys, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
FD = sys.argv[1]
B, Or, Aq, Y, V, INK, MUT, GR = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#6250d6", "#0b0b0b", "#52514e", "#c3c2b7"
plt.rcParams.update({"font.size": 7.2, "font.family": "DejaVu Sans", "pdf.fonttype": 42})
fig, ax = plt.subplots(figsize=(7.4, 4.6)); ax.axis("off"); ax.set_xlim(0, 100); ax.set_ylim(0, 62)
def box(x, y, w, h, title, body, col, fill):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1.2", fc=fill, ec=col, lw=1.1))
    ax.text(x + w / 2, y + h - 1.6, title, ha="center", va="top", fontsize=7.6, weight="bold", color=INK)
    ax.text(x + w / 2, y + h - 5.0, body, ha="center", va="top", fontsize=6.4, color=INK, linespacing=1.35)
def arrow(p, q, col=MUT, ls="-"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=9, lw=1.0, color=col, linestyle=ls, shrinkA=0, shrinkB=0))
box(12, 50.5, 36, 9.5, "Operational PV data", "Daily zone yields and 30-min power\nOperation and maintenance wash log", MUT, "#f3f2ee")
box(54, 50.5, 36, 9.5, "Open environmental data", "ERA5, CERES irradiance and AOD, CAMS PM$_{2.5}$\nVIIRS fire radiative power, rainfall, winds", MUT, "#f3f2ee")
box(12, 34.5, 36, 11.5, "Loss decomposition", "Clear-day PV model (Section 2.7)\natmospheric term: AOD\ndeposition term: partial-cleaning PM$_{2.5}$ dose", B, "#e6effb")
box(54, 34.5, 36, 11.5, "Source attribution", "72-h trajectory fire exposure (Section 2.5)\nhaze models of AOD and PM$_{2.5}$\nexact Shapley values by region (Section 2.8)", Or, "#fdece4")
box(3, 13.5, 29, 15.5, "Identifiability diagnostics", "Synthetic recoverability map\n(dose–AOD correlation × error condition)\ndiagnostic reporting criteria,\nleave-one-family-out and\nconfusion-matrix checks (Fig. 5, Table S8)", B, "#e6effb")
box(69, 13.5, 29, 15.5, "Attribution robustness", "288 specifications: transport,\nexposure form, response, dose,\ncounterfactual; rank stability,\nEIDI with bootstrap interval,\nbaselines (Fig. 6, Tables S6d–f)", Or, "#fdece4")
box(36, 13.5, 29, 15.5, "Application: maintenance decision", "Washing rule selected on 2023–2024\nheld-out replay on 2025–2026\nminimax regret over 36 soiling\nmodels; calendar and no-wash\nbaselines (Fig. 7, Table S10)", Aq, "#e3f5ee")
ax.add_patch(FancyBboxPatch((3, 1.5), 95, 7.5, boxstyle="round,pad=0.3,rounding_size=1.2", fc="white", ec=GR, lw=1.0, ls="--"))
ax.text(50.5, 7.4, "Results reported at the level the diagnostics support", ha="center", va="top", fontsize=7.4, weight="bold", color=INK)
ax.text(50.5, 4.4, "total loss with a component range  ·  modelled washing thresholds and regret  ·  source shares conditional on transport height", ha="center", va="top", fontsize=6.5, color=INK)
for x0 in (30, 72):
    arrow((x0, 50.2), (x0, 46.4))
arrow((44, 50.2), (58, 46.4)); arrow((58, 50.2), (44, 46.4))
arrow((20, 34.2), (17.5, 29.4), B); arrow((76, 34.2), (83.5, 29.4), Or); arrow((40, 34.2), (50.5, 29.4), Aq)
arrow((48.3, 40.2), (53.7, 40.2), B)
ax.text(51, 41.3, "PV\ncoefficients", ha="center", va="bottom", fontsize=5.6, color=MUT, linespacing=1.1)
for x0 in (17.5, 50.5, 83.5): arrow((x0, 13.2), (x0, 9.4), GR, "--")
fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
dpi = max(1000, int(np.ceil(7500 / fig.get_size_inches()[0]))); fig.savefig(f"{FD}/Figure_1.pdf"); fig.savefig(f"{FD}/Figure_1.png", dpi=dpi); plt.close(fig)
Image.open(f"{FD}/Figure_1.png").convert("RGB").save(f"{FD}/Figure_1.tif", compression="tiff_lzw", dpi=(dpi, dpi)); print("Figure_1 done")
# ---- graphical abstract: rev29 code with the transport-height qualification ----
F29, O25, CK, J28 = sys.argv[2:6]; src = open(F29, encoding="utf-8").read()
head = src[:src.index("# ---------- Fig 1")]; ga = src[src.index("# ---------- graphical abstract"):]
for o, n in (('c.set_title("Attributed PV loss", fontsize=8)', 'c.set_title("Attributed PV loss, 850 hPa", fontsize=8)'),
             ('fig.suptitle(', 'c.text(0.5, -0.19, "Ranking reverses in most 700 hPa\\nspecifications; the exposure–loss\\ngap keeps its sign in all 288", transform=c.transAxes, ha="center", va="top", fontsize=5.6, color=MUT)\nfig.suptitle(')):
    assert ga.count(o) == 1, o; ga = ga.replace(o, n)
sys.argv = [F29, O25, CK, "-", FD, "-", J28]; exec(compile(head + ga, F29, "exec"), {"__name__": "ga37"})