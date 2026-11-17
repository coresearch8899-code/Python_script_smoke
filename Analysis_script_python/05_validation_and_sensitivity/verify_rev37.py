"""Compare the rev37 analyses rerun from the repository with the stored outputs and check the numbers quoted in the paper.
Usage: verify_rev37.py <rerun_novelty_out> <stored_novelty_out> <package_dir>"""
import sys, json, re, docx, pandas as pd
O, S, P = sys.argv[1:4]
def flat(o, p=""):
    if isinstance(o, dict):
        for k, v in o.items(): yield from flat(v, f"{p}.{k}")
    elif isinstance(o, list):
        for i, v in enumerate(o): yield from flat(v, f"{p}[{i}]")
    elif isinstance(o, (int, float)) and not isinstance(o, bool): yield p, float(o)
bad = n = 0
for f in ("novelty37.json", "criteria_lofo.json", "washing_regret.json"):
    a, b = dict(flat(json.load(open(f"{O}/{f}")))), dict(flat(json.load(open(f"{S}/{f}"))))
    dd = [k for k in a if k in b and abs(a[k] - b[k]) > 1e-9 * max(1, abs(b[k]))]; n += len(a); bad += len(dd) + len(set(a) ^ set(b))
    if dd: print("DIFF", f, dd[:3])
for f in ("eidi_bootstrap.csv", "washing_regret.csv"):
    p, q = pd.read_csv(f"{O}/{f}"), pd.read_csv(f"{S}/{f}"); nu = p.select_dtypes("number").columns; m = float((p[nu] - q[nu]).abs().max().max())
    if m > 1e-9 or p.shape != q.shape: print("DIFF", f, m); bad += 1
print(f"rev37 analyses rerun from the repository: {n} JSON numbers and 2 CSV files compared, {bad} differences")
m = docx.Document(P + r"\03_Manuscript_HazePV_RenewableEnergy.docx"); t = " ".join(p.text for p in m.paragraphs).replace(" ", " ")
su = docx.Document(P + r"\06_Supplementary_material.docx")
N = json.load(open(f"{O}/novelty37.json")); L = json.load(open(f"{O}/criteria_lofo.json")); RG = json.load(open(f"{O}/washing_regret.json"))["summary"]
ok = []
def chk(name, cond): ok.append(bool(cond)); print(("ok  " if cond else "BAD ") + name)
A = N["A"]; r0 = lambda x: f"{x:.0f}".replace("-", "−")
chk("EIDI Thailand +53 (30–77)", f"+{A['THA']['EIDI']:.0f} points for Thailand (joint bootstrap 95 % interval {A['THA']['EIDI_CI'][0]:.0f}–{A['THA']['EIDI_CI'][1]:.0f})" in t)
chk("EIDI Myanmar −57 (−85 to −34)", f"{r0(A['MMR']['EIDI'])} points for Myanmar ({r0(A['MMR']['EIDI_CI'][0])} to {r0(A['MMR']['EIDI_CI'][1])})" in t)
chk("Laos and other intervals include zero", all(A[c]["EIDI_CI"][0] < 0 < A[c]["EIDI_CI"][1] for c in ("LAO", "OTH")) and all(A[c]["EIDI_CI"][0] > 0 or A[c]["EIDI_CI"][1] < 0 for c in ("THA", "MMR")))
chk("abstract gap +53, 30–77", f"+{A['THA']['EIDI']:.0f} points, 95 % interval {A['THA']['EIDI_CI'][0]:.0f}–{A['THA']['EIDI_CI'][1]:.0f}" in t)
o = N["B_shares"]["one-at-a-time"]; chk("one-at-a-time 64 and 18 %", f"gave {o['THA']:.0f} and {o['MMR']:.0f} %" in t)
B = N["C_blh"]; chk("BL 94 % / 22 %", f"on {100 * B['share_days_850_within_daytime_BL']:.0f} % of January–May days and the 700 hPa level on {100 * B['share_days_700_within_daytime_BL']:.0f} %" in t)
F = N["C_fit_by_transport"]; chk("LOYO 700 vs 850: 0.46 vs 0.43", f"LOYO R2 {F['p700']['AOD']['loyo_r2']:.2f} against {F['p850']['AOD']['loyo_r2']:.2f}" in t and F["p700"]["AOD"]["loyo_r2"] > F["p850"]["AOD"]["loyo_r2"])
K = N["B_skill"]; chk("proximity LOYO 0.37 vs 0.43; not for ln PM", f"LOYO R2 {K['near-field and far-field, region-agnostic']['AOD']['loyo_r2']:.2f} against {K['region-specific (main)']['AOD']['loyo_r2']:.2f}" in t
    and K["near-field and far-field, region-agnostic"]["lnPM25"]["loyo_r2"] >= K["region-specific (main)"]["lnPM25"]["loyo_r2"])
C = N["D_cams"]; chk("CAMS 148 MWh, 66 %", f"the total was {C['total']:.0f} MWh and the atmospheric fraction {C['atm_fraction']:.0f} %" in t)
chk("criteria 76 / 72 / 40 %", f"correct in {100 * L['overall_prespecified_acc']:.0f} % of cells and in {100 * L['lofo_mean_test_acc']:.0f} % on average" in t and f"but in {100 * L['per_family']['AOD noise SD 0.10']['test_acc']:.0f} % for AOD noise" in t)
chk("threshold perturbation changes no split class", L["share_cells_changed_by_threshold_perturbation"] == 0)
bl = [b["test_max_regret"] for v in RG.values() for b in v["baselines"].values()]; chk("baselines 411–1,431 kTHB", f"{min(bl):,.0f}–{max(bl):,.0f} kTHB" in t)
ch = [c for c, v in RG.items() if any(se["Dstar"] != v["minimax_Dstar"] for se in v["sensitivity"].values())]; chk("sensitivity changes the choice only at 50 kTHB", ch == ["50"])
chk("minimax beats baselines at 10 and 30 kTHB", all(RG[c]["test_max_regret_minimax"] < min(b["test_max_regret"] for b in RG[c]["baselines"].values()) for c in ("10", "30")))
chk("supplement title = manuscript title", su.paragraphs[0].text.strip() == m.paragraphs[0].text.strip())
print(f"{sum(ok)} of {len(ok)} checks passed")
