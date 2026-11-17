import sys, json, re, docx, pandas as pd, numpy as np
O, S, P = sys.argv[1:4]
def flat(o, p=""):
    if isinstance(o, dict):
        for k, v in o.items(): yield from flat(v, f"{p}.{k}")
    elif isinstance(o, list):
        for i, v in enumerate(o): yield from flat(v, f"{p}[{i}]")
    elif isinstance(o, (int, float)) and not isinstance(o, bool): yield p, float(o)
bad = 0; n = 0
for f in ("attribution_ensemble.json", "washing_regret.json", "uncertainty_decomposition.json"):
    a, b = dict(flat(json.load(open(f"{O}/novelty_out/{f}")))), dict(flat(json.load(open(f"{S}/novelty_out/{f}"))))
    dd = [k for k in a if k in b and abs(a[k] - b[k]) > 1e-9 * max(1, abs(b[k]))]; n += len(a); bad += len(dd) + len(set(a) ^ set(b))
    if dd: print("DIFF", f, dd[:3])
for f in ("identifiability_map.csv", "attribution_ensemble.csv", "washing_regret.csv"):
    p, q = pd.read_csv(f"{O}/novelty_out/{f}"), pd.read_csv(f"{S}/novelty_out/{f}"); nu = p.select_dtypes("number").columns; m = float((p[nu] - q[nu]).abs().max().max())
    if m > 1e-9 or not p.astype(str).equals(q.astype(str)) and m > 1e-9: print("DIFF", f, m); bad += 1
print(f"new analyses rerun from the repository: {n} JSON numbers and 3 CSV files compared, {bad} differences")
m = docx.Document(P + r"\03_Manuscript_HazePV_RenewableEnergy.docx"); t = " ".join(p.text for p in m.paragraphs).replace("\u00a0", " ")
E = json.load(open(f"{O}/novelty_out/attribution_ensemble.json")); RG = json.load(open(f"{O}/novelty_out/washing_regret.json"))["summary"]; U = json.load(open(f"{O}/novelty_out/uncertainty_decomposition.json"))
q = E["quantiles"]; ok = []
def chk(name, cond): ok.append(bool(cond)); print(("ok  " if cond else "BAD ") + name)
chk("ARS 69 %", f"{100 * E['ARS']['THA']:.0f} %" == "69 %" and "in 69 %" in t)
chk("700 hPa 6 %", f"{100 * E['THA_gt_MMR_by_factor']['transport']['p700']:.0f}" == "6" and "6 % of the 700 hPa" in t)
chk("variance 86 / 4 / 2", [round(100 * E["variance_share_L_THA"][k]) for k in ("transport", "counterfactual", "form")] == [86, 4, 2] and "86 % of the variance" in t)
chk("EIDI ranges", f"+{q['EIDI_THA'][0]:.0f} to +{q['EIDI_THA'][-1]:.0f}" in t and f"−{-q['EIDI_MMR'][0]:.0f} to −{-q['EIDI_MMR'][-1]:.0f}" in t)
pm = E["proximity_model"]; ps = E["proximity_shares"]; chk("proximity 35 / 58, ΔAIC +59", f"assigned {ps['THA']:.0f} % of the loss to Thailand and {ps['MMR']:.0f} %" in t and f"ΔAIC = +{pm['aic_proximity'] - pm['aic_region_nearfar']:.0f}" in t)
chk("regret 352→84, 550→168", f"from {RG['10']['test_max_regret_best']:.0f} to {RG['10']['test_max_regret_minimax']:.0f} kTHB" in t and f"from {RG['30']['test_max_regret_best']:.0f} to {RG['30']['test_max_regret_minimax']:.0f} kTHB" in t)
chk("uncertainty 81 / 10 / 6", f"contributed {100 * U['sampling_PV_coefficients']:.0f} %, the aerosol response {100 * U['aerosol_response']:.0f} % and the soiling model {100 * U['soiling_model']:.0f} %" in t)
I = pd.read_csv(f"{O}/novelty_out/identifiability_map.csv"); cm = I[I.condition == "Correct model"]; r5 = I[I.condition == "True rain rule 5 mm"]
chk("correct model split at all rho", (cm.report == "split").all()); chk("5 mm dep −16 to −27, total within 8", round(r5.relbias_dep.max()) == -16 and round(r5.relbias_dep.min()) == -27 and r5.relbias_tot.abs().max() < 8.5 and "by −16 to −27 %" in t)
print(f"{sum(ok)} of {len(ok)} text checks passed")
