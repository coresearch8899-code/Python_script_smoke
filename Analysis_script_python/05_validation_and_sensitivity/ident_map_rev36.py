"""Identifiability map of the atmospheric-soiling decomposition: dose-AOD correlation (0, 0.3, 0.5, 0.7, 0.9) crossed with five error conditions, using the synthetic recovery
machinery of synthetic_recovery_rev27.py (500 data sets, 300 bootstrap replicates, 28-day blocks per cell).
Classes, fixed before this grid was run (but after the scenario results of Table S8 were known):
  reliable     |relative bias| <= 10 % (fraction: <= 5 points) and coverage of the nominal 95 % interval >= 85 %
  conditional  |relative bias| <= 20 % (fraction: <= 10 points) and coverage >= 70 %
  unreliable   otherwise
Usage: ident_map_rev36.py <daily_csv> <out_dir> <joint_script> <synthetic_script> [sims=500] [boot=300]"""
import sys, json, numpy as np, pandas as pd
D3, OUT, JS, SY = sys.argv[1:5]; SIMS = int(sys.argv[5]) if len(sys.argv) > 5 else 500; BOOT = int(sys.argv[6]) if len(sys.argv) > 6 else 300
src = open(SY, encoding="utf-8").read(); src = src[:src.index("\nR = []\n")]
sys.argv = [SY, D3, OUT, JS, str(SIMS), str(BOOT)]; G = {"__name__": "ident"}; exec(compile(src, SY, "exec"), G)
run, build_dose, D0, RAIN, PM, PV, d, S_dep, BS, days, T, rng = (G[k] for k in ("run", "build_dose", "D0", "RAIN", "PM", "PV", "d", "S_dep", "BS", "days", "T", "rng"))
A_all = d.aod_ceres; r0 = float(np.corrcoef(PV.aod, PV.soil)[0, 1])
def tilt(target):
    a = (A_all - PV.aod.mean()) / PV.aod.std() * PV.soil.std(); lo, hi = (0.0, 20.0) if target > r0 else (-5.0, 0.0)
    for _ in range(50):
        k = (lo + hi) / 2; s = (D0 + k * a).clip(lower=0).fillna(0).reindex(days).values
        if np.corrcoef(s, PV.aod.values)[0, 1] < target: lo = k
        else: hi = k
    return (D0 + k * a).clip(lower=0).fillna(0)
def cls(b, c, frac=False):
    t1, t2 = (5, 10) if frac else (10, 20)
    return "reliable" if abs(b) <= t1 and c >= 85 else "conditional" if abs(b) <= t2 and c >= 70 else "unreliable"
rows = []
for rho in (0.0, 0.3, 0.5, 0.7, 0.9):
    Dc = tilt(rho); rc = float(np.corrcoef(Dc.reindex(days).values, PV.aod.values)[0, 1]); sc = S_dep(D0) / S_dep(Dc)
    # error conditions applied on top of the tilted dose
    D5 = tilt_ = None
    pf5 = build_dose(RAIN, PM, 5.0); Dr = build_dose(RAIN, PM, 2.0, True); PMb = np.where(np.isin(d.index.month, [3, 4]), PM * 2.0, PM); Db = build_dose(RAIN, PMb)
    k_ = (Dc - D0)                                                                        # tilt component, kept identical in truth and fit
    conds = {"Correct model": dict(dose_true=Dc, dose_fit=Dc, bs=BS * sc),
             "True rain rule 5 mm": dict(dose_true=(pf5 + k_).clip(lower=0), dose_fit=Dc, bs=BS * S_dep(D0) / S_dep((pf5 + k_).clip(lower=0))),
             "True rain rule 2 mm reset": dict(dose_true=(Dr + k_).clip(lower=0), dose_fit=Dc, bs=BS * S_dep(D0) / S_dep((Dr + k_).clip(lower=0))),
             "March-April PM2.5 bias x 2": dict(dose_true=(Db + k_).clip(lower=0), dose_fit=Dc, bs=BS * S_dep(D0) / S_dep((Db + k_).clip(lower=0))),
             "AOD noise SD 0.10": dict(dose_true=Dc, dose_fit=Dc, bs=BS * sc, aod_fit_fn=lambda: PV.aod.values + rng.normal(0, 0.10, T))}
    for name, kw in conds.items():
        r = run(f"rho {rho} | {name}", **kw); r.update(rho_target=rho, rho_actual=rc, condition=name)
        for k in ("atm", "dep", "tot"):
            r[f"relbias_{k}"] = 100 * r[f"bias_{k}"] / r[f"true_{k}"]; r[f"class_{k}"] = cls(r[f"relbias_{k}"], r[f"coverage_{k}"])
        r["class_frac"] = cls(r["bias_frac"], r["coverage_frac"], True)
        r["report"] = ("split" if r["class_atm"] == r["class_dep"] == "reliable" and r["class_frac"] == "reliable" else
                       "total with component range" if r["class_tot"] != "unreliable" else "no estimate")
        rows.append(r)
M = pd.DataFrame(rows); M.to_csv(f"{OUT}/identifiability_map.csv", index=False)
print(M.pivot_table(index="condition", columns="rho_target", values="report", aggfunc="first").to_string())
print(M.pivot_table(index="condition", columns="rho_target", values="relbias_dep").round(1).to_string()); print(M.pivot_table(index="condition", columns="rho_target", values="coverage_dep").round(0).to_string())
print(M.pivot_table(index="condition", columns="rho_target", values="relbias_tot").round(1).to_string()); print(M.groupby("rho_target").rho_actual.first().round(2).to_dict())
