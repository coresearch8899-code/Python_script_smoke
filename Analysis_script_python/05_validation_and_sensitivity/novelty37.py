"""rev37 analyses on the existing data.
A  EIDI of the main model: joint 28-day block bootstrap of loss shares (haze models and PV coefficients refitted) and exposure shares (same blocks); H0: EIDI = 0
B  baselines: exposure-proportional, one-at-a-time and region-agnostic proximity attribution; leave-one-year-out skill of the haze models behind them
C  transport plausibility: daytime BLH against the heights of 850 and 700 hPa, and haze-model fit by transport set-up (no ensemble weighting is derived)
D  cross-product check: PV model and attribution with CAMS forecast AOD instead of CERES AOD (not an independent ground truth)
Usage: novelty37.py <step3_csv> <step3_dir_with_tfe> <out_dir> <checks_28d_script>"""
import sys, json, itertools, math, warnings, numpy as np, pandas as pd, statsmodels.api as sm
warnings.filterwarnings("ignore")
D3, TD, OUT, CK = sys.argv[1:5]
src = open(CK, encoding="utf-8-sig").read(); src = src[:src.index("rows = []\nfor b in range(1000):")]
sys.argv = [CK, D3, TD, OUT]; G = {"__name__": "n37"}; exec(compile(src, CK, "exec"), G)
for k in ("d", "C", "H", "FIRE", "MET", "X0", "Zc", "COAL", "unit_values", "shapley_from", "m0", "PV", "pv_X", "COMMISSION", "AOD_BG", "CLRW", "KWP_TOT", "dose_partial", "DEP_MASK", "W_ATM", "EXC", "Hg", "uh", "PVg", "up"): globals()[k] = G[k]
rng = np.random.default_rng(37); R = {}
tfe = pd.read_csv(f"{TD}/tfe_p850.csv", index_col=0, parse_dates=True)
def eshare(idx):
    idx = idx[idx >= COMMISSION]; s = {c: float(tfe[f"tfe_{c}"].reindex(idx).sum()) for c in C}; t = sum(s.values()); return {c: 100 * s[c] / t for c in C}
# ---- A: EIDI bootstrap (28-day blocks) ----
mA0 = sm.OLS(H.aod_ceres, X0).fit(); mP0 = sm.OLS(np.log(H.pm25), X0).fit(); U0 = unit_values(mA0, mP0)
phi0, j0 = shapley_from(U0, m0.params.aod, m0.params.soil); L0 = {c: 100 * phi0[c] / j0 for c in C}; E0 = eshare(H.index)
rows = []
for b in range(1000):
    ks = rng.choice(uh, len(uh), replace=True); Hr = pd.concat([Hg[k] for k in ks])
    Xr = sm.add_constant(Hr[FIRE + MET]); mA = sm.OLS(Hr.aod_ceres, Xr).fit(); mP = sm.OLS(np.log(Hr.pm25), Xr).fit()
    Pb = pd.concat([PVg[k] for k in rng.choice(up, len(up), replace=True)]); mb = sm.OLS(Pb.ratio, pv_X(Pb)).fit()
    ph, jj = shapley_from(unit_values(mA, mP), mb.params.aod, mb.params.soil); E = eshare(Hr.index)
    rows.append({**{f"L_{c}": 100 * ph[c] / jj for c in C}, **{f"E_{c}": E[c] for c in C}})
BT = pd.DataFrame(rows)
for c in C: BT[f"EIDI_{c}"] = BT[f"L_{c}"] - BT[f"E_{c}"]
BT.to_csv(f"{OUT}/eidi_bootstrap.csv", index=False); q = lambda s: [float(s.quantile(.025)), float(s.quantile(.975))]
R["A"] = {c: dict(L=L0[c], E=E0[c], EIDI=L0[c] - E0[c], EIDI_CI=q(BT[f"EIDI_{c}"]), L_CI=q(BT[f"L_{c}"]), E_CI=q(BT[f"E_{c}"]),
                  p_two_sided=float(min(1.0, 2 * min((BT[f"EIDI_{c}"] <= 0).mean(), (BT[f"EIDI_{c}"] >= 0).mean())))) for c in C}
# ---- B: baselines ----
oat = {c: -m0.params.aod * U0[(c,)][0] - m0.params.soil * U0[(c,)][1] for c in C}; so = sum(oat.values())
R["B_shares"] = {"exposure-proportional": E0, "one-at-a-time": {c: 100 * oat[c] / so for c in C}, "Shapley": L0,
                 "one-at-a-time_sum_over_joint": float(so / j0)}
def loyo(y, X):
    e = []
    for yr in sorted(set(X.index.year)):
        tr, te = X.index.year != yr, X.index.year == yr; m = sm.OLS(y[tr], X[tr]).fit(); e.append(y[te] - m.predict(X[te]))
    e = pd.concat(e); return float(1 - (e ** 2).sum() / ((y - y.mean()) ** 2).sum())
near = {c: tfe[f"tfe24_{c}"].reindex(H.index) for c in C}; far = {c: (tfe[f"tfe_{c}"] - tfe[f"tfe24_{c}"]).reindex(H.index) for c in C}; tot = {c: tfe[f"tfe_{c}"].reindex(H.index) for c in C}
def lag2(name, s):
    return pd.DataFrame({f"{name}_0": np.log1p(s), f"{name}_1": np.log1p(s.shift(1))}, index=H.index)
XM = {"meteorology only": H[MET], "total exposure, region-agnostic": lag2("tot", sum(tot.values())).join(H[MET]),
      "near-field and far-field, region-agnostic": lag2("near", sum(near.values())).join(lag2("far", sum(far.values()))).join(H[MET]), "region-specific (main)": H[FIRE + MET]}
sk = {}
for nm, X in XM.items():
    ok = X.notna().all(1); Xc = sm.add_constant(X[ok]); sk[nm] = {}
    for ynm, y in (("AOD", H.aod_ceres[ok]), ("lnPM25", np.log(H.pm25[ok]))):
        m = sm.OLS(y, Xc).fit(); sk[nm][ynm] = dict(loyo_r2=loyo(y, Xc), aic=float(m.aic), n=int(ok.sum()))
R["B_skill"] = sk
# ---- C: transport plausibility ----
jm = d[(d.index.month <= 5) & (d.index >= "2023-01-01")]; H850, H700, ELEV = 1500.0, 3100.0, 400.0   # approximate standard heights a.s.l. in the tropics; site elevation
R["C_blh"] = dict(blh_day_median=float(jm.e5_blh_day.median()), blh_max_median=float(jm.e5_blh_max.median()), share_days_850_within_daytime_BL=float((jm.e5_blh_max >= H850 - ELEV).mean()),
                  share_days_700_within_daytime_BL=float((jm.e5_blh_max >= H700 - ELEV).mean()), heights_agl=[H850 - ELEV, H700 - ELEV])
fitby = {}
for k in ("p850", "p700", "3d850"):
    t_ = pd.read_csv(f"{TD}/tfe_{k}.csv", index_col=0, parse_dates=True); X = pd.DataFrame(index=H.index)
    for c in C:
        for L in (0, 1): X[f"lt_{c}_{L}"] = np.log1p(t_[f"tfe_{c}"].reindex(H.index).shift(L))
    X = X.join(H[MET]); ok = X.notna().all(1); Xc = sm.add_constant(X[ok]); fitby[k] = {}
    for ynm, y in (("AOD", H.aod_ceres[ok]), ("lnPM25", np.log(H.pm25[ok]))):
        m = sm.OLS(y, Xc).fit(); fitby[k][ynm] = dict(loyo_r2=loyo(y, Xc), aic=float(m.aic), r2=float(m.rsquared))
R["C_fit_by_transport"] = fitby
# ---- D: CAMS AOD cross-product check ----
d2 = d.copy(); bg2 = float(d2.loc[d2.index.month.isin([11, 12, 1]), "aod_cams"].median())
P2 = PV.copy(); P2["aod"] = d2.aod_cams.reindex(P2.index).values; P2 = P2.dropna(); m2 = sm.OLS(P2.ratio, pv_X(P2)).fit()
fy = d.index[(d.index.year.isin([2024, 2025])) & d.index.isin(G["d"].index)]
act = sum(d[f"Y_{z}"] * k for z, k in G["KWP"].items()); okd = act.notna() & d.index.year.isin([2024, 2025])
ann = lambda s: float(s[okd].groupby(s[okd].index.year).sum().mean() / 1000)
atm2 = ann(-m2.params.aod * (d.aod_cams - bg2).clip(lower=0) * d.e5_ssrd * CLRW * KWP_TOT); dep2 = ann(-m2.params.soil * d.dose * d.e5_ssrd * KWP_TOT)
H2 = H.copy(); H2["aod_ceres"] = d.aod_cams.reindex(H2.index).values; X2 = sm.add_constant(H2[FIRE + MET]); mA2 = sm.OLS(H2.aod_ceres, X2).fit(); mP2 = sm.OLS(np.log(H2.pm25), X2).fit()
G["EXC"].loc[:] = (H2.aod_ceres - bg2).clip(lower=0).values; G["H"]["aod_ceres"] = H2.aod_ceres.values
U2 = unit_values(mA2, mP2); ph2, j2 = shapley_from(U2, m2.params.aod, m2.params.soil)
R["D_cams"] = dict(beta_A=float(m2.params.aod), beta_S=float(m2.params.soil), aod_bg=bg2, atm=atm2, dep=dep2, total=atm2 + dep2, atm_fraction=100 * atm2 / (atm2 + dep2),
                   shares={c: 100 * ph2[c] / j2 for c in C}, corr_cams_ceres=float(d[["aod_cams", "aod_ceres"]].corr().iloc[0, 1]), haze_r2_AOD=float(mA2.rsquared))
json.dump(R, open(f"{OUT}/novelty37.json", "w"), indent=1, default=float); print(json.dumps(R, indent=1, default=float))
