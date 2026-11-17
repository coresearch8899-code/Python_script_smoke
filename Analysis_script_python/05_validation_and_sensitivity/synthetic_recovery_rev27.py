"""Synthetic recovery experiment: can the clear-day PV model separate atmospheric and deposition losses when the truth is known?
Yields are simulated on the real clear-day sample (same dates, zones, AOD, temperature, rain and PM2.5) with known coefficients, and the selected
model (Eq. 5, partial-cleaning dose) is refitted. Residuals are the fitted residuals resampled in 14-day blocks with the zones of a day kept
together, so their autocorrelation is kept. Each simulated data set also gets a block bootstrap with 28-day calendar blocks, the method of the paper, which gives the coverage of the reported
nominal 95 % intervals. This tests the method under stated scenarios; it is not an external validation.
Usage: synthetic_recovery_rev27.py <daily_csv> <out_dir> <joint_script> [sims=500] [boot=300]"""
import sys, json, numpy as np, pandas as pd
D3, OUT, JS = sys.argv[1:4]; SIMS = int(sys.argv[4]) if len(sys.argv) > 4 else 500; BOOT = int(sys.argv[5]) if len(sys.argv) > 5 else 300
src = open(JS, encoding="utf-8").read(); src = src[:src.index("point_lin, point_hin = losses(PV)")]
sys.argv = [JS, D3, OUT]; G = {}; exec(compile(src, JS, "exec"), G)
PV, d, X, sm, annual, CLRW, KWP_TOT, AOD_BG, CLEAN, COMMISSION = (G[k] for k in ("PV", "d", "X", "sm", "annual", "CLRW", "KWP_TOT", "AOD_BG", "CLEAN", "COMMISSION"))
days = PV.index; T = len(PV); rng = np.random.default_rng(2026)
def build_dose(rain, pm, pf=2.0, reset=False):
    wash = d.index.isin(CLEAN + pd.Timedelta(days=1)) | (d.index == COMMISSION)
    keep = (rain < pf) * 1.0 if reset else 1 - np.minimum(1.0, rain / pf)
    o = np.full(len(d), np.nan); D = 0.0
    for i in range(len(d)):
        D = 0.0 if wash[i] else D * keep[i]
        if not np.isnan(pm[i]): D += pm[i]; o[i] = D
    return pd.Series(o, index=d.index)
RAIN = d.e5_tp.fillna(0).values; PM = d.pm25.values / 1000.0; D0 = build_dose(RAIN, PM)
assert np.allclose(D0.reindex(days).values, PV.soil.values)
S_atm = G["S_atm_lin"]
S_dep = lambda dose: annual(dose * d.e5_ssrd * KWP_TOT)
S_atm_h = annual((d.aod_ceres - 0.30).clip(lower=0) * d.e5_ssrd * CLRW * KWP_TOT)
X0 = X(PV, PV.aod); iA, iS = list(X0.columns).index("aod"), list(X0.columns).index("soil")
m0 = sm.OLS(PV.ratio, X0).fit(); RES = m0.resid.values; b0 = m0.params.values.copy(); BA, BS = b0[iA], b0[iS]
blk = (days - days.min()).days // 14; ub = np.unique(blk); nb = len(ub); pos = {k: np.where(blk == k)[0] for k in ub}
def _res():
    out = np.empty(T); i = 0; draw = rng.choice(ub, 3 * nb, replace=True)
    for k in draw:
        r = RES[pos[k]]; n = min(len(r), T - i); out[i:i + n] = r[:n]; i += n
        if i >= T: break
    return out
yr = PV.yr.values; doy = days.dayofyear.values
base_rest = (lambda b: X0.values @ b)(np.where(np.isin(np.arange(len(b0)), [iA, iS]), 0, b0))     # zones, trends, temperature
blk28 = (days - days.min()).days // 28; ub28 = np.unique(blk28); pos28 = {k: np.where(blk28 == k)[0] for k in ub28}; nb28 = len(ub28)
COUNTS = rng.multinomial(nb28, np.ones(nb28) / nb28, size=BOOT)
def fit_all(Xm, y):
    idx = [pos28[k] for k in ub28]
    XtX = np.stack([Xm[i].T @ Xm[i] for i in idx]); Xty = np.stack([Xm[i].T @ y[i] for i in idx])
    b = np.linalg.solve(XtX.sum(0), Xty.sum(0))
    A = np.tensordot(COUNTS, XtX, axes=(1, 0)); c = COUNTS @ Xty
    return b, np.linalg.solve(A, c[..., None])[..., 0]
def run(name, ba=BA, bs=BS, dose_true=None, dose_fit=None, dose_fit_fn=None, extra=None, extra_fn=None, aod_true=None, aod_fit_fn=None, hinge=None, note=""):
    Dt = D0 if dose_true is None else dose_true; Dt_v = Dt.reindex(days).values
    Dfix = D0 if dose_fit is None else dose_fit; A_t = PV.aod.values if aod_true is None else aod_true
    signal = base_rest + bs * Dt_v + (hinge * np.clip(A_t - 0.30, 0, None) if hinge is not None else ba * A_t) + (0 if extra is None else extra)
    t_atm = (-hinge * S_atm_h) if hinge is not None else (-ba * S_atm); t_dep = -bs * S_dep(Dt); t = dict(atm=t_atm, dep=t_dep, tot=t_atm + t_dep, frac=100 * t_atm / (t_atm + t_dep))
    Xb = X0.values.copy(); Xb[:, iS] = Dfix.reindex(days).values; Sd = S_dep(Dfix); rows = []
    for s in range(SIMS):
        Xs = Xb
        if dose_fit_fn is not None:
            Df = dose_fit_fn(); Xs = Xs.copy(); Xs[:, iS] = Df.reindex(days).values; Sd = S_dep(Df)
        if aod_fit_fn is not None:
            Xs = Xs.copy(); Xs[:, iA] = aod_fit_fn()
        y = signal + _res() + (0 if extra_fn is None else extra_fn()); ok = ~np.isnan(Xs).any(1); b, bb = fit_all(np.nan_to_num(Xs), y)
        at, de = -b[iA] * S_atm, -b[iS] * Sd; ab, db = -bb[:, iA] * S_atm, -bb[:, iS] * Sd; tb = ab + db; fb = 100 * ab / tb
        r = dict(atm=at, dep=de, tot=at + de, frac=100 * at / (at + de))
        for k, v, tv in (("atm", ab, t["atm"]), ("dep", db, t["dep"]), ("tot", tb, t["tot"]), ("frac", fb, t["frac"])):
            lo, hi = np.quantile(v, [.025, .975]); r[f"cov_{k}"] = lo <= tv <= hi
        rows.append(r)
    O = pd.DataFrame(rows); row = dict(scenario=name, note=note)
    for k in ("atm", "dep", "tot", "frac"):
        row[f"true_{k}"] = t[k]; row[f"mean_{k}"] = O[k].mean(); row[f"bias_{k}"] = O[k].mean() - t[k]
        row[f"rmse_{k}"] = float(np.sqrt(((O[k] - t[k]) ** 2).mean())); row[f"coverage_{k}"] = 100 * O[f"cov_{k}"].mean()
    print(f"{name:34s} true {t['atm']:6.1f}/{t['dep']:6.1f}/{t['tot']:6.1f}/{t['frac']:5.1f}  est {row['mean_atm']:6.1f}/{row['mean_dep']:6.1f}/{row['mean_tot']:6.1f}/{row['mean_frac']:5.1f}  rmse {row['rmse_atm']:5.1f}/{row['rmse_dep']:5.1f}/{row['rmse_tot']:5.1f}/{row['rmse_frac']:4.1f}  cov {row['coverage_atm']:3.0f}/{row['coverage_dep']:3.0f}/{row['coverage_tot']:3.0f}/{row['coverage_frac']:3.0f}", flush=True)
    return row
R = []
R.append(run("A1 Correctly specified", note="beta values as fitted"))
R.append(run("A2 Atmospheric share lower", ba=BA * 0.5, note="beta_A halved"))
R.append(run("A3 Atmospheric share higher", bs=BS * 0.5, note="beta_S halved"))
# stronger overlap of dose and AOD: both the truth and the fitted dose are tilted towards AOD
A_all = d.aod_ceres; r0 = np.corrcoef(PV.aod, PV.soil)[0, 1]
def tilt(target):
    a = (A_all - PV.aod.mean()) / PV.aod.std() * PV.soil.std(); lo, hi = 0.0, 20.0
    for _ in range(40):
        k = (lo + hi) / 2; s = (D0 + k * a).reindex(days).values; ok = ~np.isnan(s)
        if np.corrcoef(s[ok], PV.aod.values[ok])[0, 1] < target: lo = k
        else: hi = k
    return (D0 + k * a).clip(lower=0).fillna(0)
for tg in (0.7, 0.9):
    Dc = tilt(tg); sc = S_dep(D0) / S_dep(Dc)
    R.append(run(f"B{1 if tg == 0.7 else 2} Dose-AOD correlation {tg}", bs=BS * sc, dose_true=Dc, dose_fit=Dc, note=f"baseline correlation {r0:.2f}; beta_S rescaled to keep the deposition loss"))
# rain errors in the fitted dose
R.append(run("C1 Rain error, lognormal sigma 0.7", dose_fit_fn=lambda: build_dose(RAIN * rng.lognormal(0, 0.7, len(RAIN)), PM), note="dose in the fit built from noisy rain"))
R.append(run("C2 Rain timing, one day late", dose_fit=build_dose(np.r_[0.0, RAIN[:-1]], PM), note="rain shifted by one day in the fit"))
# wrong rain rule
D5 = build_dose(RAIN, PM, 5.0); R.append(run("D1 Truth: complete cleaning at 5 mm", bs=BS * S_dep(D0) / S_dep(D5), dose_true=D5, note="fit assumes 2 mm; true deposition loss kept"))
Dr = build_dose(RAIN, PM, 2.0, True); R.append(run("D2 Truth: full reset at 2 mm", bs=BS * S_dep(D0) / S_dep(Dr), dose_true=Dr, note="fit assumes partial cleaning; true deposition loss kept"))
# dose biased in the smoke season (CAMS forecast about 3.5 times below EAC4 in March-April 2024)
PMb = np.where(np.isin(d.index.month, [3, 4]), PM * 2.0, PM); Db = build_dose(RAIN, PMb)
R.append(run("E1 PM2.5 twice as high in Mar-Apr", bs=BS * S_dep(D0) / S_dep(Db), dose_true=Db, note="fit uses the unscaled PM2.5; true deposition loss kept"))
# reference or sensor problems
R.append(run("F1 Linear drift -1.5 % a-1", extra=-0.015 * 0.583 * yr, note="absorbed by zone trends if linear"))
R.append(run("F2 Step of -3 % from 2025", extra=-0.03 * 0.583 * (days >= "2025-01-01"), note="unmodelled step"))
R.append(run("F3 Seasonal reference bias 3 %", extra=-0.03 * 0.583 * np.cos(2 * np.pi * (doy - 95) / 365.25), note="unmodelled, lowest ratio in early April"))
# aerosol response and AOD error
R.append(run("G1 Truth: hinge at AOD 0.30", hinge=-0.159, note="fit is linear"))
R.append(run("G2 AOD noise, SD 0.10", aod_fit_fn=lambda: PV.aod.values + rng.normal(0, 0.10, T), note="noise in the fitted AOD"))
# rev28: truth differs from the assumed process, low overlap, combined errors
def tilt_low(target):
    a = (A_all - PV.aod.mean()) / PV.aod.std() * PV.soil.std(); lo, hi = -5.0, 0.0
    for _ in range(40):
        k = (lo + hi) / 2; s = (D0 + k * a).reindex(days).values; ok = ~np.isnan(s)
        if np.corrcoef(s[ok], PV.aod.values[ok])[0, 1] > target: hi = k
        else: lo = k
    return (D0 + k * a).clip(lower=0).fillna(0)
Dl = tilt_low(0.2); R.append(run("H1 Dose-AOD correlation 0.2", bs=BS * S_dep(D0) / S_dep(Dl), dose_true=Dl, dose_fit=Dl, note="low overlap"))
def build_keep(keepf):
    wash = d.index.isin(CLEAN + pd.Timedelta(days=1)) | (d.index == COMMISSION); o = np.full(len(d), np.nan); D = 0.0; kp = keepf(RAIN)
    for i in range(len(d)):
        D = 0.0 if wash[i] else D * kp[i]
        if not np.isnan(PM[i]): D += PM[i]; o[i] = D
    return pd.Series(o, index=d.index)
Ds = build_keep(lambda P: np.maximum(0.3, 1 - P / 2.0)); R.append(run("H2 Truth: rain removes at most 70 %", bs=BS * S_dep(D0) / S_dep(Ds), dose_true=Ds, note="fit assumes complete cleaning at 2 mm; true deposition loss kept"))
De = build_keep(lambda P: np.exp(-P / 2.0)); R.append(run("H3 Truth: exponential rain removal", bs=BS * S_dep(D0) / S_dep(De), dose_true=De, note="non-linear removal; true deposition loss kept"))
R.append(run("H4 PM2.5 noise, lognormal sigma 0.5", dose_fit_fn=lambda: build_dose(RAIN, PM * rng.lognormal(0, 0.5, len(PM))), note="daily noise in the fitted dose"))
ud = {t: i for i, t in enumerate(pd.unique(days))}; iday = np.array([ud[t] for t in days])
R.append(run("H5 Irradiance noise 5 %", extra_fn=lambda: 0.05 * 0.583 * rng.normal(0, 1, len(ud))[iday], note="daily noise shared by the zones"))
R.append(run("H6 Combined errors", bs=BS * S_dep(D0) / S_dep(De), dose_true=De, extra=-0.015 * 0.583 * np.cos(2 * np.pi * (doy - 95) / 365.25),
             dose_fit_fn=lambda: build_dose(RAIN * rng.lognormal(0, 0.5, len(RAIN)), PM), aod_fit_fn=lambda: PV.aod.values + rng.normal(0, 0.05, T), note="exponential removal, 1.5 % seasonal bias, rain and AOD noise"))
T_ = pd.DataFrame(R); T_.to_csv(f"{OUT}/synthetic_recovery.csv", index=False)
json.dump(dict(sims=SIMS, boot=BOOT, seed=2026, corr_dose_aod=float(r0)), open(f"{OUT}/synthetic_recovery_meta.json", "w"), indent=1)
