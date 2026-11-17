"""Newey-West covariance on calendar days. Scores are summed over the zones of a day, s_d = sum_z x_zd e_zd, and the Bartlett-weighted autocovariances (lag 7) use calendar-day separation;
days without data contribute zero. Replaces statsmodels HAC on stacked rows for the pooled PV model, and on adjacent rows for the haze models (seasons are not adjacent in time).
Usage: hac_calendar_rev33.py <step3_csv> <step3_dir_with_tfe> <out_dir> <robustness_script> <step4_script>"""
import sys, json, warnings, numpy as np, pandas as pd, statsmodels.api as sm
from scipy import stats
warnings.filterwarnings("ignore")
D3, TD, OUT, ROB, ST4 = sys.argv[1:6]
s = open(ROB, encoding="utf-8").read(); s = s[:s.index("\nR = {}\n")]
sys.argv = [ROB, D3, OUT]; G = {}; exec(compile(s, ROB, "exec"), G)
METRICS, frame, X_, PRIMARY, d = (G[k] for k in ("METRICS", "frame", "X_", "PRIMARY", "d"))
def hac(m, idx, L=7):
    Xv = np.asarray(m.model.exog); e = np.asarray(m.resid); day = (idx - idx.min()).days.values; S = np.zeros((day.max() + 1, Xv.shape[1])); np.add.at(S, day, Xv * e[:, None])
    M = S.T @ S
    for l in range(1, L + 1):
        Gl = S[l:].T @ S[:-l]; M += (1 - l / (L + 1)) * (Gl + Gl.T)
    Ai = np.linalg.inv(Xv.T @ Xv); se = np.sqrt(np.diag(Ai @ M @ Ai)); nm = list(m.model.exog_names)
    return pd.DataFrame({"coef": np.asarray(m.params), "se_old_rows": np.asarray(m.get_robustcov_results("HAC", maxlags=7).bse), "se_ols": np.asarray(m.bse), "se": se, "p": 2 * stats.norm.sf(np.abs(np.asarray(m.params) / se))}, index=nm)
R = {}
P0 = frame(METRICS[PRIMARY][0]); m = sm.OLS(P0.ratio, X_(P0)).fit(); T = hac(m, P0.index); R["table3"] = T.to_dict("index")
Xk = lambda P, k: (lambda b: b.assign(aod=(P.aod - k).clip(lower=0).values) if k else b)(X_(P))
scan = []
for k in np.round(np.arange(0, 0.801, 0.05), 2):
    mh = sm.OLS(P0.ratio, Xk(P0, k)).fit(); h = hac(mh, P0.index).loc["aod"]; scan.append(dict(knot=float(k), aic=mh.aic, slope=h.coef, se=h.se, se_old_rows=h.se_old_rows))
S = pd.DataFrame(scan); S.to_csv(f"{OUT}/hinge_knot_scan_calendar_hac.csv", index=False); R["hinge"] = S[S.knot == 0.30].iloc[0].to_dict()
rows = []
for name, (s_, _) in METRICS.items():
    P = frame(s_); mm = sm.OLS(P.ratio, X_(P)).fit(); h = hac(mm, P.index).loc["soil"]; rows.append(dict(model=name, beta_S=h.coef, se=h.se, p=h.p, p_ols=float(mm.pvalues.soil)))
MS = pd.DataFrame(rows); MS.to_csv(f"{OUT}/soiling_models_calendar_hac.csv", index=False)
sea = {}
for k, mk in (("Jan-May", P0.index.month <= 5), ("Jun-Dec", P0.index.month > 5)):
    g = P0[mk]; mm = sm.OLS(g.ratio, X_(g)).fit(); h = hac(mm, g.index); sea[k] = dict(bA=h.loc["aod", "coef"], bA_se=h.loc["aod", "se"], bS=h.loc["soil", "coef"], bS_se=h.loc["soil", "se"], n=len(g))
R["season"] = sea
src = open(ST4, encoding="utf-8").read(); src = src[:src.index("def attribution(H")]
sys.argv = [ST4, D3, TD, OUT]; H_ = {}; exec(compile(src, ST4, "exec"), H_)
tfe = pd.read_csv(f"{TD}/tfe_p850.csv", index_col=0, parse_dates=True); Hf = H_["haze_frame"](tfe); FIRE, MET, C = H_["FIRE"], H_["MET"], H_["C"]
Hm = Hf.dropna(subset=FIRE + MET + ["pm25", "aod_ceres"]); R["haze"] = {}; R["lead"] = {}
for nm, y in (("logPM25", np.log(Hm.pm25)), ("AOD", Hm.aod_ceres)):
    mm = sm.OLS(y, sm.add_constant(Hm[FIRE + MET])).fit(); h = hac(mm, Hm.index); R["haze"][nm] = h.loc[FIRE].to_dict("index")
    Hl = Hm.copy(); lead = []
    for c in C:
        Hl[f"lead2_{c}"] = np.log1p(tfe[f"tfe_{c}"].reindex(Hl.index + pd.Timedelta(days=2)).values); lead.append(f"lead2_{c}")
    Hl = Hl.dropna(subset=lead); ml = sm.OLS(y.reindex(Hl.index), sm.add_constant(Hl[FIRE + lead + MET])).fit(); hl = hac(ml, Hl.index); R["lead"][nm] = hl.loc[lead + ["lt_THA_0"]].to_dict("index")
# single-series daily regressions of Section 3.1
met = pd.read_csv(f"{TD}/meteo_zone1_daily.csv", index_col=0, parse_dates=True); mt = met.join(d[["ceres_all", "e5_ssrd", "aod_ceres", "e5_tcc_day"]], how="inner"); mt = mt[(mt.n_valid_ghi >= 80) & (mt.ghi_int_kWh > 0.5)]
mt["r"] = mt.ghi_int_kWh / mt.e5_ssrd; mt["yr"] = (mt.index - mt.index[0]).days / 365.25; cl = mt[(mt.e5_tcc_day < 30) & (mt.aod_ceres < 0.4)]; mm = sm.OLS(cl.r, sm.add_constant(cl.yr)).fit(); h = hac(mm, cl.index)
R["drift"] = dict(n=len(cl), pct_per_yr=float(100 * h.loc["yr", "coef"] / h.loc["const", "coef"]), se_pct=float(100 * h.loc["yr", "se"] / h.loc["const", "coef"]), se_pct_old=float(100 * h.loc["yr", "se_old_rows"] / h.loc["const", "coef"]))
c_ = d[["ceres_clr", "e5_ssrdc", "aod_ceres", "e5_tcwv"]].dropna(); c_["ratio"] = c_.ceres_clr / c_.e5_ssrdc; mm = sm.OLS(c_.ratio, sm.add_constant(c_[["aod_ceres", "e5_tcwv"]])).fit(); h = hac(mm, c_.index)
R["clear_ratio"] = dict(n=len(c_), slope=float(h.loc["aod_ceres", "coef"]), se=float(h.loc["aod_ceres", "se"]), se_old=float(h.loc["aod_ceres", "se_old_rows"])); print(R["drift"], R["clear_ratio"])
json.dump(R, open(f"{OUT}/hac_calendar.json", "w"), indent=1, default=float)
print(T.round(5).to_string()); print(S[S.knot.isin([0, .2, .25, .3, .35, .4])].round(4).to_string()); print(MS.round(5).to_string()); print(sea)
for nm in R["haze"]: print(nm); print(pd.DataFrame(R["haze"][nm]).T[["coef", "se_old_rows", "se", "p"]].round(4).to_string()); print(pd.DataFrame(R["lead"][nm]).T[["coef", "se", "p"]].round(4).to_string())
