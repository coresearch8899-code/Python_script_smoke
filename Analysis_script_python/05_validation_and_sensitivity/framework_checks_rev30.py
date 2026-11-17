"""rev30 framework checks (no new data), corrected and extended after review: nearest-neighbour distance excludes the own day, all 15 Shapley coalitions are checked,
every soiling model is crossed with both aerosol responses, and the held-out replay uses training-period parameters and a rain rule selected on the training years only.
A  decomposition across the soiling models that pass stated criteria; best predictor versus model used for fire-linked attribution
B  stability of coefficients, total loss and atmospheric fraction with each calendar year left out
C  support of the fire counterfactuals: Mahalanobis and nearest-neighbour distance of capped exposure from the observed haze-model data
D  washing out of sample: rule and D* chosen on 2023-2024, tested on 2025-2026, for wash efficiency 100-50 % and the beta_S range
Usage: framework_checks_rev30.py <step3_csv> <step3_dir_with_tfe> <out25> <out_dir> <robustness_script> <step4_script> <block_length_joint.json>"""
import sys, json, numpy as np, pandas as pd
D3, TD, O25, OUT, ROB, ST4, BLJ = sys.argv[1:8]
s = open(ROB, encoding="utf-8").read(); s = s[:s.index("\nR = {}\n")]
sys.argv = [ROB, D3, OUT]; G = {}; exec(compile(s, ROB, "exec"), G)
d, METRICS, frame, fit, loyo_rmse, lomo_rmse, PRIMARY, E_idx, KWP_TOT, AOD_BG, CLRW, COMMISSION, TARIFF, CLEAN = (G[k] for k in
    ("d", "METRICS", "frame", "fit", "loyo_rmse", "lomo_rmse", "PRIMARY", "E_idx", "KWP_TOT", "AOD_BG", "CLRW", "COMMISSION", "TARIFF", "CLEAN"))
R = {}
fy = E_idx[(E_idx.year == 2024) | (E_idx.year == 2025)]
S_atm = float(((d.aod_ceres - AOD_BG).clip(lower=0) * d.e5_ssrd * CLRW * KWP_TOT).reindex(fy).sum() / 2 / 1000)
S_dep = lambda s_: float((s_ * d.e5_ssrd * KWP_TOT).reindex(fy).sum() / 2 / 1000)
# ---------------- A: soiling models ----------------
CALP = pd.read_csv(f"{OUT}/soiling_models_calendar_hac.csv").set_index("model").p   # calendar-day Newey-West, hac_calendar_rev33.py
P0 = frame(METRICS[PRIMARY][0]); ry0, _ = loyo_rmse(P0, soil=False); rm0, _ = lomo_rmse(P0, soil=False); rows = []
for name, (s_, rule) in METRICS.items():
    P = frame(s_); m = fit(P); ry, _ = loyo_rmse(P); rm, _ = lomo_rmse(P)
    atm, dep = -m.params.aod * S_atm, -m.params.soil * S_dep(s_)
    gy, gm = 100 * (1 - ry / ry0), 100 * (1 - rm / rm0)
    ok = (m.params.soil < 0) and (CALP[name] < 0.01) and (gy > 0) and (gm > 0)
    rows.append(dict(model=name, beta_S=m.params.soil, p=m.pvalues.soil, aic=m.aic, gain_years_pct=gy, gain_months_pct=gm, atm=atm, dep=dep, total=atm + dep,
                     atm_fraction=100 * atm / (atm + dep), passes=bool(ok), fire_linked=name.startswith("PM2.5")))
A = pd.DataFrame(rows); import statsmodels.api as sm
KN = np.round(np.arange(0.10, 0.61, 0.05), 2); hs = []
for name, (s_, rule) in METRICS.items():
    P = frame(s_); best = None; base = G["X_"](P).drop(columns=["aod"])
    for k in KN:
        Xh = base.copy(); Xh["aod"] = (P.aod - k).clip(lower=0).values; mh = sm.OLS(P.ratio, Xh).fit()
        if best is None or mh.aic < best[0]: best = (mh.aic, k, mh)
    _, k, mh = best; sa = float((((d.aod_ceres - k).clip(lower=0) - max(AOD_BG - k, 0)).clip(lower=0) * d.e5_ssrd * CLRW * KWP_TOT).reindex(fy).sum() / 2 / 1000)
    a_h, d_h = -mh.params.aod * sa, -mh.params.soil * S_dep(s_); hs.append(dict(knot=k, atm_h=a_h, dep_h=d_h, total_h=a_h + d_h, frac_h=100 * a_h / (a_h + d_h)))
A = pd.concat([A, pd.DataFrame(hs)], axis=1); A.to_csv(f"{OUT}/soiling_models_passing.csv", index=False); Y = A[A.passes]
R["A"] = dict(n_pass=int(len(Y)), total=[Y.total.min(), Y.total.max()], atm=[Y.atm.min(), Y.atm.max()], dep=[Y.dep.min(), Y.dep.max()], frac=[Y.atm_fraction.min(), Y.atm_fraction.max()],
              fire=dict(total=[Y[Y.fire_linked].total.min(), Y[Y.fire_linked].total.max()], frac=[Y[Y.fire_linked].atm_fraction.min(), Y[Y.fire_linked].atm_fraction.max()]),
              best_predictor=Y.sort_values("gain_years_pct").model.iloc[-1], passing=Y.model.tolist(),
              all_specs=dict(total=[float(min(Y.total.min(), Y.total_h.min())), float(max(Y.total.max(), Y.total_h.max()))], frac=[float(min(Y.atm_fraction.min(), Y.frac_h.min())), float(max(Y.atm_fraction.max(), Y.frac_h.max()))]),
              selected=dict(total=[float(Y.total_h.iloc[0]), float(Y.total.iloc[0])], frac=[float(Y.frac_h.iloc[0]), float(Y.atm_fraction.iloc[0])]))
# ---------------- B: year left out ----------------
rows = []
for y in sorted(set(P0.index.year)):
    m = fit(P0[P0.index.year != y]); a, b = -m.params.aod * S_atm, -m.params.soil * S_dep(METRICS[PRIMARY][0])
    rows.append(dict(year_left_out=int(y), beta_A=m.params.aod, beta_S=m.params.soil, atm=a, dep=b, total=a + b, atm_fraction=100 * a / (a + b)))
B = pd.DataFrame(rows); B.to_csv(f"{OUT}/year_left_out.csv", index=False)
R["B"] = {k: [float(B[k].min()), float(B[k].max())] for k in ("beta_A", "beta_S", "atm", "dep", "total", "atm_fraction")}
# ---------------- C: counterfactual support ----------------
src = open(ST4, encoding="utf-8").read(); src = src[:src.index("def attribution(H")]
sys.argv = [ST4, D3, TD, OUT]; H_ = {}; exec(compile(src, ST4, "exec"), H_)
Hf = H_["haze_frame"](pd.read_csv(f"{TD}/tfe_p850.csv", index_col=0, parse_dates=True)); FIRE, MET, C = H_["FIRE"], H_["MET"], H_["C"]
Hm = Hf.dropna(subset=FIRE + MET + ["pm25", "aod_ceres"]); Xo = Hm[FIRE + MET].values; mu, sd = Xo.mean(0), Xo.std(0); Zo = (Xo - mu) / sd
Vi = np.linalg.pinv(np.cov(Zo.T)); md = lambda z: np.einsum("ij,jk,ik->i", z - z.mean(0) * 0, Vi, z)
d2o = md(Zo); q99 = float(np.quantile(d2o, 0.99))
def nn(a, b):
    """Nearest-neighbour distance of row i of a to the rows of b, excluding row i of b (leave-own-day-out)."""
    D = np.sqrt(((a[:, None, :] - b[None, :, :]) ** 2).sum(2)); np.fill_diagonal(D, np.inf); return D.min(1)
import itertools
nno = nn(Zo, Zo); nnq = float(np.quantile(nno, 0.99)); sup = {}
for r_ in range(1, 5):
    for co in itertools.combinations(C, r_):
        cols = [x for x in FIRE if any(f"_{c}_" in x for c in co)]; Z = Hm[FIRE + MET].copy()
        for cc in cols: Z[cc] = np.minimum(Z[cc], Hm[cc].quantile(0.10))
        z = (Z.values - mu) / sd; ch = (Z.values != Hm[FIRE + MET].values).any(1); d2 = md(z); n_ = nn(z, Zo)
        sup["+".join(co)] = dict(days_changed=int(ch.sum()), days=int(len(z)), share_beyond_q99_mahal_pct=float(100 * (d2[ch] > q99).mean()), share_beyond_q99_nn_pct=float(100 * (n_[ch] > nnq).mean()),
                                 median_d2=float(np.median(d2[ch])), median_d2_observed=float(np.median(d2o)), median_nn=float(np.median(n_[ch])), median_nn_observed=float(np.median(nno)))
R["C"] = dict(support=sup, q99_mahal=q99, corr_lag0=Hm[[f"lt_{c}_0" for c in C]].corr().round(3).to_dict())
# ---------------- D: washing out of sample ----------------
rule = METRICS[PRIMARY][1]; _, x, thr = rule; ALL = d.index[(d.index >= COMMISSION) & (d.index <= E_idx.max())]; cut = pd.Timestamp("2024-12-31")
has_all = ALL.isin(E_idx) & d.pm25.reindex(ALL).notna().values; xv_all = x.reindex(ALL).fillna(0).values / 1000; tp_all = d.e5_tp.reindex(ALL).fillna(0).values; H_all = d.e5_ssrd.reindex(ALL).fillna(0).values
def replay_eta(D_star, eta, mask, D0=0.0, next_day=False):
    """Unit-beta soiling loss (MWh per unit -beta) and washes on the days in mask. A wash removes the fraction eta of the dose. next_day=True: the trigger uses the dose at the end of
    the previous day, the wash is done at the start of the day, then the rain and PM2.5 of the day act; next_day=False: the rain of the day is known when the trigger is checked."""
    D = float(D0); loss = 0.0; n = 0; xv, tp, H, has = xv_all[mask], tp_all[mask], H_all[mask], has_all[mask]
    for i in range(len(xv)):
        if next_day and D_star is not None and D >= D_star: D *= 1 - eta; n += 1
        D *= 1 - min(1.0, tp[i] / thr)
        if not next_day and D_star is not None and D >= D_star: D *= 1 - eta; n += 1
        D += xv[i]; loss += D * H[i] * KWP_TOT * has[i]
    return n, loss / 1000
TR, TE = ALL <= cut, ALL > cut; GRID = np.round(np.arange(0.25, 4.01, 0.25), 2); ETAS = (1.0, 0.9, 0.75, 0.5)
S0 = float(METRICS[PRIMARY][0].loc[cut]); bq = json.load(open(BLJ))["blocks"]["28"]["bS"]; m_tr = fit(P0[P0.index <= cut]); m_f = fit(P0)
BETAS = dict(training=float(m_tr.params.soil), full=float(m_f.params.soil), low=float(bq[1]), high=float(bq[0]))
cands = {k: METRICS[k] for k in ("PM2.5 dose, partial rain cleaning (primary)", "PM2.5 dose, partial rain cleaning, complete at 5 mm", "PM2.5 dose, 2 mm reset (rev24 primary)", "PM2.5 dose, 0.5 mm reset", "PM2.5 dose, 5 mm reset")}
aic_tr = {k: float(fit(frame(v[0])[lambda q: q.index <= cut]).aic) for k, v in cands.items()}; sel = min(aic_tr, key=aic_tr.get); assert sel == PRIMARY
R["D_rule"] = dict(rain_rule_selected=sel, aic_train=aic_tr, beta_train=BETAS["training"], state_31Dec2024=S0)
def variant(D0, nd):
    cache = {(e, D): (replay_eta(D, e, TR, 0.0, nd), replay_eta(D, e, TE, D0, nd)) for e in ETAS for D in GRID}; none = (replay_eta(None, 1.0, TR, 0.0, nd), replay_eta(None, 1.0, TE, D0, nd)); rows = []
    for cost in np.arange(5, 125, 5) * 1000.0:
        for e in ETAS:
            for bn, b in BETAS.items():
                best = (-b * none[0][1] * 1000 * TARIFF, None)
                for D in GRID:
                    (n, l), _ = cache[(e, D)]; c = n * cost + (-b) * l * 1000 * TARIFF
                    if c < best[0]: best = (c, D)
                Ds = best[1]
                if Ds is None: rows.append(dict(cost_THB=cost, eta=e, beta=bn, D_star=None, test_washes=0, test_saving_THB=0.0)); continue
                n, l = cache[(e, Ds)][1]; rows.append(dict(cost_THB=cost, eta=e, beta=bn, D_star=Ds, test_washes=n, test_saving_THB=(-b) * (none[1][1] - l) * 1000 * TARIFF - n * cost))
    W_ = pd.DataFrame(rows); mc = lambda q: float(q[q.test_saving_THB > 0].cost_THB.max()) if (q.test_saving_THB > 0).any() else 0.0
    return W_, {f"{e}": {b: mc(W_[(W_.eta == e) & (W_.beta == b)]) for b in BETAS} for e in ETAS}
W, main = variant(S0, True); W.to_csv(f"{OUT}/washing_out_of_sample.csv", index=False)
_, zero_same = variant(0.0, False); _, state_same = variant(S0, False); _, zero_next = variant(0.0, True)
agree = [dict(cost_THB=c, share_positive=float((g.test_saving_THB > 0).mean())) for c, g in W.groupby("cost_THB")]
R["D"] = dict(betas=BETAS, max_cost_with_test_saving=main, decision_by_cost=agree, variants=dict(zero_start_same_day=zero_same, state_start_same_day=state_same, zero_start_next_day=zero_next),
              policy_table={str(int(c / 1000)): g[g.beta == "training"].set_index("eta")[["D_star", "test_washes"]].to_dict("index") for c, g in W[W.cost_THB.isin([20000, 50000, 80000])].groupby("cost_THB")})
# ---------------- E: Table 4 on days with all terms; day counts ----------------
mh = None; base = G["X_"](P0).drop(columns=["aod"]); Xh = base.copy(); Xh["aod"] = (P0.aod - 0.30).clip(lower=0).values; import statsmodels.api as sm_; mh = sm_.OLS(P0.ratio, Xh).fit()
K_ = {"Z1": 2256.24, "Z2": 207.06, "Z3": 49.90}; Dose = METRICS[PRIMARY][0].where(d.pm25.notna())
E = pd.DataFrame({"atm": -m_f.params.aod * (d.aod_ceres - AOD_BG).clip(lower=0) * d.e5_ssrd * CLRW * KWP_TOT, "atm_h": -mh.params.aod * (d.aod_ceres - 0.30).clip(lower=0) * d.e5_ssrd * CLRW * KWP_TOT,
                  "dep": -m_f.params.soil * METRICS[PRIMARY][0] * d.e5_ssrd * KWP_TOT, "act": sum(d[f"Y_{z}"] * k for z, k in K_.items()), "dep_h": -mh.params.soil * METRICS[PRIMARY][0] * d.e5_ssrd * KWP_TOT, "aod": d.aod_ceres, "pm": d.pm25})
E = E[(E.index >= COMMISSION) & E.act.notna()]; C_ = E[E.aod.notna() & E.pm.notna()]
tb = []
for y, g in C_.groupby(C_.index.year):
    jm = g.index.month <= 5; a0, de, ac = g.atm.sum() / 1000, g.dep.sum() / 1000, g.act.sum() / 1000; ah, dh = g.atm_h.sum() / 1000, g.dep_h.sum() / 1000
    tb.append(dict(year=int(y), first=str(g.index.min().date()), last=str(g.index.max().date()), days=len(g), E_actual=ac, atm=a0, dep=de, atm_h=ah, dep_h=dh, loss_pct=100 * (a0 + de) / (ac + a0 + de), loss_pct_h=100 * (ah + dh) / (ac + ah + dh), atm_jm=g.atm[jm].sum() / 1000, dep_jm=g.dep[jm].sum() / 1000))
TB = pd.DataFrame(tb); TB.to_csv(f"{OUT}/table4_common_days.csv", index=False)
ex = E[(E.index.year == 2026) & (E.aod.isna() | E.pm.isna())]; R["E"] = dict(all_fleet_days=int(len(E)), common_days=int(len(C_)), ext_days=int(len(ex)), ext_first=str(ex.index.min().date()), ext_last=str(ex.index.max().date()), ext_act=float(ex.act.sum() / 1000), ext_dep=float(ex.dep.sum() / 1000),
    pv_rows_by_year={str(k): int(v) for k, v in P0.groupby(P0.index.year).size().items()}, pv_days=int(P0.index.nunique()), pv_first=str(P0.index.min().date()), pv_last=str(P0.index.max().date()), haze_days=int(len(Hm)), haze_first=str(Hm.index.min().date()), haze_last=str(Hm.index.max().date()),
    replay_days=int(len(ALL)), replay_loss_days=int(has_all.sum()), aod_last=str(d.aod_ceres.last_valid_index().date()), jul_clear=int(((d.e5_tcc_day < 30) & (d.ceres_all / d.ceres_clr > 0.85) & (d.e5_tp < 1)).loc["2026-06-03":"2026-07-30"].sum()), hinge_b=[float(mh.params.aod), float(mh.params.soil)])
# S5c on the same footing as the main replay; full-record break-even with the next-day trigger
def bestbe(mask, D0, beta, nd):
    _, l0 = replay_eta(None, 1.0, mask, D0, nd); b_ = 0.0
    for D in GRID:
        n, l = replay_eta(D, 1.0, mask, D0, nd)
        if n: b_ = max(b_, (-beta) * (l0 - l) * 1000 * TARIFF / n)
    return b_
sv = {str(int(c)): [float(W[(W.cost_THB == c * 1000) & (W.eta == 1.0) & (W.beta == b)].test_saving_THB.iloc[0] / 1000) for b in ("training", "full")] for c in (10, 20, 40)}
AL = np.ones(len(ALL), bool)
R["S5c"] = dict(saving=sv, D_star={str(int(c)): float(W[(W.cost_THB == c * 1000) & (W.eta == 1.0) & (W.beta == "training")].D_star.iloc[0]) for c in (10, 20, 40)}, test_optimum=[bestbe(TE, S0, BETAS["training"], True), bestbe(TE, S0, BETAS["full"], True)],
                test_optimum_zero_same=[bestbe(TE, 0.0, BETAS["training"], False), bestbe(TE, 0.0, BETAS["full"], False)], full_record_next_day=bestbe(AL, 0.0, BETAS["full"], True), full_record_same_day=bestbe(AL, 0.0, BETAS["full"], False))
json.dump(R, open(f"{OUT}/framework_checks.json", "w"), indent=1, default=float); print(R["S5c"])
print(TB.round(1).to_string()); print(json.dumps(R["E"], indent=0)); print("main", main); print("variants", json.dumps(R["D"]["variants"])); print(agree[:6], [a for a in agree if a["cost_THB"] == 100000.0]); print(R["D_rule"]["state_31Dec2024"]); print(R["D"]["policy_table"])
print(json.dumps(R["A"]["all_specs"]), json.dumps(R["C"]["support"]["THA+MMR"]))