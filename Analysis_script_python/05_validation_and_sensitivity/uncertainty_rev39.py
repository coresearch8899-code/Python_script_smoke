"""rev39: variance decomposition of the 2024-2025 total loss with coefficient-sampling variance estimated separately for every specification.
For each retained soiling model (soiling_models_passing.csv) the clear-day PV model is bootstrapped with non-overlapping 28-day calendar blocks (1,000 replicates; the same
blocks are drawn for all models in a replicate), refitting the linear model and the hinge model with its knot re-selected by AIC. Law of total variance over the 14
specifications s (7 soiling models x 2 aerosol responses): Var(T) = E_s[Var(T | s)] + Var_s(E[T | s]); the second term is split into main effects and interaction.
Usage: uncertainty_rev39.py <step3_csv> <out_dir> <robustness_script> <soiling_models_passing.csv> [B]"""
import sys, json, numpy as np, pandas as pd, statsmodels.api as sm
D3, OUT, ROB, SMP = sys.argv[1:5]; NB = int(sys.argv[5]) if len(sys.argv) > 5 else 1000
s = open(ROB, encoding="utf-8").read(); s = s[:s.index("\nR = {}\n")]
sys.argv = [ROB, D3, OUT]; G = {}; exec(compile(s, ROB, "exec"), G)
d, METRICS, frame, fit, E_idx, KWP_TOT, AOD_BG, CLRW, COMMISSION, X_ = (G[k] for k in ("d", "METRICS", "frame", "fit", "E_idx", "KWP_TOT", "AOD_BG", "CLRW", "COMMISSION", "X_"))
fy = E_idx[(E_idx.year == 2024) | (E_idx.year == 2025)]
S_atm = float(((d.aod_ceres - AOD_BG).clip(lower=0) * d.e5_ssrd * CLRW * KWP_TOT).reindex(fy).sum() / 2 / 1000)
KN = np.round(np.arange(0.10, 0.61, 0.05), 2)
S_atm_h = {k: float((((d.aod_ceres - k).clip(lower=0) - max(AOD_BG - k, 0)).clip(lower=0) * d.e5_ssrd * CLRW * KWP_TOT).reindex(fy).sum() / 2 / 1000) for k in KN}
S_dep = lambda s_: float((s_ * d.e5_ssrd * KWP_TOT).reindex(fy).sum() / 2 / 1000)
A = pd.read_csv(SMP); A = A[A.passes]; names = A.model.tolist()
def totals(P, sd):
    m = fit(P); lin = -m.params.aod * S_atm - m.params.soil * sd
    base = X_(P).drop(columns=["aod"]); best = None
    for k in KN:
        Xh = base.copy(); Xh["aod"] = (P.aod - k).clip(lower=0).values; mh = sm.OLS(P.ratio, Xh).fit()
        if best is None or mh.aic < best[0]: best = (mh.aic, k, mh)
    _, k, mh = best; return lin, -mh.params.aod * S_atm_h[k] - mh.params.soil * sd
FR = {n: frame(METRICS[n][0]) for n in names}; SD = {n: S_dep(METRICS[n][0]) for n in names}
point = {n: totals(FR[n], SD[n]) for n in names}
chk = max(max(abs(point[n][0] - r.total), abs(point[n][1] - r.total_h)) for n, r in zip(names, A.itertuples()))
blk = {n: ((FR[n].index - COMMISSION).days // 28).values for n in names}; ub = np.unique(np.concatenate(list(blk.values())))
grp = {n: {b: FR[n][blk[n] == b] for b in ub} for n in names}; rng = np.random.default_rng(39); rows = []
for b in range(NB):
    ks = rng.choice(ub, len(ub), replace=True)
    for n in names:
        Pb = pd.concat([grp[n][k] for k in ks if len(grp[n][k])]); l, h = totals(Pb, SD[n]); rows.append(dict(rep=b, soiling=n, linear=l, hinge=h))
BT = pd.DataFrame(rows); BT.to_csv(f"{OUT}/uncertainty_bootstrap_by_model.csv", index=False)
spec = pd.DataFrame([dict(soiling=n, response=r, total=point[n][0 if r == "linear" else 1], var_boot=float(BT[BT.soiling == n][r].var(ddof=1)),
                          ci=[float(BT[BT.soiling == n][r].quantile(.025)), float(BT[BT.soiling == n][r].quantile(.975))]) for n in names for r in ("linear", "hinge")])
within = float(spec.var_boot.mean()); between = float(spec.total.var(ddof=0))
main = {f: float(spec.groupby(f).total.mean().reindex(spec[f]).var(ddof=0)) for f in ("soiling", "response")}; tot = within + between
prim = names[0]; vp = spec[spec.soiling == prim].var_boot.mean()
mix = pd.concat([BT[c] for c in ("linear", "hinge")])
out = dict(B=NB, max_abs_diff_point_vs_table_S10a=float(chk), sampling_PV_coefficients=within / tot, soiling_model=main["soiling"] / tot, aerosol_response=main["response"] / tot,
           interaction=(between - sum(main.values())) / tot, total_variance=tot, sd_total=tot ** 0.5,
           ratio_var_boot_to_primary=dict(zip(spec.soiling + " | " + spec.response, (spec.var_boot / vp).round(3))), ratio_range=[float((spec.var_boot / vp).min()), float((spec.var_boot / vp).max())],
           mixture_interval_all_specs=[float(mix.quantile(.025)), float(mix.quantile(.975))], primary=prim,
           per_spec=spec.assign(ci=spec.ci.astype(str)).to_dict(orient="records"))
json.dump(out, open(f"{OUT}/uncertainty_decomposition_rev39.json", "w"), indent=1, default=float)
print(json.dumps({k: v for k, v in out.items() if k != "per_spec"}, indent=1, default=float)); print(spec.round(2).to_string())
