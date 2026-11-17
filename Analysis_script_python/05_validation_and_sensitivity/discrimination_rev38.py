"""rev38 analyses on stored outputs (no refitting).
A  Transport-model discrimination: for each transport set-up, out-of-sample skill of the haze models (novelty37.json), consistency of the arrival level with the daytime boundary
   layer, rank stability and EIDI over its 96 ensemble specifications; attribution within the set-ups that pass a physical-consistency criterion (arrival level inside the
   daily maximum boundary layer on at least half of January-May days) and within the set-up with the best out-of-sample AOD skill.
B  Failure-aware reporting: confusion matrix of the reporting level of the recoverability map against the realised error of the synthetic estimates. Truth of a cell:
   'split' if the relative RMSE of the atmospheric and of the deposition term are both <= tau; otherwise 'total' if the relative RMSE of the total is <= tau; otherwise 'none'.
   tau = 25 % (main), 20 % and 33 % (sensitivity). Binary summaries: sensitivity and specificity of 'split', false-reliable rate (share of 'split' calls that are not usable
   and share of non-usable cells called 'split'), and the same for 'any estimate' (split or total) against a usable total.
Usage: discrimination_rev38.py <novelty_out>"""
import sys, json, numpy as np, pandas as pd
NV = sys.argv[1]
N = json.load(open(f"{NV}/novelty37.json")); EN = pd.read_csv(f"{NV}/attribution_ensemble.csv"); M = pd.read_csv(f"{NV}/identifiability_map.csv")
BL = N["C_blh"]; inside = {"p850": BL["share_days_850_within_daytime_BL"], "3d850": BL["share_days_850_within_daytime_BL"], "p700": BL["share_days_700_within_daytime_BL"]}
def summ(g):
    return dict(n=int(len(g)), thai_first=float((g.L_THA > g.L_MMR).mean()), L_THA_median=float(g.L_THA.median()), L_THA_range=[float(g.L_THA.min()), float(g.L_THA.max())],
                EIDI_THA_median=float(g.EIDI_THA.median()), EIDI_THA_range=[float(g.EIDI_THA.min()), float(g.EIDI_THA.max())], EIDI_MMR_range=[float(g.EIDI_MMR.min()), float(g.EIDI_MMR.max())],
                EIDI_sign_kept=bool((g.EIDI_THA > 0).all() and (g.EIDI_MMR < 0).all()))
A = {}
for t in ("p850", "3d850", "p700"):
    A[t] = dict(loyo_AOD=N["C_fit_by_transport"][t]["AOD"]["loyo_r2"], loyo_lnPM=N["C_fit_by_transport"][t]["lnPM25"]["loyo_r2"], share_days_arrival_in_BL=inside[t],
                physically_consistent=bool(inside[t] >= 0.5), **summ(EN[EN.transport == t]))
best = max(A, key=lambda t: A[t]["loyo_AOD"]); phys = [t for t in A if A[t]["physically_consistent"]]
A["subset_physically_consistent"] = dict(setups=phys, **summ(EN[EN.transport.isin(phys)]))
A["subset_best_out_of_sample_AOD"] = dict(setups=[best], **summ(EN[EN.transport == best]))
A["all"] = summ(EN)
pd.DataFrame({k: {kk: (str(vv) if isinstance(vv, list) else vv) for kk, vv in v.items()} for k, v in A.items()}).T.to_csv(f"{NV}/transport_discrimination.csv")
def truth(df, tau):
    comp = ((100 * df.rmse_atm / df.true_atm) <= tau) & ((100 * df.rmse_dep / df.true_dep) <= tau); tot = (100 * df.rmse_tot / df.true_tot) <= tau
    return np.where(comp, "split", np.where(tot, "total", "none"))
rep = M.report.map({"split": "split", "total with component range": "total", "no estimate": "none"}).values
B = {}; rows = []
for tau in (25, 20, 33):
    tr = truth(M, tau); cm = pd.crosstab(pd.Series(rep, name="reported"), pd.Series(tr, name="truth")).reindex(index=["split", "total", "none"], columns=["split", "total", "none"], fill_value=0)
    sp_r, sp_t = rep == "split", tr == "split"; an_r, an_t = rep != "none", tr != "none"
    sd = lambda a, b: float(a.sum() / b.sum()) if b.sum() else float("nan")
    B[str(tau)] = dict(confusion=cm.to_dict(orient="index"), n_cells=int(len(rep)), exact_agreement=float((rep == tr).mean()),
        split=dict(sensitivity=sd(sp_r & sp_t, sp_t), specificity=sd(~sp_r & ~sp_t, ~sp_t), false_reliable_share_of_calls=sd(sp_r & ~sp_t, sp_r), false_reliable_rate=sd(sp_r & ~sp_t, ~sp_t)),
        any_estimate=dict(sensitivity=sd(an_r & an_t, an_t), specificity=sd(~an_r & ~an_t, ~an_t), false_reliable_share_of_calls=sd(an_r & ~an_t, an_r), false_reliable_rate=sd(an_r & ~an_t, ~an_t)),
        conservative_errors=int(((rep == "none") & (tr != "none")).sum() + ((rep == "total") & (tr == "split")).sum()),
        liberal_errors=int(((rep == "split") & (tr != "split")).sum() + ((rep == "total") & (tr == "none")).sum()))
    for i, r in enumerate(rep): rows.append(dict(tau=tau, condition=M.condition.iloc[i], rho=M.rho_target.iloc[i], reported=r, truth=tr[i]))
pd.DataFrame(rows).to_csv(f"{NV}/recoverability_confusion.csv", index=False)
R = dict(A_transport=A, B_failure=B); json.dump(R, open(f"{NV}/discrimination_rev38.json", "w"), indent=1, default=float); print(json.dumps(R, indent=1, default=float))
