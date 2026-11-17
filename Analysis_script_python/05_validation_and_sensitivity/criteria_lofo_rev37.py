"""Leave-one-family-out check of the identifiability reporting criteria. Target of a cell: the components are usable, i.e. the relative RMSE of both the atmospheric and the
deposition term is at most 25 %. A criterion classes a cell as 'split' when |relative bias| <= t for both terms and coverage >= c for both. For each error family, (t, c) is
chosen on the other four families (best accuracy; ties broken towards the prespecified t = 10 %, c = 85 %) and tested on the held-out family. Also reported: accuracy of the
prespecified criterion and the share of cells whose reporting level changes when t and c are moved by -50 % / +50 %. Usage: criteria_lofo_rev37.py <identifiability_map.csv> <out_json>"""
import sys, json, itertools, numpy as np, pandas as pd
M = pd.read_csv(sys.argv[1])
M["target"] = ((100 * M.rmse_atm / M.true_atm) <= 25) & ((100 * M.rmse_dep / M.true_dep) <= 25)
def split(df, t, c): return (df.relbias_atm.abs() <= t) & (df.relbias_dep.abs() <= t) & (df.coverage_atm >= c) & (df.coverage_dep >= c)
T, Cc = [5, 7.5, 10, 15, 20], [70, 75, 80, 85, 90]; out = {}
for fam in M.condition.unique():
    tr, te = M[M.condition != fam], M[M.condition == fam]
    sc = sorted(((float((split(tr, t, c) == tr.target).mean()), -abs(t - 10) - abs(c - 85) / 5, t, c) for t, c in itertools.product(T, Cc)), reverse=True)
    _, _, t, c = sc[0]; out[fam] = dict(t=t, c=c, train_acc=sc[0][0], test_acc=float((split(te, t, c) == te.target).mean()), test_acc_prespecified=float((split(te, 10, 85) == te.target).mean()))
res = dict(per_family=out, overall_prespecified_acc=float((split(M, 10, 85) == M.target).mean()), lofo_mean_test_acc=float(np.mean([v["test_acc"] for v in out.values()])),
           share_cells_changed_by_threshold_perturbation=float(np.mean([(split(M, 10 * f, min(99, 85 * g)) != split(M, 10, 85)).mean() for f in (0.5, 1.5) for g in (0.95, 1.05)])),
           target_share=float(M.target.mean()))
json.dump(res, open(sys.argv[2], "w"), indent=1); print(json.dumps(res, indent=1))
