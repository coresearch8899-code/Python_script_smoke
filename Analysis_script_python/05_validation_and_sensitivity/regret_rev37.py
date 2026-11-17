"""Decision robustness of the washing rule: minimax regret over 36 soiling models (3 rain rules x 3 beta_S values x 4 wash efficiencies).
The operator triggers on the primary dose (partial cleaning, complete at 2 mm), which is reset by each wash; the true soiling follows the model m and keeps the
fraction 1 - eta at a wash. Policies are thresholds D* (0.25-4.0) or no washing. The minimax-regret policy and the best-estimate policy (training beta_S,
eta = 1, primary rule) are both selected on 2023-2024 and evaluated on 2025-2026, which starts from each model's dose on 31 December 2024 under the logged washes.
Daily order as in the primary replays: trigger on the dose at the end of the previous day, wash, rain, deposition.
rev37: calendar washing every 60, 120 or 180 days and no washing are evaluated as baselines on the same models and costs, and the minimax choice is repeated without the most extreme models (50 % wash efficiency; full-reset rain rule; both).
Usage: regret_rev37.py <step3_csv> <out_dir> <robustness_script> <block_length_joint.json>"""
import sys, json, numpy as np, pandas as pd
D3, OUT, ROB, BLJ = sys.argv[1:5]
s = open(ROB, encoding="utf-8").read(); s = s[:s.index("\nR = {}\n")]
sys.argv = [ROB, D3, OUT]; G = {}; exec(compile(s, ROB, "exec"), G)
d, METRICS, frame, fit, E_idx, KWP_TOT, COMMISSION, TARIFF, PRIMARY = (G[k] for k in ("d", "METRICS", "frame", "fit", "E_idx", "KWP_TOT", "COMMISSION", "TARIFF", "PRIMARY"))
ALL = d.index[(d.index >= COMMISSION) & (d.index <= E_idx.max())]; cut = pd.Timestamp("2024-12-31"); TR, TE = ALL <= cut, ALL > cut
has = (ALL.isin(E_idx) & d.pm25.reindex(ALL).notna().values); x = d.pm25.reindex(ALL).fillna(0).values / 1000; tp = d.e5_tp.reindex(ALL).fillna(0).values; H = d.e5_ssrd.reindex(ALL).fillna(0).values
RULES = {"partial 2 mm": ("PM2.5 dose, partial rain cleaning (primary)", "partial", 2.0), "partial 5 mm": ("PM2.5 dose, partial rain cleaning, complete at 5 mm", "partial", 5.0), "reset 0.5 mm": ("PM2.5 dose, 0.5 mm reset", "reset", 0.5)}
bq = json.load(open(BLJ))["blocks"]["28"]["bS"]
MODELS = []
for rn, (mname, kind, th) in RULES.items():
    btr = float(fit(frame(METRICS[mname][0])[lambda q: q.index <= cut]).params.soil); bfull = float(fit(frame(METRICS[mname][0])).params.soil); sc = btr / bfull
    S0 = float(METRICS[mname][0].loc[cut])
    for bn, b in (("training", btr), ("low", float(bq[1]) * sc), ("high", float(bq[0]) * sc)):
        for eta in (1.0, 0.9, 0.75, 0.5): MODELS.append(dict(rule=rn, kind=kind, th=th, beta_name=bn, beta=b, eta=eta, S0=S0))
S0_obs = float(METRICS[PRIMARY][0].loc[cut])
def replay(Dstar, m, mask, start):
    Do, Dt = (0.0, 0.0) if start == "train" else (S0_obs, m["S0"]); n = 0; loss = 0.0; last = 0
    for k_, i in enumerate(np.where(mask)[0]):
        trig = (Dstar is not None and not isinstance(Dstar, str) and Do >= Dstar) or (isinstance(Dstar, str) and k_ - last >= int(Dstar[3:]))
        if trig: Do = 0.0; Dt *= 1 - m["eta"]; n += 1; last = k_
        Do *= 1 - min(1.0, tp[i] / 2.0)
        Dt = (0.0 if tp[i] >= m["th"] else Dt) if m["kind"] == "reset" else Dt * (1 - min(1.0, tp[i] / m["th"]))
        Do += x[i]; Dt += x[i]; loss += Dt * H[i] * KWP_TOT * has[i]
    return n, loss / 1000                                         # MWh per unit -beta
GRID = [None] + list(np.round(np.arange(0.25, 4.01, 0.25), 2)); CAL = ["cal60", "cal120", "cal180"]; ALLP = GRID + CAL
pre = {(str(D), j, per): replay(D, m, TR if per == "train" else TE, per) for D in ALLP for j, m in enumerate(MODELS) for per in ("train", "test")}
rows = []; summ = {}
for cost in (10_000, 20_000, 30_000, 50_000, 80_000, 100_000):
    J = {per: np.array([[pre[(str(D), j, per)][0] * cost + (-m["beta"]) * pre[(str(D), j, per)][1] * 1000 * TARIFF for j, m in enumerate(MODELS)] for D in ALLP]) for per in ("train", "test")}
    Rg = {per: J[per] - J[per][:len(GRID)].min(0, keepdims=True) for per in J}             # regret against the best dose or no-wash policy
    i_mm = int(np.argmin(Rg["train"][:len(GRID)].max(1))); j_be = next(j for j, m in enumerate(MODELS) if m["rule"] == "partial 2 mm" and m["beta_name"] == "training" and m["eta"] == 1.0); i_be = int(np.argmin(J["train"][:len(GRID), j_be]))
    sens = {}
    for sn, keep in (("without 50 % efficiency", [m["eta"] > 0.5 for m in MODELS]), ("without full-reset rule", [m["kind"] != "reset" for m in MODELS]), ("without both", [m["eta"] > 0.5 and m["kind"] != "reset" for m in MODELS])):
        kk = np.array(keep); im_ = int(np.argmin(Rg["train"][:len(GRID)][:, kk].max(1))); sens[sn] = dict(Dstar=GRID[im_], test_max_regret=float(Rg["test"][im_][kk].max() / 1000), test_max_regret_full_minimax=float(Rg["test"][i_mm][kk].max() / 1000))
    base = {p: dict(test_max_regret=float(Rg["test"][ALLP.index(p)].max() / 1000), test_mean_regret=float(Rg["test"][ALLP.index(p)].mean() / 1000)) for p in [None] + CAL}
    summ[str(cost // 1000)] = dict(minimax_Dstar=GRID[i_mm], best_estimate_Dstar=GRID[i_be],
        test_max_regret_minimax=float(Rg["test"][i_mm].max() / 1000), test_max_regret_best=float(Rg["test"][i_be].max() / 1000),
        test_mean_regret_minimax=float(Rg["test"][i_mm].mean() / 1000), test_mean_regret_best=float(Rg["test"][i_be].mean() / 1000),
        train_max_regret_minimax=float(Rg["train"][i_mm].max() / 1000), train_max_regret_best=float(Rg["train"][i_be].max() / 1000),
        test_regret_no_wash_max=float(Rg["test"][0].max() / 1000), baselines={str(k): v for k, v in base.items()}, sensitivity=sens)
    for j, m in enumerate(MODELS): rows.append(dict(cost_kTHB=cost // 1000, **{k: m[k] for k in ("rule", "beta_name", "eta")}, regret_minimax=Rg["test"][i_mm, j] / 1000, regret_best=Rg["test"][i_be, j] / 1000))
pd.DataFrame(rows).to_csv(f"{OUT}/washing_regret.csv", index=False); json.dump(dict(n_models=len(MODELS), summary=summ), open(f"{OUT}/washing_regret.json", "w"), indent=1, default=float)
print(pd.DataFrame(summ).T[["minimax_Dstar", "best_estimate_Dstar", "test_max_regret_minimax", "test_max_regret_best", "test_mean_regret_minimax", "test_mean_regret_best"]].round(1).to_string())
for c, v in summ.items(): print(c, {k: round(b["test_max_regret"]) for k, b in v["baselines"].items()}, {k: (b["Dstar"], round(b["test_max_regret"]), round(b["test_max_regret_full_minimax"])) for k, b in v["sensitivity"].items()})
