"""Why do the nominal 95 % intervals undercover in the synthetic recovery experiment? Correctly specified model, known truth; each data set is analysed with several
interval methods: non-overlapping calendar blocks (the method of the paper; 7, 14, 28, 56 d), moving blocks, normal approximation, cluster-robust t, wild cluster bootstrap.
Two residual generators: non-overlapping blocks (as in Table S8) and moving blocks with random start.
Usage: coverage_diagnostics_rev29.py <daily_csv> <out_dir> <joint_script> [sims=500]"""
import sys, json, numpy as np, pandas as pd
from scipy import stats
D3, OUT, JS = sys.argv[1:4]; SIMS = int(sys.argv[4]) if len(sys.argv) > 4 else 500
src = open(JS, encoding="utf-8").read(); src = src[:src.index("point_lin, point_hin = losses(PV)")]
sys.argv = [JS, D3, OUT]; G = {}; exec(compile(src, JS, "exec"), G)
PV, X, sm, S_atm, S_dep = G["PV"], G["X"], G["sm"], G["S_atm_lin"], G["S_dep"]
rng = np.random.default_rng(2929); T = len(PV); X0 = X(PV, PV.aod); Xm = X0.values; cols = list(X0.columns); iA, iS = cols.index("aod"), cols.index("soil")
m0 = sm.OLS(PV.ratio, X0).fit(); b0 = m0.params.values; RES = m0.resid.values; signal = Xm @ b0
dayn = (PV.index - PV.index.min()).days.values; nd = dayn.max() + 1; K = Xm.shape[1]
TRUE = dict(atm=-b0[iA] * S_atm, dep=-b0[iS] * S_dep); TRUE["tot"] = TRUE["atm"] + TRUE["dep"]; TRUE["frac"] = 100 * TRUE["atm"] / TRUE["tot"]
XtX_day = np.zeros((nd, K, K)); np.add.at(XtX_day, dayn, Xm[:, :, None] * Xm[:, None, :])
def blocks(L): return dayn // L
def losses(bA, bS):
    a, d_ = -bA * S_atm, -bS * S_dep; t = a + d_; return dict(atm=a, dep=d_, tot=t, frac=100 * a / t)
pos14 = {k: np.where(blocks(14) == k)[0] for k in np.unique(blocks(14))}; ub14 = np.array(list(pos14))
def gen_fixed():
    out = np.empty(T); i = 0
    for k in rng.choice(ub14, 3 * len(ub14), replace=True):
        r = RES[pos14[k]]; n = min(len(r), T - i); out[i:i + n] = r[:n]; i += n
        if i >= T: break
    return out
order = np.argsort(dayn, kind="stable"); st = np.searchsorted(dayn, np.arange(nd)); en = np.searchsorted(dayn, np.arange(nd) + 14)
def gen_moving():
    out = np.empty(T); i = 0
    while i < T:
        s = rng.integers(0, nd - 14); r = RES[st[s]:en[s]]; n = min(len(r), T - i)
        if n: out[i:i + n] = r[:n]; i += n
    return out
def fixed_boot(y, L, B):
    b_ = blocks(L); u = np.unique(b_); idx = [np.where(b_ == k)[0] for k in u]
    XX = np.stack([Xm[i].T @ Xm[i] for i in idx]); Xy = np.stack([Xm[i].T @ y[i] for i in idx]); C = rng.multinomial(len(u), np.ones(len(u)) / len(u), size=B)
    return np.linalg.solve(np.tensordot(C, XX, axes=(1, 0)), (C @ Xy)[..., None])[..., 0]
def moving_boot(y, B, L=14):
    Xy_day = np.zeros((nd, K)); np.add.at(Xy_day, dayn, Xm * y[:, None])
    cXX = np.r_[np.zeros((1, K, K)), np.cumsum(XtX_day, 0)]; cXy = np.r_[np.zeros((1, K)), np.cumsum(Xy_day, 0)]
    S = rng.integers(0, nd - L + 1, size=(B, int(np.ceil(nd / L)))); A = (cXX[S + L] - cXX[S]).sum(1); c = (cXy[S + L] - cXy[S]).sum(1)
    return np.linalg.solve(A, c[..., None])[..., 0]
def cr_wild(y, b, L, B):
    e = y - Xm @ b; g = blocks(L); u = np.unique(g); s = np.stack([Xm[g == k].T @ e[g == k] for k in u]); Ai = np.linalg.inv(Xm.T @ Xm); G_ = len(u)
    V = Ai @ (s.T @ s) @ Ai * G_ / (G_ - 1) * (T - 1) / (T - K)
    w = rng.choice([-1.0, 1.0], size=(B, G_)); db = (w @ s) @ Ai
    return V, db, G_
Rq = {}
def add(meth, gen, ok):
    for k, v in ok.items(): Rq.setdefault((meth, gen, k), []).append(bool(v))
def check_pct(meth, gen, bb, kind="percentile", est=None):
    L = losses(bb[:, iA], bb[:, iS]); ok = {}
    for k in TRUE:
        lo, hi = np.quantile(L[k], [.025, .975])
        if kind == "basic": lo, hi = 2 * est[k] - hi, 2 * est[k] - lo
        if kind == "normal": se = L[k].std(ddof=1); lo, hi = est[k] - 1.96 * se, est[k] + 1.96 * se
        ok[k] = lo <= TRUE[k] <= hi
    add(meth, gen, ok)
XtXi = np.linalg.inv(Xm.T @ Xm)
for s in range(SIMS):
    for gen, f in (("fixed", gen_fixed), ("moving", gen_moving)):
        y = signal + f(); b = XtXi @ (Xm.T @ y); est = losses(b[iA], b[iS])
        for L in (7, 14, 28, 56): check_pct(f"calendar {L} d, percentile", gen, fixed_boot(y, L, 300))
        bb = fixed_boot(y, 14, 1000); check_pct("calendar 14 d, percentile, B = 1000", gen, bb); check_pct("calendar 14 d, basic", gen, bb, "basic", est); check_pct("calendar 14 d, normal", gen, bb, "normal", est)
        check_pct("moving 14 d, percentile", gen, moving_boot(y, 300, 14)); check_pct("moving 28 d, percentile", gen, moving_boot(y, 300, 28))
        for L in (14, 28):
            V, db, G_ = cr_wild(y, b, L, 500); t = stats.t.ppf(.975, G_ - 1); ok = {}; se = {}
            gA, gS = np.zeros(K), np.zeros(K); gA[iA] = -S_atm; gS[iS] = -S_dep
            se["atm"] = np.sqrt(gA @ V @ gA); se["dep"] = np.sqrt(gS @ V @ gS); se["tot"] = np.sqrt((gA + gS) @ V @ (gA + gS))
            gf = 100 * (gA * est["dep"] - gS * est["atm"]) / est["tot"] ** 2; se["frac"] = np.sqrt(gf @ V @ gf)
            add(f"cluster-robust t, {L} d", gen, {k: abs(est[k] - TRUE[k]) <= t * se[k] for k in TRUE})
            bw = b[None, :] + db; check_pct(f"wild cluster, {L} d", gen, bw, "basic", est)
    if s % 50 == 0: print(s, flush=True)
T_ = pd.DataFrame([dict(method=m, generator=g, quantity=k, coverage=100 * np.mean(v)) for (m, g, k), v in Rq.items()])
P = T_.pivot_table(index=["method", "generator"], columns="quantity", values="coverage")[["atm", "dep", "tot", "frac"]].round(1); P.to_csv(f"{OUT}/coverage_diagnostics.csv"); print(P.to_string())
print("residual autocorrelation of daily mean residual, lags 1, 7, 14, 28:", [round(float(pd.Series(RES).groupby(dayn).mean().reindex(range(nd)).autocorr(l)), 3) for l in (1, 7, 14, 28)])
