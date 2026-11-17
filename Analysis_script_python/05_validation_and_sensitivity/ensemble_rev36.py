"""Attribution ensemble: 3 transport set-ups x 4 exposure forms x 2 PV responses x 3 fire-linked soiling doses x 4 counterfactuals = 288 Shapley attributions.
Outputs: rank stability (ARS), share of specifications with Thailand above Myanmar, exposure-impact decoupling index EIDI_c = L_c - E_c (signed Shapley shares, %),
variance decomposition of the shares by factor (main effects of the full factorial), and a proximity test: a haze model in which exposure enters only as
near-field (first 24 h) and far-field sums over all regions, so that a region's attributed loss depends on where along the trajectory its fires were met,
not on a region-specific coefficient. Usage: ensemble_rev36.py <step3_csv> <step3_dir_with_tfe> <out_dir> <step5_script>"""
import sys, json, itertools, math, warnings, numpy as np, pandas as pd, statsmodels.api as sm
warnings.filterwarnings("ignore")
D3, TD, OUT, ST5 = sys.argv[1:5]
src = open(ST5, encoding="utf-8").read(); src = src[:src.index("# ================= R3 Shapley attribution by set-up")]
sys.argv = [ST5, D3, TD, OUT + "/_st5_tmp"]; import os; os.makedirs(OUT + "/_st5_tmp", exist_ok=True)
G = {"__name__": "ens"}; exec(compile(src, ST5, "exec"), G)
d, C, MET, TFE, haze_frame, AOD_BG, CLRW, KWP_TOT, COMMISSION, pv_X, PV0 = (G[k] for k in ("d", "C", "MET", "TFE", "haze_frame", "AOD_BG", "CLRW", "KWP_TOT", "COMMISSION", "pv_X", "PV"))
CLEAN = G["CLEAN"]
def dose_rule(pm, kind, th):
    wash = d.index.isin(CLEAN + pd.Timedelta(days=1)) | (d.index == COMMISSION); tp = d.e5_tp.fillna(0).values; x = pm.values / 1000.0; out = np.full(len(d), np.nan); D = 0.0
    for i in range(len(d)):
        D = 0.0 if wash[i] else ((0.0 if tp[i] >= th else D) if kind == "reset" else D * (1 - min(1.0, tp[i] / th)))
        if not np.isnan(x[i]): D += x[i]; out[i] = D
    return pd.Series(out, index=d.index)
SOIL = {"partial 2 mm": ("partial", 2.0), "partial 5 mm": ("partial", 5.0), "reset 0.5 mm": ("reset", 0.5)}
PVB = PV0.drop(columns=["soil"])
COEF = {}
for sname, (kind, th) in SOIL.items():
    ds = dose_rule(d.pm25, kind, th); P = PVB.assign(soil=ds.reindex(PVB.index).values).dropna()
    ml = sm.OLS(P.ratio, pv_X(P)).fit(); mh = sm.OLS(P.ratio, pv_X(P.assign(aod_h=(P.aod - 0.30).clip(lower=0)), ("aod_h",))).fit()
    COEF[sname] = dict(dose=ds, lin=(float(ml.params.aod), float(ml.params.soil)), hin=(float(mh.params.aod_h), float(mh.params.soil)), kind=kind, th=th)
def engine(H, fire, soil, resp, cap):
    X = sm.add_constant(H[fire + MET]); mA = sm.OLS(H.aod_ceres, X).fit(); mP = sm.OLS(np.log(H.pm25), X).fit(); bA, bP = mA.predict(X), mP.predict(X)
    e = d.reindex(H.index); S_ = COEF[soil]; b_aod, b_soil = S_["lin"] if resp == "linear" else S_["hin"]; base = S_["dose"]
    def v(coal):
        if not coal: return 0.0
        Z = H[fire + MET].copy()
        for cc in [x for x in fire if x.split("_")[1] in coal]: Z[cc] = 0.0 if cap == "full removal" else np.minimum(Z[cc], H[cc].quantile(cap))
        Zc = sm.add_constant(Z, has_constant="add")
        dA = np.minimum((bA - mA.predict(Zc)).clip(lower=0), (H.aod_ceres - AOD_BG).clip(lower=0))
        dP = H.pm25 * (1 - np.exp(-(bP - mP.predict(Zc)).clip(lower=0)))
        dl = dA if resp == "linear" else (H.aod_ceres - 0.30).clip(lower=0) - (H.aod_ceres - dA - 0.30).clip(lower=0)
        atm = (-b_aod * dl * e.e5_ssrd * CLRW.reindex(H.index).values * KWP_TOT / 1000 * (H.index >= COMMISSION)).sum()
        pm_c = d.pm25.copy(); pm_c.loc[H.index] = pm_c.loc[H.index] - dP
        dep = -b_soil * (base - dose_rule(pm_c, S_["kind"], S_["th"])) * d.e5_ssrd * KWP_TOT / 1000
        dep = dep[dep.index.month.isin([1, 2, 3, 4, 5, 6]) & (dep.index >= COMMISSION)].sum()
        return float(atm + dep)
    return v, mA
def shap(v, players):
    cache = {}
    def val(S):
        k = frozenset(S)
        if k not in cache: cache[k] = v(tuple(sorted(k)))
        return cache[k]
    n = len(players); phi = {c: 0.0 for c in players}
    for c in players:
        oth = [x for x in players if x != c]
        for r in range(n):
            for S in itertools.combinations(oth, r):
                phi[c] += math.factorial(r) * math.factorial(n - r - 1) / math.factorial(n) * (val(S + (c,)) - val(S))
    return phi, val(tuple(players))
rows = []
for tr in ("p850", "p700", "3d850"):
    for form in ("log", "sqrt", "linear", "near/far"):
        H, fire = haze_frame(TFE[tr], "log" if form == "near/far" else form, nearfar=(form == "near/far"))
        Hc = H[H.index >= COMMISSION]; ex = {c: float(TFE[tr][f"tfe_{c}"].reindex(Hc.index).sum()) for c in C}; E = {c: 100 * ex[c] / sum(ex.values()) for c in C}
        for resp in ("linear", "hinge"):
            for soil in SOIL:
                for cap in (0.10, 0.05, 0.25, "full removal"):
                    v, mA = engine(H, fire, soil, resp, cap); phi, joint = shap(v, C); L = {c: 100 * phi[c] / joint for c in C}
                    rows.append(dict(transport=tr, form=form, response=resp, soiling=soil, counterfactual=str(cap), joint_MWh=joint, **{f"L_{c}": L[c] for c in C}, **{f"E_{c}": E[c] for c in C}, **{f"EIDI_{c}": L[c] - E[c] for c in C}))
    print(tr, "done", flush=True)
EN = pd.DataFrame(rows); EN.to_csv(f"{OUT}/attribution_ensemble.csv", index=False)
R = {"n_specs": len(EN)}
top = EN[[f"L_{c}" for c in C]].idxmax(axis=1).str[2:]; R["ARS"] = {c: float((top == c).mean()) for c in C}; R["share_THA_gt_MMR"] = float((EN.L_THA > EN.L_MMR).mean())
R["quantiles"] = {k: EN[k].quantile([0, .05, .25, .5, .75, .95, 1]).round(1).tolist() for k in ("L_THA", "L_MMR", "EIDI_THA", "EIDI_MMR")}
R["THA_gt_MMR_by_factor"] = {f: EN.groupby(f).apply(lambda g: float((g.L_THA > g.L_MMR).mean())).to_dict() for f in ("transport", "form", "response", "soiling", "counterfactual")}
def anova(y):
    tot = float(y.var(ddof=0)); out = {f: float(EN.groupby(f)[y.name].mean().reindex(EN[f]).var(ddof=0) / tot) for f in ("transport", "form", "response", "soiling", "counterfactual")}
    out["interactions"] = 1 - sum(out.values()); return out
R["variance_share_L_THA"] = anova(EN.L_THA); R["variance_share_EIDI_MMR"] = anova(EN.EIDI_MMR)
# ---- proximity test: region-agnostic near/far model ----
H, fire = haze_frame(TFE["p850"], "log", nearfar=True); Hm = H.copy()
N = {c: TFE["p850"][f"tfe24_{c}"].reindex(Hm.index) for c in C}; Fr = {c: (TFE["p850"][f"tfe_{c}"] - TFE["p850"][f"tfe24_{c}"]).reindex(Hm.index) for c in C}
def prox_X(Nd, Fd):
    X = pd.DataFrame(index=Hm.index)
    for L in (0, 1):
        X[f"near_{L}"] = np.log1p(sum(Nd.values()).shift(L)); X[f"far_{L}"] = np.log1p(sum(Fd.values()).shift(L))
    return X
Xp = prox_X(N, Fr).join(Hm[MET]); ok = Xp.notna().all(1); Xp = sm.add_constant(Xp[ok]); Hp = Hm[ok]
mpA = sm.OLS(Hp.aod_ceres, Xp).fit(); mpP = sm.OLS(np.log(Hp.pm25), Xp).fit(); mcA = sm.OLS(Hp.aod_ceres, sm.add_constant(Hp[fire + MET])).fit()
R["proximity_model"] = dict(aic_proximity=float(mpA.aic), aic_region_nearfar=float(mcA.aic), coef={k: float(mpA.params[k]) for k in ("near_0", "near_1", "far_0", "far_1")}, p={k: float(mpA.pvalues[k]) for k in ("near_0", "near_1", "far_0", "far_1")})
S_ = COEF["partial 2 mm"]; b_aod, b_soil = S_["lin"]; e = d.reindex(Hp.index); bA, bP = mpA.predict(Xp), mpP.predict(Xp)
q10N = {c: N[c][ok].quantile(0.10) for c in C}; q10F = {c: Fr[c][ok].quantile(0.10) for c in C}
def vprox(coal):
    if not coal: return 0.0
    Nd = {c: (np.minimum(N[c], q10N[c]) if c in coal else N[c]) for c in C}; Fd = {c: (np.minimum(Fr[c], q10F[c]) if c in coal else Fr[c]) for c in C}
    Zc = sm.add_constant(prox_X(Nd, Fd).join(Hm[MET])[ok], has_constant="add")
    dA = np.minimum((bA - mpA.predict(Zc)).clip(lower=0), (Hp.aod_ceres - AOD_BG).clip(lower=0)); dP = Hp.pm25 * (1 - np.exp(-(bP - mpP.predict(Zc)).clip(lower=0)))
    atm = (-b_aod * dA * e.e5_ssrd * CLRW.reindex(Hp.index).values * KWP_TOT / 1000 * (Hp.index >= COMMISSION)).sum()
    pm_c = d.pm25.copy(); pm_c.loc[Hp.index] = pm_c.loc[Hp.index] - dP
    dep = -b_soil * (S_["dose"] - dose_rule(pm_c, "partial", 2.0)) * d.e5_ssrd * KWP_TOT / 1000; dep = dep[dep.index.month.isin([1, 2, 3, 4, 5, 6]) & (dep.index >= COMMISSION)].sum()
    return float(atm + dep)
phi, joint = shap(vprox, C); R["proximity_shares"] = {c: 100 * phi[c] / joint for c in C}
Hc = Hp[Hp.index >= COMMISSION]; R["near_fraction_of_exposure"] = {c: float(N[c].reindex(Hc.index).sum() / (N[c] + Fr[c]).reindex(Hc.index).sum()) for c in C}
json.dump(R, open(f"{OUT}/attribution_ensemble.json", "w"), indent=1, default=float); print(json.dumps(R, indent=1, default=float))
