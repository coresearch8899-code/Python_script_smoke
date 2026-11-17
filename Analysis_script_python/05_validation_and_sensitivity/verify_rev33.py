"""Check the submitted tables against outputs regenerated from the repository scripts, and compare all regenerated outputs with the stored ones.
Usage: verify_rev33.py <package_dir> <regenerated_dir> <stored_analysis_dir>"""
import sys, os, json, re, zipfile, docx, pandas as pd
P, O, S = sys.argv[1:4]
J = lambda *a: json.load(open(os.path.join(*a)))
def flat(o, p=""):
    if isinstance(o, dict):
        for k, v in o.items(): yield from flat(v, f"{p}.{k}")
    elif isinstance(o, list):
        for i, v in enumerate(o): yield from flat(v, f"{p}[{i}]")
    elif isinstance(o, (int, float)) and not isinstance(o, bool): yield p, float(o)
n_all = 0; bad = 0
for x, y in [(f, f) for f in ("out25/R_revision_results.json", "out25/joint_bootstrap_total_loss.json", "out25/N1_rev25.json", "out25/extras_rev25.json", "checks_out/checks_rev25b.json", "checks_out/checks_rev25c.json", "checks_out/exposure_common.json", "checks_out/block_length_joint.json", "robustness_out/robustness_rev25.json", "blocks28/joint_bootstrap_total_loss.json", "blocks28/checks_rev25b.json")] + [("framework/framework_checks.json", "framework_out/framework_checks.json")]:
    a, b = dict(flat(J(O, x))), dict(flat(J(S, y))); c = set(a) & set(b); d_ = [k for k in c if abs(a[k] - b[k]) > 1e-6 * max(1, abs(b[k]))]; n_all += len(c); bad += len(d_) + len(set(a) ^ set(b))
    if d_: print("DIFF", x, d_[:3])
csvs = [("out25/" + f,) * 2 for f in ("R1_table5_single_pipeline.csv", "T14_cleaning_policies.csv", "T14c_cleaning_cost_sensitivity.csv", "R7_wash_event_validation.csv")] + [("framework/" + f, "framework_out/" + f) for f in ("washing_out_of_sample.csv", "table4_common_days.csv", "soiling_models_passing.csv", "year_left_out.csv", "coverage_diagnostics.csv")] + [("synthetic/synthetic_recovery.csv", "synthetic_out/synthetic_recovery.csv")]
for x, y in csvs:
    p, q = pd.read_csv(os.path.join(O, x)), pd.read_csv(os.path.join(S, y)); nu = p.select_dtypes("number").columns
    m = float((p[nu] - q[nu]).abs().max().max())
    if m > 1e-9: print("DIFF", x, m); bad += 1
print(f"regenerated vs stored: {n_all} JSON numbers and {len(csvs)} CSV files compared, {bad} differences")
m = docx.Document(os.path.join(P, "03_Manuscript_HazePV_RenewableEnergy.docx")); s = docx.Document(os.path.join(P, "06_Supplementary_material.docx"))
N = lambda t: t.replace(" ", " ").replace(",", "").replace("−", "-")
ok = []
def chk(name, shown, expect):
    r = str(shown) == str(expect); ok.append(r); print(("ok  " if r else "BAD ") + name, shown if r else f"{shown} != {expect}")
T4 = pd.read_csv(os.path.join(O, "framework/table4_common_days.csv")).set_index("year")
t = next(t for t in m.tables if t.rows[0].cells[0].text == "Year")
for r in t.rows[1:]:
    y = int(r.cells[0].text[:4]); v = T4.loc[y]
    chk(f"Table 4 {y}", " | ".join(N(c.text) for c in r.cells[1:5]), f"{v.E_actual:.1f} | {v.atm:.1f} | {v.dep:.1f} | {v.loss_pct:.1f} ({v.loss_pct_h:.1f})")
C = J(O, "checks_out/checks_rev25b.json"); t = next(t for t in m.tables if t.rows[0].cells[0].text.startswith("Policy"))
for r in t.rows[1:]:
    k = N(r.cells[0].text)
    if k.startswith("Dose-triggered"):
        D = [C["C_table6_dose"][c] for c in ("10", "30", "100")]
        chk("Table 7 dose washes", N(r.cells[1].text), " / ".join(str(x["washes"]) for x in D)); chk("Table 7 dose MWh", N(r.cells[2].text), " / ".join(f"{x['soiling_MWh']:.1f}" for x in D)); chk("Table 7 dose cost", N(r.cells[3].text), " / ".join(f"{x['cost']:.0f}" for x in D))
    else:
        f = next(x for x in C["C_table6_fixed"] if N(x["policy"]) == k); chk(f"Table 7 {k}", N(r.cells[2].text) + " | " + N(r.cells[3].text), f"{f['soiling_MWh']:.1f} | {f['cost_10']:.0f} / {f['cost_30']:.0f} / {f['cost_100']:.0f}")
J28 = J(O, "blocks28/joint_bootstrap_total_loss.json"); F = J(O, "framework/framework_checks.json")
for t in s.tables:
    for r in t.rows:
        k = N(r.cells[0].text)
        if k == "Linear" and len(r.cells) == 5: chk("S7b linear total", N(r.cells[3].text), f"169.7 ({J28['CI95_linear']['tot'][0]}-{J28['CI95_linear']['tot'][1]})".replace("-", "–"))
        if k.startswith("Equal-weight"): chk("S7b mixture", N(r.cells[3].text), f"{J28['CI95_pooled_tot'][0]}–{J28['CI95_pooled_tot'][1]}")
        if k in ("100 %", "90 %", "75 %", "50 %") and len(r.cells) == 5:
            e = {"100 %": "1.0", "90 %": "0.9", "75 %": "0.75", "50 %": "0.5"}[k]; mc = F["D"]["max_cost_with_test_saving"][e]
            chk(f"S10d {k}", " | ".join(c.text for c in r.cells[1:]), " | ".join(f"{mc[b] / 1000:.0f}" for b in ("training", "full", "low", "high")))
        if k.startswith("Test-period optimum"): chk("S5c test optimum", r.cells[1].text, f"{min(F['S5c']['test_optimum']) / 1000:.0f}–{max(F['S5c']['test_optimum']) / 1000:.0f}")
H = J(O, "framework/hac_calendar.json"); H0 = J(S, "framework_out/hac_calendar.json"); fa, fb = dict(flat(H)), dict(flat(H0)); chk("HAC JSON unchanged", max(abs(fa[k] - fb[k]) for k in fa) < 1e-9, True)
stars = lambda p: "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else ""
for t in m.tables:
    if t.rows[0].cells[0].text == "Quantity":
        for r in t.rows:
            k = r.cells[0].text
            if k.startswith("βA (per"): chk("Table 3 SE bA", r.cells[2].text, f"{H['table3']['aod']['se']:.3f}")
            if k.startswith("βS (per"): chk("Table 3 SE bS", r.cells[2].text, f"{H['table3']['soil']['se']:.4f}")
            if k.startswith("βA, hinge"): chk("Table 3 SE hinge", r.cells[2].text, f"{H['hinge']['se']:.3f}")
            if k.startswith("βT"): chk("Table 3 SE bT", r.cells[2].text, f"{H['table3']['t2m']['se']:.4f}")
    if t.rows[0].cells[0].text == "Term":
        for r in t.rows:
            q = re.match(r"ln\(1\+TFE(\w{3})\), lag (\d)", r.cells[0].text)
            if q:
                for cell, nm in ((r.cells[1], "logPM25"), (r.cells[2], "AOD")):
                    v = H["haze"][nm][f"lt_{q.group(1)}_{q.group(2)}"]; chk(f"Table 5 {q.group(1)} {q.group(2)} {nm}", cell.text, f"{v['coef']:.3f}".replace("-", "−") + f" ({v['se']:.3f}){stars(v['p'])}")
txt = " ".join(p.text for p in m.paragraphs).replace("\u00a0", " ")
for frag in (f"−0.135 ± {H['table3']['aod']['se']:.3f}", f"−0.159 ± {H['hinge']['se']:.3f}", f"0.0876 ± {H['table3']['soil']['se']:.4f}", f"16.3 ± {H['drift']['se_pct']:.1f} %", "THB 55,000", "94–209 MWh", "136–183 MWh", "57–76 %"): chk("text: " + frag, frag in txt, True)
A_ = pd.read_csv(os.path.join(O, "framework/soiling_models_passing.csv")); chk("retained models", int(A_.passes.sum()), 7)
W = pd.read_csv(os.path.join(O, "framework/washing_out_of_sample.csv")); W0 = pd.read_csv(os.path.join(S, "framework_out/washing_out_of_sample.csv")); chk("Fig. 7b data", W.equals(W0), True)
z7 = zipfile.ZipFile(os.path.join(P, "07_Supplementary_data.zip")).namelist(); z8 = zipfile.ZipFile(os.path.join(P, "08_Code_HazePV.zip")).namelist(); rp = open(os.path.join(os.path.dirname(P), "HazePV_github", "REPRODUCIBILITY.md"), encoding="utf-8").read()
print("REPRODUCIBILITY.md in code zip:", "REPRODUCIBILITY.md" in z8); names = sorted(set(re.findall(r"`[^`/ ]+/([^`/ ]+\.(?:csv|json))`", rp)))
b8 = {os.path.basename(n.replace("\\", "/")) for n in z8}; print("cited outputs missing from the code zip:", [n for n in names if n not in b8 and n != "joint_bootstrap_total_loss.json" and n != "checks_rev25b.json"]); print("scripts missing:", [n for n in sorted(set(re.findall(r"`([a-z0-9_A-Z]+\.py)`", rp))) if n not in b8])
print(f"{sum(ok)} of {len(ok)} table checks passed")
