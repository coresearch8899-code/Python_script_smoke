"""Compare the rev38 analysis rerun from the repository with the stored output and check the numbers quoted in the paper.
Usage: verify_rev38.py <rerun_novelty_out> <stored_novelty_out> <package_dir>"""
import sys, json, docx, pandas as pd
O, S, P = sys.argv[1:4]
def flat(o, p=""):
    if isinstance(o, dict):
        for k, v in o.items(): yield from flat(v, f"{p}.{k}")
    elif isinstance(o, list):
        for i, v in enumerate(o): yield from flat(v, f"{p}[{i}]")
    elif isinstance(o, (int, float)) and not isinstance(o, bool): yield p, float(o)
a, b = dict(flat(json.load(open(f"{O}/discrimination_rev38.json")))), dict(flat(json.load(open(f"{S}/discrimination_rev38.json"))))
eq = lambda x, y: (x != x and y != y) or abs(x - y) <= 1e-9 * max(1, abs(y))
bad = sum(not eq(a[k], b[k]) for k in a if k in b) + len(set(a) ^ set(b))
for f in ("recoverability_confusion.csv", "transport_discrimination.csv"):
    if not pd.read_csv(f"{O}/{f}").astype(str).equals(pd.read_csv(f"{S}/{f}").astype(str)): bad += 1; print("DIFF", f)
print(f"rev38 analysis rerun from the repository: {len(a)} JSON numbers and 2 CSV files compared, {bad} differences")
m = docx.Document(P + r"\03_Manuscript_HazePV_RenewableEnergy.docx"); t = " ".join(p.text for p in m.paragraphs).replace(" ", " ")
R = json.load(open(f"{O}/discrimination_rev38.json")); A, B = R["A_transport"], R["B_failure"]["25"]; N = json.load(open(f"{S}/novelty37.json")); M = pd.read_csv(f"{S}/identifiability_map.csv")
ok = []
def chk(name, cond): ok.append(bool(cond)); print(("ok  " if cond else "BAD ") + name)
chk("BL 94 % / 22 %", f"on {100 * A['p850']['share_days_arrival_in_BL']:.0f} % of January–May days and the 700 hPa level on {100 * A['p700']['share_days_arrival_in_BL']:.0f} %" in t)
chk("LOYO 0.46 / 0.43 / 0.45", f"LOYO R2 {A['p700']['loyo_AOD']:.2f} against {A['p850']['loyo_AOD']:.2f} at 850 hPa and {A['3d850']['loyo_AOD']:.2f} for three-dimensional" in t)
chk("no set-up best on both", max(("p850", "3d850", "p700"), key=lambda k: A[k]["loyo_AOD"]) not in A["subset_physically_consistent"]["setups"])
chk("Thailand first in all 192 / 6 % of 700 hPa", A["subset_physically_consistent"]["n"] == 192 and A["subset_physically_consistent"]["thai_first"] == 1 and "all 192 specifications" in t and f"{100 * A['p700']['thai_first']:.0f} % of the 700 hPa" in t)
chk("EIDI sign kept in every set-up", all(A[k]["EIDI_sign_kept"] for k in ("p850", "3d850", "p700")))
chk("11 of 12 conservative, false-reliable 6 %", B["conservative_errors"] == 11 and B["liberal_errors"] == 1 and "11 of 12 misclassified" in t and f"false-reliable rate {100 * B['split']['false_reliable_rate']:.0f} %" in t)
c9 = M[(M.condition == "Correct model") & (M.rho_target == 0.9)].iloc[0]; chk("deposition RMSE 30 % at 0.9", round(100 * c9.rmse_dep / c9.true_dep) == 30 and "reached 30 % at 0.9" in t)
chk("single removals overstate joint by 11 %", round(100 * (N["B_shares"]["one-at-a-time_sum_over_joint"] - 1)) == 11 and "by 11 %" in t)
L = json.load(open(f"{S}/criteria_lofo.json")); chk("LOFO 72 % / 40 %", f"averaged {100 * L['lofo_mean_test_acc']:.0f} % but was {100 * L['per_family']['AOD noise SD 0.10']['test_acc']:.0f} % for AOD noise" in t)
print(f"{sum(ok)} of {len(ok)} checks passed")
