"""Check the rev40 consistency edits against the stored outputs. Usage: verify_rev40.py <analysis_dir> <package_dir>"""
import sys, json, re, docx
A, P = sys.argv[1:3]
rd = lambda f: docx.Document(f"{P}/{f}")
m, su, cv, hl = rd("03_Manuscript_HazePV_RenewableEnergy.docx"), rd("06_Supplementary_material.docx"), rd("01_Cover_letter.docx"), rd("02_Highlights.docx")
T = lambda d: " ".join(p.text for p in d.paragraphs).replace(" ", " ")
t, ts, tc, th = T(m), T(su), T(cv), T(hl)
E = json.load(open(f"{A}/checks_out/exposure_common.json")); C = json.load(open(f"{A}/checks_out/checks_rev25b.json")); EN = json.load(open(f"{A}/novelty_out/attribution_ensemble.json"))
ok = []
def chk(name, cond): ok.append(bool(cond)); print(("ok  " if cond else "BAD ") + name)
mm = [E[k]["all_days"]["MMR"] for k in ("p850", "3d850", "p700")]
chk("Myanmar 71–81 % (three main set-ups, all days) in abstract, results and conclusions", f"{min(mm):.0f}–{max(mm):.0f}" == "71–81" and t.count("71–81 %") >= 3 and "69–80" not in t + tc)
chk("abstract R2 0.15 is the nested LOYO value and is reported in Section 3.3", round(C["B_nested_loyo"]["models"]["nested"]["r2_oos"], 2) == 0.15 and "(R2 = 0.15) when the rain rule was selected within the training years" in t)
q = EN["quantiles"]["EIDI_THA"]; chk("conclusions: EIDI range +13 to +63 across specifications, main +53 (30–77)", f"+{q[0]:.0f} to +{q[-1]:.0f} points; main model +53 points, 95 % interval 30–77" in t)
chk("no 'without bias' (only 'without appreciable bias')", not re.search(r"without bias", t + tc + ts))
chk("title identical in manuscript, supplement and cover letter, without 'robust source attribution'", m.paragraphs[0].text.strip() == su.paragraphs[0].text.strip() and m.paragraphs[0].text.strip() in tc and "robust source attribution" not in t + ts + tc)
chk("700 hPa reversal stated consistently ('in most 700 hPa specifications')", "frequently reversed" not in t and "often reversed" not in t and t.count("most 700 hPa specifications") >= 3)
chk("transport selection sentence (predictive skill insufficient; 850 hPa not chosen for Thai dominance)", "Predictive skill alone is insufficient to confirm a source attribution" in t and "not because it supports Thai dominance" in t)
chk("criteria limitation with 52 / 44 / 94 % and 'not a universal classifier'", "exact agreement 52 %, split sensitivity 44 %, specificity 94 %" in t and "validated or universal classifier" in t)
chk("no 'demonstrated that' / 'showed that' claims in Conclusions", not re.search(r"demonstrated that|showed that|is to show that", t + tc))
chk("percent formatting with a space in the text", not re.search(r"\d%", t))
chk("Section 3.2 cites S6f and S1b for the exposure shares", "(Supplementary Table S6f) and 50–87 % in individual seasons with hourly winds for 2021–2025 (Supplementary Table S1b)" in t)
chk("S6f note gives the per-set-up Myanmar shares", re.search(r"71 % \(850 hPa\), 73 % \(3-D\), 81 % \(700 hPa\)", ts) is not None)
chk("highlights consistent with text (57–76 %, 144–170 MWh, 288)", "144–170 MWh" in th and "57–76 %" in th and "288" in th)
chk("data availability names seeds, settings and the 288 specifications", "random seeds" in t and "288 attribution specifications" in t)
print(f"{sum(ok)} of {len(ok)} checks passed")
