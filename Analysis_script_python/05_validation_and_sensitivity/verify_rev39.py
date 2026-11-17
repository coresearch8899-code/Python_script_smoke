"""Check the rev39 numbers and wording against the regenerated outputs. Usage: verify_rev39.py <rerun_novelty_dir> <package_dir>"""
import sys, json, docx
O, P = sys.argv[1:3]
m = docx.Document(P + r"\03_Manuscript_HazePV_RenewableEnergy.docx"); t = " ".join(p.text for p in m.paragraphs).replace(" ", " ")
su = docx.Document(P + r"\06_Supplementary_material.docx"); ts = " ".join(p.text for p in su.paragraphs).replace(" ", " ")
U = json.load(open(f"{O}/uncertainty_decomposition_rev39.json")); B = json.load(open(f"{O}/discrimination_rev38.json"))["B_failure"]["25"]
ok = []
def chk(name, cond): ok.append(bool(cond)); print(("ok  " if cond else "BAD ") + name)
chk("variance shares 79 / 11 / 7", f"contributed {100 * U['sampling_PV_coefficients']:.0f} %, the aerosol response {100 * U['aerosol_response']:.0f} % and the soiling model {100 * U['soiling_model']:.0f} %" in t)
chk("point totals equal Table S10a", U["max_abs_diff_point_vs_table_S10a"] < 1e-9)
chk("exact agreement 52 %, sensitivity 44 %, specificity 94 %", f"exact agreement was {100 * B['exact_agreement']:.0f} %, the split was identified with a sensitivity of {100 * B['split']['sensitivity']:.0f} % and a specificity of {100 * B['split']['specificity']:.0f} %" in t)
chk("false-reliable rate 6 %", f"false-reliable rate {100 * B['split']['false_reliable_rate']:.0f} %" in t)
chk("precautionary reporting aid (Discussion)", "precautionary reporting aid rather than a validated classifier of identifiability" in t)
chk("forward-validation limitation", "do not independently validate the loss decomposition" in t)
chk("washing as an application (abstract, introduction, conclusions)", t.count("As an application") + t.count("as an application") >= 3)
chk("reference [1] wording", "electricity demand of public hospitals in Bangladesh [1]" in t and "increasingly being installed" not in t)
chk("supplement: evaluation target, no 'external target'", "independently defined evaluation target" in ts and "external target" not in ts)
chk("supplement: separate bootstrap stated", "separate block bootstrap of each specification" in ts)
print(f"{sum(ok)} of {len(ok)} checks passed")
