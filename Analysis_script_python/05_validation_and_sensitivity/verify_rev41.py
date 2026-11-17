"""rev41 pre-submission checks: structure (headings), cross-references (tables, figures, equations), numbers in the Abstract and Conclusions against
the tables of the paper and the stored outputs, and the Data availability statement against the package. Usage: verify_rev41.py <analysis_dir> <package_dir>"""
import sys, os, re, json, zipfile, docx
A, P = sys.argv[1:3]
m = docx.Document(f"{P}/03_Manuscript_HazePV_RenewableEnergy.docx"); su = docx.Document(f"{P}/06_Supplementary_material.docx")
ps = [p.text.replace(" ", " ") for p in m.paragraphs]; t = " ".join(ps)
end = next(i for i, x in enumerate(ps) if x.startswith("CRediT")); body = " ".join(ps[:end])
ab = ps[next(i for i, x in enumerate(ps) if x.strip() == "Abstract") + 1]; ci = next(i for i, x in enumerate(ps) if x.strip() in ("5. Conclusions", "5 Conclusions")); con = " ".join(ps[ci + 1:end])
tab = lambda d, start: next(x for x in d.tables if any(c.text.startswith(start) for c in x.rows[0].cells))
ttext = lambda x: " | ".join(c.text for r in x.rows for c in r.cells).replace(" ", " ")
ok = []
def chk(name, cond): ok.append(bool(cond)); print(("ok  " if cond else "BAD ") + name)
heads = [x.strip() for x in ps if re.match(r"^\d+(\.\d+)*\.?\s\S", x.strip()) and len(x) < 90]
chk("each section heading occurs once", len(heads) == len(set(heads)))
top = [h for h in heads if re.match(r"^\d\.\s", h)]; chk("top-level sections 1–5 in order", [h[0] for h in top] == list("12345"))
subs = [h for h in heads if re.match(r"^\d\.\d+", h)]; chk("subsections numbered consecutively", all(int(re.match(r"\d\.(\d+)", b).group(1)) == (int(re.match(r"\d\.(\d+)", a).group(1)) + 1 if a[0] == b[0] else 1) for a, b in zip(subs, subs[1:])))
first = lambda pat: [int(x) for x in re.findall(pat, body)]
for kind, pat, cap, n in (("Table", r"\bTables? (\d)\b", r"^Table (\d)\b", 7), ("Fig.", r"\bFigs?\. (\d)", r"^Fig\. (\d)\.", 7)):
    cited = first(pat); order = list(dict.fromkeys(cited)); caps = sorted({int(re.match(cap, x).group(1)) for x in ps if re.match(cap, x)})
    chk(f"{kind} 1–{n}: all have captions, all cited, first cited in order", caps == list(range(1, n + 1)) and sorted(order) == caps and order == sorted(order))
eqdef = sorted({int(re.search(r"\((\d+)\)\s*$", x).group(1)) for x in ps if "\t" in x and re.search(r"\(\d+\)\s*$", x)})
eqref = sorted({int(x) for g in re.findall(r"Eqs?\. \(([\d), (]+)\)", body) for x in re.findall(r"\d+", g)})
chk(f"equations (1)–({len(eqdef)}) defined, every cited equation exists", eqdef == list(range(1, len(eqdef) + 1)) and set(eqref) <= set(eqdef))
omath = sum(len(p._p.findall(".//{http://schemas.openxmlformats.org/officeDocument/2006/math}oMath")) for p in m.paragraphs); chk("all numbered equations (1)–(10) and the in-text EIDI and f_cs definitions are Word equations", omath == 12)
T4, T6 = ttext(tab(m, "Year")) if any(any(c.text.startswith("Year") for c in x.rows[0].cells) for x in m.tables) else "", None
J = json.load(open(f"{A}/blocks28/joint_bootstrap_total_loss.json")); SM = json.load(open(f"{A}/framework_out/framework_checks.json"))
lin, hin = J["point_linear"]["tot"], J["point_hinge"]["tot"]; pci = J["CI95_pooled_tot"]
chk("144–170 MWh and 94–209 MWh (Abstract, Conclusions) = hinge/linear totals and pooled interval", f"{hin:.0f}–{lin:.0f} MWh" in ab and f"{hin:.0f}–{lin:.0f} MWh" in con and f"{pci[0]:.0f}–{pci[1]:.0f} MWh" in ab and f"{pci[0]:.0f}–{pci[1]:.0f} MWh" in con)
a = SM["A"]["all_specs"]; chk("136–183 MWh and 57–76 % (Conclusions) = retained-model ranges", f"{a['total'][0]:.0f}–{a['total'][1]:.0f} MWh" in con and f"{a['frac'][0]:.0f}–{a['frac'][1]:.0f} %" in con)
E = json.load(open(f"{A}/checks_out/exposure_common.json")); mm = [E[k]["all_days"]["MMR"] for k in ("p850", "3d850", "p700")]
chk("Myanmar exposure 71–81 % in Abstract and Conclusions", f"{min(mm):.0f}–{max(mm):.0f} %" in ab and f"{min(mm):.0f}–{max(mm):.0f} %" in con)
RR = json.load(open(f"{A}/out25/R_revision_results.json")); th = RR["R3_shapley"]["p850"]["shapley_share"]["THA"]; N = json.load(open(f"{A}/novelty_out/novelty37.json"))["A"]["THA"]
chk("Thai share 67 % and EIDI +53 (30–77) in Abstract and Conclusions", f"{th:.0f} %" in ab and f"{th:.0f} %" in con and f"+{N['EIDI']:.0f}" in ab + con and f"{N['EIDI_CI'][0]:.0f}–{N['EIDI_CI'][1]:.0f}" in ab and f"{N['EIDI_CI'][0]:.0f}–{N['EIDI_CI'][1]:.0f}" in con)
q = json.load(open(f"{A}/novelty_out/attribution_ensemble.json"))["quantiles"]["EIDI_THA"]; chk("EIDI +13 to +63 across specifications (Conclusions)", f"+{q[0]:.0f} to +{q[-1]:.0f} points" in con)
chk("nominal interval labels: Abstract and Conclusions say 'nominal 95 % interval' for the total", "nominal 95 % interval 94–209" in ab and "nominal 95 % interval 94–209" in con)
da = next(x for x in ps if x.startswith("The external datasets are openly available"))
z = zipfile.ZipFile(f"{P}/08_Code_HazePV.zip").namelist()
chk("Data availability: code supplied as supplementary material and archive present with scripts, seeds and outputs", "provided as supplementary material" in da and os.path.exists(f"{P}/08_Code_HazePV.zip")
    and any(x.endswith("ensemble_rev36.py") for x in z) and any(x.endswith("uncertainty_rev39.py") for x in z) and "REPRODUCIBILITY.md" in z and sum(x.startswith("results/") for x in z) > 40)
chk("Data availability: DOI promised only on acceptance (no DOI claimed now)", "with a DOI on acceptance" in da and "doi.org/10.5281" not in da)
chk("no plain 'PM2.5' runs (subscripts)", not any(re.search(r"PM2\.5", r.text) for p in m.paragraphs for r in p.runs))
print(f"{sum(ok)} of {len(ok)} checks passed")
