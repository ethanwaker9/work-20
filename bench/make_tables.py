import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bench.complexity import SIZE_MODELS

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
OUT = os.path.abspath(os.path.join(ROOT, "..", "final_paper", "tables"))

ORDER = ["PBB13", "MTRS5", "CDS-MQ", "HS20", "CONCAT", "MQuorum"]
LABEL = {"PBB13": "PBB13 \\cite{pbb2013}", "MTRS5": "MTRS5 \\cite{zz2014,dtsl2021}",
         "CDS-MQ": "CDS-MQ \\cite{cds1994,bss2002}", "HS20": "HS20 \\cite{hs2020}",
         "CONCAT": "Concat \\cite{chiang2025}", "MQuorum": "\\textbf{$\\Scheme$ (ours)}"}
ANON = {"PBB13": "$\\times$", "MTRS5": "$\\times$", "CDS-MQ": "$\\checkmark$",
        "HS20": "$\\checkmark$", "CONCAT": "$\\checkmark$", "MQuorum": "$\\checkmark$"}
QROM = {"PBB13": "$\\times$", "MTRS5": "$\\times$", "CDS-MQ": "$\\times$",
        "HS20": "$\\checkmark$", "CONCAT": "$\\times$", "MQuorum": "$\\checkmark$"}
GROW = {"PBB13": "$\\lambda N n$", "MTRS5": "$\\lambda N n$", "CDS-MQ": "$\\lambda N n$",
        "HS20": "$\\lambda^{2}N$", "CONCAT": "$\\lambda t(n{+}\\sqrt{N}{+}\\lambda)$",
        "MQuorum": "$\\lambda(t(n{+}\\sqrt{N}){+}\\lambda)$"}


def load():
    return json.load(open(os.path.join(RES, "bench.json")))


def pick(rows, N, t):
    return {r["scheme"]: r for r in rows if r["N"] == N and r["t"] == t}


def fmt(x, d=1):
    if d == 1 and x >= 1000:
        return f"{x:,.0f}"
    return f"{x:,.{d}f}"


def fmt_r(x):
    if x >= 100:
        return f"{x:,.0f}"
    return f"{x:,.1f}"


def headline(rows, N=64, t=8):
    d = pick(rows, N, t)
    lines = []
    for s in ORDER:
        r = d[s]
        lines.append(" & ".join([LABEL[s], ANON[s], QROM[s], GROW[s],
                                 fmt(r["sig_bytes"] / 1024.0),
                                 fmt(r["pk_bytes"], 0),
                                 fmt(r["sign_ms"], 0), fmt(r["verify_ms"], 0),
                                 fmt(r["peak_kib"] / 1024.0)]) + " \\\\")
    return "\n".join(lines)


COMPLEX = {
 "PBB13": ("$RN(mL{+}C_h)$", "$RN(mL{+}C_h)$", "$NmL$"),
 "MTRS5": ("$RN(mL{+}C_h)$", "$RN(mL{+}C_h)$", "$NmL$"),
 "CDS-MQ": ("$RN(mL{+}C_h){+}N^{2}$", "$RN(mL{+}C_h){+}N^{2}$", "$RN(\\lambda{+}n)$"),
 "HS20": ("$\\lambda N m_{o}C_h{+}\\lambda tN$", "$\\lambda N m_{o}C_h{+}\\lambda tN$",
          "$\\lambda N m_{o}$"),
 "CONCAT": ("$t\\tau[\\Np(C_h{+}W_1){+}\\rho D(L{+}N{+}db){+}\\rho mL]$",
            "$t\\tau[\\Np(C_h{+}W_1){+}\\rho(L{+}N{+}db){+}\\rho mL]$",
            "$\\Np W_1{+}mL$"),
 "MQuorum": ("$\\tau[\\Np(C_h{+}W){+}\\rho(mL{+}Nm){+}\\rho D t(L{+}N{+}db)]$",
             "$\\tau[\\Np(C_h{+}W){+}\\rho(mL{+}Nm){+}\\rho t(L{+}N{+}db)]$",
             "$\\Np W{+}mL{+}N$"),
}


def complexity():
    return "\n".join(LABEL[s] + " & " + " & ".join(COMPLEX[s]) + " \\\\" for s in ORDER)


def grid_table(rows, key, div, dec):
    pts = sorted({(r["N"], r["t"]) for r in rows})
    out = []
    for s in ORDER:
        cells = []
        for (N, t) in pts:
            d = pick(rows, N, t)
            cells.append(fmt(d[s][key] / div, dec) if s in d else "--")
        out.append(LABEL[s] + " & " + " & ".join(cells) + " \\\\")
    hdr = " & ".join(f"$({N},{t})$" for (N, t) in pts)
    return hdr, "\n".join(out)


def ratio_table(rows):
    pts = sorted({(r["N"], r["t"]) for r in rows})
    out = []
    for s in ORDER[:-1]:
        cells = []
        for (N, t) in pts:
            d = pick(rows, N, t)
            cells.append(fmt_r(d[s]["sig_bytes"] / d["MQuorum"]["sig_bytes"])
                         if s in d else "--")
        out.append(LABEL[s] + " & " + " & ".join(cells) + " \\\\")
    return "\n".join(out)


def wrap(spec, header, body, size="\\small", tabcolsep="2.6pt"):
    return ("{" + size + "\n\\setlength{\\tabcolsep}{" + tabcolsep + "}\n"
            "\\begin{tabular}{" + spec + "}\n\\toprule\n" + header
            + " \\\\\n\\midrule\n" + body + "\n\\bottomrule\n"
            "\\end{tabular}}%")


def main():
    os.makedirs(OUT, exist_ok=True)
    rows = load()
    hd = ("Scheme & anon. & QROM & growth & $|\\sigma|$ & $|vk|$ & sign & vfy & mem")
    open(os.path.join(OUT, "tab_headline.tex"), "w").write(
        wrap("@{}lcccrrrrr@{}", hd, headline(rows)))
    open(os.path.join(OUT, "tab_complexity.tex"), "w").write(
        wrap("@{}lccc@{}", "Scheme & signing time & verification time & memory",
             complexity(), "\\scriptsize", "2.4pt"))
    npt = len(sorted({(r["N"], r["t"]) for r in rows}))
    for name, key, div, dec in [("tab_size", "sig_bytes", 1024.0, 1),
                                ("tab_mem", "peak_kib", 1024.0, 1)]:
        hdr, body = grid_table(rows, key, div, dec)
        open(os.path.join(OUT, name + ".tex"), "w").write(
            wrap("@{}l*{%d}{r}@{}" % npt, "Scheme & " + hdr, body,
                 "\\scriptsize", "1.9pt"))
    hdr, b1 = grid_table(rows, "sign_ms", 1.0, 0)
    _, b2 = grid_table(rows, "verify_ms", 1.0, 0)
    body = ("\\multicolumn{%d}{@{}l}{\\emph{signing}}\\\\\n" % (npt + 1) + b1
            + "\n\\addlinespace[2pt]\n\\multicolumn{%d}{@{}l}{\\emph{verification}}"
            "\\\\\n" % (npt + 1) + b2)
    open(os.path.join(OUT, "tab_time.tex"), "w").write(
        wrap("@{}l*{%d}{r}@{}" % npt, "Scheme & " + hdr, body, "\\scriptsize", "1.9pt"))
    npt = len(sorted({(r["N"], r["t"]) for r in rows}))
    hdr, _ = grid_table(rows, "sig_bytes", 1.0, 0)
    open(os.path.join(OUT, "tab_ratio.tex"), "w").write(
        wrap("@{}l*{%d}{r}@{}" % npt, "Scheme & " + hdr, ratio_table(rows),
             "\\scriptsize", "1.9pt"))
    print("grid points:", sorted({(r["N"], r["t"]) for r in rows}))
    if os.path.exists(os.path.join(RES, "npsweep.json")):
        sw = json.load(open(os.path.join(RES, "npsweep.json")))
        lines = []
        for r in sw:
            if r["N"] == 100:
                lines.append(" & ".join([str(r["n_party"]), str(r["tau"]),
                                         fmt(r["sig_bytes"] / 1024.0),
                                         fmt(r["sign_ms"], 0),
                                         fmt(r["verify_ms"], 0)]) + " \\\\")
        open(os.path.join(OUT, "tab_np.tex"), "w").write(
            wrap("@{}rrrrr@{}",
                 "$\\Np$ & $\\tau$ & $|\\sigma|$ (KB) & sign (ms) & verify (ms)",
                 "\n".join(lines)))
    print("tables written to", OUT)


if __name__ == "__main__":
    main()
