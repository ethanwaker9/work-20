import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from bench.complexity import ours_size, concat_size

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
FIG = os.path.abspath(os.path.join(ROOT, "..", "final_paper", "figures"))

ORDER = ["PBB13", "MTRS5", "CDS-MQ", "HS20", "CONCAT", "MQuorum"]
NAME = {"PBB13": "PBB13", "MTRS5": "MTRS5", "CDS-MQ": "CDS-MQ", "HS20": "HS20",
        "CONCAT": "Concat", "MQuorum": "MQuorum (ours)"}
STYLE = {"PBB13": ("o", "-", "#4c72b0"), "MTRS5": ("s", "-", "#dd8452"),
         "CDS-MQ": ("^", "-", "#55a868"), "HS20": ("v", "-", "#c44e52"),
         "CONCAT": ("D", "--", "#8172b3"), "MQuorum": ("*", "-", "#000000")}

plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7,
                     "xtick.labelsize": 7, "ytick.labelsize": 7,
                     "axes.grid": True, "text.usetex": False, "grid.alpha": 0.3, "grid.linewidth": 0.4,
                     "lines.linewidth": 1.1, "lines.markersize": 4,
                     "ps.useafm": True, "pdf.fonttype": 42, "ps.fonttype": 42})


def load():
    return json.load(open(os.path.join(RES, "bench.json")))


def series(rows, scheme, fix, key):
    (fk, fv), vk = fix
    pts = [(r[vk], r[key]) for r in rows if r["scheme"] == scheme and r[fk] == fv]
    pts.sort()
    return [p[0] for p in pts], [p[1] for p in pts]


def save(fig, name, tight=True):
    os.makedirs(FIG, exist_ok=True)
    eps = os.path.join(FIG, name + ".eps")
    if tight:
        fig.savefig(eps, format="eps", bbox_inches="tight", pad_inches=0.02)
    else:
        fig.savefig(eps, format="eps")
    subprocess.run(["epstopdf", eps, "--outfile=" + os.path.join(FIG, name + ".pdf")],
                   check=True)
    plt.close(fig)
    print("wrote", name)


def fig_size(rows):
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.25))
    for s in ORDER:
        m, ls, c = STYLE[s]
        x, y = series(rows, s, (("N", 64), "t"), "sig_bytes")
        axes[0].plot(x, [v / 1024 for v in y], marker=m, ls=ls, color=c, label=NAME[s])
        x, y = series(rows, s, (("t", 8), "N"), "sig_bytes")
        axes[1].plot(x, [v / 1024 for v in y], marker=m, ls=ls, color=c, label=NAME[s])
    axes[0].set_xlabel("threshold $t$ (ring size $N=64$)")
    axes[1].set_xlabel("ring size $N$ (threshold $t=8$)")
    for a in axes:
        a.set_yscale("log"); a.set_xscale("log", base=2)
        a.set_ylabel("signature size (KB)")
    axes[1].legend(ncol=3, loc="center", framealpha=0.95, handlelength=1.5,
                   columnspacing=0.8, fontsize=6.5)
    for a in axes:
        a.set_ylim(2, 6e4)
    save(fig, "fig_size")


def fig_time(rows):
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.25))
    for s in ORDER:
        m, ls, c = STYLE[s]
        x, y = series(rows, s, (("N", 64), "t"), "sign_ms")
        axes[0].plot(x, y, marker=m, ls=ls, color=c, label=NAME[s])
        x, y = series(rows, s, (("t", 8), "N"), "verify_ms")
        axes[1].plot(x, y, marker=m, ls=ls, color=c, label=NAME[s])
    axes[0].set_xlabel("threshold $t$ (ring size $N=64$)")
    axes[0].set_ylabel("signing time (ms)")
    axes[1].set_xlabel("ring size $N$ (threshold $t=8$)")
    axes[1].set_ylabel("verification time (ms)")
    for a in axes:
        a.set_yscale("log"); a.set_xscale("log", base=2)
    axes[1].legend(ncol=3, loc="upper left", framealpha=0.95, handlelength=1.5,
                   columnspacing=0.8, fontsize=6.5)
    axes[0].set_ylim(60, 3.5e4)
    axes[1].set_ylim(12, 2e5)
    save(fig, "fig_time")


def fig_3d():
    Ns = [16, 32, 64, 128, 256, 512, 1024]
    ts = [2, 4, 8, 16, 32, 64]
    X, Y = np.meshgrid(np.log2(Ns), np.log2(ts))
    Zo = np.zeros_like(X)
    Zc = np.zeros_like(X)
    for i, t in enumerate(ts):
        for j, N in enumerate(Ns):
            Zo[i, j] = ours_size(N, t)[0] / 1024.0
            Zc[i, j] = concat_size(N, t) / 1024.0
    fig = plt.figure(figsize=(3.6, 3.0))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_wireframe(X, Y, np.log10(Zc), color="#8172b3", linewidth=0.7,
                      rstride=1, cstride=1)
    ax.plot_surface(X, Y, np.log10(Zo), color="#dfe6ef", edgecolor="0.15",
                    linewidth=0.3, rstride=1, cstride=1, shade=False)
    ax.set_xlabel("$\\log_2 N$", labelpad=-6)
    ax.set_ylabel("$\\log_2 t$", labelpad=-6)
    ax.set_xticks([4, 6, 8, 10]); ax.set_yticks([1, 3, 5])
    ax.set_zticks([0.5, 1.0, 1.5, 2.0])
    ax.tick_params(pad=-2, labelsize=6)
    ax.view_init(elev=20, azim=-58)
    ax.set_box_aspect((1.1, 1.0, 0.78))
    from matplotlib.lines import Line2D
    proxies = [Line2D([0], [0], color="#8172b3", lw=1.2),
               Line2D([0], [0], color="0.15", lw=1.2)]
    ax.legend(proxies, ["Concat", "MQuorum (ours)"], loc="upper left",
              bbox_to_anchor=(0.02, 1.0), fontsize=7, framealpha=0.9,
              handlelength=1.4, borderpad=0.3)
    ax.zaxis.set_rotate_label(False)
    ax.set_zlabel("$\\log_{10}$ of size in KB", labelpad=6, rotation=90)
    fig.subplots_adjust(left=0.03, right=0.86, bottom=0.03, top=0.99)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.line.set_linewidth(0.6)
        axis.pane.set_edgecolor("0.8")
        axis._axinfo["grid"]["linewidth"] = 0.3
    save(fig, "fig_3d", tight=False)


def fig_np(rows):
    fig, ax = plt.subplots(1, 1, figsize=(3.3, 2.3))
    axes = [ax, None]
    sw = json.load(open(os.path.join(RES, "npsweep.json")))
    for N, t, mk, col in [(64, 8, "o", "#4c72b0"), (100, 50, "s", "#c44e52")]:
        pts = [(r["n_party"], r["sig_bytes"] / 1024.0, r["sign_ms"]) for r in sw
               if r["N"] == N and r["t"] == t]
        pts.sort()
        ref = [p for p in pts if p[0] == 256][0]
        xs = [p[2] / ref[2] for p in pts]
        ys = [p[1] / ref[1] for p in pts]
        axes[0].plot(xs, ys, marker=mk, color=col, label=f"$N={N}$, $t={t}$")
        for k, p in enumerate(pts):
            axes[0].annotate(str(p[0]), (xs[k], ys[k]), fontsize=6, color=col,
                             textcoords="offset points",
                             xytext=(4, 5) if k % 2 == 0 else (4, -10))
    axes[0].set_xlabel("signing time relative to $N_p=256$")
    axes[0].set_ylabel("signature size relative to $N_p=256$")
    axes[0].set_xlim(0.9, 2.05)
    axes[0].set_ylim(0.68, 1.28)
    axes[0].legend(loc="upper right")
    save(fig, "fig_np")
    fig, ax = plt.subplots(1, 1, figsize=(3.3, 2.3))
    axes = [None, ax]
    ts = [2, 4, 8, 16, 32, 64]
    wit, tree, brd = [], [], []
    for t in ts:
        sz, d, b, tau, rho, D, wbits = ours_size(64, t)
        wit.append(tau * wbits / 8192.0)
        tree.append(tau * (128 * 8 + 256) / 8192.0)
        brd.append((4 * 128 + tau * rho * (D - 1) * 16) / 8192.0)
    idx = np.arange(len(ts))
    axes[1].bar(idx, wit, 0.6, label="witness offsets", color="#4c72b0")
    axes[1].bar(idx, tree, 0.6, bottom=wit, label="tree paths", color="#dd8452")
    axes[1].bar(idx, brd, 0.6, bottom=np.array(wit) + np.array(tree),
                label="broadcast", color="#55a868")
    axes[1].set_xticks(idx); axes[1].set_xticklabels([str(v) for v in ts])
    axes[1].set_xlabel("threshold $t$ (ring size $N=64$)")
    axes[1].set_ylabel("signature size (KB)")
    axes[1].legend(loc="upper left")
    save(fig, "fig_break")


def main():
    rows = load()
    fig_size(rows)
    fig_time(rows)
    fig_3d()
    if os.path.exists(os.path.join(RES, "npsweep.json")):
        fig_np(rows)


if __name__ == "__main__":
    main()
