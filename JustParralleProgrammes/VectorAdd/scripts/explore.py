"""Explore results/run.csv -> plots/*.png.  Run: .venv/bin/python scripts/explore.py"""
import csv, statistics as st
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rows = list(csv.DictReader(open("results/run.csv")))
g = defaultdict(list)  # (kernel, len, nthreads, chunk) -> [ns]
for r in rows:
    g[(r["kernel"], int(r["len"]), int(r["nthreads"]), int(r["chunk"]))].append(int(r["time_ns"]))
med = {k: st.median(v) for k, v in g.items()}
lens = sorted({k[1] for k in med})
chunks = sorted({k[3] for k in med if k[0] == "neon_threads"})
threads = sorted({k[2] for k in med if k[0] == "neon_threads"})
gelem = lambda n, ns: n / ns  # elements per ns == Gelem/s
lab = lambda n: f"{n/1024:g}K" if n < 1 << 20 else f"{n/(1<<20):g}M"

plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.alpha": .25, "figure.dpi": 130})
C = {"scalar": "#8a8a8a", "neon": "#2a6fdb", "neon_threads": "#e8590c"}

# 1. throughput by kernel (best thread config for neon_threads) vs size
fig, ax = plt.subplots(figsize=(7, 4))
for kern in C:
    ys = []
    for n in lens:
        ts = [v for k, v in med.items() if k[0] == kern and k[1] == n]
        ys.append(gelem(n, min(ts)))
    ax.plot(range(len(lens)), ys, "o-", color=C[kern], label=kern + (" (best cfg)" if kern == "neon_threads" else ""))
ax.set_xticks(range(len(lens)), [lab(n) for n in lens])
ax.set(xlabel="vector length (elements)", ylabel="throughput (Gelem/s)", title="Throughput by kernel")
ax.legend(); fig.tight_layout(); fig.savefig("plots/throughput.png")

# 2. thread scaling: speedup over single-thread neon, one panel per size
fig, axs = plt.subplots(1, len(lens), figsize=(4 * len(lens), 3.6), sharey=False)
for ax, n in zip(axs, lens):
    base = med[("neon", n, 1, 0)]
    for c, ls in zip(chunks, ["-", "--"]):
        ax.plot(threads, [base / med[("neon_threads", n, t, c)] for t in threads], "o" + ls,
                color=C["neon_threads"], alpha=1 if ls == "-" else .6, label=f"chunk {lab(c)}")
    ax.axhline(1, color=C["neon"], lw=1)
    ax.set(title=f"len {lab(n)}", xlabel="threads", xticks=threads[::2])
axs[0].set_ylabel("speedup vs neon (1 thread)"); axs[0].legend()
fig.tight_layout(); fig.savefig("plots/thread_scaling.png")

# 3. run-to-run noise: coefficient of variation per config
fig, ax = plt.subplots(figsize=(7, 4))
for kern in C:
    cv = [st.pstdev(v) / st.mean(v) * 100 for k, v in g.items() if k[0] == kern]
    ax.scatter([kern] * len(cv), cv, color=C[kern], alpha=.6)
ax.set(ylabel="run-to-run CV (%)", title="Measurement noise per config")
fig.tight_layout(); fig.savefig("plots/noise.png")

# text summary
for n in lens:
    best = min((v, k) for k, v in med.items() if k[1] == n and k[0] == "neon_threads")
    print(f"len {n:>9}: scalar {med[('scalar', n, 1, 0)]:>10.0f}ns  neon {med[('neon', n, 1, 0)]:>10.0f}ns  "
          f"best threads={best[1][2]} chunk={best[1][3]} {best[0]:>10.0f}ns")
