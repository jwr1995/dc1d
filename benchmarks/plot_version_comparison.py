"""
Bar charts for the dc1d 0.0.4 versus 0.2.0 comparison (BACKENDS.md section 7).

Reads the raw sweep in ``results/v0.0.4_vs_v0.2.0.jsonl`` and writes three PNGs
to ``figures/``. It re-derives every number from that file, so the figures and
the tables in BACKENDS.md cannot drift apart.

Latency is the minimum median across all rounds, which is the usual estimator
for a contended machine: the fastest observation is the one least polluted by a
neighbour. Peak memory is taken from any round, because it came out identical in
all three, and the script asserts that rather than assuming it.

Needs matplotlib, which lives in the ``demo`` dependency group:

    uv run --group demo python benchmarks/plot_version_comparison.py
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).parent
RESULTS = HERE / "results" / "v0.0.4_vs_v0.2.0.jsonl"
FIGURES = HERE / "figures"

OLD, NEW = "004", "020"
LABEL = {OLD: "dc1d 0.0.4", NEW: "dc1d 0.2.0"}
COLOUR = {OLD: "#b0b0b0", NEW: "#2b7bba"}
PRETTY = {"fwd": "forward", "fwdbwd": "forward+backward"}
ORDER = [
    ("plain", "fwd"),
    ("plain", "fwdbwd"),
    ("depthwise", "fwd"),
    ("depthwise", "fwdbwd"),
    ("long-seq", "fwd"),
    ("long-seq", "fwdbwd"),
]


def load():
    """{(cfg, mode, version): {"ms": float, "mib": float}} from the raw sweep."""
    rows = [
        json.loads(line)["r"] | {"ver": json.loads(line)["ver"]}
        for line in RESULTS.read_text().splitlines()
    ]
    out = {}
    for cfg, mode in ORDER:
        for ver in (OLD, NEW):
            hits = [
                r for r in rows if r["cfg_name"] == cfg and r["mode"] == mode and r["ver"] == ver
            ]
            assert hits, f"no rows for {cfg}/{mode}/{ver}"
            peaks = {round(r["peak_mib"], 4) for r in hits}
            assert len(peaks) == 1, (
                f"peak memory varied across rounds for {cfg}/{mode}/{ver}: {peaks}"
            )
            out[cfg, mode, ver] = {
                "ms": min(m for r in hits for m in r["all_medians_ms"]),
                "mib": hits[0]["peak_mib"],
            }
    return out


def _paired_bars(ax, data, key, ylabel, title):
    xs = range(len(ORDER))
    width = 0.38
    for i, ver in enumerate((OLD, NEW)):
        vals = [data[cfg, mode, ver][key] for cfg, mode in ORDER]
        pos = [x + (i - 0.5) * width for x in xs]
        bars = ax.bar(
            pos, vals, width, label=LABEL[ver], color=COLOUR[ver], edgecolor="black", linewidth=0.4
        )
        ax.bar_label(bars, fmt="%.3g", fontsize=7, padding=2)
    ax.set_yscale("log")
    ax.set_xticks(list(xs))
    ax.set_xticklabels([f"{cfg}\n{PRETTY[mode]}" for cfg, mode in ORDER], fontsize=8)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=11)
    ax.legend(frameon=False, fontsize=9)
    ax.grid(axis="y", alpha=0.3, which="both")
    ax.set_axisbelow(True)


def main():
    FIGURES.mkdir(exist_ok=True)
    data = load()

    for key, ylabel, title, name in [
        ("ms", "milliseconds (log scale)", "Latency: lower is better", "01_latency.png"),
        ("mib", "peak MiB (log scale)", "Peak VRAM: lower is better", "02_peak_vram.png"),
    ]:
        fig, ax = plt.subplots(figsize=(9, 4.2))
        _paired_bars(ax, data, key, ylabel, title)
        fig.tight_layout()
        fig.savefig(FIGURES / name, dpi=150)
        plt.close(fig)
        print(f"wrote figures/{name}")

    # Ratios, so the depthwise memory result is not lost in a log axis.
    fig, ax = plt.subplots(figsize=(9, 4.2))
    labels = [f"{cfg}\n{PRETTY[mode]}" for cfg, mode in ORDER]
    speed = [data[c, m, OLD]["ms"] / data[c, m, NEW]["ms"] for c, m in ORDER]
    mem = [data[c, m, OLD]["mib"] / data[c, m, NEW]["mib"] for c, m in ORDER]
    xs = range(len(ORDER))
    width = 0.38
    for i, (vals, lab, col) in enumerate(
        [(speed, "speedup", "#2b7bba"), (mem, "memory ratio", "#5aa469")]
    ):
        bars = ax.bar(
            [x + (i - 0.5) * width for x in xs],
            vals,
            width,
            label=lab,
            color=col,
            edgecolor="black",
            linewidth=0.4,
        )
        ax.bar_label(bars, fmt="%.2fx", fontsize=7, padding=2)
    ax.axhline(1.0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xticks(list(xs))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("0.0.4 / 0.2.0 (higher favours 0.2.0)")
    ax.set_title("How much 0.2.0 wins by. 1.0x is parity", fontsize=11)
    ax.legend(frameon=False, fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(FIGURES / "03_ratios.png", dpi=150)
    plt.close(fig)
    print("wrote figures/03_ratios.png")


if __name__ == "__main__":
    main()
