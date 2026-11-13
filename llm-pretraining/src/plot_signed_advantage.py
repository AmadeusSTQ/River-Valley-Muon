#!/usr/bin/env python3
"""Plot Muon-to-AdamW excess validation loss relative to Muon-only runs.

The excess loss at training step t is defined as

    L_Muon-to-AdamW(t) - min_j L_Muon-only,j(t).

Negative values therefore mean that the hybrid optimizer has lower validation
loss than every Muon-only baseline recorded at the same training step. This
orientation preserves the familiar loss-curve convention: lower is better.
"""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    import numpy as np
    import pandas as pd
except ImportError as exc:
    raise SystemExit(
        "Missing data dependencies. Install them with:\n"
        "  python -m pip install pandas numpy"
    ) from exc


# Central W&B val/loss columns only. The exported MIN/MAX columns are not
# statistical error bars in this CSV and are intentionally ignored.
HYBRID_RUNS = {
    15_000: "muon15000+adamwcf - val/loss",
    16_000: "muon16000+adamw9e-5cf - val/loss",
    17_000: "muon17000+adamw9e-5 - val/loss",
    18_000: "muon18000+adamw9e-5 - val/loss",
    19_000: "muon19000+adamw9e-5 - val/loss",
}

MUON_ONLY_RUNS = [
    "muon20000cos_inf - val/loss",
    "muon20000cos - val/loss",
    "muon20000 - val/loss",
]

# Okabe-Ito-derived colors, chosen to remain distinguishable under common
# forms of color-vision deficiency.
COLORS = {
    15_000: "#0072B2",
    16_000: "#E69F00",
    17_000: "#009E73",
    18_000: "#CC79A7",
    19_000: "#D55E00",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot excess validation loss relative to Muon-only runs."
    )
    parser.add_argument("csv", type=Path, help="Path to muon20k.csv")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("handoff_excess_loss.pdf"),
        help="Output vector PDF (default: handoff_excess_loss.pdf)",
    )
    parser.add_argument(
        "--title",
        default="Late-stage AdamW handoffs reduce excess validation loss",
        help="Figure title; pass an empty string to omit it",
    )
    return parser.parse_args()


def first_sustained_win(steps: np.ndarray, excess: np.ndarray, length: int = 3) -> int | None:
    """Return the first step beginning `length` consecutive negative values."""
    wins = excess < 0
    for index in range(len(wins) - length + 1):
        if wins[index : index + length].all():
            return int(steps[index])
    return None


def load_curves(csv_path: Path) -> list[dict]:
    df = pd.read_csv(csv_path)

    required = ["iter", *MUON_ONLY_RUNS, *HYBRID_RUNS.values()]
    missing = [column for column in required if column not in df.columns]
    if missing:
        formatted = "\n  - ".join(missing)
        raise ValueError(f"CSV is missing required columns:\n  - {formatted}")

    if df["iter"].duplicated().any():
        raise ValueError("The iter column contains duplicate training steps.")
    df = df.sort_values("iter").reset_index(drop=True)

    # This lower envelope is a conservative, pointwise comparison: the hybrid
    # must beat the best available Muon-only schedule at each exact step.
    df["best_muon_only"] = df[MUON_ONLY_RUNS].min(axis=1, skipna=True)

    curves: list[dict] = []
    for switch_step, column in HYBRID_RUNS.items():
        aligned = df.loc[
            df[column].notna() & df["best_muon_only"].notna(),
            ["iter", column, "best_muon_only"],
        ].copy()
        if aligned.empty:
            raise ValueError(f"No aligned observations found for {column!r}.")
        if int(aligned["iter"].iloc[0]) != switch_step:
            raise ValueError(
                f"{column!r} starts at {int(aligned['iter'].iloc[0])}, "
                f"but {switch_step} was expected."
            )

        steps = aligned["iter"].to_numpy(dtype=int)
        excess = (
            aligned[column].to_numpy(dtype=float)
            - aligned["best_muon_only"].to_numpy(dtype=float)
        )
        crossover = first_sustained_win(steps, excess)
        curves.append(
            {
                "switch": switch_step,
                "steps": steps,
                "excess": excess,
                "crossover": crossover,
                "final_excess": float(excess[-1]),
            }
        )
    return curves


def plot(curves: list[dict], output: Path, title: str) -> None:
    try:
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
        from matplotlib.ticker import FuncFormatter, MultipleLocator
    except ImportError as exc:
        raise SystemExit(
            "Missing plotting dependency. Install it with:\n"
            "  python -m pip install matplotlib"
        ) from exc

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 13,
            "axes.linewidth": 0.8,
            "legend.fontsize": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, ax = plt.subplots(figsize=(8.2, 4.8))

    all_excess = np.concatenate([curve["excess"] for curve in curves])
    y_min = min(-0.035, float(all_excess.min()) - 0.004)
    y_max = max(0.07, float(all_excess.max()) + 0.004)

    # Quiet background cues make the sign immediately readable while keeping
    # the data lines dominant.
    ax.axhspan(0, y_max, color="#D55E00", alpha=0.035, linewidth=0)
    ax.axhspan(y_min, 0, color="#009E73", alpha=0.045, linewidth=0)
    ax.axhline(0, color="#202020", linewidth=1.25, zorder=2)

    for curve in curves:
        switch = curve["switch"]
        steps_k = curve["steps"] / 1000.0
        excess = curve["excess"]
        color = COLORS[switch]

        ax.plot(
            steps_k,
            excess,
            color=color,
            linewidth=2.0,
            marker="o",
            markersize=3.1,
            markerfacecolor="white",
            markeredgewidth=0.8,
            markevery=1,
            label=f"Switch at {switch // 1000}K",
            zorder=3,
        )

        # A conventional point plus direct text identifies the handoff without
        # relying on a potentially ambiguous icon.
        ax.scatter(
            steps_k[0],
            excess[0],
            marker="o",
            s=60,
            color=color,
            edgecolor="#202020",
            linewidth=1.0,
            zorder=5,
        )
        ax.annotate(
            f"{switch // 1000}K handoff",
            xy=(steps_k[0], excess[0]),
            xytext=(6, 8),
            textcoords="offset points",
            color=color,
            fontsize=8.5,
            fontweight="bold",
            ha="left",
            va="bottom",
        )

        # Hollow circle: first of three consecutive evaluations below zero.
        crossover = curve["crossover"]
        if crossover is not None:
            index = int(np.flatnonzero(curve["steps"] == crossover)[0])
            ax.scatter(
                crossover / 1000.0,
                excess[index],
                marker="o",
                s=74,
                facecolor="white",
                edgecolor=color,
                linewidth=2.0,
                zorder=6,
            )

    # Endpoint labels use small leader lines so the nearly identical 17K and
    # 18K results remain separately readable.
    label_offsets = {
        15_000: -0.0015,
        16_000: -0.0005,
        17_000: 0.0030,
        18_000: -0.0025,
        19_000: -0.0010,
    }
    for curve in curves:
        switch = curve["switch"]
        final_excess = curve["final_excess"]
        label_y = final_excess - label_offsets[switch]
        color = COLORS[switch]
        ax.annotate(
            f"{switch // 1000}K: {final_excess:+.3f}",
            xy=(20.0, final_excess),
            xytext=(20.28, label_y),
            color=color,
            fontsize=9,
            fontweight="bold",
            va="center",
            arrowprops={
                "arrowstyle": "-",
                "color": color,
                "linewidth": 0.9,
                "shrinkA": 0,
                "shrinkB": 3,
            },
            annotation_clip=False,
        )

    ax.set_xlim(14.8, 20.95)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Excess validation loss vs. best Muon-only (lower is better)")
    if title:
        ax.set_title(title, loc="left", pad=13, fontweight="bold")

    ax.xaxis.set_major_locator(MultipleLocator(1.0))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.0f}K"))
    ax.yaxis.set_major_locator(MultipleLocator(0.01))
    ax.yaxis.set_major_formatter(
        FuncFormatter(lambda value, _: "0" if abs(value) < 5e-7 else f"{value:+.2f}")
    )

    ax.grid(axis="y", color="#B8B8B8", linewidth=0.6, alpha=0.45)
    ax.grid(axis="x", visible=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#666666")
    ax.spines["bottom"].set_color("#666666")
    ax.tick_params(colors="#333333", width=0.8, length=3.5)

    ax.text(
        14.88,
        y_max - 0.003,
        "Hybrid worse",
        color="#A64A18",
        fontsize=9,
        fontweight="bold",
        va="top",
    )
    ax.text(
        14.88,
        y_min + 0.003,
        "Hybrid better",
        color="#007A5A",
        fontsize=9,
        fontweight="bold",
        va="bottom",
    )

    run_legend = ax.legend(
        loc="lower left",
        bbox_to_anchor=(0.0, 1.01),
        ncol=5,
        frameon=False,
        handlelength=2.2,
        columnspacing=1.25,
        borderaxespad=0,
    )
    ax.add_artist(run_legend)
    ax.legend(
        handles=[
            Line2D(
                [0], [0], marker="o", linestyle="none", color="#333333",
                markerfacecolor="white", markeredgewidth=1.6, markersize=7,
                label="Start of sustained advantage"
            ),
        ],
        loc="lower right",
        frameon=False,
        borderaxespad=0.5,
        handletextpad=0.6,
    )

    fig.subplots_adjust(left=0.13, right=0.84, bottom=0.16, top=0.78)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output,
        format="pdf",
        bbox_inches="tight",
        pad_inches=0.04,
        metadata={
            "Title": "Muon-to-AdamW excess validation loss",
            "Creator": "plot_signed_advantage.py",
        },
    )
    plt.close(fig)


def main() -> None:
    args = parse_args()
    curves = load_curves(args.csv)
    plot(curves, args.output, args.title)

    print(f"Saved vector PDF: {args.output.resolve()}")
    for curve in curves:
        crossover = curve["crossover"]
        crossover_text = "none" if crossover is None else f"{crossover / 1000:.1f}K"
        print(
            f"  {curve['switch'] // 1000}K handoff: "
            f"sustained advantage at {crossover_text}, "
            f"20K excess loss {curve['final_excess']:+.6f}"
        )


if __name__ == "__main__":
    main()
