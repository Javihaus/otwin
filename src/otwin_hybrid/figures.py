"""Figures for the hybrid digital twin comparison.

Every figure is generated from `results/curves.csv` and `results/metrics.csv`,
which `otwin_hybrid.comparison` writes. Nothing here computes a number; if a figure
disagrees with the README, the README is the bug.

Run `python -m otwin_hybrid.comparison && python -m otwin_hybrid.figures`.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

from ._paths import figures_dir, results_dir  # noqa: E402

RES = results_dir()
FIG = figures_dir()
FIG.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", context="talk", font_scale=0.85)

INK = "#1B2430"
COLOURS = {
    "soh_true": INK,
    "physics": "#1C4E73",  # blue   — first principles
    "gp": "#B5651D",  # orange — data only
    "hybrid": "#2E7D32",  # green  — physics + learned residual
    "persistence": "#9AA5B1",
    "drift": "#C0C4CC",
}
LABELS = {
    "physics": "Physics only (Wang power law)",
    "gp": "Data only (Gaussian process)",
    "hybrid": "Hybrid (physics + NN residual)",
    "persistence": "Baseline: persistence",
    "drift": "Baseline: linear drift",
}
EOL = 0.80


def _load():
    return pd.read_csv(RES / "curves.csv"), pd.read_csv(RES / "metrics.csv")


def _save(fig, name: str) -> None:
    fig.savefig(FIG / f"{name}.png", dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  figures/{name}.png")


# ---------------------------------------------------------------------------


def fig_hero(curves: pd.DataFrame, cell: str = "B0005") -> None:
    """The one figure that carries the argument.

    Two panels rather than an inset: an inset large enough to read covers the
    data it is supposed to explain, and the annotations end up fighting the
    legend for the same corner.
    """
    g = curves[curves.battery == cell]
    split = g.loc[g.split, "cycle"].max()
    cmax = g.cycle.max()

    fig, (ax, axz) = plt.subplots(
        1, 2, figsize=(15.5, 6.4), gridspec_kw={"width_ratios": [1.72, 1]}
    )

    # ---------------- main panel ----------------
    ax.axvspan(g.cycle.min(), split, color="#EEF2F6", zorder=0)
    ax.scatter(
        g.cycle,
        g.soh_true,
        s=17,
        color=INK,
        alpha=0.55,
        label="Measured SoH",
        zorder=3,
        linewidths=0,
    )
    for m in ("physics", "gp", "hybrid"):
        ax.plot(g.cycle, g[m], lw=2.6, color=COLOURS[m], label=LABELS[m], zorder=4)
    ax.plot(
        g.cycle,
        g["persistence"],
        lw=1.6,
        ls=(0, (4, 3)),
        color=COLOURS["persistence"],
        label=LABELS["persistence"],
        zorder=2,
    )

    ax.axhline(EOL, color="#B00020", lw=1.4, ls="--", zorder=1)
    ax.text(cmax - 2, EOL + 0.012, "end of life (80 %)", color="#B00020", fontsize=11, ha="right")
    ax.axvline(split, color="#66707A", lw=1.2, zorder=1)

    ax.annotate(
        "",
        xy=(g.cycle.min(), 1.048),
        xytext=(split, 1.048),
        arrowprops=dict(arrowstyle="<->", color="#66707A", lw=1.1),
    )
    ax.text(
        split / 2,
        1.053,
        "fitted here — first 40 %",
        ha="center",
        va="bottom",
        fontsize=10.5,
        color="#66707A",
    )
    ax.annotate(
        "",
        xy=(split, 1.048),
        xytext=(cmax, 1.048),
        arrowprops=dict(arrowstyle="<->", color="#66707A", lw=1.1),
    )
    ax.text(
        (split + cmax) / 2,
        1.053,
        "forecast — never seen",
        ha="center",
        va="bottom",
        fontsize=10.5,
        color="#66707A",
    )

    ax.set_xlabel("Discharge cycle")
    ax.set_ylabel("State of Health  (capacity / initial capacity)")
    ax.set_ylim(0.55, 1.09)
    ax.set_xlim(g.cycle.min() - 2, cmax + 2)
    ax.legend(loc="lower left", frameon=True, framealpha=0.94, fontsize=10.5)

    # ---------------- zoom panel ----------------
    z0 = int(split + (cmax - split) * 0.30)
    zg = g[g.cycle >= z0]
    axz.scatter(zg.cycle, zg.soh_true, s=26, color=INK, alpha=0.6, linewidths=0, zorder=3)
    for m in ("physics", "gp", "hybrid"):
        axz.plot(zg.cycle, zg[m], lw=2.6, color=COLOURS[m], zorder=4)
    axz.plot(
        zg.cycle, zg["persistence"], lw=1.6, ls=(0, (4, 3)), color=COLOURS["persistence"], zorder=2
    )
    axz.axhline(EOL, color="#B00020", lw=1.3, ls="--", zorder=1)

    lo = min(zg.soh_true.min(), zg[["physics", "hybrid"]].min().min()) - 0.025
    hi = max(zg.soh_true.max(), zg[["gp", "persistence"]].max().max()) + 0.025
    axz.set_xlim(z0, cmax)
    axz.set_ylim(lo, hi)
    axz.set_xlabel("Discharge cycle")
    axz.set_title("the last third, up close", loc="left", fontsize=12.5, color="#66707A")

    # mark the zoom region on the main panel
    ax.add_patch(
        Rectangle((z0, lo), cmax - z0, hi - lo, fill=False, ec="#66707A", lw=1.2, ls=":", zorder=5)
    )

    fig.suptitle(
        f"Cell {cell} — forecasting 60 % of life from the first 40 %.  "
        "The data-only model reverts to its mean; the physics keeps falling.",
        x=0.012,
        ha="left",
        fontsize=15,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    _save(fig, "01_hero_forecast")


def fig_all_cells(curves: pd.DataFrame) -> None:
    """Small multiples. One good cell proves nothing."""
    cells = sorted(curves.battery.unique())
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.4), sharey=True)
    for ax, cell in zip(axes.ravel(), cells, strict=False):
        g = curves[curves.battery == cell]
        split = g.loc[g.split, "cycle"].max()
        ax.axvspan(g.cycle.min(), split, color="#EEF2F6", zorder=0)
        ax.scatter(g.cycle, g.soh_true, s=11, color=INK, alpha=0.5, zorder=3, linewidths=0)
        for m in ("physics", "gp", "hybrid"):
            ax.plot(g.cycle, g[m], lw=2.1, color=COLOURS[m], zorder=4)
        ax.axhline(EOL, color="#B00020", lw=1.1, ls="--", zorder=1)
        ax.axvline(split, color="#66707A", lw=1.0, zorder=1)
        ax.set_title(cell, loc="left", fontsize=13)
        ax.set_ylim(0.5, 1.03)
    for ax in axes[1]:
        ax.set_xlabel("Discharge cycle")
    for ax in axes[:, 0]:
        ax.set_ylabel("State of Health")
    handles = [
        plt.Line2D([], [], color=COLOURS[m], lw=2.4, label=LABELS[m])
        for m in ("physics", "gp", "hybrid")
    ]
    handles.append(
        plt.Line2D([], [], marker="o", ls="", color=INK, alpha=0.5, label="Measured SoH")
    )
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, -0.035),
        fontsize=11,
    )
    fig.suptitle(
        "All four cells, same protocol — including the one where the hybrid "
        "is not the winner (B0018)",
        x=0.02,
        ha="left",
        fontsize=14.5,
    )
    fig.tight_layout()
    _save(fig, "02_all_cells")


def fig_accuracy(metrics: pd.DataFrame) -> None:
    """Skill score per model, per cell. Below 1.0 beats persistence."""
    order = ["gp", "physics", "drift", "hybrid"]
    d = metrics[metrics.model.isin(order)].copy()
    d["label"] = d.model.map(lambda m: LABELS[m].split(" (")[0])

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(13.5, 5.6), gridspec_kw={"width_ratios": [1.15, 1]}
    )

    means = d.groupby("model").skill.mean().reindex(order)
    bars = ax1.barh(
        [LABELS[m].split(" (")[0] for m in order],
        means.values,
        color=[COLOURS[m] for m in order],
        alpha=0.9,
        height=0.62,
    )
    ax1.axvline(1.0, color="#B00020", lw=1.6, ls="--")
    ax1.text(1.02, -0.62, "persistence baseline", color="#B00020", fontsize=10.5)
    for bar, v in zip(bars, means.values, strict=True):
        ax1.text(
            v + 0.03,
            bar.get_y() + bar.get_height() / 2,
            f"{v:.2f}",
            va="center",
            fontsize=12,
            fontweight="bold",
        )
    ax1.set_xlabel("Skill score  =  model RMSE ÷ persistence RMSE")
    ax1.set_title("Lower is better. Above 1.0 loses to doing nothing.", loc="left", fontsize=13)
    ax1.set_xlim(0, max(means.values) * 1.18)

    sns.stripplot(
        data=d,
        y="label",
        x="skill",
        hue="label",
        order=[LABELS[m].split(" (")[0] for m in order],
        palette=[COLOURS[m] for m in order],
        s=11,
        ax=ax2,
        legend=False,
        alpha=0.85,
        jitter=0.12,
    )
    ax2.axvline(1.0, color="#B00020", lw=1.6, ls="--")
    ax2.set_ylabel("")
    ax2.set_yticklabels([])
    ax2.set_xlabel("Skill score, per cell")
    ax2.set_title("Spread across the four cells", loc="left", fontsize=13)

    fig.suptitle(
        "A Gaussian process on cycle number is worse than assuming nothing changes",
        x=0.02,
        ha="left",
        fontsize=15,
    )
    fig.tight_layout()
    _save(fig, "03_skill")


def fig_eol(metrics: pd.DataFrame) -> None:
    """The metric an operator acts on — and where the ranking flips."""
    d = metrics.dropna(subset=["eol_error"]).copy()
    if d.empty:
        print("  (no cells reach end-of-life after the split; skipping)")
        return
    order = [m for m in ("physics", "hybrid", "drift") if m in set(d.model)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.6))

    rm = metrics[metrics.model.isin(order)].groupby("model").rmse.mean().reindex(order)
    ee = d.groupby("model").eol_error.apply(lambda s: s.abs().mean()).reindex(order)

    x = np.arange(len(order))
    ax1.bar(
        x - 0.2, rm.values / rm.max(), 0.38, label="Forecast RMSE (normalised)", color="#9AA5B1"
    )
    ax1.bar(
        x + 0.2, ee.values / ee.max(), 0.38, label="End-of-life error (normalised)", color="#B5651D"
    )
    ax1.set_xticks(x)
    ax1.set_xticklabels([LABELS[m].split(" (")[0] for m in order], fontsize=11)
    ax1.set_ylabel("relative to the worst")
    ax1.legend(frameon=False, fontsize=10.5)
    ax1.set_title("The metric decides the winner", loc="left", fontsize=13)

    for m in order:
        sub = d[d.model == m]
        ax2.scatter(
            sub.eol_true,
            sub.eol_pred,
            s=140,
            color=COLOURS[m],
            label=LABELS[m].split(" (")[0],
            zorder=3,
            alpha=0.9,
        )
    lo = min(d.eol_true.min(), d.eol_pred.min()) - 12
    hi = max(d.eol_true.max(), d.eol_pred.max()) + 12
    ax2.plot([lo, hi], [lo, hi], color=INK, lw=1.2, ls="--", zorder=1)
    ax2.fill_between(
        [lo, hi], [lo - 20, hi - 20], [lo + 20, hi + 20], color="#2E7D32", alpha=0.07, zorder=0
    )
    ax2.text(hi - 4, lo + 8, "±20 cycles", color="#2E7D32", fontsize=10.5, ha="right")
    ax2.set_xlim(lo, hi)
    ax2.set_ylim(lo, hi)
    ax2.set_xlabel("Actual end-of-life cycle")
    ax2.set_ylabel("Predicted")
    ax2.legend(frameon=False, fontsize=10.5, loc="upper left")
    ax2.set_title("When do I replace it?", loc="left", fontsize=13)

    fig.suptitle(
        "The hybrid has the lowest RMSE. The physics predicts replacement better.",
        x=0.02,
        ha="left",
        fontsize=15,
    )
    fig.tight_layout()
    _save(fig, "04_end_of_life")


def fig_residuals(curves: pd.DataFrame, cell: str = "B0005") -> None:
    """What the network is actually learning, and what is left over."""
    g = curves[curves.battery == cell]
    split = g.loc[g.split, "cycle"].max()
    res_phys = g.soh_true - g.physics_fit
    res_hyb = g.soh_true - g.hybrid_fit

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))

    axes[0].axvspan(g.cycle.min(), split, color="#EEF2F6", zorder=0)
    axes[0].axhline(0, color=INK, lw=1.1)
    axes[0].scatter(
        g.cycle,
        res_phys,
        s=16,
        color=COLOURS["physics"],
        alpha=0.75,
        label="left by the physics",
        linewidths=0,
    )
    axes[0].plot(
        g.cycle,
        g.hybrid_fit - g.physics_fit,
        lw=2.4,
        color=COLOURS["hybrid"],
        label="learned by the network",
    )
    axes[0].set_xlabel("Discharge cycle")
    axes[0].set_ylabel("SoH residual")
    axes[0].legend(frameon=False, fontsize=10)
    axes[0].set_title("The network only sees this", loc="left", fontsize=12.5)

    axes[1].axvspan(g.cycle.min(), split, color="#EEF2F6", zorder=0)
    axes[1].axhline(0, color=INK, lw=1.1)
    axes[1].scatter(g.cycle, res_hyb, s=16, color=COLOURS["hybrid"], alpha=0.75, linewidths=0)
    axes[1].set_xlabel("Discharge cycle")
    axes[1].set_title("What the hybrid still gets wrong", loc="left", fontsize=12.5)

    sns.histplot(
        res_phys.dropna(),
        bins=22,
        kde=True,
        ax=axes[2],
        color=COLOURS["physics"],
        alpha=0.45,
        label="physics",
        stat="density",
    )
    sns.histplot(
        res_hyb.dropna(),
        bins=22,
        kde=True,
        ax=axes[2],
        color=COLOURS["hybrid"],
        alpha=0.45,
        label="hybrid",
        stat="density",
    )
    axes[2].axvline(0, color=INK, lw=1.1)
    axes[2].set_xlabel("SoH residual")
    axes[2].legend(frameon=False, fontsize=10)
    axes[2].set_title("Residual distribution", loc="left", fontsize=12.5)

    fig.suptitle(
        f"Cell {cell} — the residual is small and bounded, which is why the "
        "hybrid cannot run away",
        x=0.02,
        ha="left",
        fontsize=14.5,
    )
    fig.tight_layout()
    _save(fig, "05_residuals")


def fig_horizon(curves: pd.DataFrame) -> None:
    """Error against how far ahead you are forecasting."""
    rows = []
    for cell, g in curves.groupby("battery"):
        split = g.loc[g.split, "cycle"].max()
        te = g[~g.split]
        for m in ("physics", "gp", "hybrid", "persistence", "drift"):
            rows.append(
                pd.DataFrame(
                    {
                        "battery": cell,
                        "model": m,
                        "horizon": te.cycle - split,
                        "abs_error": (te.soh_true - te[m]).abs(),
                    }
                )
            )
    d = pd.concat(rows, ignore_index=True)
    d["bin"] = (d.horizon // 15) * 15

    fig, ax = plt.subplots(figsize=(11.5, 6))
    for m in ("gp", "persistence", "physics", "drift", "hybrid"):
        s = d[d.model == m].groupby("bin").abs_error.mean()
        ax.plot(
            s.index,
            s.values,
            lw=2.6 if m in ("hybrid", "gp") else 1.8,
            color=COLOURS[m],
            label=LABELS[m],
            ls="--" if m in ("persistence", "drift") else "-",
        )
    ax.set_xlabel("Forecast horizon (cycles beyond the training window)")
    ax.set_ylabel("Mean absolute SoH error")
    ax.set_title(
        "Error grows with horizon — but not at the same rate for everyone",
        loc="left",
        fontsize=14.5,
        pad=10,
    )
    ax.legend(frameon=False, fontsize=10.5)
    _save(fig, "06_horizon")


def main() -> None:
    curves, metrics = _load()
    print("writing figures:")
    fig_hero(curves)
    fig_all_cells(curves)
    fig_accuracy(metrics)
    fig_eol(metrics)
    fig_residuals(curves)
    fig_horizon(curves)
    print(f"\n  {len(list(FIG.glob('*.png')))} figures in {FIG}")


if __name__ == "__main__":
    main()
