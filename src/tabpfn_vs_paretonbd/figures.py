"""Static figures regenerated from saved tables (results/<dataset>/<split>/figures)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from tabpfn_vs_paretonbd.evaluate import DISPLAY  # noqa: E402

COLORS = {"tabpfn": "#1f5aa6", "lgbm": "#e07b00", "glm": "#6a9f3a", "pareto_nbd": "#8b3a8b", "abe_hb": "#c2185b",
          "recency": "#7f7f7f", "base_rate": "#bbbbbb"}


def _label(m, fs):
    return DISPLAY.get(m, m) + ("" if m in ("pareto_nbd", "abe_hb") else f" [{fs}]")


def calibration(rel: pd.DataFrame, out: Path, features: str):
    fig, ax = plt.subplots(figsize=(5.2, 5))
    ax.plot([0, 1], [0, 1], color="#999", lw=1, ls="--")
    for (m, fs), g in rel.groupby(["model", "features"]):
        if (fs != features and m not in ("pareto_nbd", "abe_hb")) or m == "base_rate":
            continue
        ax.plot(g.mean_pred, g.observed, marker="o", ms=3.5, lw=1.4, color=COLORS.get(m), label=_label(m, fs))
    ax.set(xlabel="mean predicted P(inactive in 90 days)", ylabel="observed share inactive", xlim=(0, 1), ylim=(0, 1),
           title=f"Calibration, pooled holdout origins ({features} features)")
    ax.legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def learning_curve(lc: pd.DataFrame, pooled: pd.DataFrame, out: Path, metric="brier"):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    # Cap the axis at 1.3x the base-rate reference so one unstable small-sample fit cannot flatten the others;
    # points above the cap are drawn at the edge and labelled with their true value.
    base = pooled.loc[pooled.model == "base_rate", metric]
    top = 1.3 * float(base.iloc[0]) if len(base) and pd.notna(base.iloc[0]) else None
    for ax, fs in zip(axes, ["common", "expanded"]):
        g = lc[lc.features == fs].groupby(["model", "n_customers"])[metric]
        s = g.mean().unstack(0)
        lo, hi = g.min().unstack(0), g.max().unstack(0)
        for m in s.columns:
            ax.plot(s.index, s[m], marker="o", color=COLORS.get(m), label=DISPLAY.get(m, m))
            ax.fill_between(s.index, lo[m], hi[m], color=COLORS.get(m), alpha=0.12)
            if top is not None:
                for n, v in s[m][s[m] > top].items():
                    ax.annotate(f"{DISPLAY.get(m, m)}: {v:.3g} (off scale)", (n, top), xytext=(4, -10),
                                textcoords="offset points", fontsize=6, color=COLORS.get(m), arrowprops=None)
                    ax.plot([n], [top], marker="^", color=COLORS.get(m))
        for m in ("pareto_nbd", "abe_hb", "base_rate"):
            r = pooled[(pooled.model == m)]
            if len(r) and metric in r and r[metric].notna().any():
                ax.axhline(r[metric].iloc[0], color=COLORS[m], ls="--", lw=1.1,
                           label=DISPLAY[m] + (" (no labels used)" if m != "base_rate" else " (full window)"))
        ax.set_xscale("log")
        if top is not None:
            ax.set_ylim(top=top)
        ax.set(title=f"{fs} information set", xlabel="labelled training customers (log scale)")
    axes[0].set_ylabel(f"holdout {metric.replace('_', ' ')} (lower is better)")
    axes[1].legend(fontsize=7)
    fig.suptitle("Learning curve: mean over origins and seeds; band = min-max across seeds and origins", fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def by_origin(tab: pd.DataFrame, out: Path, metric="brier"):
    t = tab[tab.model != "base_rate"].copy()
    t["label"] = [_label(m, f) for m, f in zip(t.model, t.features)]
    p = t.pivot_table(index="origin", columns="label", values=metric)
    ax = p.plot(kind="bar", figsize=(10, 4), width=0.85)
    ax.set(ylabel=metric, title=f"{metric} by holdout origin (lower is better)")
    ax.legend(fontsize=6, ncol=2)
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()


def make_all(d: Path, tables: dict[str, pd.DataFrame]):
    f = d / "figures"
    f.mkdir(exist_ok=True)
    for fs in ("common", "expanded"):
        calibration(tables["reliability"], f / f"calibration_{fs}.png", fs)
    by_origin(tables["by_origin"], f / "brier_by_origin.png")
    if "learning_curve" in tables and len(tables["learning_curve"]):
        learning_curve(tables["learning_curve"], tables["pooled"], f / "learning_curve_brier.png")
        if "count_mae" in tables["learning_curve"]:
            learning_curve(tables["learning_curve"], tables["pooled"].rename(columns={"count_mae": "count_mae"}),
                           f / "learning_curve_count_mae.png", metric="count_mae")
