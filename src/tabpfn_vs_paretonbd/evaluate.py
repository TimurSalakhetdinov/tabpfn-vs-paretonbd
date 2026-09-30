"""Metrics, paired customer-clustered bootstrap, calibration and learning curves, all computed from saved predictions.

Positive class: INACTIVE (no purchase occasion in the horizon). Lower Brier / log loss is better; higher PR-AUC and
ROC-AUC are better, PR-AUC is reported with the prevalence it should be compared against.
Log loss clips probabilities to [1e-6, 1 - 1e-6] (documented; no other clipping or recalibration anywhere).

Uncertainty: the same customer appears at every evaluation origin, so rows are not independent. The bootstrap
resamples customers (with all their rows across origins) and applies the same resample to every model, giving
paired, customer-clustered intervals for pooled metrics and for differences against TabPFN.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from tabpfn_vs_paretonbd.config import SEED

EPS = 1e-6
N_BOOT = 2000
SEGMENTS = {  # prespecified before holdout scoring
    "one-time buyers (x=0)": lambda d: d.x == 0,
    "repeat buyers (x>=1)": lambda d: d.x >= 1,
    "recent (last occasion <= 30d)": lambda d: d.days_since_last <= 30,
    "31-90d since last": lambda d: (d.days_since_last > 30) & (d.days_since_last <= 90),
    "91-180d since last": lambda d: (d.days_since_last > 90) & (d.days_since_last <= 180),
    "dormant (> 180d since last)": lambda d: d.days_since_last > 180,
    "new customers (age < 90d)": lambda d: d["T"] < 90,
}
DISPLAY = {"base_rate": "Base rate", "recency": "Recency reference", "glm": "Logistic / Poisson GLM",
           "lgbm": "LightGBM", "tabpfn": "TabPFN-3.5", "pareto_nbd": "Pareto/NBD (MLE)",
           "abe_hb": "HB Pareto/NBD (Abe 2009 M1)"}


def brier(y, p):
    return np.mean((p - y) ** 2)


def log_loss(y, p):
    p = np.clip(p, EPS, 1 - EPS)
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))


def calibration_fit(y, p):
    """Calibration intercept and slope: logistic regression of y on logit(p) (slope 1, intercept 0 is ideal)."""
    from sklearn.linear_model import LogisticRegression

    lp = np.log(np.clip(p, EPS, 1 - EPS) / (1 - np.clip(p, EPS, 1 - EPS)))
    m = LogisticRegression(C=1e6, max_iter=1000).fit(lp[:, None], y)
    return float(m.intercept_[0]), float(m.coef_[0, 0])


def ece(y, p, bins=10):
    if len(np.unique(p)) < 2:
        return float(abs(np.mean(y) - np.mean(p)))
    q = pd.qcut(p, bins, labels=False, duplicates="drop")
    df = pd.DataFrame({"y": y, "p": p, "q": q}).groupby("q").agg(y=("y", "mean"), p=("p", "mean"), n=("y", "size"))
    return float((df.n * (df.y - df.p).abs()).sum() / df.n.sum())


def reliability(y, p, bins=10):
    q = pd.qcut(p, bins, labels=False, duplicates="drop")
    return (pd.DataFrame({"y": y, "p": p, "q": q}).groupby("q")
            .agg(mean_pred=("p", "mean"), observed=("y", "mean"), n=("y", "size")).reset_index(drop=True))


def classification_metrics(y, p) -> dict:
    a, b = calibration_fit(y, p)
    return {"n": len(y), "prevalence": float(np.mean(y)), "brier": float(brier(y, p)), "log_loss": float(log_loss(y, p)),
            "pr_auc": float(average_precision_score(y, p)), "roc_auc": float(roc_auc_score(y, p)),
            "mean_pred": float(np.mean(p)), "calib_intercept": a, "calib_slope": b, "ece": ece(y, p)}


def count_metrics(y, c) -> dict:
    return {"n": len(y), "mae": float(np.mean(np.abs(c - y))), "rmse": float(np.sqrt(np.mean((c - y) ** 2))),
            "bias": float(np.mean(c - y)), "pred_total": float(np.sum(c)), "obs_total": float(np.sum(y)),
            "total_ratio": float(np.sum(c) / np.sum(y))}


# --- loading ---------------------------------------------------------------------------------


def load_predictions(d: Path) -> pd.DataFrame:
    """Long table: one row per (model, features, customer, origin) joined to labels."""
    lab = pd.read_parquet(d / "labels.parquet")
    parts = []
    for f in sorted((d / "preds").glob("*.parquet")):
        model, fs = f.stem.split("__")
        parts.append(pd.read_parquet(f).assign(model=model, features=fs))
    pr = pd.concat(parts, ignore_index=True)
    out = pr.merge(lab, on=["customer_id", "origin"], how="left", validate="many_to_one")
    assert out.y_inactive.notna().all(), "every prediction must have a mature label"
    return out


def keys(df: pd.DataFrame) -> list[tuple[str, str]]:
    return sorted(set(zip(df.model, df.features)))


# --- bootstrap ---------------------------------------------------------------------------------


def _customer_weights(customers: np.ndarray, n_boot: int, seed: int) -> np.ndarray:
    """(n_boot, n_customers) multinomial resampling counts."""
    rng = np.random.default_rng(seed)
    k = len(customers)
    return rng.multinomial(k, np.full(k, 1.0 / k), size=n_boot).astype(np.float32)


def paired_bootstrap(df: pd.DataFrame, reference=("tabpfn", None), n_boot=N_BOOT, seed=SEED) -> pd.DataFrame:
    """Customer-clustered paired bootstrap of pooled Brier, log loss and count MAE, and differences vs `reference`.

    `df` holds one feature set's rows (plus the Bayesian models); the difference is model minus reference, so
    positive values mean the reference model (TabPFN) is better for loss metrics.
    """
    ks = keys(df)
    base = df[["customer_id", "origin"]].drop_duplicates().sort_values(["customer_id", "origin"])
    cust = np.sort(base.customer_id.unique())
    cidx = pd.Series(np.arange(len(cust)), index=cust)
    W = _customer_weights(cust, n_boot, seed)
    losses = {}
    for m, fs in ks:
        r = df[(df.model == m) & (df.features == fs)]
        ci = cidx[r.customer_id].to_numpy()
        per = {"n": np.bincount(ci, minlength=len(cust)).astype(np.float32)}
        y = r.y_inactive.to_numpy()
        if r.p_inactive.notna().all():
            p = r.p_inactive.to_numpy()
            per["brier"] = np.bincount(ci, (p - y) ** 2, minlength=len(cust))
            pc = np.clip(p, EPS, 1 - EPS)
            per["log_loss"] = np.bincount(ci, -(y * np.log(pc) + (1 - y) * np.log(1 - pc)), minlength=len(cust))
        if r.count_pred.notna().all():
            per["count_mae"] = np.bincount(ci, np.abs(r.count_pred.to_numpy() - r.y_count.to_numpy()),
                                           minlength=len(cust))
        losses[(m, fs)] = per
    rows = []
    ref = [k for k in ks if k[0] == reference[0] and (reference[1] is None or k[1] == reference[1])]
    ref = ref[0] if ref else None
    for k, per in losses.items():
        denom = W @ per["n"]
        for metric in ("brier", "log_loss", "count_mae"):
            if metric not in per:
                continue
            bs = (W @ per[metric]) / denom
            row = {"model": k[0], "features": k[1], "metric": metric, "estimate": float(per[metric].sum() / per["n"].sum()),
                   "ci_low": float(np.quantile(bs, 0.025)), "ci_high": float(np.quantile(bs, 0.975))}
            if ref is not None and k != ref and metric in losses[ref]:
                rb = (W @ losses[ref][metric]) / (W @ losses[ref]["n"])
                diff = bs - rb
                row.update(diff_vs_tabpfn=float(per[metric].sum() / per["n"].sum()
                                                - losses[ref][metric].sum() / losses[ref]["n"].sum()),
                           diff_ci_low=float(np.quantile(diff, 0.025)), diff_ci_high=float(np.quantile(diff, 0.975)),
                           p_tabpfn_better=float(np.mean(diff > 0)))
            rows.append(row)
    return pd.DataFrame(rows)


# --- tables --------------------------------------------------------------------------------


def metric_table(df: pd.DataFrame, by_origin: bool = False) -> pd.DataFrame:
    rows = []
    groups = [("pooled", df)] if not by_origin else [(str(o.date()), g) for o, g in df.groupby("origin")]
    for label, g in groups:
        for (m, fs), r in g.groupby(["model", "features"]):
            row = {"origin": label, "model": m, "features": fs}
            if r.p_inactive.notna().all():
                row.update(classification_metrics(r.y_inactive.to_numpy(), r.p_inactive.to_numpy()))
            if r.count_pred.notna().all():
                row.update({f"count_{k}": v for k, v in count_metrics(r.y_count.to_numpy(), r.count_pred.to_numpy()).items()})
            rows.append(row)
    return pd.DataFrame(rows)


def segment_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, f in SEGMENTS.items():
        g = df[f(df)]
        for (m, fs), r in g.groupby(["model", "features"]):
            if not r.p_inactive.notna().all() or r.y_inactive.nunique() < 2:
                continue
            y, p = r.y_inactive.to_numpy(), r.p_inactive.to_numpy()
            row = {"segment": name, "model": m, "features": fs, "n_rows": len(r),
                   "n_customers": int(r.customer_id.nunique()), "prevalence": float(y.mean()),
                   "brier": float(brier(y, p)), "log_loss": float(log_loss(y, p)), "mean_pred": float(p.mean())}
            if r.count_pred.notna().all():
                row["count_mae"] = float(np.mean(np.abs(r.count_pred - r.y_count)))
            rows.append(row)
    return pd.DataFrame(rows)


def learning_curve_table(d: Path) -> pd.DataFrame:
    lab = pd.read_parquet(d / "labels.parquet")
    rows = []
    for f in sorted((d / "learning_curve").glob("*.parquet")):
        model, fs = f.stem.split("__")
        lc = pd.read_parquet(f).merge(lab, on=["customer_id", "origin"], how="left")
        for (n, s, o), g in lc.groupby(["n_customers", "seed", "origin"]):
            y = g.y_inactive.to_numpy()
            row = {"model": model, "features": fs, "n_customers": n, "seed": s, "origin": str(o.date()),
                   "n_rows": int(g.n_rows.iloc[0]), "seconds": float(g.seconds.iloc[0])}
            if g.p_inactive.notna().all():
                row.update(brier=float(brier(y, g.p_inactive.to_numpy())), log_loss=float(log_loss(y, g.p_inactive.to_numpy())))
            if g.count_pred.notna().all():
                row["count_mae"] = float(np.mean(np.abs(g.count_pred - g.y_count)))
            rows.append(row)
    return pd.DataFrame(rows)


def runtime_table(d: Path) -> pd.DataFrame:
    if not (d / "runtime.jsonl").exists():
        return pd.DataFrame()
    rt = pd.read_json(d / "runtime.jsonl", lines=True)
    rt = rt.drop_duplicates(subset=["model", "features", "origin"], keep="last")
    return rt.groupby(["model", "features"]).agg(seconds_total=("seconds", "sum"), seconds_per_origin=("seconds", "mean"),
                                                  origins=("origin", "nunique")).reset_index()


def write_tables(d: Path) -> dict[str, pd.DataFrame]:
    df = load_predictions(d)
    t = d / "tables"
    t.mkdir(exist_ok=True)
    out = {"pooled": metric_table(df), "by_origin": metric_table(df, by_origin=True), "segments": segment_table(df)}
    boots = []
    bayes = df[df.model.isin(["pareto_nbd", "abe_hb"])]
    for fs in ("common", "expanded"):
        sub = pd.concat([df[(df.features == fs) & ~df.model.isin(["pareto_nbd", "abe_hb"])], bayes])
        if len(sub):
            boots.append(paired_bootstrap(sub, reference=("tabpfn", fs)).assign(comparison=fs))
    out["bootstrap"] = pd.concat(boots, ignore_index=True)
    if (d / "learning_curve").exists():
        out["learning_curve"] = learning_curve_table(d)
    out["runtime"] = runtime_table(d)
    rel = []
    for (m, fs), r in df.groupby(["model", "features"]):
        if r.p_inactive.notna().all() and r.p_inactive.nunique() > 1:
            rel.append(reliability(r.y_inactive.to_numpy(), r.p_inactive.to_numpy()).assign(model=m, features=fs))
    out["reliability"] = pd.concat(rel, ignore_index=True)
    for k, v in out.items():
        v.to_csv(t / f"{k}.csv", index=False)
    return out


def summary_json(d: Path) -> dict:
    return json.loads((d / "manifest.json").read_text())
