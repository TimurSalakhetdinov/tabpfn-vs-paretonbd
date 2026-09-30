"""Customer snapshots at forecast origins: features from history before the origin, labels after it.

Conventions (identical for every model and dataset):
- time unit: days; an origin t is a midnight timestamp; history is occasions with date < t;
- the label window is [t, t + H), so it holds H calendar days;
- a customer enters the snapshot at t only if they had at least one occasion before t;
- x (frequency) counts REPEAT occasions: occasions before t minus the first one;
- t_x (recency) is days from first to last occasion; T (customer age) is days from first occasion to t.
  These are exactly the Pareto/NBD sufficient statistics, measured from the FIRST OBSERVED occasion.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

COMMON_FEATURES = ["x", "t_x", "T"]
EXPANDED_FEATURES = COMMON_FEATURES + [
    "days_since_last", "n_30", "n_90", "n_180", "n_365", "trend_90",
    "gap_mean", "gap_std", "gap_last",
    "amount_total", "amount_mean", "amount_last", "units_mean", "tx_per_occasion",
    "first_seen_at_data_start",
]
# origin_month is computed for audit but is not a predictor: in development pilots it let tree models and
# logistic regression tie the level of risk to a single same-month origin a year earlier (see
# docs/dataset-selection.md, "Development pilots"). Removed before any holdout scoring.
AUDIT_ONLY = ["origin_month"]
FEATURE_SETS = {"common": COMMON_FEATURES, "expanded": EXPANDED_FEATURES}
ID_COLS = ["customer_id", "origin"]
TARGETS = ["y_inactive", "y_count"]


def snapshot(occ: pd.DataFrame, origin: pd.Timestamp, horizon_days: int, data_start: pd.Timestamp,
             data_end: pd.Timestamp | None = None) -> pd.DataFrame:
    """One row per customer with history before `origin`. Labels are NaN if the window is not fully observed."""
    origin = pd.Timestamp(origin)
    hist = occ[occ["date"] < origin].sort_values(["customer_id", "date"])
    if hist.empty:
        return pd.DataFrame()
    g = hist.groupby("customer_id", sort=True)
    first, last = g["date"].min(), g["date"].max()
    n = g.size()
    days = lambda s: s.dt.days.astype(float)  # noqa: E731
    f = pd.DataFrame({"x": (n - 1).astype(float), "t_x": days(last - first), "T": days(origin - first)})
    f["days_since_last"] = f["T"] - f["t_x"]
    age = days(origin - hist["date"])
    for w in (30, 90, 180, 365):  # occasions in the last w days before the origin
        f[f"n_{w}"] = hist.assign(a=age.values <= w).groupby("customer_id")["a"].sum().astype(float)
    prev = hist.assign(a=(age.values > 90) & (age.values <= 180)).groupby("customer_id")["a"].sum()
    f["trend_90"] = f["n_90"] - prev.astype(float)
    gaps = hist.assign(gap=hist.groupby("customer_id")["date"].diff().dt.days)
    gg = gaps.groupby("customer_id")["gap"]
    f["gap_mean"], f["gap_std"], f["gap_last"] = gg.mean(), gg.std(), gg.last()
    f["amount_total"] = g["amount"].sum()
    f["amount_mean"] = g["amount"].mean()
    f["amount_last"] = g["amount"].last()
    f["units_mean"] = g["units"].mean()
    f["tx_per_occasion"] = g["n_tx"].mean()
    f["first_seen_at_data_start"] = ((first - data_start).dt.days < 60).astype(float)
    f["origin_month"] = float(origin.month)
    f = f.reset_index()
    f.insert(1, "origin", origin)

    end = origin + pd.Timedelta(days=horizon_days)
    mature = data_end is None or end - pd.Timedelta(days=1) <= pd.Timestamp(data_end)
    fut = occ[(occ["date"] >= origin) & (occ["date"] < end)]
    cnt = fut.groupby("customer_id").size()
    f["y_count"] = f["customer_id"].map(cnt).fillna(0).astype(float)
    f["y_inactive"] = (f["y_count"] == 0).astype(float)
    if not mature:
        f[["y_count", "y_inactive"]] = np.nan
    f["label_end"] = end
    return f


def build_snapshots(occ: pd.DataFrame, origins: list[pd.Timestamp], horizon_days: int,
                    data_end: str | pd.Timestamp) -> pd.DataFrame:
    start = occ["date"].min()
    parts = [snapshot(occ, o, horizon_days, start, pd.Timestamp(data_end)) for o in origins]
    return pd.concat([p for p in parts if len(p)], ignore_index=True)
