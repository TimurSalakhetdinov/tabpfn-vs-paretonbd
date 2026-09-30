# /// script
# requires-python = ">=3.11"
# ///
"""Dataset-selection profiling: executed feasibility checks for each candidate.

Run: uv run --with pyreadr python scripts/profile_datasets.py
Writes docs/profiling/profile.json and prints a compact table. pyreadr is only needed to read the
Complete Journey .rds candidate; it is not a project dependency.
"""

import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tabpfn_vs_paretonbd import data  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "docs" / "profiling"
HORIZONS = [30, 60, 90]


def occasion_profile(occ: pd.DataFrame) -> dict:
    per = occ.groupby("customer_id").agg(first=("date", "min"), last=("date", "max"), n=("date", "size"))
    start, end = occ["date"].min(), occ["date"].max()
    days = (end - start).days + 1
    gaps = occ.sort_values(["customer_id", "date"]).groupby("customer_id")["date"].diff().dt.days.dropna()
    return {
        "first_date": str(start.date()), "last_date": str(end.date()), "span_days": days,
        "customers": len(per), "occasions": len(occ),
        "occasions_per_customer_median": float(per["n"].median()),
        "share_customers_repeat": round(float((per["n"] >= 2).mean()), 3),
        "share_first_obs_in_first_60d": round(float((per["first"] < start + pd.Timedelta(days=60)).mean()), 3),
        "interpurchase_days_median": float(gaps.median()) if len(gaps) else None,
        "interpurchase_days_p90": float(gaps.quantile(0.9)) if len(gaps) else None,
    }


def origin_grid(occ: pd.DataFrame, horizon: int, step_days: int = 30, min_history: int = 90) -> list[dict]:
    """For each candidate origin: customers observed before it and share inactive over the next `horizon` days.

    An origin is usable only if origin + horizon <= last observed date (label fully observed).
    """
    start, end = occ["date"].min(), occ["date"].max()
    rows = []
    o = start + pd.Timedelta(days=min_history)
    while o + pd.Timedelta(days=horizon) <= end:
        hist = occ[occ["date"] < o]
        fut = occ[(occ["date"] >= o) & (occ["date"] < o + pd.Timedelta(days=horizon))]
        base = hist["customer_id"].unique()
        ret = np.isin(base, fut["customer_id"].unique())
        rows.append({"origin": str(o.date()), "n_base": len(base), "p_inactive": round(float(1 - ret.mean()), 3)})
        o += pd.Timedelta(days=step_days)
    return rows


def protocol_check(grid: list[dict], horizon: int, n_eval: int = 3) -> dict:
    """Last `n_eval` monthly origins as holdout; count training origins whose label window ends before the first."""
    if len(grid) < n_eval + 1:
        return {"feasible": False, "usable_origins": len(grid)}
    first_eval = pd.Timestamp(grid[-n_eval]["origin"])
    train = [g for g in grid if pd.Timestamp(g["origin"]) + pd.Timedelta(days=horizon) <= first_eval]
    return {"feasible": len(train) >= 3, "usable_origins": len(grid), "eval_origins": [g["origin"] for g in grid[-n_eval:]],
            "mature_train_origins": len(train), "eval_base_customers": [g["n_base"] for g in grid[-n_eval:]]}


def completejourney_occasions() -> pd.DataFrame:
    import pyreadr

    df = list(pyreadr.read_r(str(data.fetch("completejourney"))).values())[0]
    df = df[df["sales_value"] > 0]
    df = df.assign(customer_id=df["household_id"].astype(str), date=df["transaction_timestamp"].dt.normalize())
    return df.groupby(["customer_id", "date"], as_index=False).agg(n_tx=("basket_id", "nunique"), units=("quantity", "sum"),
                                                                    amount=("sales_value", "sum"))


def donations_profile() -> dict:
    z = zipfile.ZipFile(data.fetch("donations_1995"))
    txt = z.read("1995_cohort_binary.txt").decode("utf-16")  # file is UTF-16 with BOM
    m = np.array([[int(v) for v in line.split()] for line in txt.splitlines() if line.strip()])
    m = m[:, -12:]  # 1995..2006; leading columns hold the supporter ID
    years = list(range(1995, 2007))
    rates = {y: round(float(m[:, i].mean()), 3) for i, y in enumerate(years)}
    return {"donors": int(m.shape[0]), "grain": "annual binary incidence", "years": [1995, 2006],
            "annual_donation_rate": rates, "first_col_all_one": bool((m[:, 0] == 1).all())}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    result = {}
    occ_or, log_or = data.online_retail_occasions(return_log=True)
    cm, log_cm = data.cdnow_occasions("master", return_log=True)
    cs, log_cs = data.cdnow_occasions("sample", return_log=True)
    cj = completejourney_occasions()
    for name, occ, log in [("online_retail_ii", occ_or, log_or), ("cdnow_master", cm, log_cm),
                           ("cdnow_sample", cs, log_cs), ("completejourney", cj, {})]:
        prof = {"cleaning_log": log, **occasion_profile(occ), "horizons": {}}
        for h in HORIZONS:
            g = origin_grid(occ, h)
            prof["horizons"][h] = {"grid": g, "protocol": protocol_check(g, h)}
        result[name] = prof
    raw = data.load_online_retail_raw()
    country = raw.dropna(subset=["CustomerID"]).groupby("CustomerID")["Country"].agg(lambda s: s.mode().iat[0])
    result["online_retail_ii"]["share_customers_uk"] = round(float((country == "United Kingdom").mean()), 3)
    per = occ_or.groupby("customer_id").agg(n=("date", "size"), amt=("amount", "sum"), units=("units", "sum"))
    result["online_retail_ii"]["amount_per_occasion_median"] = float(occ_or["amount"].median())
    result["online_retail_ii"]["units_per_occasion_median"] = float(occ_or["units"].median())
    result["online_retail_ii"]["top1pct_customers_share_of_amount"] = round(
        float(per["amt"].nlargest(max(1, len(per) // 100)).sum() / per["amt"].sum()), 3)
    result["donations_1995"] = donations_profile()
    (OUT / "profile.json").write_text(json.dumps(result, indent=1, default=str) + "\n")

    print(f"{'dataset':18} {'span':>5} {'cust':>6} {'occ':>7} {'rep%':>5} {'H':>3} {'orig':>4} {'trainO':>6} {'evalN':>16} {'pInact@eval':>12}")
    for name in ["online_retail_ii", "cdnow_master", "cdnow_sample", "completejourney"]:
        p = result[name]
        for h in HORIZONS:
            pr = p["horizons"][h]["protocol"]
            g = p["horizons"][h]["grid"]
            pin = ",".join(str(x["p_inactive"]) for x in g[-3:])
            print(f"{name:18} {p['span_days']:5} {p['customers']:6} {p['occasions']:7} {p['share_customers_repeat']:5} {h:3} "
                  f"{pr['usable_origins']:4} {pr.get('mature_train_origins', 0):6} {str(pr.get('eval_base_customers')):>16} {pin:>12}")
    print(json.dumps(result["donations_1995"], indent=1))


if __name__ == "__main__":
    main()
