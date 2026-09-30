"""Run models at development or holdout origins and save predictions (one parquet per model and feature set).

Layout: results/<dataset>/<split>/
    labels.parquet                       customer_id, origin, y_inactive, y_count, segment columns
    preds/<model>__<features>.parquet    customer_id, origin, p_inactive, count_pred
    runtime.jsonl                        one line per (model, features, origin) with seconds and sizes
    manifest.json                        protocol, origins, training windows, row counts, excluded immature labels
    learning_curve/<model>__<features>.parquet   n_customers, seed, origin, customer_id, p_inactive, count_pred
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from tabpfn_vs_paretonbd import data
from tabpfn_vs_paretonbd.config import (
    ABE_SETTINGS,
    LEARNING_CURVE_CUSTOMERS,
    LEARNING_CURVE_SEEDS,
    PROTOCOLS,
    RHAT_MAX,
    SEED,
    TRAIN_WINDOW_ORIGINS,
)
from tabpfn_vs_paretonbd.features import FEATURE_SETS, build_snapshots

RESULTS = data.ROOT / "results"
SUPERVISED = ["base_rate", "recency", "glm", "lgbm", "tabpfn"]
BAYES = ["pareto_nbd", "abe_hb"]
FAMILIES = {"references": ["base_rate", "recency"], "sklearn": ["glm", "lgbm"], "tabpfn": ["tabpfn"],
            "pareto_nbd": ["pareto_nbd"], "abe_hb": ["abe_hb"]}


def out_dir(dataset: str, split: str) -> Path:
    d = RESULTS / dataset / split
    (d / "preds").mkdir(parents=True, exist_ok=True)
    return d


def eval_origins(dataset: str, split: str) -> list[pd.Timestamp]:
    p = PROTOCOLS[dataset]
    return [pd.Timestamp(o) for o in (p.dev_origins if split == "dev" else p.holdout_origins)]


def load_snapshots(dataset: str) -> pd.DataFrame:
    cache = data.INTERIM / f"snapshots_{dataset}.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    p = PROTOCOLS[dataset]
    occ = data.load_occasions(dataset)
    s = build_snapshots(occ, p.origins, p.horizon_days, p.data_end)
    data.INTERIM.mkdir(parents=True, exist_ok=True)
    s.to_parquet(cache)
    return s


def split_at(snaps: pd.DataFrame, dataset: str, origin: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Training rows: the most recent TRAIN_WINDOW_ORIGINS origins whose label windows ended by `origin`."""
    p = PROTOCOLS[dataset]
    mature = p.train_origins(origin)
    use = mature[-TRAIN_WINDOW_ORIGINS:] if TRAIN_WINDOW_ORIGINS else mature
    train = snaps[snaps.origin.isin(use)]
    test = snaps[snaps.origin == origin]
    # guard rails: every training label window ends on or before the forecast origin and is observed
    assert (train.label_end <= origin).all() and train.y_inactive.notna().all()
    assert test.y_inactive.notna().all(), "evaluation labels must be mature"
    immature = snaps[(snaps.origin < origin) & ~snaps.origin.isin(mature)]
    info = {"origin": str(origin.date()), "train_origins": [str(t.date()) for t in use], "n_train": len(train),
            "n_train_customers": int(train.customer_id.nunique()), "n_test": len(test),
            "excluded_immature_rows_before_origin": len(immature)}
    return train, test, info


def _log_runtime(d: Path, rec: dict) -> None:
    with open(d / "runtime.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\n")


def write_labels_and_manifest(dataset: str, split: str) -> None:
    d = out_dir(dataset, split)
    snaps = load_snapshots(dataset)
    infos, labs = [], []
    for o in eval_origins(dataset, split):
        _, test, info = split_at(snaps, dataset, o)
        infos.append(info)
        labs.append(test[["customer_id", "origin", "y_inactive", "y_count", "x", "days_since_last", "T"]])
    lab = pd.concat(labs, ignore_index=True)
    lab.to_parquet(d / "labels.parquet")
    p = PROTOCOLS[dataset]
    manifest = {"dataset": dataset, "split": split, "horizon_days": p.horizon_days, "data_end": p.data_end,
                "train_window_origins": TRAIN_WINDOW_ORIGINS, "origins": infos, "seed": SEED,
                "raw_sha256": {k: data.SOURCES[k].sha256 for k in data.SOURCES if k.startswith(dataset.split("_")[0])},
                "positive_class": "inactive: no purchase occasion in [origin, origin + horizon)"}
    (d / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")


def run_supervised(dataset: str, split: str, models: list[str], feature_sets=("common", "expanded")) -> None:
    from tabpfn_vs_paretonbd.models.tabular import fit_predict

    d = out_dir(dataset, split)
    snaps = load_snapshots(dataset)
    for m in models:
        for fs in feature_sets:
            cols = FEATURE_SETS[fs]
            parts = []
            for o in eval_origins(dataset, split):
                train, test, info = split_at(snaps, dataset, o)
                t0 = time.time()
                p, c = fit_predict(m, train, test, cols)
                sec = time.time() - t0
                parts.append(pd.DataFrame({"customer_id": test.customer_id.to_numpy(), "origin": o,
                                           "p_inactive": p, "count_pred": c}))
                _log_runtime(d, {"model": m, "features": fs, **info, "seconds": round(sec, 2),
                                 "host": platform.machine()})
                print(f"{dataset}/{split} {m}/{fs} {info['origin']}: {sec:.1f}s", flush=True)
            pd.concat(parts, ignore_index=True).to_parquet(d / "preds" / f"{m}__{fs}.parquet")


def run_bayes(dataset: str, split: str, model: str) -> None:
    """Pareto/NBD (MLE) or Abe HB M1, fitted at each origin on the unlabelled history of every customer in the base.

    These models use no labelled outcomes; they see the same x, t_x, T as the common feature set, for all
    customers observed before the origin (the evaluation population itself).
    """
    from tabpfn_vs_paretonbd.models.abe_hb import AbeHB
    from tabpfn_vs_paretonbd.models.pareto_nbd import ParetoNBD

    d = out_dir(dataset, split)
    snaps = load_snapshots(dataset)
    h = PROTOCOLS[dataset].horizon_days
    parts, diags = [], []
    for o in eval_origins(dataset, split):
        test = snaps[snaps.origin == o]
        x, t_x, T = (test[c].to_numpy(float) for c in ("x", "t_x", "T"))
        t0 = time.time()
        if model == "pareto_nbd":
            m = ParetoNBD().fit(x, t_x, T)
            p, se = m.prob_no_purchase(h, x, t_x, T, seed=SEED)
            c = m.expected_count(h, x, t_x, T)
            diag = {"params": dict(zip(["r", "alpha", "s", "beta"], map(float, m.params))), "converged": m.converged, "at_bound": m.at_bound,
                    "loglik": m.loglik, "max_mc_se": float(se.max())}
        else:
            m = AbeHB(seed=SEED, **ABE_SETTINGS).fit(x, t_x, T)
            pr = m.predict(h, t_x, T)
            p, c = pr["p_no_purchase"], pr["expected_count"]
            diag = m.diagnostics()
            rhats = {k: v["rhat"] for k, v in diag.items() if isinstance(v, dict) and "rhat" in v}
            diag["max_rhat"] = max(rhats.values())
            diag["converged"] = bool(diag["max_rhat"] < RHAT_MAX)
            m.eta_draws = None  # free memory
        sec = time.time() - t0
        parts.append(pd.DataFrame({"customer_id": test.customer_id.to_numpy(), "origin": o, "p_inactive": p,
                                   "count_pred": c}))
        diags.append({"origin": str(o.date()), "seconds": round(sec, 1), **diag})
        _log_runtime(d, {"model": model, "features": "common", "origin": str(o.date()), "n_fit": len(test),
                         "seconds": round(sec, 2), "host": platform.machine()})
        print(f"{dataset}/{split} {model} {o.date()}: {sec:.1f}s", flush=True)
    # an unconverged MCMC fit at any origin is kept for audit but never enters the results tables
    ok = all(dg.get("converged", True) for dg in diags) if model == "abe_hb" else True
    target = d / "preds" if ok else d / "unconverged"
    target.mkdir(exist_ok=True)
    (d / "unconverged" if ok else d / "preds").joinpath(f"{model}__common.parquet").unlink(missing_ok=True)
    pd.concat(parts, ignore_index=True).to_parquet(target / f"{model}__common.parquet")
    if not ok:
        print(f"{dataset}/{split} {model}: NOT CONVERGED (max R-hat >= {RHAT_MAX}); saved to unconverged/", flush=True)
    (d / f"diagnostics_{model}.json").write_text(json.dumps(diags, indent=1, default=float) + "\n")


def run_learning_curve(dataset: str, split: str, models: list[str], feature_sets=("common", "expanded"),
                       sizes=LEARNING_CURVE_CUSTOMERS, seeds=LEARNING_CURVE_SEEDS, skip_existing: bool = False) -> None:
    """Training-customer learning curve: sample whole customers (all their training rows), fixed seeds.

    With skip_existing, a model/feature-set file that is already saved is kept (resume after an interruption;
    sampling is seeded, so a resumed curve equals an uninterrupted one)."""
    from tabpfn_vs_paretonbd.models.tabular import fit_predict

    d = out_dir(dataset, split) / "learning_curve"
    d.mkdir(exist_ok=True)
    snaps = load_snapshots(dataset)
    for m in models:
        for fs in feature_sets:
            if skip_existing and (d / f"{m}__{fs}.parquet").exists():
                print(f"LC {m}/{fs}: already saved, skipped", flush=True)
                continue
            cols, parts = FEATURE_SETS[fs], []
            for o in eval_origins(dataset, split):
                train, test, _ = split_at(snaps, dataset, o)
                cust = np.sort(train.customer_id.unique())
                for n in sizes:
                    if n > len(cust):
                        continue
                    for s in seeds:
                        pick = np.random.default_rng([SEED, s, n]).choice(cust, size=n, replace=False)
                        tr = train[train.customer_id.isin(pick)]
                        if tr.y_inactive.nunique() < 2:
                            continue
                        t0 = time.time()
                        kw = {"seed": SEED + s} if m == "tabpfn" else {}
                        p, c = fit_predict(m, tr, test, cols, **kw)
                        parts.append(pd.DataFrame({"n_customers": n, "n_rows": len(tr), "seed": s, "origin": o,
                                                   "customer_id": test.customer_id.to_numpy(), "p_inactive": p,
                                                   "count_pred": c, "seconds": time.time() - t0}))
                        print(f"LC {m}/{fs} {o.date()} n={n} s={s}: {time.time() - t0:.1f}s", flush=True)
            pd.concat(parts, ignore_index=True).to_parquet(d / f"{m}__{fs}.parquet")


def run_family_subprocess(dataset: str, split: str, family: str, learning_curve: bool = False) -> None:
    """Run one model family in a fresh interpreter (see models/tabular.py for why)."""
    cmd = [sys.executable, "-m", "tabpfn_vs_paretonbd.cli", "run-family", dataset, split, family]
    if learning_curve:
        cmd.append("--learning-curve")
    subprocess.run(cmd, check=True, env={**os.environ})
