"""Development-only pilot: which training window (last K matured origins) and feature set to freeze.

Run each model family in its own process (torch and LightGBM ship different libomp copies that
deadlock on macOS in one process):
    UV_NO_EDITABLE=1 uv run python scripts/pilots/context_pilot.py online_retail_ii baselines
    UV_NO_EDITABLE=1 uv run python scripts/pilots/context_pilot.py online_retail_ii tabpfn
Only development origins are scored. Appends to docs/evidence/context-pilot-<dataset>.json.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score

from tabpfn_vs_paretonbd import data
from tabpfn_vs_paretonbd.config import PROTOCOLS, SEED
from tabpfn_vs_paretonbd.features import FEATURE_SETS, build_snapshots

ds, which = sys.argv[1], sys.argv[2]
p = PROTOCOLS[ds]
occ = data.load_occasions(ds)
snaps = build_snapshots(occ, p.origins, p.horizon_days, p.data_end)
sets = {"common": FEATURE_SETS["common"], "expanded": FEATURE_SETS["expanded"],
        "expanded_no_month": [c for c in FEATURE_SETS["expanded"] if c != "origin_month"]}
windows = [1, 3, 6, None]  # None = all matured origins
out = Path("docs/evidence") / f"context-pilot-{ds}.json"
rows = json.loads(out.read_text()) if out.exists() else []
rows = [r for r in rows if r["model"] not in ({"lr", "lgbm"} if which == "baselines" else {"tabpfn"})]

if which == "tabpfn":
    from tabpfn import TabPFNClassifier
    from tabpfn.constants import ModelVersion

    def fit_predict(X, y, Xt):
        m = TabPFNClassifier.create_default_for_version(ModelVersion.V3_5, device="auto", random_state=SEED)
        return m.fit(X.to_numpy(), y.to_numpy()).predict_proba(Xt.to_numpy())[:, 1]
    models = {"tabpfn": fit_predict}
else:
    from lightgbm import LGBMClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import FunctionTransformer, StandardScaler
    slog = FunctionTransformer(lambda a: np.sign(a) * np.log1p(np.abs(a)))

    def lr(X, y, Xt):
        m = make_pipeline(slog, StandardScaler(), LogisticRegression(max_iter=2000))
        return m.fit(X.fillna(-1), y).predict_proba(Xt.fillna(-1))[:, 1]

    def lgbm(X, y, Xt):
        m = LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=40,
                           verbose=-1, random_state=SEED)
        return m.fit(X, y).predict_proba(Xt)[:, 1]
    models = {"lr": lr, "lgbm": lgbm}

rng = np.random.default_rng(SEED)
for eo in map(pd.Timestamp, p.dev_origins):
    tr_all = snaps[snaps.origin.isin(p.train_origins(eo))]
    te = snaps[snaps.origin == eo]
    y = te.y_inactive.astype(int)
    avail = sorted(tr_all.origin.unique())
    for k in windows:
        tr = tr_all[tr_all.origin.isin(avail if k is None else avail[-k:])]
        if which == "tabpfn" and len(tr) > 10_000:  # context cap: whole customers sampled
            cust = tr.customer_id.unique()
            rng.shuffle(cust)
            keep, n = [], 0
            sizes = tr.groupby("customer_id").size()
            for c in cust:
                if n + sizes[c] > 10_000:
                    break
                keep.append(c)
                n += sizes[c]
            tr = tr[tr.customer_id.isin(keep)]
        for fs, cols in sets.items():
            if which == "tabpfn" and (fs == "expanded" or k == 6):
                continue  # time budget: origin_month already rejected on the fast baselines
            for name, f in models.items():
                t0 = time.time()
                q = f(tr[cols], tr.y_inactive.astype(int), te[cols])
                r = {"model": name, "eval_origin": str(eo.date()), "window": k or "all", "features": fs,
                     "n_train": len(tr), "brier": round(brier_score_loss(y, q), 4), "auc": round(roc_auc_score(y, q), 4),
                     "mean_pred": round(float(q.mean()), 3), "prevalence": round(float(y.mean()), 3),
                     "seconds": round(time.time() - t0, 1)}
                rows.append(r)
                print(r, flush=True)
                out.write_text(json.dumps(rows, indent=1) + "\n")
