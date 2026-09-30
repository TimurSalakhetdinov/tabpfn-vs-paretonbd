"""Development-only LightGBM grid (fixed budget: 6 configurations), frozen window policy, both feature sets.

Selection rule (declared before running): lowest mean development Brier across the three development origins and
both feature sets. The same parameters are used for the Poisson count regressor.
Writes docs/evidence/lgbm-grid-<dataset>.json.
"""

import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from tabpfn_vs_paretonbd.evaluate import brier
from tabpfn_vs_paretonbd.features import FEATURE_SETS
from tabpfn_vs_paretonbd.models.tabular import lgbm
from tabpfn_vs_paretonbd.pipeline import eval_origins, load_snapshots, split_at

ds = sys.argv[1]
snaps = load_snapshots(ds)
grid = [{"n_estimators": n, "learning_rate": 0.03, "num_leaves": L, "min_child_samples": 40}
        for n, L in itertools.product([150, 300], [7, 15, 31])]
rows = []
for i, prm in enumerate(grid):
    for o in eval_origins(ds, "dev"):
        train, test, _ = split_at(snaps, ds, o)
        for fs, cols in FEATURE_SETS.items():
            p, c = lgbm(train, test, cols, params=prm)
            rows.append({"config": i, **prm, "origin": str(o.date()), "features": fs,
                         "brier": brier(test.y_inactive.to_numpy(), p),
                         "count_mae": float(np.mean(np.abs(c - test.y_count.to_numpy())))})
df = pd.DataFrame(rows)
summ = df.groupby("config")[["brier", "count_mae"]].mean()
best = int(summ.brier.idxmin())
print(summ.round(4))
print("selected", grid[best])
Path(f"docs/evidence/lgbm-grid-{ds}.json").write_text(json.dumps({"grid": grid, "rows": rows, "selected": grid[best],
                                                                    "rule": "min mean dev Brier"}, indent=1) + "\n")
