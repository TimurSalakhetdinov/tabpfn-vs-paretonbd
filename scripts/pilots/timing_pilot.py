"""Development-only pilot: snapshot sizes and TabPFN-3.5 runtime vs context size on one dev origin."""
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss

from tabpfn_vs_paretonbd import data
from tabpfn_vs_paretonbd.config import PROTOCOLS, SEED
from tabpfn_vs_paretonbd.features import FEATURE_SETS, build_snapshots

ds = sys.argv[1] if len(sys.argv) > 1 else "online_retail_ii"
p = PROTOCOLS[ds]
occ = data.load_occasions(ds)
t0 = time.time(); snaps = build_snapshots(occ, p.origins, p.horizon_days, p.data_end); t_snap = time.time() - t0
eo = pd.Timestamp(p.dev_origins[-1])
tr = snaps[snaps.origin.isin(p.train_origins(eo))]; te = snaps[snaps.origin == eo]
out = {"dataset": ds, "snapshot_build_s": round(t_snap, 1), "rows_per_origin": snaps.groupby("origin").size().astype(int).to_dict(),
       "dev_eval_origin": str(eo.date()), "n_train_rows": len(tr), "n_train_customers": int(tr.customer_id.nunique()), "n_test": len(te), "runs": []}
from tabpfn import TabPFNClassifier
from tabpfn.constants import ModelVersion

cols = FEATURE_SETS["expanded"]
rng = np.random.default_rng(SEED)
for n in [1000, 3000, 10000]:
    idx = rng.choice(len(tr), size=min(n, len(tr)), replace=False)
    m = TabPFNClassifier.create_default_for_version(ModelVersion.V3_5, device="auto", random_state=SEED)
    t0 = time.time(); m.fit(tr[cols].iloc[idx].to_numpy(), tr.y_inactive.iloc[idx].to_numpy().astype(int)); fs = time.time() - t0
    t0 = time.time(); pr = m.predict_proba(te[cols].to_numpy())[:, 1]; ps = time.time() - t0
    out["runs"].append({"context": len(idx), "fit_s": round(fs, 1), "predict_s": round(ps, 1),
                        "dev_brier": round(brier_score_loss(te.y_inactive, pr), 4), "dev_logloss": round(log_loss(te.y_inactive, pr), 4),
                        "maxrss_gb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9, 2)})
    print(out["runs"][-1], flush=True)
out["rows_per_origin"] = {str(k.date()): v for k, v in out["rows_per_origin"].items()}
Path("docs/evidence").mkdir(parents=True, exist_ok=True)
Path(f"docs/evidence/timing-pilot-{ds}.json").write_text(json.dumps(out, indent=1) + "\n")
print(json.dumps({k: v for k, v in out.items() if k != "runs"}, indent=1))
