"""Check that TabPFN-3.5 runs locally: fit tiny classifier + regressor, record checkpoint identity.

Writes docs/evidence/tabpfn35-verification.json. Needs the TabPFN-3.5 weights (see README).
"""
import hashlib
import importlib.metadata as md
import json
import platform
import time
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

load_dotenv()
import torch  # noqa: E402
from tabpfn import TabPFNClassifier, TabPFNRegressor  # noqa: E402
from tabpfn.constants import ModelVersion  # noqa: E402
from tabpfn.model_loading import get_cache_dir  # noqa: E402

rng = np.random.default_rng(0)
X = rng.normal(size=(300, 4))
y_cls = (X[:, 0] + rng.normal(scale=0.5, size=300) > 0).astype(int)
y_reg = X[:, 0] * 2 + np.sin(X[:, 1]) + rng.normal(scale=0.1, size=300)
out = {"tabpfn_version": md.version("tabpfn"), "torch": torch.__version__, "platform": platform.platform(),
       "model_version_enum": str(ModelVersion.V3_5)}
for name, cls, y in [("classifier", TabPFNClassifier, y_cls), ("regressor", TabPFNRegressor, y_reg)]:
    t0 = time.time()
    m = cls.create_default_for_version(ModelVersion.V3_5, device="auto", random_state=0)
    m.fit(X[:200], y[:200])
    fit_s = time.time() - t0
    t0 = time.time()
    if name == "classifier":
        p = m.predict_proba(X[200:])[:, 1]
        score = {"toy_accuracy": float(np.mean((p > 0.5) == y[200:]))}
    else:
        score = {"toy_mae": float(np.mean(np.abs(m.predict(X[200:]) - y[200:])))}
    out[name] = {"model_path": "<tabpfn cache>/" + Path(str(getattr(m, "model_path", ""))).name, "fit_s": round(fit_s, 2),
                 "predict_s": round(time.time() - t0, 2), **score}
out["cache_checkpoints"] = [
    {"file": p.name, "bytes": p.stat().st_size, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
    for p in sorted(Path(get_cache_dir()).rglob("*.safetensors"))]
print(json.dumps(out, indent=2))
Path("docs/evidence/tabpfn35-verification.json").write_text(json.dumps(out, indent=2) + "\n")
