"""Validate Pareto/NBD (MLE) and Abe (2009) HB M1 on the CDNOW 1/10 sample (2,357 customers).

Calibration: first purchase 1997-01-01..03-31, calibration ends 1997-09-30 (39 weeks); holdout 39 weeks to 1998-06-30.
Published references:
- Pareto/NBD MLE (Fader, Hardie and Lee 2005): r=0.553, alpha=10.578, s=0.606, beta=11.669 (weeks), LL=-9594.98.
- Abe (2009) Table 3, M1 posterior medians (weeks): log-lambda mean -3.58, log-mu mean -3.59,
  var log-lambda 1.40, var log-mu 1.87, cov 0.06 (corr 0.04). Actual holdout repeat occasions: 1788.
Writes docs/evidence/bayes-validation-cdnow_sample.json.
"""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from tabpfn_vs_paretonbd import data
from tabpfn_vs_paretonbd.models.abe_hb import AbeHB
from tabpfn_vs_paretonbd.models.pareto_nbd import ParetoNBD

occ = data.load_occasions("cdnow_sample")
cal_end = pd.Timestamp("1997-10-01")  # history strictly before this midnight = through 1997-09-30
g = occ[occ.date < cal_end].groupby("customer_id").date
first, last, n = g.min(), g.max(), g.size()
wk = lambda s: s.dt.days / 7.0  # noqa: E731
x, t_x, T = (n - 1).to_numpy(float), wk(last - first).to_numpy(), wk(cal_end - pd.Timedelta(days=1) - first).to_numpy()
hold = occ[(occ.date >= cal_end) & (occ.date <= "1998-06-30")].groupby("customer_id").size()
y = hold.reindex(n.index, fill_value=0).to_numpy(float)
h = 39.0
res = {"n_customers": len(x), "actual_holdout_occasions": int(y.sum()), "time_unit": "weeks"}

p = ParetoNBD().fit(x, t_x, T)
ec = p.expected_count(h, x, t_x, T)
res["pareto_nbd"] = {"params": dict(zip(["r", "alpha", "s", "beta"], np.round(p.params, 3).tolist())),
                     "loglik": round(p.loglik, 2), "pred_holdout_occasions": round(float(ec.sum()), 1),
                     "corr_individual": round(float(np.corrcoef(ec, y)[0, 1]), 3)}
print(res["pareto_nbd"], flush=True)

t0 = time.time()
m = AbeHB(n_chains=4, burnin=5000, draws=20000, thin=20, seed=20260929, time_unit=1.0).fit(x, t_x, T)
pr = m.predict(h, t_x, T)
res["abe_hb_m1"] = {"diagnostics": m.diagnostics(), "seconds": round(time.time() - t0, 1),
                    "pred_holdout_occasions": round(float(pr["expected_count"].sum()), 1),
                    "corr_individual": round(float(np.corrcoef(pr["expected_count"], y)[0, 1]), 3),
                    "mean_p_alive": round(float(pr["p_alive"].mean()), 3),
                    "p_no_purchase_vs_actual_share_zero": [round(float(pr["p_no_purchase"].mean()), 3),
                                                           round(float((y == 0).mean()), 3)]}
res["abe_published_m1"] = {"beta_log_lambda": -3.58, "beta_log_mu": -3.59, "var_log_lambda": 1.40,
                           "var_log_mu": 1.87, "cov_log_lambda_mu": 0.06, "corr": 0.04}
print(json.dumps(res["abe_hb_m1"], indent=1))
Path("docs/evidence/bayes-validation-cdnow_sample.json").write_text(json.dumps(res, indent=1) + "\n")
