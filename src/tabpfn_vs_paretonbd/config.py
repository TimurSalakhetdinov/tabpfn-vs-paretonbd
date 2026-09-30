"""Frozen evaluation protocol. Fixed after dataset profiling, before any model was scored on holdout origins.

Positive class for classification: INACTIVE = no purchase occasion in [origin, origin + horizon).
Displayed as "90-day inactivity risk"; its complement is "probability of returning within 90 days".
Secondary target: number of purchase occasions in the same window (not revenue, not lifetime value).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

SEED = 20260929
HORIZON_DAYS = 90


def monthly(start: str, end: str) -> list[pd.Timestamp]:
    return list(pd.date_range(start, end, freq="MS"))


@dataclass(frozen=True)
class Protocol:
    dataset: str
    first_origin: str          # earliest snapshot origin (needs some history before it)
    last_origin: str           # latest origin whose label window is fully observed
    data_end: str              # last calendar day with observed data (inclusive)
    dev_origins: tuple[str, ...]      # development evaluation origins (tuning, pilots)
    holdout_origins: tuple[str, ...]  # untouched final evaluation origins
    horizon_days: int = HORIZON_DAYS
    notes: dict = field(default_factory=dict)

    @property
    def origins(self) -> list[pd.Timestamp]:
        return monthly(self.first_origin, self.last_origin)

    def train_origins(self, eval_origin: pd.Timestamp | str) -> list[pd.Timestamp]:
        """Snapshot origins whose label window [t, t + H) has fully ended by eval_origin."""
        e = pd.Timestamp(eval_origin)
        return [t for t in self.origins if t + pd.Timedelta(days=self.horizon_days) <= e]


PROTOCOLS: dict[str, Protocol] = {
    # Online Retail II: data 2009-12-01 .. 2011-12-09. Label of 2011-09-01 ends 2011-11-29 (inclusive).
    "online_retail_ii": Protocol(
        "online_retail_ii", "2010-03-01", "2011-09-01", "2011-12-09",
        dev_origins=("2011-01-01", "2011-02-01", "2011-03-01"),
        holdout_origins=("2011-07-01", "2011-08-01", "2011-09-01"),
        notes={"role": "primary"},
    ),
    # CDNOW full cohort: first purchases Q1 1997, data to 1998-06-30. Origins start after acquisition closes.
    "cdnow_master": Protocol(
        "cdnow_master", "1997-04-01", "1998-04-01", "1998-06-30",
        dev_origins=("1997-09-01", "1997-10-01", "1997-11-01"),
        holdout_origins=("1998-02-01", "1998-03-01", "1998-04-01"),
        notes={"role": "bayesian bridge / fallback"},
    ),
}

# --- frozen after development pilots (docs/dataset-selection.md), before any holdout scoring -------------
# Supervised training window: snapshots from the most recent matured origin only. On all three development
# origins this beat 3, 6 and all matured origins for logistic regression, LightGBM and TabPFN
# (docs/evidence/context-pilot-online_retail_ii.json): the inactivity level drifts with season and base ageing.
TRAIN_WINDOW_ORIGINS = 1
# TabPFN context cap: largest training table passed to TabPFN (customers sampled as whole groups).
TABPFN_MAX_CONTEXT = 10_000
# LightGBM: chosen from a 6-config grid on development origins (docs/evidence/lgbm-grid-online_retail_ii.json).
LGBM_PARAMS = {"n_estimators": 150, "learning_rate": 0.03, "num_leaves": 7, "min_child_samples": 40}
# Abe (2009) HB M1 sampler settings, validated on the CDNOW sample (docs/evidence/bayes-validation-cdnow_sample.json).
# Convergence gate (fixed before holdout): Abe HB predictions enter the results only if every hyperparameter's
# split-R-hat is below RHAT_MAX at every origin; otherwise they are saved to unconverged/ and reported as a failure.
RHAT_MAX = 1.1
# Frozen 2026-09-30: on Online Retail II dev even 4x longer chains gave max R-hat 1.18, so Abe HB is expected to
# fail the gate there and is reported as non-converged (docs/dataset-selection.md, section 6).
ABE_SETTINGS = {"n_chains": 4, "burnin": 5000, "draws": 20000, "thin": 20, "time_unit": 7.0}
# Learning curve: number of distinct training customers (all their matured snapshots are kept).
LEARNING_CURVE_CUSTOMERS = (50, 100, 250, 500, 1000, 2000)
LEARNING_CURVE_SEEDS = (0, 1, 2)
# Learning curves run on the primary dataset only: on CDNOW each TabPFN fit predicts 23,570 customers, and the
# 108-fit curve would take ~6 h on the development machine (estimated from docs/evidence/timing-pilot-*.json).
LEARNING_CURVE_DATASETS = ("online_retail_ii",)
