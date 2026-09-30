"""Supervised models on customer snapshots: references, logistic/Poisson GLMs, LightGBM and TabPFN-3.5.

Every model exposes `fit_predict(train, test, cols) -> (p_inactive, count_pred)`; either output may be None
when the model does not provide it. Count predictions are clipped at zero for every model (only TabPFN's
regressor can go negative). No recalibration is applied to any model.

LightGBM and TabPFN must not run in the same process on macOS: LightGBM (via scikit-learn's bundled libomp) and
torch ship different OpenMP runtimes that deadlock or crash when both are initialised. The pipeline therefore
runs each model family in its own subprocess; this module imports those libraries lazily.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tabpfn_vs_paretonbd.config import LGBM_PARAMS, SEED, TABPFN_MAX_CONTEXT

FAMILY = {"base_rate": "references", "recency": "references", "glm": "sklearn", "lgbm": "sklearn",
          "tabpfn": "tabpfn"}


def _signed_log(a):
    return np.sign(a) * np.log1p(np.abs(a))


def _glm_prep():
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import FunctionTransformer, StandardScaler

    return make_pipeline(FunctionTransformer(_signed_log), SimpleImputer(strategy="median", add_indicator=True),
                         StandardScaler())


def base_rate(train, test, cols):
    """Training-window prevalence and mean count, the same for every test customer."""
    n = len(test)
    return np.full(n, train.y_inactive.mean()), np.full(n, train.y_count.mean())


def recency(train, test, cols):
    """Monotone (isotonic) map from days since last occasion to inactivity rate and mean count, fitted on training."""
    from sklearn.isotonic import IsotonicRegression

    d, dt = train.days_since_last.to_numpy(), test.days_since_last.to_numpy()
    p = IsotonicRegression(increasing=True, out_of_bounds="clip").fit(d, train.y_inactive).predict(dt)
    c = IsotonicRegression(increasing=False, out_of_bounds="clip").fit(d, train.y_count).predict(dt)
    return p, c


def glm(train, test, cols):
    """Logistic regression for inactivity; Poisson regression for counts (same preprocessing)."""
    from sklearn.linear_model import LogisticRegression, PoissonRegressor
    from sklearn.pipeline import make_pipeline

    X, Xt = train[cols], test[cols]
    clf = make_pipeline(_glm_prep(), LogisticRegression(max_iter=5000)).fit(X, train.y_inactive.astype(int))
    reg = make_pipeline(_glm_prep(), PoissonRegressor(alpha=1e-4, max_iter=5000)).fit(X, train.y_count)
    return clf.predict_proba(Xt)[:, 1], reg.predict(Xt)


def lgbm(train, test, cols, params: dict | None = None):
    """LightGBM binary classifier and Poisson count regressor with the frozen small-grid parameters."""
    from lightgbm import LGBMClassifier, LGBMRegressor

    prm = dict(LGBM_PARAMS if params is None else params)
    X, Xt = train[cols], test[cols]
    clf = LGBMClassifier(**prm, verbose=-1, random_state=SEED).fit(X, train.y_inactive.astype(int))
    reg = LGBMRegressor(**prm, objective="poisson", verbose=-1, random_state=SEED).fit(X, train.y_count)
    return clf.predict_proba(Xt)[:, 1], reg.predict(Xt)


def cap_context(train: pd.DataFrame, max_rows: int = TABPFN_MAX_CONTEXT, seed: int = SEED) -> pd.DataFrame:
    """Subsample whole customers until the training table fits the TabPFN context cap."""
    if len(train) <= max_rows:
        return train
    rng = np.random.default_rng(seed)
    sizes = train.groupby("customer_id").size()
    order = rng.permutation(sizes.index.to_numpy())
    keep = order[np.cumsum(sizes.loc[order].to_numpy()) <= max_rows]
    return train[train.customer_id.isin(keep)]


# Test rows attend only to the training context, so predicting in chunks gives the same values; it only bounds
# peak MPS memory (CDNOW scores 23,570 customers per origin). Smaller test sets are predicted in one call.
TABPFN_TEST_CHUNK = 8_000


def _chunked(predict, Xt, chunk: int | None = None):
    chunk = chunk or TABPFN_TEST_CHUNK
    if len(Xt) <= chunk:
        return predict(Xt)
    return np.concatenate([predict(Xt[i:i + chunk]) for i in range(0, len(Xt), chunk)])


def tabpfn(train, test, cols, seed: int = SEED, tasks=("clf", "reg")):
    """TabPFN-3.5 (local weights, ModelVersion.V3_5) classifier for inactivity and regressor for counts."""
    from tabpfn import TabPFNClassifier, TabPFNRegressor
    from tabpfn.constants import ModelVersion

    train = cap_context(train, seed=seed)
    X, Xt = train[cols].to_numpy(float), test[cols].to_numpy(float)
    p = c = None
    if "clf" in tasks:
        m = TabPFNClassifier.create_default_for_version(ModelVersion.V3_5, device="auto", random_state=seed)
        m.fit(X, train.y_inactive.astype(int).to_numpy())
        p = _chunked(lambda a: m.predict_proba(a)[:, 1], Xt)
    if "reg" in tasks:
        m = TabPFNRegressor.create_default_for_version(ModelVersion.V3_5, device="auto", random_state=seed)
        m.fit(X, train.y_count.to_numpy(float))
        c = _chunked(m.predict, Xt)
    return p, c


MODELS = {"base_rate": base_rate, "recency": recency, "glm": glm, "lgbm": lgbm, "tabpfn": tabpfn}


def fit_predict(name, train, test, cols, **kw):
    p, c = MODELS[name](train, test, cols, **kw)
    if c is not None:
        c = np.clip(np.asarray(c, float), 0, None)
    return p, c
