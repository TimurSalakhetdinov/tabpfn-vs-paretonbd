"""Pareto/NBD and Abe HB: published values, P(no purchase) vs P(alive), and metric orientation."""

import numpy as np
import pandas as pd
import pytest

from tabpfn_vs_paretonbd import data
from tabpfn_vs_paretonbd.evaluate import brier, classification_metrics, log_loss, paired_bootstrap
from tabpfn_vs_paretonbd.models.abe_hb import AbeHB, _p_alive, split_rhat
from tabpfn_vs_paretonbd.models.pareto_nbd import ParetoNBD


def cdnow_sample_cbs():
    if not (data.RAW / data._FILES["cdnow_sample"]).exists():
        pytest.skip("CDNOW sample not fetched")
    occ = data.load_occasions("cdnow_sample")
    cal = pd.Timestamp("1997-10-01")
    g = occ[occ.date < cal].groupby("customer_id").date
    wk = lambda s: s.dt.days / 7.0  # noqa: E731
    return ((g.size() - 1).to_numpy(float), wk(g.max() - g.min()).to_numpy(),
            wk(cal - pd.Timedelta(days=1) - g.min()).to_numpy())


def test_pareto_nbd_matches_published_cdnow_estimates():
    x, t_x, T = cdnow_sample_cbs()
    m = ParetoNBD().fit(x, t_x, T)
    # Fader, Hardie and Lee (2005): r=0.553, alpha=10.578, s=0.606, beta=11.669, LL=-9594.98
    np.testing.assert_allclose(m.params, [0.553, 10.578, 0.606, 11.669], atol=2e-3)
    assert abs(m.loglik - (-9594.98)) < 0.05


def test_prob_no_purchase_is_not_p_alive():
    m = ParetoNBD(r=0.55, alpha=10.6, s=0.61, beta=11.7)
    x, t_x, T = np.array([0.0, 2, 10]), np.array([0.0, 20, 38]), np.array([39.0, 39, 39])
    pa = m.p_alive(x, t_x, T)
    p0_short, _ = m.prob_no_purchase(0.01, x, t_x, T)
    p0, se = m.prob_no_purchase(13, x, t_x, T, n_draws=20000)
    assert np.all(p0_short > 0.99)  # nothing can happen in an instant
    assert np.all(p0 > 1 - pa)  # dead customers never buy, alive ones may not buy either
    assert np.all(np.abs(p0 - (1 - pa)) > 0.01)  # the two quantities genuinely differ
    assert np.all(np.diff(p0) < 0) and np.all(se < 0.01)  # more history -> more likely to return


def test_abe_p_alive_formula():
    lam, mu = np.array([0.5]), np.array([0.1])
    # recent purchase (t_x = T) means certainly alive
    assert _p_alive(lam, mu, np.array([10.0]), np.array([10.0]))[0] == pytest.approx(1.0)
    pa = _p_alive(lam, mu, np.array([0.0]), np.array([10.0]))[0]
    s = 0.6
    assert pa == pytest.approx(1 / (1 + mu[0] / s * (np.exp(s * 10) - 1)))


@pytest.mark.slow
def test_abe_hb_recovers_simulated_hyperparameters():
    rng = np.random.default_rng(1)
    n, T = 3000, 52.0
    beta, G = np.array([-2.0, -3.5]), np.array([[0.6, 0.1], [0.1, 0.5]])
    eta = rng.multivariate_normal(beta, G, size=n)
    lam, mu = np.exp(eta[:, 0]), np.exp(eta[:, 1])
    tau = rng.exponential(1 / mu)
    x, t_x = np.zeros(n), np.zeros(n)
    for i in range(n):
        t, end = 0.0, min(T, tau[i])
        while True:
            t += rng.exponential(1 / lam[i])
            if t > end:
                break
            x[i] += 1
            t_x[i] = t
    m = AbeHB(n_chains=3, burnin=2000, draws=4000, thin=4, seed=0, time_unit=1.0).fit(x, t_x, np.full(n, T))
    dg = m.diagnostics()
    assert abs(dg["beta_log_lambda"]["median"] - beta[0]) < 0.15
    assert dg["var_log_lambda"]["q025"] < G[0, 0] < dg["var_log_lambda"]["q975"] * 1.1
    assert max(dg[k]["rhat"] for k in ("beta_log_lambda", "var_log_lambda")) < 1.1


def test_split_rhat_detects_non_mixing():
    rng = np.random.default_rng(0)
    good = rng.normal(size=(4, 1000))
    bad = good + np.arange(4)[:, None] * 3
    assert split_rhat(good) < 1.01 and split_rhat(bad) > 1.5


def test_metric_orientation_positive_class_is_inactive():
    y = np.array([1, 1, 0, 0])  # 1 = inactive
    good, bad = np.array([0.9, 0.8, 0.2, 0.1]), np.array([0.1, 0.2, 0.8, 0.9])
    assert brier(y, good) < brier(y, bad) and log_loss(y, good) < log_loss(y, bad)
    m = classification_metrics(y, good)
    assert m["pr_auc"] == 1 and m["roc_auc"] == 1 and m["prevalence"] == 0.5


def test_paired_bootstrap_sign_and_clustering():
    rng = np.random.default_rng(0)
    n = 400
    base = pd.DataFrame({"customer_id": np.tile(np.arange(n).astype(str), 3),
                         "origin": np.repeat(pd.to_datetime(["2020-01-01", "2020-02-01", "2020-03-01"]), n)})
    y = rng.random(len(base)) < 0.5
    rows = []
    for m, noise in [("tabpfn", 0.2), ("lgbm", 0.4)]:
        p = np.clip(np.where(y, 0.7, 0.3) + rng.normal(0, noise, len(base)), 0.01, 0.99)
        rows.append(base.assign(model=m, features="common", p_inactive=p, count_pred=np.nan, y_inactive=y.astype(float),
                                y_count=0.0))
    df = pd.concat(rows, ignore_index=True)
    b = paired_bootstrap(df, reference=("tabpfn", "common"), n_boot=300)
    r = b[(b.model == "lgbm") & (b.metric == "brier")].iloc[0]
    assert r.diff_vs_tabpfn > 0 and r.diff_ci_low > 0 and r.p_tabpfn_better > 0.95


def test_tabpfn_chunked_prediction_matches_single_call():
    from tabpfn_vs_paretonbd.models import tabular as tb

    X = np.arange(25, dtype=float).reshape(-1, 1)
    calls = []

    def predict(a):
        calls.append(len(a))
        return a[:, 0] * 2

    assert np.array_equal(tb._chunked(predict, X, chunk=10), X[:, 0] * 2)
    assert calls == [10, 10, 5]
    calls.clear()
    tb._chunked(predict, X, chunk=100)
    assert calls == [25]
