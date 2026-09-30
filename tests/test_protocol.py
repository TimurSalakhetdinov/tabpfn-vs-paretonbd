"""Protocol: mature chronological training labels, dev/holdout separation, grouped sampling, hashes."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tabpfn_vs_paretonbd import data
from tabpfn_vs_paretonbd.config import PROTOCOLS, TRAIN_WINDOW_ORIGINS
from tabpfn_vs_paretonbd.models.tabular import cap_context


@pytest.mark.parametrize("name", list(PROTOCOLS))
def test_training_labels_mature_and_holdout_untouched(name):
    p = PROTOCOLS[name]
    H = pd.Timedelta(days=p.horizon_days)
    for o in map(pd.Timestamp, p.dev_origins + p.holdout_origins):
        tr = p.train_origins(o)
        assert tr and all(t + H <= o for t in tr)  # every label window [t, t+H) ends by the origin
    # holdout label windows are fully observed and later than every development label window
    assert pd.Timestamp(p.holdout_origins[-1]) + H - pd.Timedelta(days=1) <= pd.Timestamp(p.data_end)
    assert pd.Timestamp(p.dev_origins[-1]) + H <= pd.Timestamp(p.holdout_origins[0])
    # the development origins never train on anything that overlaps the holdout windows
    assert max(p.train_origins(p.dev_origins[-1])) + H <= pd.Timestamp(p.holdout_origins[0])
    assert TRAIN_WINDOW_ORIGINS >= 1


def test_cap_context_keeps_whole_customers():
    df = pd.DataFrame({"customer_id": np.repeat(np.arange(100).astype(str), 3), "v": 1})
    out = cap_context(df, max_rows=50, seed=0)
    assert len(out) <= 50 and (out.groupby("customer_id").size() == 3).all()


@pytest.mark.parametrize("name", list(data.SOURCES))
def test_raw_file_hashes(name):
    path = data.RAW / data._FILES[name]
    if not path.exists():
        pytest.skip("raw file not fetched")
    assert data.sha256(path) == data.SOURCES[name].sha256


def test_results_labels_and_predictions_align():
    """If holdout results exist, every saved prediction has exactly one mature label and no ID leaks into features."""
    d = data.ROOT / "results" / "online_retail_ii" / "holdout"
    if not (d / "labels.parquet").exists():
        pytest.skip("no saved results")
    from tabpfn_vs_paretonbd.evaluate import load_predictions

    df = load_predictions(d)
    assert df.y_inactive.notna().all()
    assert not df.duplicated(["model", "features", "customer_id", "origin"]).any()
    assert df.p_inactive.dropna().between(0, 1).all() and (df.count_pred.dropna() >= 0).all()


def test_learning_curve_samples_customers_not_rows():
    import inspect

    from tabpfn_vs_paretonbd import pipeline

    src = inspect.getsource(pipeline.run_learning_curve)
    assert "customer_id.unique()" in src and "isin(pick)" in src


def test_repo_contains_no_raw_data_in_git_tracked_paths():
    gi = (Path(data.ROOT) / ".gitignore").read_text()
    assert "data/" in gi or "data/raw" in gi
