"""Feature cutoffs, label windows, censoring, purchase counting and predictor hygiene."""

import numpy as np
import pandas as pd
import pytest

from tabpfn_vs_paretonbd.data import OCCASION_COLS, clean_online_retail_lines
from tabpfn_vs_paretonbd.features import AUDIT_ONLY, EXPANDED_FEATURES, FEATURE_SETS, ID_COLS, TARGETS, snapshot

D = pd.Timestamp


def occ(rows):
    return pd.DataFrame([(c, D(d), 1, 1, 10.0) for c, d in rows], columns=OCCASION_COLS)


BASE = occ([("a", "2020-01-01"), ("a", "2020-01-11"), ("a", "2020-03-01"),
            ("b", "2020-02-01"),
            ("c", "2020-03-01")])  # c's first occasion is exactly on the origin -> not in the base


def snap(o, h=30, end="2020-12-31", data=BASE):
    return snapshot(data, D(o), h, data.date.min(), D(end)).set_index("customer_id")


def test_sufficient_statistics_and_units():
    s = snap("2020-03-01")
    assert list(s.index) == ["a", "b"]  # c has no history strictly before the origin
    a = s.loc["a"]
    assert (a.x, a.t_x, a["T"]) == (1.0, 10.0, 60.0)  # repeat occasions, days first->last, days first->origin
    assert s.loc["b"].x == 0 and s.loc["b"].t_x == 0


def test_features_ignore_everything_on_or_after_origin():
    extra = pd.concat([BASE, occ([("a", "2020-03-01"), ("a", "2020-03-05"), ("b", "2020-06-01")])])
    f1, f2 = snap("2020-03-01"), snap("2020-03-01", data=extra)
    cols = FEATURE_SETS["expanded"]
    pd.testing.assert_frame_equal(f1[cols], f2[cols])
    assert f2.loc["a", "y_count"] == 3  # 2020-03-01 (BASE) + extra 03-01 row + 03-05


def test_label_window_is_half_open():
    s = snap("2020-02-01", h=29)  # window [02-01, 03-01): a's 03-01 occasion falls outside
    assert s.loc["a", "y_count"] == 0 and s.loc["a", "y_inactive"] == 1
    s = snap("2020-02-01", h=30)
    assert s.loc["a", "y_count"] == 1 and s.loc["a", "y_inactive"] == 0


def test_immature_labels_are_censored():
    s = snap("2020-12-15", h=30, end="2020-12-31")
    assert s.y_inactive.isna().all() and s.y_count.isna().all()
    s = snap("2020-12-02", h=30, end="2020-12-31")  # last label day 12-31 == data end -> mature
    assert s.y_inactive.notna().all()


def test_no_ids_labels_or_future_columns_in_predictors():
    forbidden = set(ID_COLS) | set(TARGETS) | {"label_end"} | set(AUDIT_ONLY)
    for cols in FEATURE_SETS.values():
        assert not forbidden & set(cols)
    assert FEATURE_SETS["common"] == ["x", "t_x", "T"]
    assert set(FEATURE_SETS["common"]) <= set(EXPANDED_FEATURES)


def test_online_retail_cleaning_rules():
    lines = pd.DataFrame({
        "Invoice": ["1", "1", "2", "C3", "4", "5", "6", "6"],
        "StockCode": ["x", "y", "x", "x", "x", "x", "x", "x"],
        "Quantity": [2, 1, 1, -1, 0, 1, 3, 3],
        "Price": [1.0, 2.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
        "InvoiceDate": pd.to_datetime(["2020-01-01 09:00", "2020-01-01 09:00", "2020-01-01 17:00", "2020-01-02 10:00",
                                       "2020-01-03 10:00", "2020-01-04 10:00", "2020-01-05 10:00", "2020-01-05 10:00"]),
        "CustomerID": [7.0, 7.0, 7.0, 7.0, 7.0, np.nan, 7.0, 7.0],
        "Country": ["UK"] * 8,
    })
    occ_, _, log = clean_online_retail_lines(lines)
    # two invoices on 2020-01-01 collapse into one occasion; cancellation, zero qty/price and missing-ID dropped
    assert len(occ_) == 1
    r = occ_.iloc[0]
    assert (r.customer_id, r.n_tx, r.units, r.amount) == ("7", 2, 4, 5.0)
    assert log["lines_cancellation"] == 1 and log["lines_missing_customer"] == 1


@pytest.mark.parametrize("o", ["2020-02-01", "2020-03-01"])
def test_snapshot_is_deterministic(o):
    pd.testing.assert_frame_equal(snap(o), snap(o))
