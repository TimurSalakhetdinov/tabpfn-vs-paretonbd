"""TabPFN vs Pareto/NBD local app. Launch with `tabpfn-vs-paretonbd app` (or `streamlit run src/tabpfn_vs_paretonbd/app.py`).

Every number shown comes from saved artifacts under results/ (predictions, tables, runtime logs). The customer view
is retrospective: it shows what each model said at a past forecast origin next to what then happened.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(os.environ.get("TABPFN_VS_PARETONBD_ROOT", Path.cwd())).resolve()
RESULTS = ROOT / "results"
DISPLAY = {"base_rate": "Base rate", "recency": "Recency reference", "glm": "Logistic / Poisson GLM",
           "lgbm": "LightGBM", "tabpfn": "TabPFN-3.5", "pareto_nbd": "Pareto/NBD (MLE)",
           "abe_hb": "HB Pareto/NBD (Abe 2009 M1)"}
DATASET_NAMES = {"online_retail_ii": "UCI Online Retail II (UK online wholesaler, 2009-2011)",
                 "cdnow_master": "CDNOW full cohort (1997-1998)"}

st.set_page_config(page_title="TabPFN vs Pareto/NBD", layout="wide")


@st.cache_data
def load(dataset: str, split: str):
    d = RESULTS / dataset / split
    t = {p.stem: pd.read_csv(p) for p in (d / "tables").glob("*.csv")}
    lab = pd.read_parquet(d / "labels.parquet")
    preds = []
    for f in sorted((d / "preds").glob("*.parquet")):
        m, fs = f.stem.split("__")
        preds.append(pd.read_parquet(f).assign(model=m, features=fs))
    pr = pd.concat(preds, ignore_index=True).merge(lab, on=["customer_id", "origin"])
    manifest = json.loads((d / "manifest.json").read_text())
    diags = {p.stem.replace("diagnostics_", ""): json.loads(p.read_text()) for p in d.glob("diagnostics_*.json")}
    figs = {p.stem: p for p in (d / "figures").glob("*.png")}
    return t, pr, manifest, diags, figs


@st.cache_data
def occasions(dataset: str):
    try:
        from tabpfn_vs_paretonbd import data

        return data.load_occasions(dataset)
    except Exception:  # raw data not fetched: the timeline is optional
        return None


def name(m, fs=None):
    return DISPLAY.get(m, m) + (f" [{fs}]" if fs and m not in ("pareto_nbd", "abe_hb") else "")


# CDNOW row-level predictions are not redistributed; without a local run only its aggregate tables exist.
_with_tables = sorted((d for d in RESULTS.glob("*") if (d / "holdout" / "tables").exists()), key=lambda d: d.name != "online_retail_ii") if RESULTS.exists() else []
available = [d.name for d in _with_tables if (d / "holdout" / "labels.parquet").exists()]
tables_only = [d.name for d in _with_tables if d.name not in available]
if not available:
    st.error("No saved holdout results found under results/. Run `tabpfn-vs-paretonbd run online_retail_ii` first.")
    st.stop()

st.sidebar.title("TabPFN vs Pareto/NBD")
dataset = st.sidebar.selectbox("Dataset", available, format_func=lambda d: DATASET_NAMES.get(d, d))
tables, preds, manifest, diags, figs = load(dataset, "holdout")
H = manifest["horizon_days"]
st.sidebar.markdown(
    f"**Target:** {H}-day inactivity = no purchase occasion in [origin, origin + {H} days).  \n"
    f"Shown as risk, and its complement as *probability of returning within {H} days*. "
    "This is not permanent churn.")
if tables_only:
    st.sidebar.caption("Aggregate tables only (row-level predictions not redistributed; run locally to explore): "
                       + ", ".join(f"results/{d}/holdout/tables/" for d in tables_only))
st.sidebar.markdown("**Holdout origins:** " + ", ".join(o["origin"] for o in manifest["origins"]))
page = st.sidebar.radio("View", ["Measured results", "Customer at a past origin", "Model comparison",
                                 "Risk ranking", "Downloads"])

pooled = tables["pooled"]
boot = tables.get("bootstrap", pd.DataFrame())
def mcmc_failures(dg):
    """Origins whose MCMC fit failed the R-hat gate (MLE entries carry an optimiser flag, not an R-hat)."""
    return [o for o in dg if isinstance(o, dict) and "max_rhat" in o and o.get("converged") is False] \
        if isinstance(dg, list) else []


unconverged = [DISPLAY.get(m, m) for m, dg in diags.items() if mcmc_failures(dg)]

if page == "Measured results":
    st.title("When does TabPFN-3.5 beat Bayesian customer models?")
    st.caption(f"{DATASET_NAMES.get(dataset, dataset)} · measured on untouched holdout origins · "
               "every number is read from saved predictions")
    if unconverged:
        st.warning(", ".join(unconverged) + " failed the MCMC convergence check (split-R-hat < 1.1) on this dataset "
                   "and is not scored here; see Model comparison → Compute cost and failures.")
    for fs, label in [("common", "Same inputs: frequency, recency, customer age"),
                      ("expanded", "Richer inputs: recent windows, gaps, trends, monetary history")]:
        st.subheader(label)
        rows = pooled[(pooled.features == fs) | pooled.model.isin(["pareto_nbd", "abe_hb"])].copy()
        b = boot[(boot.comparison == fs)] if len(boot) else pd.DataFrame()
        rows["Model"] = [name(m, f) for m, f in zip(rows.model, rows.features)]
        show = rows[["Model", "brier", "log_loss", "pr_auc", "prevalence", "roc_auc", "calib_slope", "count_mae",
                     "count_bias", "count_total_ratio"]].rename(columns={
                         "brier": "Brier", "log_loss": "Log loss", "pr_auc": "PR-AUC", "prevalence": "Prevalence",
                         "roc_auc": "ROC-AUC", "calib_slope": "Calib. slope", "count_mae": "Count MAE",
                         "count_bias": "Count bias", "count_total_ratio": "Pred/obs total"})
        if len(b):
            bb = b[b.metric == "brier"].set_index("model")
            ci = [f"{bb.loc[m, 'diff_vs_tabpfn']:+.4f} [{bb.loc[m, 'diff_ci_low']:+.4f}, {bb.loc[m, 'diff_ci_high']:+.4f}]"
                  if m in bb.index and pd.notna(bb.loc[m, "diff_vs_tabpfn"]) else "reference" if m == "tabpfn" else ""
                  for m in rows.model]
            show.insert(2, "Brier minus TabPFN (95% CI)", ci)
        st.dataframe(show.sort_values("Brier").round(4), hide_index=True, use_container_width=True)
    st.caption("Positive class: inactive. Lower Brier/log loss is better. PR-AUC should be read against prevalence. "
               "Differences: model minus TabPFN, paired bootstrap resampling customers across all holdout origins "
               "(positive = TabPFN better). Pareto/NBD models use no labels: they are fitted on the unlabelled "
               "transaction history of every customer in the base at each origin.")
    c1, c2 = st.columns(2)
    if "learning_curve_brier" in figs:
        c1.image(str(figs["learning_curve_brier"]), caption="Learning curve (labelled training customers)")
    if "calibration_expanded" in figs:
        c2.image(str(figs["calibration_expanded"]), caption="Calibration, expanded features")

elif page == "Customer at a past origin":
    st.title("One customer, one past forecast origin")
    st.info("Retrospective view: predictions were made with data before the origin; the outcome is what happened "
            "afterwards. Nothing here is a recommendation to contact or discount a customer.")
    origins = sorted(preds.origin.unique())
    o = st.selectbox("Forecast origin", origins, format_func=lambda t: str(pd.Timestamp(t).date()))
    sub = preds[preds.origin == o]
    tp = sub[(sub.model == "tabpfn") & (sub.features == "expanded")].sort_values("p_inactive")
    how = st.radio("Pick a customer", ["highest TabPFN risk", "lowest TabPFN risk", "one-time buyers", "any"],
                   horizontal=True)
    pool = {"highest TabPFN risk": tp.tail(50)[::-1], "lowest TabPFN risk": tp.head(50),
            "one-time buyers": tp[tp.x == 0], "any": tp}[how]
    cid = st.selectbox("Customer (public benchmark ID)", pool.customer_id.tolist()[:500])
    rows = sub[sub.customer_id == cid]
    lab = rows.iloc[0]
    c1, c2, c3, c4 = st.columns(4)
    t = rows[(rows.model == "tabpfn") & (rows.features == "expanded")].iloc[0]
    c1.metric(f"TabPFN P(return within {H}d)", f"{1 - t.p_inactive:.0%}")
    c2.metric("TabPFN expected occasions", f"{t.count_pred:.2f}")
    bay = rows[rows.model == "abe_hb"]
    if len(bay):
        c3.metric("HB Pareto/NBD P(return)", f"{1 - bay.p_inactive.iloc[0]:.0%}")
    c4.metric("Observed afterwards", f"{int(lab.y_count)} occasions" + (" (inactive)" if lab.y_inactive else ""))
    occ = occasions(dataset)
    if occ is not None:
        h = occ[occ.customer_id == cid].copy()
        h["period"] = np.where(h.date < pd.Timestamp(o), "history (used)", "after origin (not used)")
        import altair as alt

        ch = alt.Chart(h).mark_circle(size=80).encode(
            x=alt.X("date:T", title="purchase occasions"), y=alt.Y("amount:Q", title="amount"),
            color=alt.Color("period:N", scale=alt.Scale(range=["#1f5aa6", "#e07b00"])), tooltip=["date:T", "amount:Q"])
        rule = alt.Chart(pd.DataFrame({"d": [pd.Timestamp(o)]})).mark_rule(color="black").encode(x="d:T")
        end = alt.Chart(pd.DataFrame({"d": [pd.Timestamp(o) + pd.Timedelta(days=H)]})).mark_rule(
            color="gray", strokeDash=[4, 4]).encode(x="d:T")
        st.altair_chart(ch + rule + end, use_container_width=True)
        st.caption("Solid line: forecast origin. Dashed: end of the label window.")
        from tabpfn_vs_paretonbd.features import EXPANDED_FEATURES, snapshot

        f = snapshot(occ, pd.Timestamp(o), H, occ.date.min()).set_index("customer_id").loc[cid, EXPANDED_FEATURES]
        st.markdown("**Features known at the cutoff** (days; amounts in source currency)")
        st.dataframe(f.to_frame("value").T.round(2), hide_index=True, use_container_width=True)
    st.markdown("**All models at this origin**")
    tab = rows.assign(Model=[name(m, f) for m, f in zip(rows.model, rows.features)],
                      **{f"P(return within {H}d)": 1 - rows.p_inactive, "Expected occasions": rows.count_pred})
    st.dataframe(tab[["Model", f"P(return within {H}d)", "Expected occasions"]].round(3), hide_index=True)

elif page == "Model comparison":
    st.title("Model comparison")
    c1, c2 = st.columns(2)
    for c, k, cap in [(c1, "calibration_common", "Calibration, common inputs"),
                      (c2, "calibration_expanded", "Calibration, expanded inputs"),
                      (c1, "learning_curve_brier", "Learning curve, Brier"),
                      (c2, "learning_curve_count_mae", "Learning curve, count MAE"),
                      (c1, "brier_by_origin", "Brier by holdout origin")]:
        if k in figs:
            c.image(str(figs[k]), caption=cap)
    st.subheader("Prespecified segments (sample sizes shown)")
    seg = tables["segments"]
    seg = seg.assign(Model=[name(m, f) for m, f in zip(seg.model, seg.features)])
    pick = st.selectbox("Segment", seg.segment.unique())
    st.dataframe(seg[seg.segment == pick][["Model", "n_rows", "n_customers", "prevalence", "brier", "log_loss",
                                           "count_mae"]].sort_values("brier").round(4), hide_index=True)
    st.subheader("Per-origin results")
    bo = tables["by_origin"]
    st.dataframe(bo.assign(Model=[name(m, f) for m, f in zip(bo.model, bo.features)])[
        ["origin", "Model", "n", "prevalence", "brier", "pr_auc", "count_mae"]].round(4), hide_index=True)
    st.subheader("Compute cost and failures")
    if "runtime" in tables:
        rt = tables["runtime"]
        st.dataframe(rt.assign(Model=[name(m, f) for m, f in zip(rt.model, rt.features)])[
            ["Model", "origins", "seconds_per_origin", "seconds_total"]].round(1), hide_index=True)
    fails = tables["runtime"].get("failures", pd.Series(dtype=float)).sum() if "runtime" in tables else 0
    st.markdown(f"Failed fits recorded: **{int(fails)}** (the pipeline stops on any failure, so a completed run "
                "has none). Hardware and runtimes: see README.")
    for m, dg in diags.items():
        bad = mcmc_failures(dg)
        flagged = [o["origin"] for o in dg if isinstance(o, dict) and "max_rhat" not in o
                   and o.get("converged") is False] if isinstance(dg, list) else []
        if flagged:
            st.info(f"{DISPLAY.get(m, m)}: the optimiser reported a non-success status at {', '.join(flagged)} "
                    "(see `at_bound` and `loglik` in the diagnostics below). The predictions are kept and scored.")
        if bad:
            st.error(f"{DISPLAY.get(m, m)} did not converge (split-R-hat ≥ 1.1 at "
                     + ", ".join(f"{o['origin']} (max {o['max_rhat']:.2f})" for o in bad)
                     + "). Its predictions are excluded from every table on this dataset.")
    for m, dg in diags.items():
        with st.expander(f"{DISPLAY.get(m, m)} fit diagnostics"):
            st.json(dg)

elif page == "Risk ranking":
    st.title("Ranking customers by inactivity risk")
    st.warning("Ranking by risk is not targeting advice. Without randomised treatment data we cannot say who would "
               "respond to an intervention; high-risk customers may be the least persuadable.")
    o = st.selectbox("Origin", sorted(preds.origin.unique()), format_func=lambda t: str(pd.Timestamp(t).date()))
    opts = sorted(set(zip(preds.model, preds.features)))
    m, fs = st.selectbox("Model", opts, format_func=lambda k: name(*k),
                         index=opts.index(("tabpfn", "expanded")) if ("tabpfn", "expanded") in opts else 0)
    r = preds[(preds.origin == o) & (preds.model == m) & (preds.features == fs)].sort_values("p_inactive", ascending=False)
    r["decile"] = np.minimum(np.arange(len(r)) * 10 // len(r) + 1, 10)
    g = r.groupby("decile").agg(customers=("customer_id", "size"), mean_predicted=("p_inactive", "mean"),
                                observed=("y_inactive", "mean"))
    g["lift"] = g.observed / r.y_inactive.mean()
    st.dataframe(g.round(3), use_container_width=True)
    st.caption("Decile 1 = highest predicted inactivity risk. Observed = share inactive in the following window.")

else:
    st.title("Downloads")
    st.markdown("Prediction tables for every model at every holdout origin, joined to observed outcomes.")
    csv = preds.to_csv(index=False).encode()
    st.download_button("All holdout predictions (CSV)", csv, f"tabpfn_vs_paretonbd_{dataset}_holdout_predictions.csv")
    for k, v in tables.items():
        st.download_button(f"Table: {k}", v.to_csv(index=False).encode(), f"{dataset}_{k}.csv", key=k)
    st.json(manifest)
