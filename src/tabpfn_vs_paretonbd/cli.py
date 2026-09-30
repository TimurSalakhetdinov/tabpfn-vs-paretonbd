"""TabPFN vs Pareto/NBD command line. Run `tabpfn-vs-paretonbd --help` (or `UV_NO_EDITABLE=1 uv run tabpfn-vs-paretonbd --help`)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import typer

app = typer.Typer(add_completion=False, no_args_is_help=True, help=__doc__)

DATASETS = ["online_retail_ii", "cdnow_master"]


@app.command()
def fetch(dataset: str = typer.Argument("all", help="online_retail_ii | cdnow_master | cdnow_sample | all")):
    """Download raw public data into data/raw and verify SHA-256 hashes."""
    from tabpfn_vs_paretonbd import data

    names = ["online_retail_ii", "cdnow_master", "cdnow_sample"] if dataset == "all" else [dataset]
    for n in names:
        p = data.fetch(n)
        typer.echo(f"{n}: {p} sha256 ok")


@app.command()
def snapshots(dataset: str):
    """Build (and cache) customer snapshots at every protocol origin; print the per-origin summary."""
    from tabpfn_vs_paretonbd.pipeline import load_snapshots

    s = load_snapshots(dataset)
    typer.echo(s.groupby("origin").agg(rows=("customer_id", "size"), inactive=("y_inactive", "mean")).to_string())


@app.command("run-family")
def run_family(dataset: str, split: str, family: str,
               learning_curve: bool = typer.Option(False, help="also run the training-customer learning curve"),
               skip_existing: bool = typer.Option(False, help="learning curve: keep already-saved model/feature files")):
    """Run one model family (references | sklearn | tabpfn | pareto_nbd | abe_hb) in this process."""
    from tabpfn_vs_paretonbd import pipeline

    assert split in ("dev", "holdout")
    pipeline.write_labels_and_manifest(dataset, split)
    if family in ("pareto_nbd", "abe_hb"):
        pipeline.run_bayes(dataset, split, family)
        return
    models = pipeline.FAMILIES[family]
    if learning_curve:
        pipeline.run_learning_curve(dataset, split, [m for m in models if m in ("glm", "lgbm", "tabpfn")],
                                    skip_existing=skip_existing)
    else:
        # the references only use days since last occasion (= T - t_x), i.e. the common information set
        pipeline.run_supervised(dataset, split, models, ("common",) if family == "references" else ("common", "expanded"))


@app.command()
def run(dataset: str, split: str = typer.Argument("holdout", help="dev | holdout"),
        families: str = typer.Option("references,sklearn,tabpfn,pareto_nbd,abe_hb", help="comma-separated"),
        learning_curve: bool = typer.Option(True, help="run learning curves for sklearn and tabpfn (primary dataset only)")):
    """Full evaluation for one dataset and split; each model family runs in a fresh subprocess."""
    from tabpfn_vs_paretonbd import pipeline
    from tabpfn_vs_paretonbd.config import LEARNING_CURVE_DATASETS

    learning_curve = learning_curve and dataset in LEARNING_CURVE_DATASETS
    for fam in families.split(","):
        pipeline.run_family_subprocess(dataset, split, fam)
        if learning_curve and fam in ("sklearn", "tabpfn"):
            pipeline.run_family_subprocess(dataset, split, fam, learning_curve=True)
    report(dataset, split)


@app.command()
def report(dataset: str, split: str = typer.Argument("holdout", help="dev | holdout")):
    """Regenerate every table and figure from saved predictions (no model is refitted)."""
    from tabpfn_vs_paretonbd import evaluate, figures, pipeline

    d = pipeline.out_dir(dataset, split)
    tables = evaluate.write_tables(d)
    figures.make_all(d, tables)
    pooled = tables["pooled"]
    cols = [c for c in ["model", "features", "brier", "log_loss", "pr_auc", "prevalence", "roc_auc", "count_mae",
                        "count_bias"] if c in pooled]
    typer.echo(pooled[cols].round(4).to_string(index=False))


@app.command()
def smoke(n_customers: int = typer.Option(300, help="customers sampled for training and for testing"),
          tabpfn: bool = typer.Option(True, help="include TabPFN-3.5 (downloads weights on first use)")):
    """Fast end-to-end check (a few minutes): fast tests, then each model family on a small sample of one
    development origin, each in its own subprocess. Writes nothing to results/."""
    subprocess.run([sys.executable, "-m", "pytest", "-q", "-m", "not slow"], check=True)
    for m in ["base_rate", "recency", "glm", "lgbm", "pareto_nbd"] + (["tabpfn"] if tabpfn else []):
        subprocess.run([sys.executable, "-m", "tabpfn_vs_paretonbd.cli", "smoke-one", m, "--n-customers", str(n_customers)],
                       check=True)
    typer.echo("smoke ok")


@app.command("smoke-one", hidden=True)
def smoke_one(model: str, n_customers: int = 300):
    import time

    import numpy as np
    import pandas as pd

    from tabpfn_vs_paretonbd import evaluate, pipeline
    from tabpfn_vs_paretonbd.config import PROTOCOLS, SEED
    from tabpfn_vs_paretonbd.features import FEATURE_SETS

    ds = "online_retail_ii"
    o = pd.Timestamp(PROTOCOLS[ds].dev_origins[-1])
    train, test, _ = pipeline.split_at(pipeline.load_snapshots(ds), ds, o)
    rng = np.random.default_rng(SEED)
    train = train[train.customer_id.isin(rng.choice(train.customer_id.unique(), n_customers, replace=False))]
    test = test[test.customer_id.isin(rng.choice(test.customer_id.unique(), n_customers, replace=False))]
    t0 = time.time()
    if model == "pareto_nbd":
        from tabpfn_vs_paretonbd.models.pareto_nbd import ParetoNBD

        x, t_x, T = (test[c].to_numpy(float) for c in ("x", "t_x", "T"))
        m = ParetoNBD().fit(x, t_x, T)
        p, _ = m.prob_no_purchase(PROTOCOLS[ds].horizon_days, x, t_x, T, n_draws=1000, seed=SEED)
        c = m.expected_count(PROTOCOLS[ds].horizon_days, x, t_x, T)
    else:
        from tabpfn_vs_paretonbd.models.tabular import fit_predict

        p, c = fit_predict(model, train, test, FEATURE_SETS["common"])
    y = test.y_inactive.to_numpy()
    assert np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all() and np.isfinite(c).all(), model
    typer.echo(json.dumps({"model": model, "origin": str(o.date()), "n_train": len(train), "n_test": len(test),
                           "brier": round(evaluate.brier(y, p), 4), "seconds": round(time.time() - t0, 1)}))


@app.command("app")
def run_app():
    """Launch the local Streamlit app."""
    path = Path(__file__).with_name("app.py")
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(path)], check=True)


if __name__ == "__main__":
    app()
