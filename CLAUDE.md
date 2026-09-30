# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

The local project brief `opus-churn-tabpfn.md` (not tracked in git) is the source of truth. Always prefix commands with `UV_NO_EDITABLE=1`: on macOS the hidden `.pth` file breaks editable installs.

```bash
uv sync --frozen
UV_NO_EDITABLE=1 uv run pytest -m "not slow"                  # fast tests
UV_NO_EDITABLE=1 uv run pytest tests/test_features.py::test_name  # single test
UV_NO_EDITABLE=1 uv run ruff check .
UV_NO_EDITABLE=1 uv run tabpfn-vs-paretonbd smoke                      # tests + each model on 300 customers
UV_NO_EDITABLE=1 uv run tabpfn-vs-paretonbd run <dataset> <dev|holdout> [--families references,sklearn,tabpfn,pareto_nbd,abe_hb]
UV_NO_EDITABLE=1 uv run tabpfn-vs-paretonbd report <dataset> <split>   # tables/figures from saved preds only
UV_NO_EDITABLE=1 uv run tabpfn-vs-paretonbd app
```

- Datasets: `online_retail_ii` (primary) and `cdnow_master`.
- `run` launches each model family as a separate `run-family` subprocess, because LightGBM's libomp and torch's libomp deadlock in a single process on macOS.
- The frozen protocol lives in `src/tabpfn_vs_paretonbd/config.py`. Do not change it after holdout scoring.
- Abe HB predictions go to `unconverged/` when any R-hat is ≥ `RHAT_MAX`, and are never reported.

## Project

TabPFN vs Pareto/NBD is a TabPFN-3.5 hackathon entry and portfolio project. It is a customer inactivity (will they buy again?) prediction workbench that answers: *When does TabPFN-3.5 improve on specialised Bayesian customer models (Pareto/NBD), especially as labelled history grows?*

- Deadline: 6 October 2026, 23:59 CEST (Prior Labs hackathon, https://platform.priorlabs.ai/hackathon-3.5). Reviewable candidate by 3 October, freeze by 5 October. If time runs short, cut optional scope, not evaluation quality.
- Judging: 50% TabPFN showcase, 30% creativity/practical value, 20% technical quality/reproducibility.
- The repo must be public-release-ready under Apache-2.0. Code licensing is separate from dataset, dependency and model-weight licensing.
- Work autonomously. Ask the user only about real blockers: credentials, spending, legal acceptance, publishing/submission, or unclear rights. Final submission and any public URL stay pending until a human does them.

## Work order

1. Dataset research phase (mandatory, comes first): shortlist 3–5 candidates (UCI Online Retail II, CDNOW from brucehardie.com, and at least one other dated longitudinal dataset). Run real profiling plus small baseline/TabPFN pilots, then record the results and the choice (primary + fallback) in `docs/dataset-selection.md`. Report this as a milestone and keep going without stopping.
2. Implement Pareto/NBD and Abe (2009) HB from the papers. This repo is independent of the author's earlier Abe replication (which had no TabPFN); do not import, copy or link its code. Never present a standard Pareto/NBD as Abe's hierarchical correlated model.
3. Build one end-to-end baseline + TabPFN run with saved predictions.
4. Add the Bayesian comparator, calibration and learning curves.
5. Build the app, reproduce the results, and prepare the release/submission artifacts.
6. Add a second dataset only if the core deliverables are done.

## Invariants that span the codebase

These rules apply to data prep, features, models, metrics and the app together. Breaking one invalidates the results.

**Target definitions**
- Primary target: no purchase within a fixed future horizon (initially 90 days, frozen after profiling). Show it as "N-day inactivity risk" and its complement, "probability of returning within N days". Never call it permanent churn for noncontractual data.
- Secondary target: purchase count over the same horizon. Never call it revenue or LTV.
- Pareto/NBD probability-alive is **not** P(purchase within horizon). Derive that probability from the predictive count distribution or posterior predictive simulation. If that isn't possible, compare on counts only and mark classification as unavailable.

**Temporal validity**
- Build supervised examples as (snapshot features at an origin, outcome over the following horizon). Features use only information available up to the cutoff.
- At each evaluation origin, every training label window must have fully ended. Exclude immature/censored labels and log how many were excluded.
- Never randomly split snapshot rows. Use chronological splits with no overlapping label windows across train/validation boundaries.
- Deployment estimand: future behavior of existing customers. The same customer can appear at earlier training origins and later evaluation origins. Claims about unseen customers need a separate customer-disjoint test.
- Customer IDs are for grouping and joins only and must never be predictors.
- Fit preprocessing, calibration, thresholds and hyperparameters on development data only. Freeze the design, then evaluate once on untouched holdout origins. If any outcome influenced design, classify it as a development result.
- Aggregate transaction line items to purchase occasions before counting. Document same-day transactions, cancellations/refunds, first-purchase conditioning and time units, and apply them identically across all models.

**Models**: base-rate/recency references, logistic regression, one boosting library (fixed small validation budget), a TabPFN-3.5 classifier and regressor, and Pareto/NBD. Verify and pin the actual TabPFN-3.5 model ID/version from official docs; package defaults are not evidence of 3.5 use. Compare on the common information set first (frequency, recency, customer age), then on a predeclared expanded feature set, and report the two comparisons separately. Learning curves sample by customer, not by snapshot row. Disclose that Pareto/NBD uses different information (unlabelled transactions).

**Metrics**: set the positive class explicitly. Report Brier score, log loss, calibration, and PR-AUC with prevalence; ROC-AUC is secondary. For counts, report MAE, aggregate predicted vs. observed, and bias. Uncertainty must be paired and customer-clustered, and must account for dependence across origins. Report runtime and compute cost. Do not publish unconverged MCMC results.

**Honesty constraints**: every displayed number must come from saved, measured artifacts. Report unfavorable results. Make no uplift, causal ROI, "revenue saved" or prize-guarantee claims. Never switch datasets because a competitor model wins.

## Required deliverables

The brief's "Reproducibility and required deliverables" and "Critical tests" sections list them in full. Key items: `docs/dataset-selection.md` and `docs/hackathon-compliance.md` (clause → file → evidence → pass/fail/pending); reproducible fetch scripts with raw-data hashes and split manifests; locked environment; Python package + CLI + lightweight local app; saved predictions from which the report tables regenerate; license/third-party notices; a submission description of at most 5,000 characters; a 90-second demo script; and a portfolio case-study draft.
