# TabPFN vs Pareto/NBD

**When does TabPFN-3.5 beat specialized Bayesian customer models at predicting whether a customer comes back?**

This project is a benchmark and a small local workbench. At monthly forecast origins, it predicts two things for
every existing customer:

- **90-day inactivity risk**, `P(no purchase occasion in the next 90 days)`, shown with its complement, the
  probability of returning within 90 days.
- **The number of purchase occasions** in those 90 days.

It compares TabPFN-3.5 (classifier and regressor, untuned) with reference rules, logistic regression, tuned
LightGBM, the Pareto/NBD model (maximum likelihood), and Abe's (2009) hierarchical Bayesian Pareto/NBD (MCMC).
The comparison uses customer-grouped learning curves, calibration, and a customer-clustered paired bootstrap on
untouched holdout origins.

**Note:** The data is noncontractual, so "inactive for 90 days" isn't churn, and nothing here estimates the effect
of contacting a customer. The app ranks risk. It doesn't claim that high-risk customers benefit from an intervention.

## Measured answer

**Short answer:** TabPFN-3.5 matches the Bayesian models only when it gets richer features than they can use, and it
learns best from few labeled customers. With the Pareto/NBD's own three inputs, the Bayesian models are as good or
better on both datasets.

In the following tables, a difference is the other model minus TabPFN-3.5, with a 95% customer-clustered paired
bootstrap interval. A negative Brier difference means that the other model is better.

### Online Retail II (primary dataset)

All numbers come from `results/online_retail_ii/holdout/`: three untouched origins (2011-07-01, 2011-08-01, and
2011-09-01), 15,433 customer-origin rows, and 61.8% inactive.

| Model (information set) | Brier | Δ versus TabPFN [95% CI] | PR-AUC | ECE | Count MAE | Δ count MAE versus TabPFN |
|---|---|---|---|---|---|---|
| **TabPFN-3.5 (expanded)** | 0.1684 | reference | **0.858** | 0.060 | **0.561** | reference |
| GLM (expanded) | **0.1679** | −0.0005 [−0.0015, +0.0005] | 0.857 | 0.048 | 0.576 | +0.014 [+0.010, +0.018] |
| LightGBM (expanded) | 0.1685 | +0.0002 [−0.0005, +0.0009] | 0.855 | 0.051 | 0.608 | +0.047 [+0.042, +0.052] |
| Pareto/NBD, no labels (versus TabPFN expanded) | 0.1689 | +0.0005 [−0.0008, +0.0019] | 0.850 | **0.030** | 0.601 | +0.040 [+0.034, +0.046] |
| TabPFN-3.5 (common) | 0.1769 | reference | 0.847 | 0.076 | 0.588 | reference |
| LightGBM (common) | 0.1726 | −0.0043 [−0.0060, −0.0027] | 0.847 | 0.043 | 0.636 | +0.048 [+0.041, +0.055] |
| GLM (common) | 0.1766 | −0.0003 [−0.0026, +0.0020] | 0.840 | 0.042 | 0.600 | +0.012 [+0.001, +0.021] |
| Pareto/NBD, no labels (versus TabPFN common) | 0.1689 | −0.0080 [−0.0094, −0.0067] | 0.850 | 0.030 | 0.601 | +0.013 [+0.006, +0.019] |
| Recency reference | 0.1909 | | 0.823 | 0.056 | 0.733 | |
| Base rate | 0.2382 | | 0.603 | 0.038 | 0.912 | |

- **Same three inputs** (frequency, recency, and customer age): the unlabeled Pareto/NBD is better calibrated and
  has a lower Brier score than TabPFN-3.5. TabPFN-3.5 still has a lower count MAE.
- **Expanded inputs:** TabPFN-3.5, the GLM, LightGBM, and Pareto/NBD are statistically tied on Brier score.
  TabPFN-3.5 has the best PR-AUC and ROC-AUC (0.816), and a significantly lower count MAE than every other model.
- **Weaknesses of TabPFN-3.5 on this dataset:**
  - It has the worst calibration (ECE 0.060–0.076).
  - It under-forecasts total occasions (predicted/observed 0.80–0.83, compared with 0.96 for Pareto/NBD).

  Both weaknesses follow from training on the most recent matured origin, whose inactivity rate differs from the
  holdout's rate. For details, see "prevalence shift" in `docs/dataset-selection.md`. No recalibration was added
  after the holdout was scored.
- **Abe (2009) HB didn't converge on Online Retail II.** The maximum split-R-hat was 1.48 at 2011-07-01; the other
  two origins passed. By the frozen rule, it isn't scored on this dataset (`diagnostics_abe_hb.json`).

#### Learning curve

The learning curve shows the Brier score by the number of labeled training customers, as the mean of three origins
and three seeds.

![Two panels of holdout Brier score against the number of labeled training customers, from 50 to 2,000 on a log
scale. With expanded features, TabPFN-3.5 has the lowest score at every size and reaches the Pareto/NBD line at
about 2,000 customers.](results/online_retail_ii/holdout/figures/learning_curve_brier.png)

| Model (information set) | 50 | 100 | 250 | 500 | 1,000 | 2,000 |
|---|---|---|---|---|---|---|
| TabPFN-3.5 (expanded) | **0.190** | **0.179** | **0.177** | **0.172** | **0.170** | **0.168** |
| GLM (expanded) | 0.208 | 0.198 | 0.184 | 0.173 | 0.171 | 0.169 |
| LightGBM (expanded) | 0.239 | 0.185 | 0.185 | 0.180 | 0.175 | 0.170 |
| TabPFN-3.5 (common) | 0.199 | 0.183 | 0.183 | 0.178 | 0.179 | 0.177 |
| GLM (common) | 0.191 | 0.180 | 0.181 | 0.178 | 0.176 | 0.176 |

With expanded features, TabPFN-3.5 is the best supervised model at every size. Its advantage is largest with 50–250
labeled customers, and it reaches the label-free Pareto/NBD level (0.169) at about 2,000 customers. The
unregularized Poisson GLM is unstable with 50 customers and expanded features: its median count MAE is 9.8, and
one seed reaches about 10⁶.

#### Compute

On an Apple M1 with 16 GB, TabPFN-3.5 takes about 200 s per origin on MPS for the classifier and regressor, with
about 4,800 training rows. LightGBM takes under 1 s, Pareto/NBD about 100 s, and Abe HB about 350 s. Some runs
overlapped with other jobs, so treat these runtimes as upper bounds.

### CDNOW (the Bayesian models' home dataset)

All numbers come from `results/cdnow_master/holdout/tables/`. Only aggregate tables are committed. The holdout has
three origins (1998-02-01, 1998-03-01, and 1998-04-01), 70,710 customer-origin rows, and 84.8% inactive. TabPFN-3.5
sees a sample of 10,000 of the 23,570 training customers because of its context cap; the other models see all of
them.

| Model | Brier | Δ versus TabPFN [95% CI] | Count MAE | Δ count MAE versus TabPFN (common / expanded) |
|---|---|---|---|---|
| **Abe (2009) HB, converged** (max R-hat 1.06) | 0.0968 | versus common −0.0012 [−0.0017, −0.0007]; versus expanded +0.0004 [+0.0001, +0.0008] | 0.298 | −0.034 / −0.023 |
| Pareto/NBD | 0.0971 | versus common −0.0009 [−0.0014, −0.0004]; versus expanded +0.0007 [+0.0003, +0.0011] | **0.297** | −0.036 / −0.025 |
| TabPFN-3.5 (common) | 0.0980 | reference | 0.333 | reference |
| TabPFN-3.5 (expanded) | 0.0964 | reference | 0.321 | reference |
| GLM (expanded) | **0.0961** | −0.0003 [−0.0006, +0.0001] | 0.336 | +0.015 |
| LightGBM (expanded) | 0.0963 | −0.0001 [−0.0004, +0.0002] | 0.335 | +0.014 |

On CDNOW, the hierarchical Bayesian model converges and is the best model on the common inputs, for both
inactivity and counts. With expanded features, TabPFN-3.5 moves slightly ahead of both Bayesian models on Brier
score and ties with the GLM and LightGBM. The Bayesian models still forecast counts best. TabPFN-3.5 takes about
950 s per origin on MPS because it scores 23,570 customers per origin in chunks. Abe HB takes about 1,160 s on
the CPU.

### When TabPFN-3.5 beats the Bayesian models

In these two datasets, TabPFN-3.5 needs information that the Pareto/NBD can't use: recent-window, gap, trend, and
monetary features. With few labeled customers, it's the strongest supervised option. With the Pareto/NBD's own
sufficient statistics, the Bayesian models were as good or better on both datasets.

## Quickstart

Before you begin, make sure that you have the following:

- macOS or Linux, and Python 3.11.
- [uv](https://docs.astral.sh/uv/).
- About 3 GB of free RAM, and about 1 GB of disk for the TabPFN-3.5 weights.

The weights download on first use. They're under a non-commercial license that you accept when you download them.
For details, see `THIRD_PARTY_NOTICES.md`.

To install the project, fetch the primary dataset, and run a quick check, run the following commands:

```bash
uv sync --frozen
export UV_NO_EDITABLE=1                             # macOS: see "Known issues"
uv run tabpfn-vs-paretonbd fetch online_retail_ii   # UCI download, SHA-256 checked
uv run tabpfn-vs-paretonbd smoke                    # tests + every model on 300 customers
uv run tabpfn-vs-paretonbd app                      # local Streamlit app on the committed results
```

The smoke check takes a few minutes. To regenerate everything under `results/`, run the full evaluation:

```bash
uv run tabpfn-vs-paretonbd run online_retail_ii holdout     # all families and learning curves
uv run tabpfn-vs-paretonbd fetch cdnow_master               # rights unclear: you fetch it; never redistributed
uv run tabpfn-vs-paretonbd run cdnow_master holdout         # no learning curve (compute budget)
uv run tabpfn-vs-paretonbd report online_retail_ii holdout  # tables and figures from saved predictions only
```

The `run` command starts each model family in a fresh subprocess:
`run-family DATASET SPLIT FAMILY`, where `FAMILY` is one of `references`, `sklearn`, `tabpfn`, `pareto_nbd`, or
`abe_hb`. Measured runtimes are in each split's `tables/runtime.csv`.

## Models and information sets

| Model | Uses labels? | Inputs |
|---|---|---|
| Base rate; recency (isotonic on days since last purchase) | Yes | Common set only |
| Logistic or Poisson GLM; LightGBM (6-configuration grid on development data only) | Yes | Common or expanded |
| **TabPFN-3.5** classifier and regressor, default settings | Yes (in context) | Common or expanded |
| Pareto/NBD (MLE) | **No**: fitted to the unlabeled history of the customers being scored | `x`, `t_x`, `T` |
| Abe (2009) HB M1 (MCMC; published only if every split-R-hat is below 1.1) | No | `x`, `t_x`, `T` |

The information sets are defined as follows:

- **Common set:** `x` (repeat occasions), `t_x` (days from first to last occasion), and `T` (days from first
  occasion to origin). These are the Pareto/NBD sufficient statistics.
- **Expanded set:** adds recency, counts in the last 30, 90, 180, and 365 days, trend, gap statistics, monetary and
  basket features, and a left-truncation flag (`src/tabpfn_vs_paretonbd/features.py`).
- **Bayesian models:** they see different information: no labels, but all customers' unlabeled histories. Both
  comparisons show them as the same reference.

## Protocol

These rules were frozen before any holdout scoring (`src/tabpfn_vs_paretonbd/config.py` and
`docs/dataset-selection.md`):

- A purchase occasion is one customer-day. Cancellations, missing IDs, non-positive lines, and duplicates are
  dropped.
- Features use history strictly before the origin. The label window is `[origin, origin + 90 d)`.
- The training rows for an evaluation origin come from the most recent origin whose 90-day label window has fully
  ended. Immature labels are excluded and counted in `manifest.json`.
- Online Retail II uses development origins 2011-01-01, 2011-02-01, and 2011-03-01, and holdout origins
  2011-07-01, 2011-08-01, and 2011-09-01. CDNOW uses development origins 1997-09-01, 1997-10-01, and 1997-11-01,
  and holdout origins 1998-02-01, 1998-03-01, and 1998-04-01.
- Uncertainty comes from a paired bootstrap that resamples customers (2,000 draws), so a customer's rows at all
  three origins move together.
- The learning curve varies the number of training *customers* from 50 to 2,000, with three seeds each.

## Repository map

```
src/tabpfn_vs_paretonbd/  data.py (fetch, hashes, cleaning) · features.py (snapshots) · config.py (frozen protocol)
                          models/ (tabular.py, pareto_nbd.py, abe_hb.py) · pipeline.py · evaluate.py · figures.py
                          cli.py · app.py
results/<ds>/<split>/     manifest.json · labels.parquet · preds/ · learning_curve/ · tables/ · figures/
                          diagnostics_*.json
docs/                     dataset-selection.md · hackathon-compliance.md · submission-description.md
                          case-study.md · evidence/ · profiling/
scripts/                  profiling, development pilots, Bayesian validation
tests/                    cutoffs, censoring, maturity, grouping, published Pareto/NBD values, metric orientation
```

## TabPFN-3.5 verification

`scripts/verify_tabpfn35.py` records the package version (`tabpfn` 9.0.0), `ModelVersion.V3_5`, and the checkpoint
file and hash (`tabpfn-v3.5-20260909.safetensors`, SHA-256 `ece4d67e…`) in
`docs/evidence/tabpfn35-verification.json`. The models are created with
`TabPFNClassifier.create_default_for_version(ModelVersion.V3_5)` and the matching `TabPFNRegressor` call.

## Known issues

- **macOS and uv:** files in a cloned repository can carry the `hidden` flag, and Python then skips the
  editable-install `.pth` file. To avoid this, set `UV_NO_EDITABLE=1`, which installs the package as a normal copy on
  each `uv run`.
- **macOS OpenMP:** LightGBM (through scikit-learn's bundled libomp) and PyTorch's libomp can deadlock in one
  process. The pipeline therefore runs each model family in its own subprocess. If you combine them in one process
  yourself, set `OMP_NUM_THREADS=1`.

## Data and licenses

The code is under Apache-2.0 (`LICENSE`). Data and weights aren't included; for details, see
`THIRD_PARTY_NOTICES.md`.

- Online Retail II is under CC BY 4.0 (Chen 2012, UCI, doi:10.24432/C5CG6D).
- CDNOW (Fader and Hardie 2001) has no stated license and is never redistributed. For CDNOW, only aggregate tables
  are committed.

## Prior work

The author previously replicated Abe (2009) on CDNOW, without TabPFN. This project is a separate codebase: both
Bayesian models are implemented here from the original papers and validated against the published CDNOW estimates
(`docs/evidence/bayes-validation-cdnow_sample.json`). The comparison with TabPFN-3.5 is this project's
contribution.
