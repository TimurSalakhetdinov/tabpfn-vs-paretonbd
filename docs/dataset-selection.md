# Dataset selection and frozen protocol

Status: **frozen on 2026-09-29, before any holdout origin was scored.** Every number in this file is either a
documented claim (marked *doc*) or the output of code in this repository (marked *run*, with the script that
produced it). Development results below influenced design choices; they are therefore not used as final
evidence. Final evidence comes only from the holdout origins listed in §5.

## 1. Question and task

*When does TabPFN-3.5 improve on specialised Bayesian customer models, especially as labelled history becomes
available?* All candidates are noncontractual (no observed cancellation), so the supported endpoint is
**horizon inactivity**, not permanent churn:

- positive class `y_inactive = 1`: no purchase occasion in `[origin, origin + 90 days)`;
- shown as "90-day inactivity risk" and its complement "probability of returning within 90 days";
- secondary target `y_count`: number of purchase occasions in the same window;
- a purchase occasion is one customer on one calendar day with ≥ 1 valid purchase (BTYD convention).

Pareto/NBD's P(alive) is **not** the same quantity as P(no purchase in 90 days); the Bayesian models are scored
on the latter, computed from their posterior or MLE (see `models/pareto_nbd.py`, `models/abe_hb.py`).

## 2. Shortlist and sources

| Candidate | Source (direct URL) | Version / SHA-256 of downloaded file | Rights | Grain |
|---|---|---|---|---|
| **UCI Online Retail II** | `https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip` | `572e3627…67bfb` | CC BY 4.0 (*doc*, UCI page); attribution Chen (2012), doi:10.24432/C5CG6D | invoice line |
| **CDNOW master** (full cohort) | `https://www.brucehardie.com/datasets/CDNOW_master.zip` | `94081977…aaff49` | no licence stated; publicly posted for research use. **Not redistributed**; fetched by the user | customer × day record |
| CDNOW 1/10 sample (Abe 2009 data) | `https://www.brucehardie.com/datasets/CDNOW_sample.zip` | `7c487a00…a282f` | as above | customer × day record |
| dunnhumby / 84.51 *The Complete Journey* | `completejourney` R package, `data/transactions.rds` | `1fa07000…e0007` | R package CC0 (*doc*, DESCRIPTION file, checked 2026-09-29); data originate from 84.51 | basket line |
| Hardie 1995 donation cohort | `https://www.brucehardie.com/datasets/1995_cohort_binary.zip` | `12e5b706…320b5` | no licence stated | donor × year binary |

Full hashes are in `src/tabpfn_vs_paretonbd/data.py::SOURCES`; `tabpfn-vs-paretonbd fetch` downloads and verifies them.

## 3. Executed feasibility checks

Code: `scripts/profile_datasets.py` → `docs/profiling/profile.json` (*run*). Cleaning for Online Retail II
(`data.clean_online_retail_lines`, tested in `tests/test_features.py`):

| Online Retail II step | Lines |
|---|---|
| both sheets, after removing the overlapping period (2010-12-01 .. 2010-12-09 appears in both) | 1,044,848 |
| dropped: missing `Customer ID` | 235,287 |
| dropped: cancellation invoices (`C…`) | 18,446 |
| dropped: non-positive quantity or price | 70 |
| dropped: exact duplicate lines | 11,620 |
| **used** | **779,425** → 33,107 occasions, 5,878 customers |

Documentation states 1,067,371 lines (*doc*). The overlap rule (keep sheet 1 only before the first date of sheet 2) removes the remaining 22,523 lines.

| Check (*run*) | Online Retail II | CDNOW master | Complete Journey | Donations 1995 |
|---|---|---|---|---|
| span | 2009-12-01 .. 2011-12-09 (739 d) | 1997-01-01 .. 1998-06-30 (546 d) | 2017-01-01 .. 2018-01-01 (366 d) | 1995 .. 2006, annual |
| customers / occasions | 5,878 / 33,107 | 23,570 / 67,591 | 2,469 / 126,703 | 11,104 donors |
| repeat customers | 71.3 % | 48.9 % | 98.6 % | — |
| median inter-purchase gap | 31 d (p90 147 d) | 34 d (p90 200 d) | 3 d (p90 13 d) | — |
| first seen in first 60 days (left truncation) | 22.3 % | 70.5 % (Q1-1997 acquisition cohort) | 87.9 % | cohort by construction |
| 90-day inactivity rate at origins | 0.38 → 0.67 (drifts) | 0.77 → 0.90+ | ≈ 0.04 | annual incidence 0.51 → 0.17 |
| mature 90-day origins | 19 monthly (2010-03 .. 2011-09) | 13 monthly | 7 (2 mature training origins before a 3-origin holdout) | — |

Other checks executed: Online Retail II is 91 % UK customers and wholesale-heavy (median 166 units and
£313 per occasion; the top 1 % of customers account for 31.8 % of revenue). First observed purchase ≠ true
acquisition: customers seen in the first 60 days of Online Retail II are left-truncated, and the flag
`first_seen_at_data_start` is part of the expanded feature set. CDNOW is a true first-purchase cohort (all
customers acquired in Q1 1997), so `x, t_x, T` are measured from true acquisition there.

## 4. Decision

- **Primary: UCI Online Retail II.** It has real dated transactions, clear CC BY 4.0 rights, 19 matured monthly
  origins (enough for development and a separate untouched holdout), a customer base that grows through the period, and
  a practical wholesale retention story for the demo. Limitation: its inactivity prevalence drifts with season
  and base ageing (0.38 → 0.67), which dominates calibration (see §6).
- **Fallback and Bayesian bridge: CDNOW master.** It is the dataset of the prior Abe (2009) project, and the
  published Pareto/NBD estimates let us validate both Bayesian implementations. It is scored as a second
  evaluation dataset because it is cheap once the pipeline exists and gives the Bayesian models their "home"
  setting (a single cohort, stationary behaviour). Rights are unclear, so data are never redistributed.
- **Rejected: Complete Journey.** With one year of data and 90-day labels, only about 2 matured training origins
  exist before a 3-origin holdout. The inactivity rate is about 4 % (households are selected as frequent shoppers), and
  87.9 % of households are left-truncated at the start. That is too few positives and origins for a fair temporal
  comparison.
- **Rejected: 1995 donations.** It is annual binary incidence with no within-year dates, so no 90-day
  horizon or purchase-count target can be built. The Pareto/NBD comparison would require the discrete-time BG/BB model instead.

The horizon stays at 90 days because it is feasible on both chosen datasets: the median gap is about 1 month,
the p90 gap is 5–7 months, and 30 days would be dominated by order-cycle noise in wholesale data. That was checked, not assumed.

## 5. Frozen protocol (`src/tabpfn_vs_paretonbd/config.py`)

| | Online Retail II | CDNOW master |
|---|---|---|
| snapshot origins | monthly 2010-03-01 .. 2011-09-01 | monthly 1997-04-01 .. 1998-04-01 |
| development origins | 2011-01-01, 2011-02-01, 2011-03-01 | 1997-09-01, 1997-10-01, 1997-11-01 |
| **holdout origins (untouched)** | **2011-07-01, 2011-08-01, 2011-09-01** | **1998-02-01, 1998-03-01, 1998-04-01** |
| holdout test rows | 5,041 + 5,143 + 5,249 | 23,570 per origin (whole cohort) |

- A training row at origin t is usable for evaluation origin e only if t + 90 d ≤ e, so the label is fully
  observed before the forecast is made. Immature rows are counted and excluded (`manifest.json`).
- Features use history strictly before the origin (tested). **Common** features are `x, t_x, T`, the
  Pareto/NBD sufficient statistics (in days). **Expanded** features add recency, frequency, monetary and basket
  features and the left-truncation flag. `origin_month` is computed for audit only (§6).
- The Bayesian models use no labels. At each origin they are fitted on the unlabelled `x, t_x, T` of the
  evaluation population.
- Metrics: Brier (primary), log loss, calibration intercept and slope, ECE, ROC and PR AUC, count MAE, RMSE and bias.
  Customer-clustered paired bootstrap (2,000 draws) for differences against TabPFN.
- Learning curve: 50 … 2,000 training customers, whole customers sampled, seeds 0, 1 and 2. This runs on Online Retail II
  only. On CDNOW each TabPFN fit predicts 23,570 customers, so the curve would take about 6 h (compute budget, decided
  before holdout).
- Seed 20260929; TabPFN-3.5 checkpoint `tabpfn-v3.5-20260909.safetensors`, sha256 `ece4d67e…0be3`.

## 6. Development pilots (they influenced design, so they are not final evidence)

All pilots use Online Retail II development origins only.

**Prevalence shift.** With the frozen one-origin window, the development origins 2011-01 / 02 / 03 train on
origins 2010-10 / 11 / 12, whose inactivity rates are 0.47 / 0.56 / 0.67. The targets are 0.68 / 0.67 / 0.65, so
the first two development origins carry a large level shift and the third almost none.
Discrimination is stable (ROC AUC ≈ 0.75–0.78), so Brier differences are largely a level (calibration) effect.
This is disclosed, not corrected: recalibrating to the target period would need its labels.

**Training window** (`scripts/pilots/context_pilot.py` → `docs/evidence/context-pilot-online_retail_ii.json`,
dev Brier at 2011-01 / 02 / 03; lower is better):

| model / features | last 1 matured origin | last 3 | all matured |
|---|---|---|---|
| logistic, common | 0.220 / 0.186 / 0.172 | 0.234 / 0.213 / 0.181 | 0.236 / 0.224 / 0.202 |
| logistic, expanded without month | 0.210 / 0.176 / 0.174 | 0.224 / 0.197 / 0.172 | 0.220 / 0.198 / 0.176 |
| LightGBM, expanded without month | 0.231 / 0.194 / 0.176 | 0.237 / 0.203 / 0.173 | 0.231 / 0.203 / 0.177 |
| TabPFN-3.5, common | 0.232 / 0.195 / 0.172 | 0.241 / 0.218 / 0.176 | 0.261 / 0.247 / 0.213 |
| TabPFN-3.5, expanded without month | 0.218 / 0.185 / 0.171 | 0.228 / 0.206 / 0.200 | 0.260 / 0.214 / 0.198 |

Frozen: `TRAIN_WINDOW_ORIGINS = 1`. It is best or tied for every family, and this matches the drift story: the
most recent labels are closest to the target period. The TabPFN cells for 2011-03 finished after the decision
was frozen, and they agree with it. TabPFN windows larger than 1 are capped at 10,000 context rows.

**`origin_month` removed.** With multi-origin windows it let the model extrapolate seasonal level to unseen
months (logistic regression, 3 origins: Brier 0.65 at 2011-02 and 2011-03). With one origin it is constant.
It stays in the snapshot for audit only (`features.AUDIT_ONLY`).

**LightGBM grid** (`scripts/pilots/lgbm_grid.py`, 6 configurations, mean dev Brier on expanded features
0.1953 … 0.2024): frozen `n_estimators=150, learning_rate=0.03, num_leaves=7, min_child_samples=40`. Logistic
regression and TabPFN use defaults (no tuning).

**TabPFN context** (`scripts/pilots/timing_pilot.py`, 2011-03 origin, all matured origins): 1k / 3k / 10k rows
of context took 26 / 39 / 128 s, with peak RSS of 2.3–2.5 GB on Apple-silicon MPS. The cap is set at 10,000 rows
(whole customers). With the one-origin window, Online Retail II never reaches it (≤ 5,000 rows); CDNOW does.

**Bayesian validation** (`scripts/validation/validate_bayes_cdnow.py` →
`docs/evidence/bayes-validation-cdnow_sample.json`, CDNOW 1/10 sample, 39-week calibration):

- Pareto/NBD MLE reproduces the published r, α, s, β.
- Abe HB M1 (4 chains × 20,000 draws after 5,000 burn-in, thinned by 20; 219 s):

| quantity | ours (median) | Abe (2009) | R-hat |
|---|---|---|---|
| E log λ | −3.53 | −3.58 | 1.06 |
| E log μ | −3.61 | −3.59 | 1.02 |
| var log λ | 1.34 | 1.40 | 1.03 |
| var log μ | 2.87 (95 % 0.93–6.85) | 1.87 | 1.07 |
| cov | 0.19 | 0.06 | 1.02 |

The dropout variance mixes slowly (ESS ≈ 61) and is weakly identified, as in the original paper. Holdout
occasions predicted: Abe 1,603, Pareto/NBD 1,666, actual 1,788.

**Pareto/NBD on Online Retail II.** Unbounded MLE diverged (s, β → ∞, NaN at two development origins), so the
optimiser is now bounded in log-space and a boundary hit is reported (`at_bound` in `diagnostics_pareto_nbd.json`).
At 2011-02 the fit converged inside the bounds with s ≈ 9 and β ≈ 8.5 × 10⁴ days: the dropout rate is identified
only through s/β. This is a property of the data (a young, growing wholesale base), not a tuning choice.

**Pareto/NBD optimiser flag on CDNOW holdout 1998-03.** L-BFGS-B returned `success=False` because of precision loss in
the line search, not a boundary hit (`at_bound=False`). A post-hoc check (2026-09-30) found the fit at the optimum:
the largest absolute gradient in log-parameters was 2 × 10⁻⁴ against a negative log-likelihood of 1.19 × 10⁵, and a
restarted L-BFGS-B left the parameters unchanged (r 0.605, α 90.6, s 0.366, β 50.3; in line with the neighbouring
origins). The predictions are kept. The app shows the flag rather than hiding it.

**Abe HB on Online Retail II did not converge (decision frozen before holdout, 2026-09-30).** With the frozen
settings, the development runs reached a max split-R-hat of 2.7 (2011-01), 2.7 (2011-02) and 1.8 (2011-03), all on
the population mean of log μ, with ESS ≈ 3. A 4× longer test at 2011-03 (burn-in 20,000, 80,000 draws, thinned by
80; 887 s) still gave max R-hat 1.18 (β log μ 1.18, cov 1.12, ESS 26). The dropout process is barely identified in
this young, growing customer base (see the Pareto/NBD note above). The frozen rule is therefore: Abe HB keeps
`ABE_SETTINGS` on every split, and any origin with R-hat ≥ `RHAT_MAX` = 1.1 is saved to `unconverged/` and reported
as a failure, never as a score. On CDNOW, where the settings were validated, the same gate applies. Longer chains
were not pursued further because repeated tuning against the gate is a design choice that would need its own
development budget.

## 7. Known limitations

- This is one retailer and one period. The holdout covers three consecutive origins whose label windows overlap, so the
  bootstrap is clustered by customer but origins are not independent.
- Prevalence drift between training and target periods affects every supervised model, TabPFN included.
- CDNOW rights are unclear, so users must fetch it themselves, and results tables contain no raw rows.
- "Inactive for 90 days" is not churn. Many wholesale customers reorder seasonally.
