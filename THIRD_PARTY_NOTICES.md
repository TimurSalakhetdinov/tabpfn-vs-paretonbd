# Third-party notices

TabPFN vs Pareto/NBD source code is released under the Apache License 2.0 (`LICENSE`). The Apache licence covers only
the code in this repository. Data, model weights and dependencies keep their own terms, listed below.
Nothing in this file is legal advice.

## Model weights: TabPFN-3.5 (not included)

- Package: `tabpfn` 9.0.0 (Apache-2.0, © 2026 Prior Labs GmbH; see its `NOTICE` and `THIRD-PARTY-NOTICES.md`).
- Weights: `tabpfn-v3.5-20260909.safetensors` (sha256 `ece4d67eadfea42eb0e610df5189bea60cb7f31073d81e9c7a019b76eacf0be3`).
  The package documentation says the weights are released under a **non-commercial licence**:
  https://huggingface.co/Prior-Labs/tabpfn_3_5/blob/main/LICENSE .
- The weights are **not redistributed**. `tabpfn` downloads them on first use to the user's cache. Each user
  accepts the model licence themselves (the package may ask for browser-based acceptance).
- Consequence: TabPFN vs Pareto/NBD code may be reused under Apache-2.0, but running it with TabPFN-3.5 weights is subject
  to the weight licence. Any commercial use of the weights needs separate terms from Prior Labs.

## Datasets (not included; fetched by `tabpfn-vs-paretonbd fetch`, SHA-256 verified)

| Dataset | Terms | Attribution |
|---|---|---|
| UCI Online Retail II | CC BY 4.0 | Chen, D. (2012). *Online Retail II* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5CG6D |
| CDNOW master and 1/10 sample (brucehardie.com) | no licence stated; posted publicly for research and teaching. **Not redistributed.** | Fader, P. S. and Hardie, B. G. S. (2001). Forecasting Repeat Sales at CDNOW: A Case Study. *Interfaces* 31(3), S94–S107. |
| Complete Journey (profiling only) | `completejourney` R package: CC0; data from 84.51° | Boehmke, B., `completejourney` R package; 84.51° *The Complete Journey*. |
| 1995 donation cohort (profiling only) | no licence stated. **Not redistributed.** | Fader, Hardie and Shang (2010). *Marketing Science* 29(6). |

Committed result files: for Online Retail II (CC BY 4.0), `results/online_retail_ii/` holds per-customer
predictions, labels and summary statistics, attributed as above. For CDNOW, only aggregate tables, figures and
diagnostics are committed. Per-customer CDNOW files (`labels.parquet`, `preds/`, `learning_curve/`) are
git-ignored and rebuilt locally after `tabpfn-vs-paretonbd fetch cdnow_master`.

## Method references (independent implementations, no code copied)

- Schmittlein, Morrison and Colombo (1987); Fader and Hardie (2005), "A Note on Deriving the Pareto/NBD Model and Related Expressions", for the likelihood,
  P(alive) and expected transactions. The implementation is checked against published CDNOW estimates.
- Abe, M. (2009). "Counting Your Customers" One by One: A Hierarchical Bayes Extension to the Pareto/NBD Model.
  *Marketing Science* 28(3), 541–553. Model M1 (no covariates) is implemented from the paper.

## Python dependencies

These are installed from PyPI by `uv sync` and locked in `uv.lock`. They are not vendored. Main runtime dependencies
and their declared licences: numpy, scipy, pandas, scikit-learn, python-dotenv (BSD-3-Clause); pyarrow, requests,
tabpfn, streamlit (Apache-2.0); torch 2.14 (metadata: Apache-2.0 and bundled components); lightgbm (MIT); matplotlib (PSF-based
Matplotlib licence); typer (MIT); openpyxl (MIT).
Run `uv pip list` or check each package's metadata for the exact versions and texts.
