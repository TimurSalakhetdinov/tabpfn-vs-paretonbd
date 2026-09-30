# Submission description (paste into the form, ≤ 5,000 characters)

<!-- The text between the markers is what goes into the form. Check its length with:
     python3 -c "s=open('docs/submission-description.md').read(); print(len(s.split('<!-- BEGIN -->')[-1].split('<!-- END -->')[0].strip()))" -->

<!-- BEGIN -->
TabPFN vs Pareto/NBD: when does TabPFN-3.5 beat specialised Bayesian customer models?

Retention analysts often model repeat buying with Pareto/NBD, a 1987 probabilistic customer-base model that needs no labels, and its hierarchical Bayesian extension (Abe 2009). This project asks when TabPFN-3.5, learning in context from labelled examples, predicts better, and how that depends on the inputs and the number of labelled customers.

TASK
At monthly forecast origins, for every existing customer, predict (1) 90-day inactivity risk, P(no purchase occasion in the next 90 days), shown with its complement, the probability of returning; and (2) the number of purchase occasions in those 90 days. In noncontractual retail this is not "churn", and nothing here estimates the effect of contacting a customer.

DATA
Primary: UCI Online Retail II (CC BY 4.0), a UK online wholesaler, 2009–2011; about 5,000 customers per origin. Secondary: the CDNOW full cohort (Fader & Hardie), the Bayesian models' home dataset (23,570 customers), which is fetched by the user and never redistributed. Line items are aggregated to customer-days; cancellations, missing IDs and duplicates are dropped; raw files are SHA-256 checked.

MODELS
TabPFN-3.5 classifier and regressor (ModelVersion.V3_5, default settings, no tuning, local MPS); base-rate and recency references; logistic/Poisson GLM; LightGBM (6-configuration grid on development data only); Pareto/NBD by maximum likelihood; and Abe's (2009) hierarchical Bayesian Pareto/NBD by MCMC, published only if every split-R-hat < 1.1. The Bayesian models use no labels: they are fitted to all customers' unlabelled histories. Supervised models are compared on the Pareto/NBD inputs only (frequency, recency, customer age), then on a predeclared expanded set (recent windows, gaps, trends, monetary history).

PROTOCOL
Features use history strictly before the origin. Training labels come from the latest origin whose 90-day window has fully ended, and immature rows are excluded and counted. The design (horizon, features, grid, context cap, convergence rule) was frozen on development origins; three holdout origins were then scored once. Uncertainty comes from a paired bootstrap that resamples customers across all origins. Learning curves vary labelled training customers from 50 to 2,000, with three seeds.

MEASURED RESULTS (holdout)
Online Retail II (3 origins, 15,433 customer-origin rows, 61.8 % inactive). Differences are other model minus TabPFN, with 95 % CIs.
- Same three inputs: the label-free Pareto/NBD beats TabPFN-3.5 on Brier, 0.169 vs 0.177 (−0.008 [−0.009, −0.007]), and is best calibrated.
- Expanded inputs: TabPFN-3.5 (0.1684) ties with the GLM, LightGBM and Pareto/NBD (all CIs include 0). It has the best PR-AUC (0.858 at prevalence 0.618) and the best count MAE (0.561; others 0.576–0.608, all significant).
- Learning curve: TabPFN-3.5 is the best supervised model at every size from 50 to 2,000 labelled customers (Brier 0.190 vs 0.208 GLM and 0.239 LightGBM at 50) and reaches the Pareto/NBD level at about 2,000.
- Weaknesses: TabPFN-3.5 has the worst calibration (ECE 0.06–0.08) and under-forecasts total purchases by 17–20 %, because of prevalence drift between the training and target periods.
- Abe HB did not converge here (R-hat 1.48), so it is not scored.
CDNOW (70,710 rows): Abe HB converges and wins on the common inputs (Brier 0.0968 vs 0.0980; count MAE 0.298 vs 0.333). With expanded features TabPFN-3.5 edges ahead on Brier (0.0964); the Bayesian models keep the best count forecasts.
Answer: TabPFN-3.5 wins when it can use richer features and when labels are scarce. On the Pareto/NBD's own inputs, the Bayesian models are as good or better.

WHAT YOU CAN RUN
`uv sync --frozen`, `tabpfn-vs-paretonbd fetch online_retail_ii`, then `tabpfn-vs-paretonbd smoke` (tests plus every model on a 300-customer sample, a few minutes) and `tabpfn-vs-paretonbd app`: a local Streamlit workbench that shows one customer's history at a past origin, every model's prediction next to what happened, calibration, decile lift, compute cost and convergence failures. `tabpfn-vs-paretonbd run <dataset> holdout` regenerates everything; `tabpfn-vs-paretonbd report` rebuilds every table from the saved predictions. Every model family runs in its own subprocess.

LIMITS
One retailer and one period; the holdout origins overlap in time, so they are not independent. TabPFN sees at most 10,000 training rows (whole customers), which matters on CDNOW. It ranks risk, not persuadability: there is no treatment data, so there are no uplift or ROI claims. Code is Apache-2.0; TabPFN-3.5 weights are under Prior Labs' non-commercial licence and are downloaded by the user.

The author previously replicated Abe (2009) without TabPFN; this project is a new codebase whose Bayesian models are implemented from the papers and validated against published CDNOW estimates.
<!-- END -->
