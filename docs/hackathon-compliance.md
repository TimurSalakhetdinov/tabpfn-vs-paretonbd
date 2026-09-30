# Hackathon compliance: TabPFN-3.5 Hackathon (Prior Labs)

Terms source: https://platform.priorlabs.ai/hackathon-3.5 ("Terms & Conditions" dialog). They were re-read on
**2026-09-29**, extracted from the page's published JS bundle because the page renders client-side. The SHA-256 of the
terms text as read is `907131e56d9f547407d15b33e3248645fbb020c71b46fc54f97d82c8e29f2a2d`.
On 2026-09-30 every clause was compared again with a saved copy of that text (same hash); no requirement was missed.
**Re-check before submitting**: the organiser may amend the terms (clause 7.7).

Status key: **pass** = satisfied and evidenced in this repo · **pending** = needs a human action or a later
step · **n/a** = not applicable.

| Clause | Requirement (paraphrased) | Status | Evidence |
|---|---|---|---|
| 2.1–2.2 | Participant is 18+ with one valid Prior Labs account | **pass** | Confirmed by the participant on 2026-09-30: 18 or older, with one Prior Labs account (stimur55@gmail.com). The agent did not create or use an account |
| 2.4 | Accept the terms to participate | **pass** | Accepted by the participant in the browser (confirmed 2026-09-30); the agent did not accept them |
| 3.1 | Built with TabPFN-3.5 as a core part | **pass** | `models/tabular.py::tabpfn` uses `ModelVersion.V3_5` classifier and regressor; verification of the checkpoint and version in `docs/evidence/tabpfn35-verification.json`; TabPFN is the reference model in every comparison, the learning curve and the app |
| 3.2 | Repo contains code and instructions to run; input data included or at a public URL | **pass (local)** / **pending (public repo)** | README quickstart; `tabpfn-vs-paretonbd fetch` downloads from the public URLs listed in `docs/dataset-selection.md` with SHA-256 checks; `tabpfn-vs-paretonbd smoke` run on 2026-09-30 with TabPFN-3.5 included (26 tests passed, every model fitted on a 300-customer sample, 36 s); all app pages rendered headlessly on both datasets with no exceptions; saved predictions under `results/` regenerate every table with `tabpfn-vs-paretonbd report` |
| 3.3 | Rights to the code and data used and published | **pass, with disclosed limits** | Code: our own, Apache-2.0. Online Retail II: CC BY 4.0, attributed. CDNOW: no licence stated, so it is **not redistributed**, and only aggregate results are committed (`.gitignore`, `THIRD_PARTY_NOTICES.md`). TabPFN weights are not redistributed (non-commercial weight licence). All code is written for this project; no third-party code is vendored |
| 3.4 | API credits | **n/a** | Runs locally with downloaded weights; no API calls and no cost |
| 3.5(i) | Link to a **public** source repository under **Apache-2.0** | **pass (licence)** / **pending (public URL)** | `LICENSE` (Apache-2.0), `pyproject.toml` `license = "Apache-2.0"`. The repo is local only; publishing needs the user's go-ahead |
| 3.5(ii) | Description detailed enough for a third-party developer | **pass** | `docs/submission-description.md`, 4,918 characters with both datasets' holdout results (≤ 5,000; the form enforces `max(5000)`, checked in the page bundle) |
| 3.5 optional | Video link, social handles | **pending (human)** | Script ready in `docs/demo-script.md`; recording and upload are left to the user |
| 3.6 | Submit by **6 Oct 2026, 23:59 CEST** | **pending (human)** | Package targets reviewable by 3 Oct and frozen by 5 Oct |
| 3.7 | Latest submission counts; separate entries per project | n/a | One entry |
| 4.2 | Judging: showcase 50 %, creativity and value 30 %, technical quality and reproducibility 20 % | — | See "How the entry addresses the criteria" below |
| 5.2 | Organiser may share the entry with attribution | noted | Nothing in the repo prevents this; CDNOW rows are not included |

**Submission form fields** (from the page bundle): full name, country (optional), repository URL, project
description (1–5,000 characters), YouTube URL (optional), X handle (optional), LinkedIn URL (optional).

## How the entry addresses the criteria (claims limited to what is measured)

- **Showcase (50 %)**: TabPFN-3.5 is used for both targets, a classifier for 90-day inactivity and a regressor for
  the purchase count. It is used without tuning and compared on a frozen, untouched holdout against tuned
  LightGBM, logistic regression and two Bayesian customer-base models. There is also a learning curve over labelled
  training customers.
- **Creativity and value (30 %)**: a benchmark of a tabular foundation model against a hierarchical
  Bayesian marketing model (Abe 2009), with an inspectable app for retention analysts. It makes no causal or uplift
  claims.
- **Technical quality (20 %)**: locked environment (`uv.lock`), hashed data, mature-label guards, tests, saved
  predictions and a manifest per split, and one command to regenerate every table from the saved predictions.

## Not claimed

We do not claim 100 % compliance while the rows above are pending, we do not claim a prize, and we do not claim
that TabPFN "wins": see the measured results in the README.
