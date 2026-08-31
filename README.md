# Causal Uplift Modeling for Targeted Marketing

Identifying *who a treatment actually persuades* — not just who is likely
to convert — using real randomized experiment data (Criteo-Uplift) and a
progression of causal ML methods from classical to tree-based.

## Problem Framing

Standard predictive targeting asks "who is likely to convert?" and ranks
customers accordingly. This is the wrong question for a targeting
decision: it ranks **sure things** (people who'd convert regardless of
treatment) just as highly as genuinely **persuadable** people, wasting
budget on the former and potentially missing the latter. It can also fail
to flag **sleeping dogs** — customers whose likelihood of converting
actually *decreases* under treatment.

The right question is causal: **who does the treatment change the
behavior of?** This project builds and compares four approaches to
answering that question, on real randomized experiment data, with an
emphasis on validating every result before trusting it.

## Data & Experiment Validity

**Dataset:** Criteo-Uplift Prediction Dataset (Criteo AI Lab) — ~13.9M
rows from a real, randomized advertising experiment. Users were randomly
exposed to advertising (`treatment=1`) or not (`treatment=0`), with
`conversion` as the binary outcome. Loaded via `scikit-uplift`'s built-in
fetcher for reliability (the direct download link is unstable).

Before trusting any causal estimate, two validity checks were run:

- **Sample Ratio Mismatch (SRM) check:** confirmed the observed 85/15
  treatment/control split matches the documented experimental design
  (chi-square p = 0.999, no mismatch detected).
- **Covariate balance check:** confirmed treatment and control groups are
  statistically comparable on all 12 pre-treatment features (all
  standardized mean differences well under the 0.1 threshold), meaning
  randomization worked as intended and any effect measured isn't
  confounded by pre-existing group differences.

Both checks passed, giving a solid foundation for the causal estimates
that follow.

## Methodology

Four approaches were built and compared, each addressing a specific
weakness in the one before it.

### 1. Naive Baseline (the mistake we set out to disprove)
A standard logistic regression predicting `P(conversion | features)`,
deliberately **excluding treatment** — representing the common real-world
mistake of targeting based on likelihood-to-convert alone.

- **AUC-ROC: 0.9405** — looks strong at first glance.
- **PR-AUC: 0.1902** — reveals the real story: on this rare-event problem
  (~0.29% conversion rate), ROC-AUC is inflated by the large number of
  easy true negatives. PR-AUC is the more honest metric here, and the gap
  between the two is a useful diagnostic for any rare-event classification
  problem.

### 2. T-learner
Two separate models — one trained on treated customers, one on control —
with the individual treatment effect estimated as the difference between
their predictions for the same customer.

- Initial version (logistic regression base learner) predicted a
  uniformly positive effect for 100% of customers — a limitation of
  linear models' inability to express heterogeneity, not evidence the ad
  helps everyone.
- Switched to a regularized XGBoost base learner (with a proper held-out
  test set), revealing a genuine negatively-affected subgroup (~13.45% of
  customers) that survived the fix to overfitting — evidence the finding
  is real, not an artifact of an unconstrained model.

### 3. X-learner
Extends the T-learner by imputing each customer's missing counterfactual
outcome using the *other* group's model, then combining both estimates
via propensity weighting.

- The propensity-weighting theory predicts `tau_control` (built from the
  majority treatment arm's response model) should be the more reliable,
  lower-variance component, and should therefore dominate the combined
  estimate under this dataset's 85/15 imbalance. Empirically, the
  opposite held: `tau_control` showed *wider* tails (higher std, larger
  range) than `tau_treatment` — the theoretical justification for trusting
  it more didn't hold up when checked directly.
- Despite that, X-learner still outperformed T-learner on the Qini metric
  (1058.6 vs 963.3 — see comparison below), so the propensity-weighted
  combination net-improved ranking quality even though its core
  reliability assumption broke down in isolation. A legitimate,
  dataset-specific finding about the gap between an uplift method's
  theoretical justification and its practical behavior.

### 4. Causal Forest
A tree-based method that splits directly on treatment effect
heterogeneity, rather than on prediction error — no propensity weighting
or imputation required. Trained on a stratified 2M-row subsample (see
Limitations).

- Raw CATE predictions showed a near-even 49.43%/50.57% positive/negative
  split — investigated further rather than taken at face value.
- Filtering to the ~0.71% of customers with a meaningfully large
  estimated effect (|CATE| > 0.05) revealed the real, actionable signal:
  **73.4% positive, 26.6% negative** — a small, high-confidence segment
  worth targeting (or actively avoiding), buried under a much larger
  population where the estimated effect is statistically present but
  practically negligible.
- Top heterogeneity-driving features: f9, f8, f4.

## Model Comparison — Qini Curve Evaluation

Accuracy-style metrics (precision/recall/AUC on conversion) measure the
wrong thing for this problem — a model can rank converters perfectly and
still be useless for targeting, because it says nothing about
*incremental* conversions caused by treatment. The **Qini curve**
answers the right question instead: if we targeted only the top X% of
customers ranked by a model's uplift score, how many extra conversions
would that have actually bought us, versus the control group's baseline
rate?

All four models were evaluated on the identical held-out test set
(2,795,919 rows) for a fair comparison.

| Model | Qini Coefficient |
|---|---|
| Naive baseline | 1110.0 |
| X-learner | 1058.6 |
| T-learner | 963.3 |
| Causal Forest | 618.5 |

**Two findings worth stating explicitly rather than glossing over:**

1. **The naive baseline scored highest.** Investigation confirmed this is
   real, not a bug: on this specific dataset, naive predicted conversion
   probability correlates 0.43–0.51 with each causal model's uplift
   score, because the segment of customers most likely to convert anyway
   substantially overlaps with the segment that's genuinely persuadable.
   This is a **dataset-specific finding, not a general result** — it
   should not be read as "naive targeting is a good strategy" in general,
   only that on Criteo's specific ad-response data, response propensity
   and treatment persuadability aren't as separable as the theory
   typically assumes.
2. **Causal Forest's last-place finish is confounded by a training-data
   disadvantage**, not a clean verdict on the method. It's the only
   model trained on 2M rows instead of the full ~11.2M (see Limitations)
   — roughly 18% of what the other three models received.

## Limitations & Future Work

- **Causal Forest was trained on a 2M-row subsample**, not the full
  ~11.2M-row training set, due to local compute constraints (an 8GB RAM
  machine). Two attempts to close this gap (a 5M-row run, and a
  full-scale run) both failed — the full-scale run hit a background
  execution timeout, and the 5M-row run's silent death (no exit code, no
  crash report, memory visibly tight beforehand) is most consistent with
  an out-of-memory kill, though this was not definitively confirmed.
  Given this, the Qini comparison's causal forest ranking should be read
  as directionally informative, not definitive, until re-run on a machine
  with more available memory.
- **TARNet** (a deep learning approach using a shared-representation
  neural network) was scoped as a comparison model but not implemented,
  given project time constraints. It remains a natural next step,
  particularly to test whether a more flexible representation surfaces
  different heterogeneity patterns than the tree-based Causal Forest.
- **An LLM-based reporting layer** (auto-generating plain-English
  business summaries from model output) was scoped but not built, for
  the same reason.
- A full engineering log of every issue hit during this project — from
  environment/dependency conflicts to methodology corrections — is kept
  in `DECISIONS_AND_ISSUES_LOG.md`, documenting not just what was built
  but why, and what was tried and rejected along the way.

## Business Takeaway

For a company running this exact experiment, the actionable
recommendation is: **don't target based on predicted conversion
probability alone.** Use a causal approach to separate persuadable
customers from sure-things and sleeping dogs. On this dataset, the
Causal Forest's magnitude-filtered analysis identified a specific,
high-confidence segment (~0.7% of customers) where targeting decisions
matter most — a small enough segment to act on cheaply, with a clear,
defensible 73/27 split between customers who benefit from the campaign
and customers who respond negatively to it.

## Repo Structure

```
causal-uplift-marketing/
├── README.md
├── DECISIONS_AND_ISSUES_LOG.md   # full engineering/methodology log
├── requirements.txt
├── data/
├── notebooks/
├── src/
│   ├── data/            # loading, SRM check, covariate balance check
│   ├── models/          # naive baseline, T-learner, X-learner, causal forest
│   └── evaluation/       # Qini curve computation and comparison plotting
├── reports/
└── figures/
    └── qini_comparison.png
```
