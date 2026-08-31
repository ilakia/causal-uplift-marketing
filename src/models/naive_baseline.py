"""
Phase 1: Naive baseline - the mistake we're setting out to disprove.

A response model (P(conversion | features)) tells you who is likely to
convert, but says nothing about who converts *because of* treatment. Ranking
users by this score and targeting the top of the list is a common real-world
approach, but it targets users who were going to convert anyway (or even
users treatment would repel) just as readily as it targets the persuadable
ones. Later phases build uplift models (e.g. T-learner, X-learner, causal
trees) that estimate the treatment effect directly and are what should
actually drive a targeting decision.
"""

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split

_RANDOM_STATE = 42


def train_naive_baseline(
    df: pd.DataFrame,
    feature_cols: list,
    treatment_col: str = "treatment",
    target_col: str = "conversion",
) -> tuple[LogisticRegression, pd.DataFrame]:
    """Train the naive "ignore treatment" baseline.

    This is intentionally the wrong approach for targeting decisions: it
    fits a plain response model, P(conversion | features), using only
    feature_cols and never sees the treatment column. It will happily rank
    a user highest even if that user would have converted regardless of
    treatment (or would convert less if treated) - because it has no
    concept of a counterfactual. We build this on purpose, as the baseline
    later uplift models (T-learner, X-learner, causal trees, etc.) need to
    beat to justify their added complexity.

    Uses the same train/test split (80/20, stratified on treatment_col,
    random_state=42) as t_learner.py, x_learner.py, and causal_forest.py -
    given identical inputs, this produces the exact same held-out test_df,
    so this model's predictions land on the same rows as the other three
    and are directly comparable (e.g. on the same Qini curve). Stratifying
    on treatment_col rather than target_col is what makes that possible;
    it costs nothing here since the model never uses treatment_col as a
    feature anyway.

    Returns (model, test_df), and prints AUC-ROC and PR-AUC (average
    precision) on that held-out test set. PR-AUC matters here because
    conversion is rare (well under 1% base rate): ROC-AUC can look strong
    even when the model isn't very useful for picking out the small
    positive class, since it's diluted by the (easy) true-negative
    majority. PR-AUC is more sensitive to that imbalance and gives a more
    honest read on ranking quality for the converters we actually care
    about.
    """
    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=_RANDOM_STATE, stratify=df[treatment_col]
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(train_df[feature_cols], train_df[target_col])

    y_pred_proba = model.predict_proba(test_df[feature_cols])[:, 1]
    auc = roc_auc_score(test_df[target_col], y_pred_proba)
    pr_auc = average_precision_score(test_df[target_col], y_pred_proba)
    print(f"Naive baseline AUC-ROC: {auc:.4f}")
    print(f"Naive baseline PR-AUC:  {pr_auc:.4f}")

    return model, test_df
