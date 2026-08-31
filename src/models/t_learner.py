"""
Phase 2: T-learner - the simplest uplift model that actually looks at treatment.

Unlike the naive baseline (a single P(conversion | features) model that never
sees treatment), a T-learner fits two separate response models - one on
control, one on treatment - and estimates each user's individual treatment
effect (ITE) as the difference between their predicted outcomes under each
model. This directly targets what we actually care about for a targeting
decision: who converts *because of* treatment, not just who's likely to
convert.

Known weakness: the control group here is the minority class (~15% of
users, since this experiment used an 85/15 treatment/control split), so
control_model is fit on far less data than treatment_model. Less data means
noisier probability estimates, which can inject spurious variance into the
ITE (a difference of two noisy estimates is noisier than either alone).
This is one of the standard motivations for X-learners and other approaches
that explicitly correct for treatment/control group-size imbalance.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

_RANDOM_STATE = 42


def train_t_learner(
    df: pd.DataFrame,
    feature_cols: list,
    treatment_col: str = "treatment",
    target_col: str = "conversion",
) -> tuple[XGBClassifier, XGBClassifier, pd.DataFrame]:
    """Train a T-learner: two independent response models, one per arm.

    df is first split into train/test (stratified on treatment_col, so both
    splits preserve the 85/15 treatment/control ratio). control_model and
    treatment_model are fit ONLY on the training portion - control_model on
    rows where treatment_col == 0, treatment_model on rows where
    treatment_col == 1. Neither model ever sees treatment_col as a feature -
    the treatment signal comes entirely from which model made the
    prediction.

    Evaluating predict_ite() on these same training rows would let each
    model's fit to its own training noise leak into the ITE, understating
    how much of that noise is real signal. Returning the untouched test_df
    alongside the fitted models makes it possible (and easy) to evaluate
    ITE and calibration on data neither model has seen.

    Uses XGBClassifier (gradient-boosted trees) as the base learner rather
    than a linear model, since a linear response surface has little room to
    flip sign across sub-populations - it mostly just shifts up or down,
    which is what produced a suspicious 100% positive / 0% negative ITE
    split with logistic regression. Trees can capture nonlinear
    interactions and are more likely to surface real treatment-effect
    heterogeneity, including any negatively-affected subgroup. Given the
    rare positive class (~0.3% conversion) and largely unbounded default
    tree depth, we constrain the trees (max_depth=4, min_child_weight=10,
    subsample=0.8, colsample_bytree=0.8) so a handful of rare-event leaves
    can't dominate the fit and inflate ITE into implausibly extreme values.

    Returns (control_model, treatment_model, test_df), where test_df is the
    held-out split (still containing treatment_col and target_col) that
    predict_ite() and calibration checks should be run against.
    """
    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=_RANDOM_STATE, stratify=df[treatment_col]
    )

    control_df = train_df[train_df[treatment_col] == 0]
    treatment_df = train_df[train_df[treatment_col] == 1]

    xgb_params = dict(
        eval_metric="logloss",
        max_depth=4,
        min_child_weight=10,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=_RANDOM_STATE,
    )

    control_model = XGBClassifier(**xgb_params)
    control_model.fit(control_df[feature_cols], control_df[target_col])

    treatment_model = XGBClassifier(**xgb_params)
    treatment_model.fit(treatment_df[feature_cols], treatment_df[target_col])

    return control_model, treatment_model, test_df


def predict_ite(
    control_model: XGBClassifier, treatment_model: XGBClassifier, X: pd.DataFrame
) -> np.ndarray:
    """Estimate individual treatment effect (ITE) for each row of X.

    Both models score the SAME input X, so the only thing that differs
    between the two predictions is which arm's response surface produced
    them. ITE = P(conversion | X, treatment_model) - P(conversion | X, control_model).

    Also prints summary stats of the resulting ITE distribution.
    """
    p_control = control_model.predict_proba(X)[:, 1]
    p_treatment = treatment_model.predict_proba(X)[:, 1]
    ite = p_treatment - p_control

    pct_positive = (ite > 0).mean() * 100
    pct_negative = (ite < 0).mean() * 100

    print(f"ITE mean: {ite.mean():.4f}")
    print(f"ITE min:  {ite.min():.4f}")
    print(f"ITE max:  {ite.max():.4f}")
    print(f"% positive effect: {pct_positive:.2f}%")
    print(f"% negative effect: {pct_negative:.2f}%")

    return ite
