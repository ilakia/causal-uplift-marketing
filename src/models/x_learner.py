"""
Phase 3: X-learner - closing the T-learner's variance gap on an imbalanced split.

The T-learner estimates ITE as a straight difference of two independently
fit response models, control_model and treatment_model. On our 85/15
treatment/control split, control_model is fit on the minority class, so it
is the noisier of the two - and that noise flows directly into every ITE
estimate everywhere in feature space, since the T-learner has no way to
lean more on one model than the other.

The X-learner fixes this in two steps:

1. It imputes an individual treatment effect for every unit using the
   OTHER arm's outcome model as a stand-in for the missing counterfactual
   (treated units: actual outcome minus control_model's prediction; control
   units: treatment_model's prediction minus actual outcome), then fits a
   second-stage regression on each set of imputed effects (tau_treatment,
   tau_control).

2. It combines tau_control(x) and tau_treatment(x) using the propensity
   score P(treatment=1|x), rather than averaging them blindly.

Why this specifically helps on an imbalanced split like ours: tau_control
is fit on effects imputed as treatment_model(X_control) - y_control. Since
treatment_model was trained on the LARGE (85%) treatment group, it
generalizes well almost everywhere in feature space, including the region
where control units live - so tau_control's imputed training targets are
built on a reliable base and tau_control itself tends to be trustworthy
across most of the input space. tau_treatment, by contrast, is fit on
effects imputed as y_treated - control_model(X_treated); control_model was
trained on the SMALL (15%) control group, so in regions of feature space
dominated by treated units it has to extrapolate, making tau_treatment's
training targets - and therefore tau_treatment itself - less reliable
exactly where treatment is most prevalent.

The propensity-weighted combination, tau(x) = p(x) * tau_control(x) +
(1 - p(x)) * tau_treatment(x), automatically leans on tau_control (the more
reliable estimator here) in proportion to how likely a unit was to be
treated. With random assignment at a near-constant ~85% treatment rate,
that means roughly 85% weight on the more reliable tau_control and only
~15% on the noisier tau_treatment - directly correcting for the arm-size
imbalance that made the T-learner's control_model noisy in the first place.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier, XGBRegressor

_RANDOM_STATE = 42

_XGB_CLASSIFIER_PARAMS = dict(
    eval_metric="logloss",
    max_depth=4,
    min_child_weight=10,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=_RANDOM_STATE,
)

_XGB_REGRESSOR_PARAMS = dict(
    max_depth=4,
    min_child_weight=10,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=_RANDOM_STATE,
)


def train_x_learner(
    df: pd.DataFrame,
    feature_cols: list,
    treatment_col: str = "treatment",
    target_col: str = "conversion",
) -> tuple[dict, pd.DataFrame]:
    """Train an X-learner: two response models, two effect models, one propensity model.

    Uses the same train/test split (80/20, stratified on treatment_col,
    random_state=42) and the same regularized XGBoost setup as
    t_learner.py - given identical inputs, this produces the exact same
    held-out test_df as train_t_learner(), so results are directly
    comparable.

    Stage 1: control_model and treatment_model are fit exactly like the
    T-learner, on the control and treatment subsets of the training split.

    Stage 2: imputed treatment effects -
        treated units: d_treated = y_treated - control_model(X_treated)
        control units: d_control = treatment_model(X_control) - y_control

    Stage 3: tau_treatment and tau_control are regressions (XGBRegressor)
    fit on those imputed effects (tau_treatment on treated units'
    features -> d_treated, tau_control on control units' features ->
    d_control). propensity_model is a classifier predicting treatment_col
    from feature_cols, used to combine tau_control/tau_treatment in
    predict_ite_x().

    Returns (models, test_df), where models is a dict with keys
    'control_model', 'treatment_model', 'tau_control', 'tau_treatment',
    'propensity_model', and test_df is the held-out split (still containing
    treatment_col and target_col) for evaluation.
    """
    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=_RANDOM_STATE, stratify=df[treatment_col]
    )

    control_df = train_df[train_df[treatment_col] == 0]
    treatment_df = train_df[train_df[treatment_col] == 1]

    # Stage 1: response models, one per arm.
    control_model = XGBClassifier(**_XGB_CLASSIFIER_PARAMS)
    control_model.fit(control_df[feature_cols], control_df[target_col])

    treatment_model = XGBClassifier(**_XGB_CLASSIFIER_PARAMS)
    treatment_model.fit(treatment_df[feature_cols], treatment_df[target_col])

    # Stage 2: impute each unit's treatment effect using the other arm's model.
    d_treated = treatment_df[target_col] - control_model.predict_proba(treatment_df[feature_cols])[:, 1]
    d_control = treatment_model.predict_proba(control_df[feature_cols])[:, 1] - control_df[target_col]

    # Stage 3: regress imputed effects on features, one model per arm.
    tau_treatment = XGBRegressor(**_XGB_REGRESSOR_PARAMS)
    tau_treatment.fit(treatment_df[feature_cols], d_treated)

    tau_control = XGBRegressor(**_XGB_REGRESSOR_PARAMS)
    tau_control.fit(control_df[feature_cols], d_control)

    # Propensity model: P(treatment=1 | X), used to weight tau_control vs tau_treatment.
    propensity_model = XGBClassifier(**_XGB_CLASSIFIER_PARAMS)
    propensity_model.fit(train_df[feature_cols], train_df[treatment_col])

    models = {
        "control_model": control_model,
        "treatment_model": treatment_model,
        "tau_control": tau_control,
        "tau_treatment": tau_treatment,
        "propensity_model": propensity_model,
    }

    return models, test_df


def predict_ite_x(models: dict, X: pd.DataFrame) -> np.ndarray:
    """Estimate individual treatment effect (ITE) for each row of X.

    Combines tau_control(x) and tau_treatment(x) using the propensity
    score p(x) = P(treatment=1 | x):

        ITE(x) = p(x) * tau_control(x) + (1 - p(x)) * tau_treatment(x)

    This leans on tau_control (fit using the well-generalizing
    treatment_model's imputations) in proportion to how likely a unit was
    to be treated, and on tau_treatment (fit using the more extrapolation-
    prone control_model's imputations) otherwise - see module docstring
    for why that specifically helps when treatment/control is imbalanced.

    Also prints summary stats of the resulting ITE distribution.
    """
    p = models["propensity_model"].predict_proba(X)[:, 1]
    tau_c = models["tau_control"].predict(X)
    tau_t = models["tau_treatment"].predict(X)

    ite = p * tau_c + (1 - p) * tau_t

    pct_positive = (ite > 0).mean() * 100
    pct_negative = (ite < 0).mean() * 100

    print(f"ITE mean: {ite.mean():.4f}")
    print(f"ITE min:  {ite.min():.4f}")
    print(f"ITE max:  {ite.max():.4f}")
    print(f"% positive effect: {pct_positive:.2f}%")
    print(f"% negative effect: {pct_negative:.2f}%")

    return ite
