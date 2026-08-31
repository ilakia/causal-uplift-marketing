"""
Phase 3: Causal Forest - tree-based CATE estimation via econml's CausalForestDML.

Unlike the X-learner, which estimates ITE by combining tau_control(x) and
tau_treatment(x) using an estimated propensity score P(treatment=1|x) as a
proxy for how much to trust each meta-learner in different regions of
feature space, a causal forest never estimates or weights by propensity at
all. It grows trees that split directly on treatment-effect heterogeneity:
each split is chosen (via the generalized random forest's honest,
gradient-based splitting criterion) to maximize the difference in
treatment effect between the resulting child nodes, so leaves are regions
of feature space with similar CATE - not regions with similar outcome or
similar propensity. A unit's predicted effect is then a similarity-
weighted local average of honest treatment-effect estimates from its
neighboring trees.

This sidesteps the X-learner's whole imbalanced-arms correction machinery
(propensity-weighted combination of two separately-fit, noisy imputed-
effect regressions) since heterogeneity is what the forest is built to
find directly, not a byproduct of combining two other models.

We use econml's CausalForestDML, which residualizes conversion and
treatment against features using first-stage nuisance models (model_y,
model_t) before fitting the forest on the residuals - the "double machine
learning" (DML) part, and what makes the forest's splits about treatment
effect rather than about predicting conversion or treatment directly.
Since assignment here is fully randomized (see Phase 1's SRM and
covariate-balance checks), model_t's job is comparatively easy - it
should recover something close to the constant ~85% treatment rate seen
in x_learner.py's propensity model; model_y still needs to be a real
predictor of conversion so its residuals capture the outcome variation
that isn't explained by features alone.
"""

import numpy as np
import pandas as pd
from econml.dml import CausalForestDML
from joblib import parallel_backend
from sklearn.model_selection import train_test_split
from threadpoolctl import threadpool_limits
from xgboost import XGBClassifier

_RANDOM_STATE = 42

_XGB_CLASSIFIER_PARAMS = dict(
    eval_metric="logloss",
    max_depth=4,
    min_child_weight=10,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=_RANDOM_STATE,
)

# CausalForestDML's cross-fitting (nuisance models) and tree building both
# go through joblib.Parallel, which by default schedules work across
# multiple native thread/process pools (its own "loky" workers, plus each
# worker independently spinning up xgboost's OpenMP pool and numpy's
# BLAS pool). On macOS those pools collide - concurrent entry into two
# different libraries' thread pools from the same process corrupts memory
# and segfaults deep in native code (observed both in xgboost's DMatrix
# construction and econml's Cython tree fit, depending on which pool got
# there first). Neither switching joblib to threading-only nor pinning
# n_jobs=1 on the individual models was sufficient on its own - only
# collapsing every native pool down to a single thread (threadpoolctl,
# which reaches into already-loaded BLAS/OpenMP libraries regardless of
# import order) reliably fixed it. This trades away the forest's
# parallelism for stability; it's still fast enough here because the
# dataset only has 12 features.
_PARALLEL_BACKEND = "threading"


def train_causal_forest(
    df: pd.DataFrame,
    feature_cols: list,
    treatment_col: str = "treatment",
    target_col: str = "conversion",
    sample_size: int = None,
) -> tuple[CausalForestDML, pd.DataFrame]:
    """Train a causal forest: DML-residualized nuisance models + an honest forest.

    Uses the same train/test split (80/20, stratified on treatment_col,
    random_state=42) as t_learner.py and x_learner.py, computed from the
    FULL df before any subsampling - given identical inputs, this produces
    the exact same held-out test_df regardless of sample_size, so results
    stay directly comparable across all three methods.

    If sample_size is given, the training split (not test_df) is further
    subsampled down to that many rows, stratified on treatment_col to
    preserve the ~85/15 treatment/control ratio (random_state=42). This
    exists because our thread-pool workaround for the OpenMP crash (see
    _PARALLEL_BACKEND/threadpool_limits below) forces the whole fit to run
    single-threaded, and a single-threaded honest-forest fit over the full
    ~11.2M-row training split is impractically slow - 2M rows keeps the
    forest fast enough to run in a normal foreground call while still
    being a large, representative training set.

    model_y and model_t are the same regularized XGBClassifier setup used
    for the response models in t_learner.py/x_learner.py (max_depth=4,
    min_child_weight=10, subsample=0.8, colsample_bytree=0.8), predicting
    conversion and treatment respectively from feature_cols; discrete_outcome
    and discrete_treatment tell CausalForestDML to residualize using each
    model's predicted probabilities rather than raw regression output.

    The forest itself is left at econml's defaults (honest=True,
    min_samples_split=10, min_samples_leaf=5, n_estimators=100): honesty
    (splitting on one subsample, estimating effects on another) plus a
    minimum leaf size is how GRF-style forests regularize instead of
    capping tree depth the way the boosted response models do - depth
    caps aren't the standard recommendation here since averaging across
    many honest trees is what controls variance.

    Prints the forest's feature importances (how much each feature
    contributes to treatment-effect heterogeneity, not to predicting
    conversion) right after fitting, since they're a property of the
    fitted model rather than of any particular evaluation set.

    Returns (model, test_df), where test_df is the held-out split (still
    containing treatment_col and target_col) for evaluation via
    predict_cate().
    """
    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=_RANDOM_STATE, stratify=df[treatment_col]
    )

    if sample_size is not None and sample_size < len(train_df):
        train_df, _ = train_test_split(
            train_df, train_size=sample_size, random_state=_RANDOM_STATE, stratify=train_df[treatment_col]
        )

    model = CausalForestDML(
        model_y=XGBClassifier(**_XGB_CLASSIFIER_PARAMS),
        model_t=XGBClassifier(**_XGB_CLASSIFIER_PARAMS),
        discrete_outcome=True,
        discrete_treatment=True,
        random_state=_RANDOM_STATE,
    )
    with parallel_backend(_PARALLEL_BACKEND), threadpool_limits(limits=1):
        model.fit(
            Y=train_df[target_col],
            T=train_df[treatment_col],
            X=train_df[feature_cols],
        )

    print_feature_importances(model, feature_cols)

    return model, test_df


def predict_cate(model: CausalForestDML, X: pd.DataFrame) -> np.ndarray:
    """Estimate conditional average treatment effect (CATE) for each row of X.

    Unlike predict_ite()/predict_ite_x(), this doesn't combine two
    separately-fit response (or effect) models - the forest was trained
    to output a treatment-effect estimate directly, so this is a single
    call to the fitted model rather than an arithmetic combination step.

    Also prints summary stats of the resulting CATE distribution, in the
    same format as predict_ite()/predict_ite_x(), for direct comparison
    against the T-learner and X-learner on the same held-out test set.
    """
    with parallel_backend(_PARALLEL_BACKEND), threadpool_limits(limits=1):
        cate = model.effect(X)

    # econml returns effect() as shape (n, 1) for a single outcome/treatment;
    # flatten to 1D to match predict_ite()/predict_ite_x()'s return shape.
    cate = cate.ravel()

    pct_positive = (cate > 0).mean() * 100
    pct_negative = (cate < 0).mean() * 100

    print(f"CATE mean: {cate.mean():.4f}")
    print(f"CATE min:  {cate.min():.4f}")
    print(f"CATE max:  {cate.max():.4f}")
    print(f"% positive effect: {pct_positive:.2f}%")
    print(f"% negative effect: {pct_negative:.2f}%")

    return cate


def print_feature_importances(model: CausalForestDML, feature_cols: list) -> pd.Series:
    """Print and return the forest's feature importances, sorted descending.

    These measure how much each feature drives treatment-effect
    heterogeneity (i.e. how often/how impactfully it's split on), which is
    a different question from how much a feature predicts conversion -
    the causal forest never fits a model of conversion directly.
    """
    importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)

    print("Feature importances (drivers of treatment-effect heterogeneity):")
    for feature, importance in importances.items():
        print(f"  {feature}: {importance:.4f}")

    return importances
