"""
Phase 0: Data loading and initial validity checks.

Before we model anything, we must verify the experiment itself is trustworthy.
This is where your A/B testing knowledge plugs in directly:
    - Sample Ratio Mismatch (SRM) check: are treatment/control split as expected?
    - Covariate balance: do treatment and control groups look similar on
      pre-treatment features? (if not, randomization may be broken)

Data loading uses scikit-uplift's built-in fetcher, which downloads and
caches the Criteo-Uplift dataset automatically (no manual CSV download
needed - avoids the flaky direct-download link from Criteo's own servers).

TODO (we'll build this together, step by step):
    1. load_criteo_data() -> DONE, uses sklift's fetch_criteo
    2. check_sample_ratio_mismatch() -> chi-square test on treatment counts
    3. check_covariate_balance() -> compare feature distributions by group
"""

from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import pandas as pd
from scipy.stats import chisquare

# scikit-uplift's fetch_criteo() points at an S3 bucket that now returns
# 403 Forbidden, so we pull from Criteo's own official mirror instead and
# cache it at the path data/README.md already documents for manual download.
_CRITEO_URL = "https://criteostorage.blob.core.windows.net/criteo-research-datasets/criteo-uplift-v2.1.csv.gz"
_RAW_DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "raw" / "criteo_uplift.csv.gz"

_DTYPES = {"exposure": "Int8", "treatment": "Int8", "conversion": "Int8", "visit": "Int8"}


def load_criteo_data() -> pd.DataFrame:
    """Load the Criteo-Uplift dataset, downloading it if not already cached.

    First call downloads and caches the data locally (~300MB compressed) at
    data/raw/criteo_uplift.csv.gz; subsequent calls load from the local
    cache instantly.

    Returns a single DataFrame with columns f0-f11 (features), treatment,
    conversion, visit, exposure.
    """
    if not _RAW_DATA_PATH.exists():
        _RAW_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        urlretrieve(_CRITEO_URL, _RAW_DATA_PATH)

    return pd.read_csv(_RAW_DATA_PATH, dtype=_DTYPES)


def check_sample_ratio_mismatch(
    df: pd.DataFrame, treatment_col: str = "treatment", expected_treatment_ratio: float = 0.85
) -> dict:
    """Check whether the treatment/control split matches the expected ratio.

    An SRM here would mean something is wrong with the randomization itself -
    a critical pre-check before trusting any downstream causal estimate.
    """
    n_treatment = int((df[treatment_col] == 1).sum())
    n_control = int((df[treatment_col] == 0).sum())
    n_total = n_treatment + n_control

    observed = [n_treatment, n_control]
    expected = [n_total * expected_treatment_ratio, n_total * (1 - expected_treatment_ratio)]

    chi2_stat, p_value = chisquare(f_obs=observed, f_exp=expected)

    return {
        "observed": {"treatment": n_treatment, "control": n_control},
        "expected": {"treatment": expected[0], "control": expected[1]},
        "chi2_statistic": chi2_stat,
        "p_value": p_value,
        "srm_detected": p_value < 0.001,
    }


def check_covariate_balance(df: pd.DataFrame, feature_cols: list, treatment_col: str = "treatment") -> pd.DataFrame:
    """Compare feature distributions between treatment and control groups.

    If groups differ significantly on pre-treatment features, randomization
    may not have worked as intended (or there's selection bias to account for).

    We use the standardized mean difference (SMD) here rather than a t-test.
    With a dataset this large (~14M rows, 85/15 split still leaves millions
    per arm), a t-test's p-value is a function of sample size as much as
    effect size: even a trivial, practically meaningless difference in means
    will come back as "significant" (p < 0.001) simply because n is huge.
    SMD instead measures effect size directly, independent of sample size,
    so it tells us whether an imbalance is large enough to matter rather
    than just detectable.
    """
    treatment_mask = df[treatment_col] == 1
    control_mask = df[treatment_col] == 0

    rows = []
    for col in feature_cols:
        treatment_vals = df.loc[treatment_mask, col]
        control_vals = df.loc[control_mask, col]

        mean_treatment = treatment_vals.mean()
        mean_control = control_vals.mean()
        pooled_std = np.sqrt((treatment_vals.std() ** 2 + control_vals.std() ** 2) / 2)

        smd = (mean_treatment - mean_control) / pooled_std

        rows.append(
            {
                "feature": col,
                "mean_treatment": mean_treatment,
                "mean_control": mean_control,
                "smd": smd,
                "balanced": abs(smd) < 0.1,
            }
        )

    return pd.DataFrame(rows).set_index("feature")
