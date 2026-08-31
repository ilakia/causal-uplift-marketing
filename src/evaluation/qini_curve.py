"""
Phase 5: Qini curve - evaluating uplift models by ranking, not accuracy.

Accuracy/precision/recall on conversion prediction measures the wrong thing
for this problem: a model can rank converters perfectly and still be useless
for targeting, because it says nothing about *incremental* conversions -
the ones treatment actually caused. The Qini curve instead asks: if we
targeted only the top X% of customers as ranked by a model's uplift score,
how many extra conversions would that have bought us, versus what the
control group's baseline rate would predict for that same number of
treated customers?

At each percentile x (customers sorted by predicted uplift, descending):

    Qini(x) = (treatment conversions in top x)
              - (control conversions in top x) * (treatment count in top x / control count in top x)

The second term rescales the control group's observed conversions up to
treatment-group size, giving an estimate of how many conversions the top-x
treated customers would have produced anyway, absent treatment. The
difference is the incremental gain actually attributable to treatment
within that slice - this only works because treatment/control assignment
is randomized (see Phase 1's SRM/covariate-balance checks), so control's
conversion rate is a valid stand-in for "what treatment customers would
have done without treatment."

The Qini coefficient collapses the whole curve to one number: the area
between the model's curve and the random-targeting diagonal (a straight
line from (0%, 0) to (100%, total incremental gain) - what you'd expect,
in expectation, from targeting a random x% rather than using the model's
ranking at all). A model that ranks no better than random scores ~0; a
model that concentrates persuadable customers at the top of its ranking
scores higher.

A naive response model (P(conversion | features), never trained on
treatment - see naive_baseline.py) has nothing to say about incremental
effect, but you can still feed its predicted conversion probability into
compute_qini_curve() as if it were an uplift score. That's the point:
doing so demonstrates concretely (not just by assertion) that ranking by
"who's likely to convert" performs close to random on the metric that
actually matters for targeting, which is why t_learner.py, x_learner.py,
and causal_forest.py exist.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_FIGURES_DIR = Path(__file__).resolve().parents[2] / "figures"

# Fixed categorical order (dataviz skill's validated default palette) - hues
# are assigned by position in results_dict, never re-picked per curve's rank,
# so the same model always gets the same color across repeated runs.
_CATEGORICAL_COLORS = [
    "#2a78d6",  # blue
    "#eb6834",  # orange
    "#1baf7a",  # aqua
    "#eda100",  # yellow
    "#e87ba4",  # magenta
    "#008300",  # green
    "#4a3aa7",  # violet
    "#e34948",  # red
]
_MUTED_INK = "#898781"
_SECONDARY_INK = "#52514e"
_PRIMARY_INK = "#0b0b0b"
_GRIDLINE = "#e1e0d9"
_AXIS_LINE = "#c3c2b7"
_SURFACE = "#fcfcfb"


def compute_qini_curve(y_true, treatment, uplift_scores, n_points: int = 100) -> dict:
    """Compute a Qini curve and its summary coefficient for one model's ranking.

    y_true, treatment, and uplift_scores must all be aligned (same order,
    same length) and drawn from the same held-out test set: y_true is the
    actual conversion outcome, treatment is the actual treatment
    assignment (1/0), and uplift_scores is the model's predicted uplift
    (or, for the naive baseline, its predicted conversion probability -
    see module docstring).

    Customers are sorted by uplift_scores descending, then at each of
    n_points evenly spaced percentiles (plus the mandatory (0%, 0) start),
    we compute the cumulative incremental gain among the top x% via:

        gain(x) = treat_conversions(x) - control_conversions(x) * (treat_n(x) / control_n(x))

    using running cumulative sums over the sorted order. Where control_n(x)
    is 0 (possible in the earliest percentiles before any control unit has
    been reached), the ratio term is treated as 0 - there's no control
    baseline yet to subtract, so gain(x) reduces to the raw treatment
    conversions seen so far.

    The Qini coefficient is the area between this curve and the random-
    targeting diagonal (the straight line from (0, 0) to (100%, gain(100%))
    - what random targeting would achieve in expectation at every
    percentile), computed via the trapezoidal rule over the percentile
    fraction. Positive means the model ranks customers better than random;
    0 means no better than random; the theoretical max is bounded by how
    concentrated real treatment effects are in this population.

    Returns a dict with keys 'percentiles' (ndarray, 0-100, length
    n_points+1), 'gains' (ndarray of cumulative incremental-gain counts,
    same length), and 'qini_coefficient' (float).
    """
    y_true = np.asarray(y_true).astype(int)
    treatment = np.asarray(treatment).astype(int)
    uplift_scores = np.asarray(uplift_scores, dtype=float)

    order = np.argsort(-uplift_scores)
    y_sorted = y_true[order]
    t_sorted = treatment[order]

    is_treat = t_sorted == 1
    is_control = ~is_treat

    cum_treat_conversions = np.cumsum(y_sorted * is_treat)
    cum_control_conversions = np.cumsum(y_sorted * is_control)
    cum_treat_n = np.cumsum(is_treat)
    cum_control_n = np.cumsum(is_control)

    with np.errstate(divide="ignore", invalid="ignore"):
        size_ratio = np.where(cum_control_n > 0, cum_treat_n / cum_control_n, 0.0)
    gains_full = cum_treat_conversions - cum_control_conversions * size_ratio

    n = len(y_sorted)
    percentiles = np.linspace(0, 100, n_points + 1)
    row_idx = np.clip(np.round(percentiles[1:] / 100 * n).astype(int), 1, n) - 1
    gains = np.concatenate(([0.0], gains_full[row_idx]))

    model_area = np.trapz(gains, percentiles / 100)
    diagonal_area = 0.5 * gains[-1]
    qini_coefficient = model_area - diagonal_area

    return {
        "percentiles": percentiles,
        "gains": gains,
        "qini_coefficient": float(qini_coefficient),
    }


def plot_qini_comparison(results_dict: dict, save_path: Path = None) -> dict:
    """Plot Qini curves for multiple models on one chart and print their coefficients.

    results_dict maps model_name -> (y_true, treatment, uplift_scores),
    where all three arrays come from the SAME held-out test set across
    every model (only the uplift_scores differ) - that's what makes the
    curves directly comparable and lets them share a single random-
    targeting diagonal.

    Saves the chart to figures/qini_comparison.png (or save_path, if
    given) and prints each model's Qini coefficient, ranked descending,
    so the models can be compared numerically as well as visually.

    Returns a dict of model_name -> qini_coefficient.
    """
    if save_path is None:
        save_path = _FIGURES_DIR / "qini_comparison.png"
    save_path = Path(save_path)

    fig, ax = plt.subplots(figsize=(9, 6.5), facecolor=_SURFACE)
    ax.set_facecolor(_SURFACE)

    qini_coefficients = {}
    diagonal_end = None

    for i, (model_name, (y_true, treatment, uplift_scores)) in enumerate(results_dict.items()):
        curve = compute_qini_curve(y_true, treatment, uplift_scores)
        qini_coefficients[model_name] = curve["qini_coefficient"]
        diagonal_end = curve["gains"][-1]  # identical across models sharing the same test set

        color = _CATEGORICAL_COLORS[i % len(_CATEGORICAL_COLORS)]
        ax.plot(
            curve["percentiles"],
            curve["gains"],
            color=color,
            linewidth=2,
            solid_capstyle="round",
            label=f"{model_name} (Qini = {curve['qini_coefficient']:.1f})",
        )

    ax.plot(
        [0, 100],
        [0, diagonal_end],
        color=_MUTED_INK,
        linestyle="--",
        linewidth=1.5,
        label="Random targeting",
    )

    ax.set_xlabel("% of population targeted (ranked by predicted uplift)", color=_SECONDARY_INK)
    ax.set_ylabel("Cumulative incremental conversions", color=_SECONDARY_INK)
    ax.set_title("Qini curve comparison — held-out test set", color=_PRIMARY_INK, fontsize=13, fontweight="bold")
    ax.grid(True, color=_GRIDLINE, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(_AXIS_LINE)
    ax.tick_params(colors=_MUTED_INK)
    legend = ax.legend(frameon=False, loc="best")
    for text in legend.get_texts():
        text.set_color(_PRIMARY_INK)

    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)

    print(f"Saved Qini curve comparison to {save_path}")
    print("\nQini coefficients (higher is better; 0 = no better than random targeting):")
    for model_name, coef in sorted(qini_coefficients.items(), key=lambda kv: -kv[1]):
        print(f"  {model_name}: {coef:.4f}")

    return qini_coefficients
