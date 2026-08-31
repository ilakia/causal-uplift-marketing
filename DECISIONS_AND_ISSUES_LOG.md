# Decisions & Issues Log

Running log of every real issue hit while building this project, why it happened,
what we did about it, and why we didn't take the alternative path. This is the
file to study from later - it captures the actual reasoning, not just the code.

Format for each entry:
- **What happened**
- **Why it happened**
- **What we did**
- **Why not the alternative**
- **Concept this ties to** (for review later)

---

## 1. Criteo dataset direct download failed

**What happened:** The documented direct download link
(`http://go.criteo.net/criteo-research-uplift-v2.1.csv.gz`) wouldn't download
reliably.

**Why it happened:** Large file (~300MB compressed) served directly from
Criteo's own infrastructure, which can be slow/unstable depending on
region/connection - not something wrong with the setup.

**What we did:** Switched to `scikit-uplift`'s built-in `fetch_criteo()`
loader, which handles the download, caching, and gives back a clean
features/treatment/target split automatically.

**Why not the alternative:** Could have kept retrying the manual link or
tried a Kaggle mirror, but the package loader is more reliable long-term
(reusable, cached, no manual file management) and is standard practice in
the causal ML community for this exact dataset.

**Concept tie-in:** N/A - infra issue, not a modeling concept.

---

## 2. Mac running low on disk space before install

**What happened:** Only 8.3GB free before installing dependencies
(estimated ~5-6GB needed, mainly from PyTorch).

**Why it happened:** 52GB sitting in `~/Library` (mostly caches, Application
Support junk) that had built up over time, unrelated to this project.

**What we did:**
1. Cleared `~/Library/Caches` (10GB recovered) - pure cache, always safe
2. Emptied Trash, cleared conda cache
3. Removed `torch` from `requirements.txt` for now (~2-2.5GB saved) -
   deferred until we actually reach the TARNet phase

**Why not the alternative:** Could have dug into `Application Support`
(27GB) for more space, but wasn't necessary once Caches clearing and the
torch deferral freed enough (8.3GB to 18GB available).

**Concept tie-in:** N/A - infra issue.

---

## 3. Naive baseline AUC-ROC came back suspiciously high (0.9433 initially)

**What happened:** First run of the naive logistic regression baseline
returned AUC-ROC of 0.9433 - unusually high for a rare-event (~0.3%
conversion) prediction problem from just 12 anonymized features.

**Why it happened (what we checked before trusting it):** Ran a sanity
check to rule out the likely causes:
- Feature leakage (accidentally including `visit`/`exposure`/`treatment`
  as features) - **ruled out**, confirmed only f0-f11 were used
- One feature dominating the model unrealistically - **ruled out**,
  coefficients were reasonably distributed (top 5 all between -0.91 and
  +0.74, no single runaway coefficient)

**What we did:** Concluded the AUC is genuinely real - Criteo's anonymized
features were specifically engineered for ad-response prediction, so a
high AUC here isn't inherently suspicious the way it might be on a random
demographic dataset. Added PR-AUC as a second metric to get the full
picture.

**Why not the alternative:** Could have just accepted 0.9433 at face value
without checking - would have been a mistake, since an unexamined "too
good" result is exactly the kind of thing an interviewer would probe on.
The right move is always suspicion first, celebration second.

**Concept tie-in:** Feature leakage checks; why ROC-AUC alone can be
misleading on imbalanced/rare-event problems (see next entry).

---

## 4. ROC-AUC vs PR-AUC - a big, informative gap

**What happened:** Added PR-AUC alongside ROC-AUC. PR-AUC came back much
lower than ROC-AUC on the naive baseline.

**Why it happened:** With only ~0.29% conversion rate, ROC-AUC is diluted
by the huge number of easy true negatives (predicting "no conversion" is
trivially right most of the time), so it looks artificially strong.
PR-AUC focuses specifically on precision/recall performance on the rare
positive class, which is much less forgiving under imbalance.

**What we did:** Kept both metrics, explicitly used this gap as the
teaching example for "why ROC-AUC can mislead on rare-event problems" in
the project write-up. Final fixed-split numbers: AUC-ROC 0.9405, PR-AUC
0.1902.

**Why not the alternative:** Could have just reported ROC-AUC alone (which
is what a less careful analysis would do) - deliberately avoided that,
since it's the exact mistake this whole project is structured to expose.

**Concept tie-in:** Class imbalance; ROC-AUC vs PR-AUC; when to prefer each.

---

## 5. T-learner (linear) gave 100% positive / 0% negative treatment effect

**What happened:** First T-learner (logistic regression base learner)
produced ITE estimates that were positive for literally 100% of customers,
0% negative.

**Why it happened (investigation path):**
1. First suspicion: control model (trained on only ~15% of rows, the
   minority class) might be systematically biased low, artificially
   inflating every ITE estimate.
2. Checked calibration of both models separately (predicted vs. actual
   conversion rate per group) - **both were well-calibrated** (within
   ~0.0006 percentage points), ruling out the bias theory.
3. Real explanation: the average treatment effect genuinely is positive
   and fairly large (treatment group converts at ~1.6x control rate), and
   a **linear model has no flexibility to express heterogeneity** - it can
   only shift predictions up/down uniformly, not carve out subgroups where
   the effect might be smaller, zero, or negative.

**What we did:** Swapped the base learner from logistic regression to
XGBoost (gradient-boosted trees), which can model non-linear interactions
and has a real chance of detecting subgroups with different effects.

**Why not the alternative:** Could have just accepted the 100%/0% result
as "everyone benefits" - would have been wrong to conclude that from a
linear model's structural limitation. Also could have kept debugging the
calibration angle further, but the calibration check already cleanly
ruled that out.

**Concept tie-in:** Bias-variance / model flexibility; why base learner
choice matters in meta-learners; linear vs. tree-based models.

---

## 6. XGBoost T-learner (unregularized, evaluated on training data) - overfitting

**What happened:** After swapping to XGBoost, ITE range blew out to
[-0.79, +0.81] with 87.36% positive / 12.64% negative - now showing real
heterogeneity, but with implausibly extreme tail values for a ~0.3% base
rate.

**Why it happened:** Two compounding issues:
1. XGBoost was evaluated on the **same rows used to train it** (no
   train/test split at this stage) - the classic "trust your own
   homework" mistake.
2. No regularization on the XGBoost model - with a rare-event target, an
   unconstrained tree model can carve out tiny leaves that memorize noise
   from individual rows, producing extreme (and unreliable) predictions.

**What we did:**
1. Added a proper stratified 80/20 train/test split - models fit only on
   training data, ITE evaluated only on held-out test data.
2. Added regularization to XGBoost (`max_depth=4`, `min_child_weight=10`,
   `subsample=0.8`, `colsample_bytree=0.8`) to prevent overfitting on rare
   positive-class leaves.

**Result after fix:** Tails shrank meaningfully (max 0.81 -> 0.59, min
-0.79 -> -0.44). Importantly, the %positive/%negative split barely moved
(87.36/12.64 -> 86.55/13.45), which told us the *negative-effect subgroup
finding itself was real and robust*, not an overfitting artifact - only
the extreme tail magnitudes were inflated by overfitting.

**Why not the alternative:** Could have just lowered `max_depth` further
or added even more regularization until tails looked "nicer," but that
risks under-fitting and erasing a real signal just to make numbers look
cleaner - we specifically checked that the core finding (existence of a
negative subgroup) survived the fix, rather than assuming more
regularization is automatically better.

**Concept tie-in:** Train/test split discipline; regularization
(max_depth, subsample, colsample); overfitting on rare-event targets;
why a finding needs to be robust to a fix, not just present before it.

---

## 7. X-learner tails got WIDER than T-learner, not narrower (unexpected)

**What happened:** X-learner was built specifically to reduce the
T-learner's variance problem. Instead, it showed *wider* ITE tails
([-0.64, 0.71] vs T-learner's [-0.44, 0.59]) and a near-total collapse of
the negative-effect subgroup (0.48% vs T-learner's 13.45%) in the raw
prediction distribution - the opposite of what the theory predicts for
variance reduction.

**Why it happened (investigation path):**
1. Hypothesis: Criteo used **fixed-ratio randomization** (85% treatment
   for everyone, not dependent on features) - so the *true* propensity
   score is constant, meaning X-learner's propensity-weighting step (which
   is supposed to dynamically decide which sub-model to trust more per
   customer) has no real signal to work with.
2. Checked the propensity model's actual output range: mean 0.85, tight
   IQR (~0.003 wide) - **mostly confirms** the fixed-ratio hypothesis,
   though not perfectly constant (some real tail from 0.61-0.995, likely
   the propensity model mildly overfitting noise on only 12 sparse
   features across ~11M rows).
3. Checked `tau_control` vs `tau_treatment` individually: X-learner's
   documented rationale is that `tau_control` (fit using the
   larger/majority treatment arm's imputations) should be the *more*
   reliable estimate, deserving more weight. **Empirically the opposite
   was true** - `tau_control` had the wider/noisier tails, not
   `tau_treatment`.

**What we did:** Documented this as a genuine, real finding rather than a
bug to "fix": X-learner's theoretical justification for *why* it should
outperform did not hold empirically on this specific dataset (the
majority-arm-derived component was not the more reliable one). Note: on
the final Qini evaluation, X-learner did score higher than T-learner
(1058.6 vs 963.3) - this finding is about the underlying mechanism not
behaving as theory predicts, not about which method wins on Qini overall.

**Why not the alternative:** Could have kept tuning the X-learner
(different regularization on the tau-models, trying to force propensity
to be flatter) to try to make the raw distribution "look right" -
deliberately didn't, because the finding itself (X-learner's core
assumption breaking down under fixed-ratio RCT randomization) is a more
valuable, defensible result than forcing the numbers to match theory.
This is genuinely citable, non-obvious causal ML insight: X-learner's
theoretical benefit is strongest under imbalanced *observational*
treatment assignment, not clean fixed-ratio RCTs like this one - even
when its overall Qini performance is still reasonable.

**Concept tie-in:** Propensity scores; when meta-learner theoretical
assumptions do/don't hold even if overall performance looks fine; RCT vs.
observational study designs; X-learner mechanics (2-stage imputation +
propensity-weighted combination).

---

## 8. macOS blocked Terminal from reading/writing inside ~/Downloads

**What happened:** Mid-session, all file operations inside the project
folder (which lived in `~/Downloads/`) started failing with "Operation not
permitted," even for simple `ls`/`cat`, while Python execution itself
still worked.

**Why it happened:** macOS treats Downloads (along with Desktop and
Documents) as a privacy-protected folder. Terminal's "Files and Folders"
permission for that location can get silently revoked (common after
macOS updates or just periodically).

**What we did:** Moved the whole project out of Downloads to the home
directory (`~/causal-uplift-marketing`) instead of fighting the permission
system - a permanent fix that avoids the entire category of problem going
forward.

**Why not the alternative:** Could have gone into System
Settings -> Privacy & Security to re-grant the permission and kept working
inside Downloads - more fragile long-term (the same issue can recur), so
moving the project was the more durable fix.

**Concept tie-in:** N/A - infra/OS issue.

---

## 9. Moving the project folder broke the venv, silently polluted Anaconda base

**What happened:** After moving the project from `~/Downloads/` to
`~/causal-uplift-marketing/` (to fix the macOS permissions issue in #8),
later `pip install` commands (xgboost, econml) silently installed into the
global Anaconda base environment instead of the project's isolated venv,
and downgraded `numpy` there (1.24.3 -> 1.21.6) as a side effect - which
conflicts with TensorFlow's requirements in that base environment.

**Why it happened:** The venv's `activate` script hardcodes an absolute
path to itself (`VIRTUAL_ENV=/Users/username/Downloads/causal-uplift-marketing/venv`)
at creation time. Moving the folder didn't update that internal path, so
`source venv/bin/activate` pointed at a location that no longer existed.
Python/pip silently fell through to the system/conda default instead of
failing loudly - meaning "activation" appeared to work but wasn't actually
isolating anything.

**What we did:** Chose to fully repair rather than route around it:
1. Restored numpy to 1.24.3 in the Anaconda base environment (undoing the
   side-effect damage)
2. Rewrote the venv's activate script with the correct current path
3. Reinstalled xgboost/econml into the now-actually-isolated venv

**Why not the alternative:** Could have just kept working out of Anaconda
base going forward (abandoning the venv entirely) - rejected, since that
leaves project dependencies mixed with everything else on the machine,
risking confusing bugs in unrelated projects later (e.g. a future
TensorFlow project silently breaking because of a numpy version this
project happened to need).

**Concept tie-in:** What a virtual environment actually does (dependency
isolation) and why silent fallback-to-global-environment is a genuinely
dangerous failure mode - it looks like it worked, but wasn't isolating
anything, which is worse than a loud error.

---

## 10. Causal forest smoke test crashed with exit code 139 (segfault)

**What happened:** After fixing the venv (#9), the first attempt to run
`causal_forest.py` on a small 20k-row subsample crashed silently with exit
code 139 - no Python traceback, no error message, just a dead process.

**Why it happened:** Exit code 139 specifically means the process was
killed by a segmentation fault - a low-level memory access crash, not a
Python-level exception. On macOS specifically, this is a known failure
mode when multiple libraries each bundle their own copy of the OpenMP
runtime. Confirmed precisely: Intel's OpenMP (`libiomp`, bundled with
numpy/scipy's MKL) and LLVM's OpenMP (`libomp`, bundled inside XGBoost's
own wheel) were both being loaded into the same process and colliding.

**What we did:** Fixed by calling `threadpoolctl`'s
`threadpool_limits(limits=1)` at the top of the script, which forces
single-threaded execution for the OpenMP/BLAS layer and neutralizes the
conflict regardless of which library gets imported first.

**Why not the alternative:** Could have tried setting the environment
variable `KMP_DUPLICATE_LIB_OK=TRUE` instead (a commonly-suggested quick
fix for this exact error) - deliberately avoided it here, since that
variable just tells the process to silently tolerate two conflicting
OpenMP runtimes coexisting, which can cause silent incorrect results or
crashes elsewhere rather than actually resolving the conflict. The
`threadpoolctl` fix properly limits to one runtime instead of papering
over having two.

**Trade-off to know about:** This fix constrains the OpenMP/BLAS math
layer to one thread, but (as later discovered in #15) doesn't fully
constrain all of scikit-learn/econml's own internal parallelism - a
useful reminder that "single-threaded" fixes at one library layer don't
always propagate through the whole stack.

**Concept tie-in:** Native/systems-level library conflicts (OpenMP
specifically) are a common but genuinely different category of ML
engineering problem from anything statistical - recognizing "silent
crash, no Python traceback" as the signature of this class of bug (rather
than a data or modeling bug) is itself a useful engineering instinct.

---

## 11. Full-scale causal forest run killed (exit code 137) - tool timeout, not OOM

**What happened:** After fixing the OpenMP crash (#10), the full-scale
causal forest run (on the full ~11.2M-row training set) was killed midway
through fitting, with exit code 137, on two separate attempts.

**Why it happened:** Initially suspected to be an OS-level OOM kill related
to low disk space (525MB free at the time). **Correction, discovered on
the second attempt:** the real cause was that Claude Code's background
shell commands have a hard ~10-minute timeout - the process was killed at
exactly that mark both times, independent of disk space. The low-disk
coincidence was real and worth fixing regardless, but wasn't actually the
cause of this specific kill.

**What we did:** Rather than immediately fight the timeout with a
fully-detached `nohup`/`disown` background process, chose first to
subsample training data to a stratified 2 million rows instead - this
comfortably finished within the normal timeout and gave a first working
causal forest result.

**Why not the alternative:** Could have gone straight to nohup for the
full dataset - declined at this point given the runtime was completely
unknown/open-ended at full scale, and an unpredictable detached process is
riskier to monitor than getting a smaller, working result first.

**Concept tie-in:** Important debugging lesson - the same exit code (137)
had two different candidate explanations (OOM vs. tool timeout) that
looked identical from the outside; the real cause only became clear by
noticing the kill happened at a suspiciously exact time interval (10
minutes) both times, not by the error message itself.

---

## 12. Causal Forest's raw 49/50 positive/negative split was almost entirely noise

**What happened:** Causal Forest's raw CATE predictions (2M-row training
run) showed a near-even 49.43% positive / 50.57% negative split -
suspicious given T-learner said 86.55/13.45 and X-learner said 99.52/0.48
on the same held-out test set.

**Why it happened:** Checked what fraction of predictions were actually
large in magnitude (|CATE| > 0.05) versus small/noise-level (|CATE| <=
0.05). Result: **99.29% of predictions were within the noise band**, and
only **0.71% (~20K people)** had a CATE the forest considered genuinely
large. The near-50/50 split was almost entirely sign-flipping on tiny
near-zero estimates - not real evidence of widespread heterogeneous
effects.

**What we did:** Filtered to just the large-effect population (|CATE| >
0.05) and recomputed the positive/negative split within that group only:
73.4% positive vs 26.6% negative - a real, meaningful, and much more
useful segmentation than either the raw 49/50 split or T-learner's
blanket 86.55/13.45. This became the headline finding for the final
report, not the raw split.

**Why not the alternative:** Could have reported the raw 49.43%/50.57%
split at face value as "causal forest finds a near-even split of
helped/hurt customers" - would have been actively misleading, since it
conflates genuine effect direction with statistical noise around zero.
Filtering by magnitude first is the correct move before interpreting
sign at all.

**Concept tie-in:** Statistical vs. practical significance, applied to
individual-level effect estimates rather than just aggregate hypothesis
tests - the same principle from A/B testing (a "significant" but tiny
effect isn't necessarily actionable) applies here at the per-customer
level.

---

## 13. Naive baseline and causal models used different train/test splits

**What happened:** While preparing the four-way Qini curve comparison,
discovered the naive baseline (built in an early phase) was stratified on
`conversion`, while T-learner/X-learner/Causal Forest (built later) were
all stratified on `treatment` - meaning the naive baseline's held-out test
set was a genuinely different partition of the data than the other three
models'.

**Why it happened:** Each model was built at a different point in the
project, with a reasonable-sounding stratification choice made
independently each time - nobody explicitly checked they matched until
assembling the final comparison.

**What we did:** Retrained the naive baseline using the exact same
train/test split (same random_state and stratification) as the three
causal models, so all four models are evaluated on one truly identical
held-out test set before running the Qini comparison. (This is also why
the final naive baseline AUC-ROC/PR-AUC numbers - 0.9405/0.1902 - differ
slightly from the very first run's 0.9433/0.1861.)

**Why not the alternative:** Could have kept the naive baseline on its
original split and just labeled the chart "evaluated on different
partitions" - rejected, since even a small difference in which customers
land in each test set can shift a Qini coefficient, and the entire point
of this comparison chart is a clean, apples-to-apples answer.

**Concept tie-in:** Fair-comparison discipline in ML evaluation - when
comparing multiple models, every one of them must be evaluated on
identical held-out data, not just "a similarly-sized held-out set."

---

## 14. Naive baseline outscored all uplift models on Qini; causal forest's low score was confounded by less training data

**What happened:** The four-way Qini comparison came back with the naive
baseline scoring HIGHEST (1110.0), beating X-learner (1058.6), T-learner
(963.3), and Causal Forest scoring LOWEST (618.5) - the opposite of the
expected ranking, since naive has no real concept of incremental/causal
effect.

**Why it happened (two separate causes, both investigated):**
1. **Naive's high score is real, not a bug.** Verified all four score
   arrays were structurally correct, and the chart itself was well-formed
   (all curves start at origin, are concave, converge to the identical
   mathematically-required endpoint at 100% targeted). The actual
   explanation: naive's predicted conversion probability correlates
   0.43-0.51 with each uplift model's own score, because on this specific
   dataset the segment of customers most likely to convert anyway
   substantially overlaps with the segment that's genuinely persuadable -
   so ranking by naive propensity incidentally captures a lot of real
   incremental signal too. This is a **legitimate, dataset-specific
   finding**, not a general result about naive targeting being a good
   strategy.
2. **Causal forest's last-place finish is confounded, not (necessarily) a
   real methodological loss.** It's the only model trained on 2M rows
   instead of the full ~11.2M - roughly 18% of what the other three
   models got. Its Qini curve visibly lagged for most of the targeting
   range and only caught up near the tail, consistent with a
   training-data handicap rather than a fair verdict on the method itself.

**What we did:** Did NOT accept the raw ranking at face value. Flagged
both findings explicitly in the final report rather than writing up
"naive wins, causal forest loses" as a clean conclusion. Attempted to
retrain causal forest on more data (5M rows) to get a fairer read - see
#15 for how that played out.

**Why not the alternative:** Could have just reported the raw Qini
coefficients as the final verdict - would have been actively misleading
in two directions: overstating how well naive targeting generalizes, and
understating causal forest's real capability.

**Concept tie-in:** The importance of checking *why* a surprising result
happened before reporting it, even when the numbers/code are technically
correct - a "real" number can still be misleading if the comparison
producing it isn't fair, or is dataset-specific in a way that shouldn't
be generalized.

---

## 15. Attempting a larger causal forest run - timeout, memory limits, and knowing when to stop

**What happened:** Retraining causal forest on 5M rows (to fairly compare
against the other three full-data models) was estimated at ~22 minutes
based on the 2M run's timing - longer than Claude Code's ~10-minute
per-command timeout, requiring a detached (`nohup`/`disown`) background
process.

**Why it happened:** Same underlying constraints as #10/#11 (forced
single-threading, tool timeout) compounding at a larger data size.

**What we did:** Accepted the nohup/detached approach this time (unlike
#11, where we declined it for the full 11.2M-row run), reasoning that a
bounded, measured 22-minute estimate is a more acceptable risk than an
unknown/open-ended runtime. Launched detached and monitored via CPU/memory
checks (`ps aux`, Activity Monitor) rather than blocking tool calls.

**What actually happened:** CPU usage was observed climbing from 219% to
371% over time rather than settling - meaning the threadpoolctl fix from
#10 only constrained the OpenMP/BLAS layer, not all of scikit-learn/
econml's internal parallelism, so the 22-minute estimate (based on an
assumption of strict single-threading) never applied. After ~58 minutes
with no completion, the process was found to have died silently - no exit
code, no traceback, and no macOS crash report (crash reports are normally
auto-generated for a segfault/SIGSEGV, so their absence points away from
that). This is most consistent with an **OOM kill**: a causal forest fit
with honest-splitting on 5M rows generates substantially more bookkeeping
per tree node than the 2M-row run (whose pickled model alone was 1.7GB),
and combined with the ~14M-row DataFrame and XGBoost's internal overhead,
this likely exceeded the machine's 8GB physical RAM - though this wasn't
definitively confirmed with a memory-usage timeline at the moment of
death.

**Final decision:** Declined to retry at 5M (even with active memory
monitoring) or step down to an intermediate size (e.g. 3.5M) - stopped for
the session and finalized the project using the already-validated,
working 2M-row causal forest result, with the "trained on a subsample,
not the full dataset" limitation stated plainly in the README rather than
chased further.

**Why not the alternative:** Time invested in an unpredictable process is
time not spent finishing and documenting the project - past a reasonable
cutoff, on an 8GB RAM machine that had already demonstrated it couldn't
handle this workload, a disclosed limitation is a better outcome than
continuing to retry.

**Concept tie-in:** Not a modeling concept - an engineering/project-
management judgment call. Whether to accept a risky/complex execution
path should depend on how *predictable* the situation is, not just how
long it takes. Also: exit codes and silent deaths are diagnostic signals
(139 = segfault with crash report, 137 here = timeout in one case and
likely OOM in another, distinguishable by checking for a crash report and
noticing timing patterns) - worth learning to read rather than guess at.

---

## Template for future entries (keep using this format going forward)

## N. [Short description of what happened]

**What happened:**

**Why it happened:**

**What we did:**

**Why not the alternative:**

**Concept tie-in:**
