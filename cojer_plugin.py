"""
CoJER plug-in for drug_target_interaction.ipynb.

Self-contained (no hidimstat import): the null conformal p-value simulation of
utils_conformal.py (Gazin et al., AISTATS 2023) is re-implemented here. Calibration and
the post hoc bound come from the `sanssouci` package (pip install sanssouci).

Typical use
-----------
    # once, outside the repetition loop (depends only on n, m, delta, B, k_max)
    thr = cojer_thresholds_sanssouci(n, m, delta=0.1, B=10000, k_max=int(0.02 * m), n_jobs=-1)

    # for each repetition
    pvals_d = conformal_pvals_discrete(calib_score, test_score)
    sorted_p, bound = cojer_fdp_bound_sets(pvals, pvals_d, thr)   # bound for every prefix
    fdp_grid = cojer_bound_on_grid(sorted_p, bound, grid)         # bound for {p <= t}, t in grid
"""
import os
import numpy as np
import scipy.stats as stat
from scipy.special import betainc, betaincinv


# ---------------------------------------------------------------------------------------
# Null conformal p-values (from utils_conformal.py)
# ---------------------------------------------------------------------------------------
def get_conditional_distrib_conformal(n_cal):
    """Distribution of a null conformal p-value conditionally on the calibration set:
    p takes values i/(n_cal+1) with Dirichlet-distributed probabilities."""
    U = stat.dirichlet.rvs(alpha=np.ones(n_cal + 1))[0]
    p_values = (1 + np.arange(n_cal + 1)) / (n_cal + 1)
    return stat.rv_discrete(values=(p_values, U))


def get_null_pvals_conformal(B, n_cal, n_test):
    """B draws of n_test null conformal p-values (one calibration draw per row),
    each row sorted increasingly. Shape (B, n_test)."""
    pval0 = np.zeros((B, n_test))
    for b in range(B):
        pval0[b] = get_conditional_distrib_conformal(n_cal).rvs(size=n_test)
    return np.sort(pval0, axis=1)


# ---------------------------------------------------------------------------------------
# CoJER as in the paper: empirical template + lambda-calibration (sanssouci)
# ---------------------------------------------------------------------------------------
def _simulate_null_chunk(n_cal, n_test, k_keep, n_rows, seed):
    """n_rows draws of n_test null conformal p-values; keeps only the k_keep smallest of each
    row (sorted). Same law as get_null_pvals_conformal, but vectorized: instead of
    rv_discrete.rvs, uniforms are mapped through the cumulative Dirichlet weights."""
    rng = np.random.default_rng(seed)
    out = np.empty((n_rows, k_keep))
    values = (1 + np.arange(n_cal + 1)) / (n_cal + 1)
    for b in range(n_rows):
        cdf = np.cumsum(rng.dirichlet(np.ones(n_cal + 1)))
        idx = np.minimum(np.searchsorted(cdf, rng.random(n_test)), n_cal)
        row = values[idx]
        out[b] = np.partition(row, k_keep - 1)[:k_keep] if k_keep < n_test else row
        out[b].sort()
    return out


def cojer_thresholds_sanssouci(n, m, delta=0.1, B=10000, k_max=None, seed=0, n_jobs=1):
    """
    Calibrated threshold family t_1 <= ... <= t_{k_max} controlling the JER at level delta.

    The learned template is the empirical one: for each rank k, the quantiles over the B
    simulated null samples (column-wise sort of the null p-values). `calibrate_jer` then
    picks, by dichotomy, the template quantile level whose estimated JER is <= delta.

    k_max must be chosen a priori (not after looking at the data). In neuroimaging,
    k_max = 2% of m (e.g. 1000 for m = 50000).

    Only the k_max smallest null p-values of each draw matter for the JER over ranks
    1..k_max, so only those are stored: memory is O(B * k_max) instead of O(B * m)
    (about 30 MB instead of 1.4 GB for B = 10000, m = 18033, k_max = 360).
    The simulation (the expensive step) is split over `n_jobs` processes with joblib
    (n_jobs=-1: all cores); results are reproducible for a given (seed, n_jobs).
    """
    from joblib import Parallel, delayed
    from sanssouci.lambda_calibration import calibrate_jer
    k_max = m if k_max is None else min(k_max, m)
    n_chunks = max(1, abs(n_jobs) if n_jobs != -1 else (os.cpu_count() or 1))
    n_chunks = min(n_chunks * 4, B)                 # a few chunks per worker for load balance
    rows = [B // n_chunks + (i < B % n_chunks) for i in range(n_chunks)]
    seeds = np.random.SeedSequence(seed).spawn(n_chunks)
    chunks = Parallel(n_jobs=n_jobs)(
        delayed(_simulate_null_chunk)(n, m, k_max, r, sd) for r, sd in zip(rows, seeds)
    )
    pval0 = np.vstack(chunks)                       # (B, k_max), sorted within rows
    template = np.sort(pval0, axis=0)               # (B, k_max), sorted within columns
    return np.asarray(calibrate_jer(delta, template, pval0, k_max))


def _estimate_jer_light(template, pval0, k_max, k_min=0, chunk=500):
    """Memory-light drop-in for sanssouci.lambda_calibration.estimate_jer.
    The JER event for one null draw is {exists k <= k_max : p_(k) < t_k}; since the template
    row is non-decreasing this is exactly what estimate_jer computes with searchsorted/sign
    on (B, m) integer arrays, but here only boolean (chunk, k_max) blocks are built."""
    t = np.asarray(template)[k_min:k_max]
    n_viol = 0
    for i in range(0, pval0.shape[0], chunk):
        n_viol += int(np.any(pval0[i:i + chunk, k_min:k_max] < t, axis=1).sum())
    return n_viol / pval0.shape[0]


def _calibrate_light(delta, template, pval0, k_max):
    """sanssouci.calibrate_jer (same dichotomy, same fallbacks) with the light JER estimator."""
    import sanssouci.lambda_calibration as lc
    original = lc.estimate_jer
    lc.estimate_jer = _estimate_jer_light
    try:
        return np.asarray(lc.calibrate_jer(delta, template, pval0, k_max))
    finally:
        lc.estimate_jer = original


def cojer_thresholds_multi(n, m, delta=0.1, B=10000, k_max_list=None, seed=0, n_jobs=1):
    """
    Calibrate one threshold family per k_max in k_max_list, from ONE shared simulation.

    Returns {k_max: thresholds}. Each family controls the JER at level delta on its own.
    Intended for a sensitivity analysis: reporting all of them is valid, picking the best one
    after looking at the data is not (k_max must be chosen a priori).
    Peak memory is dominated by the largest k_max: about 3 * B * max(k_max) * 8 bytes
    (about 4-6 GB for B = 10000 and k_max = m = 18033).
    """
    from joblib import Parallel, delayed
    k_list = sorted({min(int(k), m) for k in k_max_list})
    k_top = k_list[-1]
    n_chunks = max(1, abs(n_jobs) if n_jobs != -1 else (os.cpu_count() or 1))
    n_chunks = min(n_chunks * 4, B)
    rows = [B // n_chunks + (i < B % n_chunks) for i in range(n_chunks)]
    seeds = np.random.SeedSequence(seed).spawn(n_chunks)
    pval0 = np.vstack(Parallel(n_jobs=n_jobs)(
        delayed(_simulate_null_chunk)(n, m, k_top, r, sd) for r, sd in zip(rows, seeds)))
    out = {}
    for k in k_list:
        sub = pval0[:, :k]                       # k smallest null p-values of each draw
        template = np.sort(sub, axis=0)          # column-wise sort = first k columns of the full template
        out[k] = _calibrate_light(delta, template, sub, k)
        del template
    return out


# ---------------------------------------------------------------------------------------
# Post hoc FDP bounds
# ---------------------------------------------------------------------------------------
def conformal_pvals_discrete(calib_score, test_score):
    """Non-randomized conformal p-values (the ones whose null law matches the template).
    Small score = evidence against H0, as in the notebook."""
    cs = np.sort(calib_score)
    return (1 + np.searchsorted(cs, test_score, side="right")) / (len(cs) + 1)


def cojer_fdp_bound_sets(pvals_smoothed, pvals_discrete, thresholds):
    """
    CoJER FDP upper bound for every prefix S_s = {the s smallest smoothed p-values}.

    The sets are defined by the (randomized, smoothed) p-values used by the other methods,
    but the threshold family is evaluated on the matching discrete p-values, whose null law
    is the one simulated for the template.

    Returns (sorted_smoothed, bound) where bound[s-1] is the FDP bound for the s first rejections.
    """
    from sanssouci.post_hoc_bounds import curve_max_fp
    order = np.argsort(pvals_smoothed, kind="stable")
    # curve_max_fp needs sorted input: running max of the discrete p-values along the order
    # is a monotone upper envelope, hence conservative.
    pd_sorted = np.maximum.accumulate(np.asarray(pvals_discrete)[order])
    curve = curve_max_fp(pd_sorted, np.asarray(thresholds))   # max #false positives, per prefix
    s = np.arange(1, len(order) + 1)
    return np.asarray(pvals_smoothed)[order], curve / s


def cojer_bound_on_grid(sorted_smoothed, bound_prefix, grid):
    """Evaluate the prefix bounds on {p <= t}, t in grid (0 when the set is empty)."""
    R = np.searchsorted(sorted_smoothed, np.atleast_1d(grid), side="right")
    return np.where(R > 0, bound_prefix[np.maximum(R, 1) - 1], 0.0)


def cojer_fdp_bound_sanssouci(pvals, thresholds, grid):
    """Simpler variant using a single p-value vector (no smoothed/discrete distinction)."""
    from sanssouci.post_hoc_bounds import curve_max_fp
    p = np.sort(np.asarray(pvals))
    curve = curve_max_fp(p, np.asarray(thresholds))
    R = np.searchsorted(p, np.atleast_1d(grid), side="right")
    return np.where(R > 0, curve[np.maximum(R, 1) - 1] / np.maximum(R, 1), 0.0)


# ---------------------------------------------------------------------------------------
# Optional: one-parameter threshold families, without sanssouci
# ---------------------------------------------------------------------------------------
def calibrate_cojer_thresholds(n, m, delta=0.1, B=1000, template="linear", kmax=None, seed=0):
    """
    Threshold family with a single parameter lambda, calibrated on B null simulations so that
    P(exists k <= kmax: p_(k) < t_k) <= delta.
      template="linear": t_k = lambda * k / m                  (Simes-type)
      template="beta"  : t_k = lambda-quantile of Beta(k, m+1-k)  (order-statistic type)
    Returns (thresholds, lambda).
    """
    K = m if kmax is None else min(kmax, m)
    np.random.seed(seed)
    sims = get_null_pvals_conformal(B, n, m)[:, :K]
    ks = np.arange(1, K + 1)
    if template == "linear":
        stats_ = (sims * m / ks).min(axis=1)
    elif template == "beta":
        stats_ = betainc(ks, m + 1 - ks, sims).min(axis=1)
    else:
        raise ValueError(template)
    lam = np.sort(stats_)[int(np.floor(delta * B))]
    thr = lam * ks / m if template == "linear" else betaincinv(ks, m + 1 - ks, lam)
    return np.minimum(thr, 1.0), lam


def cojer_fdp_bound(pvals, thresholds, grid):
    """Pure-numpy FDP bound on {p <= t}, t in grid, for any non-decreasing threshold family:
    V(S) = min_{0<=k<=|S|} ( |S ∩ {p >= t_{k+1}}| + k ). Same output as
    cojer_fdp_bound_sanssouci, vectorized instead of linear-time."""
    p = np.sort(np.asarray(pvals))
    grid = np.atleast_1d(grid)
    K = len(thresholds)
    below = np.searchsorted(p, thresholds, side="left")   # #{p < t_{k+1}}, k = 0..K-1
    ks = np.arange(K)
    R = np.searchsorted(p, grid, side="right")            # |S_t|
    out = np.zeros(len(grid))
    for i in range(0, len(grid), 100):                    # blocks, to bound memory
        r = R[i:i + 100, None]
        out[i:i + 100] = np.minimum(
            r, (r - np.minimum(r, below[None, :])) + ks[None, :]
        ).min(axis=1)
    return out / np.maximum(R, 1)
