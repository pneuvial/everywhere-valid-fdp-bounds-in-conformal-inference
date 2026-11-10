#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")            # non-interactive: figures are written to PDF, never displayed
import matplotlib.pyplot as plt
from pathlib import Path
FIG_DIR = Path('all_figures'); FIG_DIR.mkdir(exist_ok=True)
plt.rcParams.update({'xtick.labelsize': 16, 'ytick.labelsize': 16})   # readable on slides
from ecdf_upper_bound import uniform_ecdf_upper_bound, uniform_FDP_upper_bound
from cojer_plugin import (cojer_thresholds_multi, cojer_fdp_bound_sets,
                          cojer_bound_on_grid, conformal_pvals_discrete)
def benjamini_hochberg(pvals, alpha=0.1):
    """
    Performs the Benjamini-Hochberg procedure to control the False Discovery Rate.

    Parameters:
    - pvals (list or numpy array): Array of p-values.
    - alpha (float): Significance level (default is 0.05).

    Returns:
    - numpy array: Boolean array indicating whether each hypothesis is rejected.
    """
    m = len(pvals)
    sorted_indices = np.argsort(pvals)
    sorted_pvals = np.array(pvals)[sorted_indices]
    thresholds = np.array([alpha * (i + 1) / m for i in range(m)])
    
    # Determine the largest index for which the p-value is below the threshold
    reject = sorted_pvals <= thresholds
    max_index = np.where(reject)[0].max() if reject.any() else -1
    
    # Create a boolean array to indicate which p-values are rejected
    rejected = np.zeros(m, dtype=bool)
    #print(max_index)
    if max_index >= 0:
        rejected[sorted_indices[:max_index + 1]] = True
        return rejected, sorted_pvals[max_index]
    else:
        return rejected, 0
    


# In[2]:


test_data = pd.read_csv('test_199.csv')
calib_data = pd.read_csv('calib_199.csv')
test_data.rename(columns={'test_pred': 'pred', 'test_true': 'true'}, inplace=True)
calib_data.rename(columns={'calib_pred': 'pred', 'calib_true': 'true'}, inplace=True)
all_data = pd.concat([calib_data, test_data], ignore_index=True)
n = calib_data.shape[0]
m = test_data.shape[0]
n_total = m+n


# In[3]:


print(m,n)


# In[4]:


upper_MC_HC = uniform_ecdf_upper_bound(n = n, m = m, delta= 0.1, lower = 0.01, upper = 0.5, method = 'MC-THC')

# ---- CoJER: threshold calibration (once; depends only on n, m, delta, B, k_max) ----
COJER_B = 10000
# k_max as a fraction of m, fixed a priori and reported side by side:
#   5% m : assumes sparsity (relevant sets contain at most ~10% of m, q_max = 0.5)
#   m    : no assumption on k_max
COJER_KFRAC = 0.05            
COJER_KFRACS_ALL = [0.05, 0.50, 1.00]    # used in the matched-pairs plot (intro_a_matched)
COJER_KMAX = {f: min(m, int(f * m)) for f in COJER_KFRACS_ALL}
def _thr_path(k):
    return Path(f'cojer_thr_n{n}_m{m}_B{COJER_B}_k{k}.npy')   # one cache file per k_max
_missing = [k for k in COJER_KMAX.values() if not _thr_path(k).exists()]
if _missing:   # one shared simulation for all the k_max still to calibrate
    _res = cojer_thresholds_multi(n, m, delta=0.1, B=COJER_B, k_max_list=_missing, n_jobs=-1)
    for k, thr in _res.items():
        np.save(_thr_path(k), thr)
    del _res
cojer_thr_all = {f: np.load(_thr_path(COJER_KMAX[f])) for f in COJER_KFRACS_ALL}
COJER_LABEL = {f: (r"CoJER ($k_{max}$=m)" if f == 1.0 else rf"CoJER ($k_{{max}}$={f} m)")
               for f in COJER_KFRACS_ALL}
COJER_NAME = {f: ("CoJER (k_max=m)" if f == 1.0 else f"CoJER (k_max={f} m)") for f in COJER_KFRACS_ALL}   # plain text, for print
COJER_MARKER = {f: m_ for f, m_ in zip(COJER_KFRACS_ALL, ['^', 'D', 'v', 'P'])}

# def V(xhat, y):
#     return 10*y - xhat
def V(yhat, y, c):
    M = 10  # Large constant M
    return M * (y > c) + c * (y <= c) - yhat

# def V(x,y):
#     return y-x
### compute (true) FDP if reject all pval < alpha
def FDP(alpha, pvals, truth):
    R = sum(pvals <= alpha)
    FR = R - sum(truth[pvals<=alpha])
    return FR/np.maximum(R,1)

grid = np.linspace(0,1,500)

experiment_results = []
alphas_bh = []
for _ in range(repeat_num):
    np.random.seed(_) 
    shuffled_indices = np.random.permutation(n_total)
    test_indices = shuffled_indices[:m]
    calib_indices = shuffled_indices[m:]
    test_data = all_data.iloc[test_indices].reset_index(drop=True)
    calib_data = all_data.iloc[calib_indices].reset_index(drop=True)
    
    #c = np.percentile(calib_data['true'], 70) # set the threshold. This is c

    test_score = V(test_data['pred'].values, test_data['q8'].values, test_data['q8'].values)
    calib_score = V(calib_data['pred'].values, calib_data['true'].values, calib_data['q8'].values)
    ## conformal p value
    #pvals = (np.random.rand(m) + np.sum(calib_score[:, None] < test_score[None, :], axis=0)) / (n + 1)
    pvals = np.zeros(m)
    for idx in range(m):
        pvals[idx] = (np.random.rand()* (1+np.sum( calib_score == test_score[idx])) + np.sum( calib_score < test_score[idx]))/(n+1)
    for alpha_bh in [0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95]:
    #for alpha_bh in [0.1, 0.2, 0.3,0.4,0.5]:
    #for alpha_bh in np.linspace(0.01, 0.5, 50):
        _, rejection_threshold = benjamini_hochberg(pvals, alpha = alpha_bh)
        if np.sum(pvals <= rejection_threshold) > 0.05*m:
            break
    print(np.sum(pvals <= rejection_threshold), rejection_threshold, alpha_bh)
    alphas_bh.append(alpha_bh)
    T = len(grid)
    FDPs = np.zeros(T)
    FDP_simulation = np.zeros(T)
    FDP_estimation = np.zeros(T)
    non_nulls = test_data['true'] > test_data['q8']
    non_nulls = non_nulls.astype(int)
    for i in range(T):
        FDPs[i] = FDP(grid[i], pvals, non_nulls)
        FDP_estimation[i] = m * grid[i]/np.maximum(sum(pvals <= grid[i]),1)
    FDP_simulation = uniform_FDP_upper_bound(pvals, n = n, m = m, delta= 0.1, lower = 0.01, upper = 0.5, method = 'MC-THC', boost = True)(grid)
    # CoJER: same sets {pvals <= t}; thresholds evaluated on the discrete p-values
    pvals_d = conformal_pvals_discrete(calib_score, test_score)
    R_bh = int(np.sum(pvals <= rejection_threshold))
    cojer_extra = {}
    for f in COJER_KFRACS_ALL:
        _sp_f, _bp_f = cojer_fdp_bound_sets(pvals, pvals_d, cojer_thr_all[f])
        cojer_extra[f'FDP_cojer_k{int(100 * f)}'] = cojer_bound_on_grid(_sp_f, _bp_f, grid)
        cojer_extra[f'FDP_BH_cojer_k{int(100 * f)}'] = _bp_f[R_bh - 1] if R_bh > 0 else 0.0
    FDP_BH = FDP(rejection_threshold, pvals, non_nulls)
    FDP_BH_upper_bound = m * upper_MC_HC(rejection_threshold)/np.maximum(sum(pvals <= rejection_threshold),1)
    experiment_results.append({
        'FDP_simulation': FDP_simulation, 
        'FDPs': FDPs,  
        'FDP_estimator': FDP_estimation,
        'FDP_BH': FDP_BH,
        'nominal_level': alphas_bh,
        'rejection_threshold': rejection_threshold,
        'FDP_BH_upper_bound': FDP_BH_upper_bound,
        **cojer_extra
    })

experiments_df = pd.DataFrame(experiment_results)


# In[ ]:


true_fdp = experiments_df["FDP_BH"].to_numpy()
upper = np.minimum(experiments_df["FDP_BH_upper_bound"].to_numpy(), 1.0)
nominal = np.asarray(alphas_bh)

order = np.argsort(true_fdp)

plt.figure(figsize=(10, 6))
plt.plot(true_fdp[order], label="True FDP", lw=2, marker="o")
plt.plot(upper[order], label="Proposed upper bound", lw=2, linestyle="--", marker="x")
for f in COJER_KFRACS_ALL:
    _uc = np.minimum(experiments_df[f"FDP_BH_cojer_k{int(100 * f)}"].to_numpy(), 1.0)
    plt.plot(_uc[order], label=COJER_LABEL[f], lw=2, linestyle="-.", marker=COJER_MARKER[f])
plt.plot(nominal[order], label="Nominal level", lw=2, linestyle=":", marker="s")

plt.xlabel("Experiments sorted by true FDP", fontsize=14)
plt.ylabel("Value", fontsize=14)
plt.title("Comparison after sorting by true FDP", fontsize=15)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(FIG_DIR / 'fdp_sorted_comparison.pdf', format='pdf', bbox_inches='tight')
plt.close()


# ---- CoJER: sensitivity to k_max (BH sets) ----
_true = experiments_df["FDP_BH"].to_numpy()
_gaps, _labels = [], []
print("k_max (% of m) | coverage | mean gap | median gap | simultaneous validity (grid)")
for f in COJER_KFRACS_ALL:
    _u = np.minimum(experiments_df[f"FDP_BH_cojer_k{int(100 * f)}"].to_numpy(), 1.0)
    _simult = sum(np.all(np.array(r[f"FDP_cojer_k{int(100 * f)}"]) >= np.array(r["FDPs"]))
                  for _, r in experiments_df.iterrows())
    print(f"{int(100 * f):>13d}% | {np.mean(_u >= _true):.3f}    | {np.mean(_u - _true):.3f}    | "
          f"{np.median(_u - _true):.3f}      | {_simult}/{len(experiments_df)}")
    _gaps.append(_u - _true); _labels.append(f"{int(100 * f)}%")
_up = np.minimum(experiments_df["FDP_BH_upper_bound"].to_numpy(), 1.0)
_gaps.append(_up - _true); _labels.append("MC-THC")
_labels = [(r"CoJER" "\n" r"($k_{max}$=m)" if l == "100%" else r"CoJER" "\n" rf"($k_{{max}}$={l} m)") if l != "MC-THC" else l
           for l in _labels]
plt.figure(figsize=(9, 5.5))
plt.boxplot(_gaps, tick_labels=_labels, showmeans=True)
plt.axhline(0, color="black", linestyle="--", lw=2)
plt.ylabel("Gap to realized FDP (BH sets)", fontsize=14)
plt.title("Gap to realized FDP: CoJER vs proposed bound", fontsize=15)
plt.grid(True, axis="y", alpha=0.3); plt.tight_layout()
plt.savefig(FIG_DIR / 'cojer_kmax_sensitivity.pdf', format='pdf', bbox_inches='tight')
plt.close()


# In[8]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Inputs
true_fdp = experiments_df["FDP_BH"].to_numpy()
upper = np.minimum(experiments_df["FDP_BH_upper_bound"].to_numpy(), 1.0)
upper_cojer = {f: np.minimum(experiments_df[f"FDP_BH_cojer_k{int(100 * f)}"].to_numpy(), 1.0)
               for f in COJER_KFRACS_ALL}
nominal = np.asarray(alphas_bh)

# Basic summaries
gap_upper = upper - true_fdp
gap_cojer = {f: upper_cojer[f] - true_fdp for f in COJER_KFRACS_ALL}
for f in COJER_KFRACS_ALL:
    print(f"Coverage of {COJER_NAME[f]}:    {np.mean(gap_cojer[f] >= 0):.3f}")
    print(f"Mean gap ({COJER_NAME[f]} - true):          {np.mean(gap_cojer[f]):.3f}")
    print(f"Median gap ({COJER_NAME[f]} - true):        {np.median(gap_cojer[f]):.3f}")
gap_nominal = nominal - true_fdp

cover_upper = np.mean(gap_upper >= 0)
cover_nominal = np.mean(gap_nominal >= 0)

mean_gap_upper = np.mean(gap_upper)
mean_gap_nominal = np.mean(gap_nominal)

median_gap_upper = np.median(gap_upper)
median_gap_nominal = np.median(gap_nominal)

print(f"Coverage of proposed upper bound: {cover_upper:.3f}")
print(f"Coverage of selected BH level:    {cover_nominal:.3f}")
print(f"Mean gap (upper - true):          {mean_gap_upper:.3f}")
print(f"Mean gap (BH level - true):       {mean_gap_nominal:.3f}")
print(f"Median gap (upper - true):        {median_gap_upper:.3f}")
print(f"Median gap (BH level - true):     {median_gap_nominal:.3f}")

# ---------- Figure ----------
fig, axes = plt.subplots(1, 2, figsize=(17, 6.5))

# ---------------- Panel 1: calibration scatter ----------------
ax = axes[0]

mn = min(true_fdp.min(), upper.min(), nominal.min(), *[u.min() for u in upper_cojer.values()])
mx = max(true_fdp.max(), upper.max(), nominal.max(), *[u.max() for u in upper_cojer.values()])
pad = 0.03 * (mx - mn + 1e-12)
lo, hi = mn - pad, mx + pad

ax.scatter(
    true_fdp, nominal,
    alpha=0.75, s=55,
    label=r"Selected BH level $\widehat{\alpha}_{\mathrm{BH}}$"
)
ax.scatter(
    true_fdp, upper,
    alpha=0.75, s=55,
    label="Proposed FDP upper bound"
)
for f in COJER_KFRACS_ALL:
    ax.scatter(true_fdp, upper_cojer[f], alpha=0.75, s=55, marker=COJER_MARKER[f], label=COJER_LABEL[f])
ax.plot([lo, hi], [lo, hi], "k--", lw=2, label=r"$y=x$")
fs = 16
ax.set_xlabel("Realized FDP", fontsize=fs)
ax.set_ylabel("Selected BH level or FDP upper bound", fontsize=fs)
ax.set_title("(a) Calibration against realized FDP", fontsize=18)
ax.set_xlim(lo, hi)
ax.set_ylim(lo, hi)
ax.tick_params(axis="both", labelsize=fs)
ax.legend(fontsize=fs-4)
ax.grid(True, alpha=0.3)

# ax.text(
#     0.04, 0.96,
#     f"Coverage, proposed bound: {cover_upper:.2f}\n"
#     f"Coverage, selected BH level: {cover_nominal:.2f}",
#     transform=ax.transAxes,
#     va="top",
#     fontsize=12,
#     bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8)
# )

# ---------------- Panel 2: gap distribution ----------------
ax = axes[1]

gap_df = pd.DataFrame({
    r"Selected BH level $\widehat{\alpha}_{\mathrm{BH}}$": gap_nominal,
    "Proposed FDP upper bound": gap_upper,
})

ax.boxplot(
    [gap_nominal, gap_upper] + [gap_cojer[f] for f in COJER_KFRACS_ALL],
    tick_labels=[
        "Selected\nBH level",
        "Proposed\nFDP bound",
    ] + [COJER_LABEL[f].replace(" (", "\n(") for f in COJER_KFRACS_ALL],
    showmeans=True
)

ax.axhline(0, color="black", linestyle="--", lw=2)
ax.set_ylabel("Gap to realized FDP", fontsize=fs)
ax.set_title("(b) Gap relative to realized FDP", fontsize=18)
ax.tick_params(axis="y", labelsize=fs)
ax.tick_params(axis="x", labelsize=fs - 3)
ax.grid(True, axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig(
    Path("all_figures") / "BH_better_visualization.pdf",
    format="pdf",
    bbox_inches="tight"
)
plt.close()


# In[ ]:


results = []

for idx, row in experiments_df.iterrows():
    fdps = np.array(row['FDPs'])
    fdp_simulation = np.array(row['FDP_simulation'])
    
    # Check if FDP_simulation is always greater than or equal to FDPs
    is_always_larger = np.all(fdp_simulation >= fdps)
    results.append(is_always_larger)

# Displaying or saving the dataframe
print(sum(results))
for f in COJER_KFRACS_ALL:
    print(f'{COJER_NAME[f]}, simultaneous validity:',
          sum(np.all(np.array(r[f'FDP_cojer_k{int(100 * f)}']) >= np.array(r['FDPs'])) for _, r in experiments_df.iterrows()))


# In[17]:


grid = np.linspace(0, 1, 500)
print(grid.size)
# Plotting the differences for each of the 100 experiments
plt.figure(figsize=(10, 8))

for idx, row in experiments_df.iterrows():
    # Assuming each value in the columns is a list of FDP values for the corresponding grid points
    fdps = np.array(row['FDPs'])
    fdp_simulation = np.array(row['FDP_simulation'])
    difference = fdp_simulation - fdps
    plt.plot(grid, difference, alpha=0.3, linewidth=0.8)  # Plot with some transparency for better visualization

plt.xlabel('Threshold (alpha)', fontsize=22)
plt.ylabel('FDP upper bound - FDP', fontsize=22)
plt.grid(True)
plt.xticks(fontsize=22)
plt.yticks(fontsize=22)

plt.xlim(-0.001, 0.5)
plt.ylim(-0.05, 0.15)
plt.tight_layout()
plt.savefig(Path('all_figures')/'intro_b.pdf', format = 'pdf')

plt.xlim(-0.001, 1.001)
plt.savefig(Path('all_figures')/'intro_b_full.pdf', format = 'pdf')

plt.close()


for f in COJER_KFRACS_ALL:
    _k = int(100 * f)
    plt.figure(figsize=(10, 8))
    for idx, row in experiments_df.iterrows():
        plt.plot(grid, np.array(row[f'FDP_cojer_k{_k}']) - np.array(row['FDPs']), alpha=0.3, linewidth=0.8)
    plt.xlabel('Threshold (alpha)', fontsize=22)
    plt.ylabel(f'{COJER_LABEL[f]} FDP upper bound - FDP', fontsize=22)
    plt.grid(True)
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    plt.ylim(-0.05, 0.15)
    plt.tight_layout()

    plt.xlim(-0.001, 0.5)
    plt.savefig(Path('all_figures')/f'intro_b_cojer_k{_k}.pdf', format='pdf')

    plt.xlim(-0.001, 1.0001)
    plt.savefig(Path('all_figures')/f'intro_b_cojer_k{_k}_full.pdf', format='pdf')
    plt.close()


# # Single Experiment

# In[10]:


test_data = pd.read_csv('test_199.csv')
calib_data = pd.read_csv('calib_199.csv')
test_data.rename(columns={'test_pred': 'pred', 'test_true': 'true'}, inplace=True)
calib_data.rename(columns={'calib_pred': 'pred', 'calib_true': 'true'}, inplace=True)
all_data = pd.concat([calib_data, test_data], ignore_index=True)
n = calib_data.shape[0]
m = test_data.shape[0]
n_total = m+n

upper_MC_HC = uniform_ecdf_upper_bound(n = n, m = m, delta= 0.1, lower = 0.01, upper = 0.5, method = 'MC-THC')

def V(yhat, y, c):
    return 10 * y - yhat

def V(yhat, y, c):
    M = 10  # Large constant M
    return M * (y > c) + c * (y <= c) - yhat


# def V(x,y):
#     return y-x
### compute (true) FDP if reject all pval < alpha
def FDP(alpha, pvals, truth):
    R = sum(pvals <= alpha)
    FR = R - sum(truth[pvals<=alpha])
    return FR/np.maximum(R,1)

grid = np.linspace(0,1,10000)


#np.random.seed(123) 
shuffled_indices = np.random.permutation(n_total)
test_indices = shuffled_indices[:m]
calib_indices = shuffled_indices[m:]
test_data = all_data.iloc[test_indices].reset_index(drop=True)
calib_data = all_data.iloc[calib_indices].reset_index(drop=True)

# set the threshold as the 70th percentile

test_score = V(test_data['pred'].values, test_data['q8'].values, test_data['q8'].values)
calib_score = V(calib_data['pred'].values, calib_data['true'].values, calib_data['q8'].values)
## conformal p value
#pvals = (np.random.rand(m) + np.sum(calib_score[:, None] < test_score[None, :], axis=0)) / (n + 1)
pvals = np.zeros(m)
for idx in range(m):
    pvals[idx] = (np.random.rand()* (1+np.sum( calib_score == test_score[idx])) + np.sum( calib_score < test_score[idx]))/(n+1)
_, rejection_threshold = benjamini_hochberg(pvals)

T = len(grid)
FDPs = np.zeros(T)
FDP_estimation = np.zeros(T)
non_nulls = test_data['true'] > test_data['q8']
non_nulls = non_nulls.astype(int)
for i in range(T):
    FDPs[i] = FDP(grid[i], pvals, non_nulls)
    FDP_estimation[i] = m * grid[i]/np.maximum(sum(pvals <= grid[i]),1)
# ---- bounds of the article, with ITS hyper-parameters (lower=0.01, upper=0.5) ----
_art = dict(n=n, m=m, delta=0.1, lower=0.01, upper=0.5, boost=True)
FDP_THC = uniform_FDP_upper_bound(pvals, method='MC-THC', **_art)(grid)
FDP_HC = uniform_FDP_upper_bound(pvals, method='MC-HC', **_art)(grid)
FDP_KS = uniform_FDP_upper_bound(pvals, method='MC-KS', **_art)(grid)
FDP_baseline = uniform_FDP_upper_bound(pvals, method='KS', **_art)(grid)

# ---- HC / THC restricted to [0, k_max/m], to be matched with CoJER ----
_m0 = dict(n=n, m=m, delta=0.1, boost=True)
FDP_HC_01 = uniform_FDP_upper_bound(pvals, method='MC-HC', lower=0, upper=1, **_m0)(grid)       # vs CoJER k_max=m
FDP_THC_050 = uniform_FDP_upper_bound(pvals, method='MC-THC', lower=0, upper=0.5, **_m0)(grid)  # vs CoJER k_max=50% m
FDP_THC_005 = uniform_FDP_upper_bound(pvals, method='MC-THC', lower=0, upper=0.05, **_m0)(grid) # vs CoJER k_max=5% m

# ---- CoJER ----
pvals_d = conformal_pvals_discrete(calib_score, test_score)
FDP_cojer_all = {}
for _f in COJER_KFRACS_ALL:
    _spf, _bpf = cojer_fdp_bound_sets(pvals, pvals_d, cojer_thr_all[_f])
    FDP_cojer_all[_f] = cojer_bound_on_grid(_spf, _bpf, grid)


# In[11]:


# Number of rejections for each threshold
num_rejections = np.array([np.sum(pvals <= t) for t in grid])
_pos = num_rejections > 0          # log x-axis: drop the empty rejection sets
fs = 15


def plot_intro_a(curves, fname, original = False):
    """curves: list of (label, y over grid, matplotlib kwargs). x = number of selections, log scale."""
    plt.figure()
    for label, y, kw in curves:
        plt.plot(num_rejections[_pos], np.asarray(y)[_pos], label=label, **kw)
    if original:
        plt.xlim(-100, 5000)
    else:
        plt.xscale('log')
        plt.xlim(1, num_rejections.max())
    plt.xlabel('Number of Selections', fontsize=fs)
    plt.ylabel('Values', fontsize=fs)
    plt.xticks(fontsize=fs)
    plt.yticks(fontsize=fs)
    plt.title('True FDP and several FDP upper bounds', fontsize=fs)
    plt.legend(fontsize=fs - 4)
    plt.savefig(Path('all_figures') / fname, format='png', bbox_inches='tight')
    plt.close()


_true = ('True FDP', FDPs, {})
_article = [('MC-THC', FDP_THC, {}),
            ('MC-HC', FDP_HC, {}),
            ('MC-KS', FDP_KS, {}),
            ('baseline', FDP_baseline, {'linestyle': '--'})]

# 1) original figure of the article (hyper-parameters lower=0.01, upper=0.5)
plot_intro_a([_true] + _article, 'intro_a_orig.png', original=True)

# 1) original figure of the article (hyper-parameters lower=0.01, upper=0.5), log x
plot_intro_a([_true] + _article, 'intro_a.png')

# 2) same + CoJER 
plot_intro_a([_true] + _article
             + [('CoJER', FDP_cojer_all[COJER_KFRAC], {})],
             'intro_a_cojer.png')

# 3) matched pairs: HC / THC[0, k_max/m] (solid) vs CoJER (dashed), same colour per pair
_pairs = [  # (HC/THC label, HC/THC curve, CoJER k_max fraction, colour)
    ('MC-HC', FDP_HC_01, 1.00, 'tab:green'),
    (r'MC-THC [0, 0.5]', FDP_THC_050, 0.50, 'tab:orange'),
    (r'MC-THC [0, 0.05]', FDP_THC_005, 0.05, 'tab:red'),
]
_curves = [('True FDP', FDPs, {'color': 'k'})]
for _lab, _y, _f, _c in _pairs:
    _curves.append((_lab, _y, {'color': _c, 'linestyle': '-', 'linewidth': 2}))
    _curves.append((COJER_LABEL[_f], FDP_cojer_all[_f], {'color': _c, 'linestyle': '--', 'linewidth': 2}))
plot_intro_a(_curves, 'intro_a_matched.png')
