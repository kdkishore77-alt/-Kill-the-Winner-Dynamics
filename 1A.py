"""
1A.py -- main three-condition experiment (KtW-active | fixed-mortality | no-mortality).
Uses the shared model in plankton_model.py.   QUICK=1 python 1A.py  runs a tiny smoke test.

Changes relative to the submitted version (each tagged with the reviewer point it answers):
  [R1-3,R2] model moved to plankton_model.py with a full plain-language description;
            parameter table with units/status exported to parameters.csv
  [R2]      20 replicates (was 10); 95% CIs and a replicate-convergence figure
  [R1-6]    extended 2400-generation run (5 replicates) with checkpoint table
  [R1-7]    rank-abundance curves + exponential-vs-power-law fits
  [R1-10,R2] isotropy diagnostic + axis-aligned, equal-aspect rasters; new spatially
            explicit local-diversity / nutrient maps (Fig. 5); patch-size statistic
  [R2]      p_fixed calibrated per FED cell (the cells that can actually die), realised
            mortality of all conditions reported so "same quantity of mortality" is checkable
  [R2]      viability threshold (10/20/40 cells) recorded every generation
  [both]    paired design: for replicate k all three conditions start from the SAME lattice
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.stats import mannwhitneyu

from plankton_model import (PlanktonEcosystemSimulation as Sim, PARAMETER_TABLE, calibrate_fixed_rate,
                            rank_abundance, fit_rank_abundance, local_shannon, patch_stats, isotropy)

QUICK = os.environ.get('QUICK') == '1'
L, M = 70, 12
MU_MAIN = 0.0            # main runs use mu = 0 (Table 1 of the revision states this); mu is swept in sensitivity.py
GENERATIONS = 60 if QUICK else 1200
N_REPLICATES = 3 if QUICK else 20
N_PILOT, PILOT_GENS = (2, 40) if QUICK else (5, 400)
EXT_GENS, EXT_REPS = (100, 2) if QUICK else (2400, 5)
CONDS = [('KtW_ON', 'ktw'), ('Fixed_Mort', 'fixed'), ('KtW_OFF', 'none')]

plt.rcParams.update({'font.family': 'serif', 'font.size': 11, 'axes.spines.top': False,
                     'axes.spines.right': False, 'figure.dpi': 150, 'savefig.dpi': 300,
                     'savefig.bbox': 'tight'})
COL = {'KtW_ON': '#2166AC', 'Fixed_Mort': '#D6604D', 'KtW_OFF': '#F4A82E'}
LAB = {'KtW_ON': 'KtW-active', 'Fixed_Mort': 'Fixed mortality', 'KtW_OFF': 'No mortality'}

# --------------------------------------------------------------------------
# Parameter table (units / status) -> parameters.csv                    [R2]
# --------------------------------------------------------------------------
pd.DataFrame(PARAMETER_TABLE, columns=['name', 'symbol', 'value', 'units', 'status']).to_csv('parameters.csv', index=False)

# --------------------------------------------------------------------------
# Step 1: calibrate p_fixed so fixed-mortality kills as many FED cells per
# generation as KtW does (same "quantity" of mortality, no density-dependence).
# --------------------------------------------------------------------------
print('STEP 1: calibrating p_fixed')
P_FIXED = calibrate_fixed_rate(N_PILOT, PILOT_GENS, grid_size=L, num_strains=M, mutation_rate=MU_MAIN)
print(f'  p_fixed = {P_FIXED:.5f} deaths per fed cell per generation')

# --------------------------------------------------------------------------
# Step 2: main experiment (paired initial lattices)
# --------------------------------------------------------------------------
print('STEP 2: main experiment')
runs = {c: [] for c, _ in CONDS}
for rep in range(N_REPLICATES):
    for cname, mode in CONDS:
        sim = Sim(grid_size=L, num_strains=M, mutation_rate=MU_MAIN, mode=mode,
                  fixed_mort_rate=P_FIXED, seed=rep).run(GENERATIONS)
        runs[cname].append(sim)
    print(f'  replicate {rep + 1}/{N_REPLICATES} done')

# --------------------------------------------------------------------------
# Step 3: raw data + summary statistics
# --------------------------------------------------------------------------
HIST = dict(shannon_diversity='history_diversity', richness='history_richness',
            richness_ge10='history_richness10', richness_ge40='history_richness40',
            dominant_fraction='history_dominant', top3_fraction='history_top3',
            vacancies='history_vacancies', mean_nutrient='history_nutrient',
            lysis_events='history_lysis', fed_cells='history_fed')
rows = []
for cname, sims in runs.items():
    for rep, s in enumerate(sims):
        for g in range(GENERATIONS):
            d = dict(condition=cname, replicate=rep, generation=g)
            d.update({k: getattr(s, v)[g] for k, v in HIST.items()})
            rows.append(d)
pd.DataFrame(rows).to_csv('simulation_data.csv', index=False)

def final_mean(sims, attr, n=50):          # per-replicate mean over the last n generations
    return np.array([np.mean(getattr(s, attr)[-n:]) for s in sims])

def ci95(x):
    x = np.asarray(x, float)
    h = 1.96 * x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else np.nan
    return x.mean() - h, x.mean() + h

summ = []
for cname, sims in runs.items():
    H = final_mean(sims, 'history_diversity')
    lo, hi = ci95(H)
    ps = [patch_stats(s.grid, M) for s in sims]
    iso = pd.DataFrame([isotropy(s.grid) for s in sims]).mean().to_dict()
    ra = pd.DataFrame([fit_rank_abundance(rank_abundance(s.grid, M)) for s in sims]).mean().to_dict()
    summ.append(dict(condition=cname, n_replicates=len(sims), p_fixed=P_FIXED,
                     shannon_mean=H.mean(), shannon_sd=H.std(ddof=1), shannon_ci95_lo=lo, shannon_ci95_hi=hi,
                     richness_ge20_mean=final_mean(sims, 'history_richness').mean(),
                     richness_ge10_mean=final_mean(sims, 'history_richness10').mean(),
                     richness_ge40_mean=final_mean(sims, 'history_richness40').mean(),
                     dominant_pct_mean=100 * final_mean(sims, 'history_dominant').mean(),
                     top3_pct_mean=100 * final_mean(sims, 'history_top3').mean(),
                     vacancies_mean=final_mean(sims, 'history_vacancies').mean(),
                     mean_nutrient=final_mean(sims, 'history_nutrient').mean(),
                     realised_death_rate_per_fed_cell=np.mean([s.mean_lysis_rate() for s in sims]),
                     realised_death_rate_per_occupied_cell=np.mean([s.mean_lysis_rate_per_occupied() for s in sims]),
                     mean_patch_size_cells=np.mean(ps),
                     iso_H=iso['H'], iso_V=iso['V'], iso_Dplus=iso['D_plus'], iso_Dminus=iso['D_minus'],
                     **{f'rankabund_{k}': v for k, v in ra.items()}))
pd.DataFrame(summ).to_csv('simulation_summary.csv', index=False)

# paired tests on final Shannon diversity (same replicate index = same starting lattice)
Hk, Hf, Hn = (final_mean(runs[c], 'history_diversity') for c in ('KtW_ON', 'Fixed_Mort', 'KtW_OFF'))
tests = []
for name, other in (('KtW_vs_Fixed', Hf), ('KtW_vs_None', Hn), ('Fixed_vs_None', None)):
    a, b = (Hk, other) if other is not None else (Hf, Hn)
    u = mannwhitneyu(a, b, alternative='two-sided') if len(a) > 1 else None
    tests.append(dict(comparison=name, mean_diff=float(a.mean() - b.mean()),
                      paired_diff_sd=float((a - b).std(ddof=1)) if len(a) > 1 else np.nan,
                      mannwhitney_U=None if u is None else float(u.statistic),
                      p_value=None if u is None else float(u.pvalue)))
pd.DataFrame(tests).to_csv('simulation_tests.csv', index=False)
print(pd.DataFrame(summ)[['condition', 'shannon_mean', 'shannon_ci95_lo', 'shannon_ci95_hi',
                          'realised_death_rate_per_fed_cell']].to_string(index=False))

# --------------------------------------------------------------------------
# Step 4: figures
# --------------------------------------------------------------------------
gens = np.arange(GENERATIONS)
def band(ax, x, sims, attr, cname, scale=1.0):
    a = np.array([getattr(s, attr) for s in sims]) * scale
    m, sd = a.mean(0), a.std(0, ddof=1) if len(a) > 1 else 0 * a.mean(0)
    ax.plot(x, m, color=COL[cname], lw=1.8, label=LAB[cname])
    ax.fill_between(x, m - sd, m + sd, color=COL[cname], alpha=0.2)

def dress(ax, ylabel, xmax=None):
    ax.set_xlabel('Generation'); ax.set_ylabel(ylabel); ax.set_xlim(0, xmax or GENERATIONS)
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(4)); ax.legend(loc='best')

fig, axs = plt.subplots(2, 2, figsize=(11, 8.5)); fig.subplots_adjust(hspace=0.38, wspace=0.30)
for ax, (attr, yl, sc, ttl) in zip(axs.flat, [
        ('history_richness', 'Strain richness (≥20 cells)', 1, 'Strain richness'),
        ('history_diversity', "Shannon diversity H′", 1, 'Shannon diversity'),
        ('history_dominant', 'Dominant strain (% grid)', 100, 'Dominant strain'),
        ('history_top3', 'Top-3 strain share (% grid)', 100, 'Top-3 strain share')]):
    for c, _ in CONDS: band(ax, gens, runs[c], attr, c, sc)
    dress(ax, yl); ax.set_title(ttl)
for letter, ax in zip('ABCD', axs.flat): ax.text(-0.10, 1.04, letter, transform=ax.transAxes, fontsize=13, fontweight='bold')
fig.suptitle(f'Figure 1. Diversity metrics (mean ± 1 s.d., n={N_REPLICATES} replicates)', y=1.01)
fig.savefig('fig1_diversity_metrics.pdf'); fig.savefig('fig1_diversity_metrics.png')

fig, ax = plt.subplots(figsize=(6.5, 4.5))
for c, _ in CONDS: band(ax, gens, runs[c], 'history_vacancies', c)
dress(ax, 'Vacant cells'); ax.set_title(f'Figure 2. Vacant cells (n={N_REPLICATES})')
fig.savefig('fig2_vacancies.pdf'); fig.savefig('fig2_vacancies.png')

# Fig 3: realised mortality (shows the "same quantity of mortality" claim explicitly)
fig, ax = plt.subplots(figsize=(6.5, 4.5))
for c in ('KtW_ON', 'Fixed_Mort'):
    a = np.array([np.array(s.history_lysis) / np.maximum(np.array(s.history_fed), 1) for s in runs[c]])
    ax.plot(gens, a.mean(0), color=COL[c], label=LAB[c])
dress(ax, 'Deaths per fed cell per generation'); ax.set_title('Figure 3. Realised mortality of the two mortality conditions')
fig.savefig('fig3_realised_mortality.pdf'); fig.savefig('fig3_realised_mortality.png')

# Fig 4: final grids. Plain axis-aligned raster, square cells, no transform. [R1-10]
cmap = plt.get_cmap('tab20', M + 1)
fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
for ax, (c, _) in zip(axes, CONDS):
    im = ax.imshow(runs[c][-1].grid, cmap=cmap, vmin=0, vmax=M, interpolation='nearest', aspect='equal')
    ax.set_title(LAB[c]); ax.set_xticks([]); ax.set_yticks([])
cb = fig.colorbar(im, ax=axes, fraction=0.015, pad=0.02, ticks=range(M + 1))
cb.ax.set_yticklabels(['Vacant'] + [f'Strain {i}' for i in range(1, M + 1)], fontsize=8)
fig.suptitle('Figure 4. Final lattices (last replicate); 1 cell = 1 square pixel', y=1.02)
fig.savefig('fig4_spatial_grids.pdf'); fig.savefig('fig4_spatial_grids.png')

# Fig 5: spatially explicit maps of local diversity (11x11 window) and nutrient   [R2]
fig, axes = plt.subplots(2, 3, figsize=(13, 8))
for j, (c, _) in enumerate(CONDS):
    s = runs[c][-1]
    im1 = axes[0, j].imshow(local_shannon(s.grid, M), cmap='viridis', vmin=0, vmax=np.log(6), aspect='equal')
    im2 = axes[1, j].imshow(s.nutrients, cmap='magma', vmin=0, vmax=max(0.5, s.nutrients.max()), aspect='equal')
    axes[0, j].set_title(LAB[c])
    for a in axes[:, j]: a.set_xticks([]); a.set_yticks([])
fig.colorbar(im1, ax=axes[0, :], fraction=0.02, label="Local Shannon H′ (11×11 window)")
fig.colorbar(im2, ax=axes[1, :], fraction=0.02, label='Local nutrient')
fig.suptitle('Figure 5. Spatially explicit local diversity (top) and nutrient (bottom), generation ' + str(GENERATIONS), y=0.98)
fig.savefig('fig5_spatial_maps.pdf'); fig.savefig('fig5_spatial_maps.png')

# Fig 6: rank-abundance                                                     [R1-7]
fig, ax = plt.subplots(figsize=(6.5, 4.5))
for c, _ in CONDS:
    ra = [rank_abundance(s.grid, M) for s in runs[c]]
    k = max(len(r) for r in ra)
    mean = np.nanmean([np.pad(r, (0, k - len(r)), constant_values=np.nan) for r in ra], axis=0)
    ax.semilogy(np.arange(1, k + 1), mean, 'o-', color=COL[c], label=LAB[c])
ax.set_xlabel('Strain rank'); ax.set_ylabel('Relative abundance'); ax.legend()
ax.set_title('Figure 6. Rank-abundance at final generation (mean of replicates)')
fig.savefig('fig6_rank_abundance.pdf'); fig.savefig('fig6_rank_abundance.png')

# Fig 7: is n enough?  running mean of final H' with 95% CI vs number of replicates  [R2]
fig, ax = plt.subplots(figsize=(6.5, 4.5))
for c, _ in CONDS:
    H = final_mean(runs[c], 'history_diversity')
    n = np.arange(2, len(H) + 1)
    if len(n):
        mu_ = [H[:k].mean() for k in n]; hw = [1.96 * H[:k].std(ddof=1) / np.sqrt(k) for k in n]
        ax.plot(n, mu_, color=COL[c], label=LAB[c]); ax.fill_between(n, np.array(mu_) - hw, np.array(mu_) + hw, color=COL[c], alpha=0.2)
ax.set_xlabel('Number of replicates'); ax.set_ylabel("Final Shannon H′ (mean, 95% CI)"); ax.legend()
fig.savefig('fig7_replicate_convergence.pdf'); fig.savefig('fig7_replicate_convergence.png')

# --------------------------------------------------------------------------
# Step 5: run-length check (2400 generations)                             [R1-6]
# --------------------------------------------------------------------------
print('STEP 5: extended runs')
ext = {c: [Sim(grid_size=L, num_strains=M, mutation_rate=MU_MAIN, mode=m, fixed_mort_rate=P_FIXED,
               seed=rep).run(EXT_GENS) for rep in range(EXT_REPS)] for c, m in CONDS}
cps = [g for g in (300, 600, 1200, 1800, 2400) if g <= EXT_GENS] or [EXT_GENS]
rows = [dict(condition=c, generation=g, shannon_mean=np.mean([np.mean(s.history_diversity[max(0, g - 50):g]) for s in ext[c]]),
             richness_mean=np.mean([np.mean(s.history_richness[max(0, g - 50):g]) for s in ext[c]]),
             richness_ge10_mean=np.mean([np.mean(s.history_richness10[max(0, g - 50):g]) for s in ext[c]]))
        for c in ext for g in cps]
pd.DataFrame(rows).to_csv('extended_run_checkpoints.csv', index=False)
fig, axs = plt.subplots(1, 2, figsize=(12, 4.3))
for c, _ in CONDS:
    for ax, attr, yl in ((axs[0], 'history_diversity', "Shannon H′"), (axs[1], 'history_richness', 'Richness (≥20 cells)')):
        band(ax, np.arange(EXT_GENS), ext[c], attr, c); dress(ax, yl, EXT_GENS)
fig.suptitle(f'Figure S1. Extended run to {EXT_GENS} generations (n={EXT_REPS})', y=1.02)
fig.savefig('figS1_extended_run.pdf'); fig.savefig('figS1_extended_run.png')
print('All outputs written.')
