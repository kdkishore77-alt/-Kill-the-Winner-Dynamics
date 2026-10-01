"""
sensitivity.py -- one-at-a-time (OAT) robustness analyses, using the SAME model
(plankton_model.py) as the main experiment.

    python sensitivity.py --analysis tier1        # rho*, delta_v, V*         (original three)
    python sensitivity.py --analysis tier2        # every other model parameter          [R2]
    python sensitivity.py --analysis grid         # L = 50, 70, 100, 140                 [R1-8, R2]
    python sensitivity.py --analysis tradeoff     # alpha and exponent of K = alpha r^k  [R2]
    python sensitivity.py --analysis tradeoff_matched  # exponent k with alpha rescaled so the mean K_s is unchanged
    python sensitivity.py --analysis mutation     # mu = 0, .001, .01, .05               [R1-5]
    python sensitivity.py --analysis all --reps 10 --gens 600 --workers 8
    QUICK=1 python sensitivity.py --analysis tier1   # tiny smoke test

Changes relative to the submitted version:
  * The submitted sensitivity script was NOT the main model: it let viral particles diffuse
    (main model: no viral transport), re-seeded numpy's global generator to 42 inside every
    constructor (so all replicates shared one random stream after initialisation), and
    compared KtW only with the no-mortality ablation. It now imports the main model.
  * The decisive control (fixed mortality) is included in every sweep, and p_fixed is
    RE-CALIBRATED at every parameter value so the fixed condition always kills as many fed
    cells per generation as KtW does at that value.
  * The three conditions share the starting lattice of each replicate (paired design).
  * Default is 10 replicates x 600 generations (was 5 x 600); the 600-generation horizon is
    justified by extended_run_checkpoints.csv written by 1A.py.
  * Collapse criterion (Delta H' < 0.05) is applied to KtW - Fixed (decisive) and KtW - None.
"""
import argparse, os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor

from plankton_model import PlanktonEcosystemSimulation as Sim, calibrate_fixed_rate

QUICK = os.environ.get('QUICK') == '1'
COLLAPSE = 0.05
CONDS = [('KtW_ON', 'ktw'), ('Fixed_Mort', 'fixed'), ('KtW_OFF', 'none')]
COL = {'KtW_ON': '#2166AC', 'Fixed_Mort': '#D6604D', 'KtW_OFF': '#F4A82E'}

# baselines (Table 1); keys are Sim() keyword names
BASE = dict(grid_size=70, num_strains=12, mutation_rate=0.0, alpha=2.0, tradeoff_exp=2.0,
            nutrient_inflow=0.10, nutrient_max=3.0, delta_v=0.8, lambda_v=0.98, rho_star=0.35,
            v_star=1.0, v_ref=2.0, p_max=0.95, p_d=0.30, t_s=3, eta=2, phi0=0.70)

SWEEPS = {
    'tier1': {'rho_star': [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50],
              'delta_v': [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6],
              'v_star': [0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8]},
    'tier2': {'v_ref': [1.0, 1.5, 2.0, 3.0, 4.0], 'p_max': [0.5, 0.7, 0.95, 1.0],
              'lambda_v': [0.90, 0.95, 0.98, 0.99], 'p_d': [0.1, 0.2, 0.3, 0.4, 0.5],
              't_s': [1, 2, 3, 5, 8], 'eta': [1, 2, 3, 4],
              'nutrient_inflow': [0.05, 0.075, 0.10, 0.15, 0.20], 'nutrient_max': [1.5, 3.0, 6.0],
              'phi0': [0.3, 0.5, 0.7, 0.9], 'num_strains': [6, 12, 24]},
    'grid': {'grid_size': [50, 70, 100, 140]},
    # tradeoff_exp = 0 removes the trade-off altogether (K_s = alpha for every strain)
    'tradeoff': {'alpha': [0.5, 1.0, 2.0, 3.0, 4.0], 'tradeoff_exp': [0.0, 1.0, 2.0, 3.0]},
    'mutation': {'mutation_rate': [0.0, 0.001, 0.01, 0.05]},
    # Exponent test with the mean half-saturation constant held at its baseline value:
    # alpha is rescaled for every k, so only the SHAPE of the trade-off changes (k = 0 gives
    # identical K_s for all strains, i.e. no trade-off). The plain 'tradeoff' sweep above changes
    # both shape and level of K_s; with k = 0 and alpha = 2, K_s = 2 exceeds the initial nutrient
    # and the lattice empties, so that point is not informative about the trade-off.
    'tradeoff_matched': {'tradeoff_exp': [0.0, 1.0, 2.0, 3.0]},
}
LABEL = {'rho_star': 'ρ*', 'delta_v': 'δ_v', 'v_star': 'V*', 'v_ref': 'V_ref', 'p_max': 'p_max',
         'lambda_v': 'λ_v', 'p_d': 'p_d', 't_s': 'T_s', 'eta': 'η', 'nutrient_inflow': 'I_n',
         'nutrient_max': 'N_max', 'phi0': 'φ_0', 'num_strains': 'M', 'grid_size': 'L',
         'alpha': 'α', 'tradeoff_exp': 'trade-off exponent k', 'mutation_rate': 'μ'}


def matched_alpha(k, base=BASE):
    """alpha such that the mean K_s over the (fixed-seed) strain traits equals its baseline value."""
    r = np.random.default_rng(42).uniform(0.15, 0.45, size=base['num_strains'] + 1)[1:]
    return float(base['alpha'] * np.mean(r ** base['tradeoff_exp']) / np.mean(r ** k))


def run_point(job):
    """One (parameter value, condition, replicate) simulation. Top-level for multiprocessing."""
    param, value, cname, mode, rep, p_fixed, gens, kw = job
    sim = Sim(mode=mode, fixed_mort_rate=p_fixed, seed=rep, **kw).run(gens)
    last = slice(-50, None)
    return dict(parameter=param, value=value, condition=cname, replicate=rep,
                shannon=float(np.mean(sim.history_diversity[last])),
                richness=float(np.mean(sim.history_richness[last])),
                richness_ge10=float(np.mean(sim.history_richness10[last])),
                dominant_pct=100 * float(np.mean(sim.history_dominant[last])),
                vacancies=float(np.mean(sim.history_vacancies[last])),
                realised_death_rate=sim.mean_lysis_rate(), p_fixed=p_fixed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--analysis', default='tier1', choices=list(SWEEPS) + ['all'])
    ap.add_argument('--reps', type=int, default=10)
    ap.add_argument('--gens', type=int, default=600)
    ap.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) - 1))
    a = ap.parse_args()
    if QUICK: a.reps, a.gens = 2, 40
    pilot = (2, 40) if QUICK else (4, 300)
    names = list(SWEEPS) if a.analysis == 'all' else [a.analysis]

    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for analysis in names:
            print(f'=== {analysis}: {a.reps} replicates x {a.gens} generations ===')
            jobs, meta = [], []
            for param, values in SWEEPS[analysis].items():
                if QUICK: values = values[:2]
                for v in values:
                    kw = dict(BASE); kw[param] = v
                    if analysis == 'tradeoff_matched':
                        kw['alpha'] = matched_alpha(v)
                    if param == 'grid_size':                       # keep run time reasonable
                        pass
                    # re-calibrate the fixed-mortality rate for THIS parameter setting
                    pf = calibrate_fixed_rate(*pilot, **kw)
                    for cname, mode in CONDS:
                        for rep in range(a.reps):
                            jobs.append((param, v, cname, mode, rep, pf, a.gens, kw))
            raw = pd.DataFrame(list(ex.map(run_point, jobs, chunksize=1)))
            raw.to_csv(f'sensitivity_{analysis}_raw.csv', index=False)

            g = raw.groupby(['parameter', 'value', 'condition'])
            piv = g['shannon'].mean().unstack('condition')
            sd = g['shannon'].std().unstack('condition')
            out = piv.copy()
            out['dH_vs_fixed'] = piv['KtW_ON'] - piv['Fixed_Mort']
            out['dH_vs_none'] = piv['KtW_ON'] - piv['KtW_OFF']
            out['status_vs_fixed'] = np.where(out['dH_vs_fixed'] >= COLLAPSE, 'ROBUST', 'COLLAPSED')
            out['status_vs_none'] = np.where(out['dH_vs_none'] >= COLLAPSE, 'ROBUST', 'COLLAPSED')
            out['p_fixed_used'] = raw.groupby(['parameter', 'value'])['p_fixed'].mean()
            out['richness_KtW'] = g['richness'].mean().unstack('condition')['KtW_ON']
            out['richness_Fixed'] = g['richness'].mean().unstack('condition')['Fixed_Mort']
            out['richness_None'] = g['richness'].mean().unstack('condition')['KtW_OFF']
            out.reset_index().to_csv(f'sensitivity_{analysis}_summary.csv', index=False)
            print(out[['KtW_ON', 'Fixed_Mort', 'KtW_OFF', 'dH_vs_fixed', 'dH_vs_none',
                       'status_vs_fixed', 'status_vs_none']].round(3).to_string())

            params = list(SWEEPS[analysis])
            fig, axs = plt.subplots(1, len(params), figsize=(4.2 * len(params), 3.8), squeeze=False)
            for ax, p in zip(axs[0], params):
                for c, _ in CONDS:
                    m = piv.xs(p, level='parameter')[c]; s = sd.xs(p, level='parameter')[c]
                    ax.plot(m.index, m.values, 'o-', color=COL[c], label=c.replace('_', ' '))
                    ax.fill_between(m.index, m - s, m + s, color=COL[c], alpha=0.2)
                ax.axvline(BASE[p], color='gray', ls='--'); ax.set_xlabel(LABEL[p])
                ax.set_ylabel("Final Shannon H′" if ax is axs[0][0] else '')
            axs[0][0].legend(fontsize=8)
            fig.suptitle(f'Sensitivity ({analysis}): mean ± s.d., n={a.reps}, {a.gens} generations')
            fig.tight_layout(); fig.savefig(f'sensitivity_{analysis}.png', dpi=300); plt.close(fig)
    print('Done.')


if __name__ == '__main__':
    main()
