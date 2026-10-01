"""
plankton_model.py -- single shared implementation of the three-rule plankton
cellular automaton, imported by BOTH 1A.py (main experiment) and
sensitivity.py (robustness analyses), so the two can never drift apart.

Model in one paragraph (all times are in dimensionless "generations"):
  * L x L periodic lattice. Each cell is vacant (0) or holds exactly ONE
    strain (1..M). A cell can never hold two strains.
  * Rule 1 (growth): a cell of strain s sees nutrient N. If N >= K_s it "feeds":
    it removes K_s from N, its starvation counter resets, and it may replicate
    with probability P_g = r_s * N / (K_s + N). If N < K_s it does not feed and
    its starvation counter goes up by one; after T_s consecutive unfed
    generations the cell becomes vacant. Cells are processed in random order.
  * Rule 2 (mortality; only for FED cells), one of three conditions:
        'ktw'   : viral load V_s(x,y) rises by delta_v when the local density of
                  s (in a (2*eta+1)^2 window) exceeds rho*; if V > V* the cell
                  is lysed with probability min(p_max, V / V_ref).
        'fixed' : each fed cell dies with constant probability p_fixed.
        'none'  : no mortality.
  * Rule 3 (transport): nutrients diffuse by a 3x3 kernel (periodic) and are
    topped up by a constant inflow, capped at N_max; viral loads decay by
    lambda_v each generation and do NOT move.
  * Trade-off: K_s = alpha * r_s ** TRADEOFF_EXP (paper: exponent 2).
  * Offspring go to a random cell of the 3x3 neighbourhood (which includes the
    parent's own cell). Empty target -> occupied. Target held by a DIFFERENT
    strain -> replaced with probability p_d. p_d does not depend on either
    strain's traits (deliberate simplification: competitive differences enter
    only through Rule 1).
  * Mutation: with probability mu an offspring is relabelled to a uniformly
    random strain of the M existing strains (undirected relabelling; no new
    trait values are created, so it acts like a low-rate random immigration).

Random numbers: every replicate owns a private numpy Generator, so results do
not depend on global state and replicates are statistically independent.
"""
import numpy as np
from scipy.signal import convolve2d
from scipy.ndimage import uniform_filter, label
from scipy.stats import entropy

# Baseline parameter values (Table 1) with units / status, exported to CSV by 1A.py
PARAMETER_TABLE = [
    # name, symbol, value, units, status
    ("grid_size", "L", 70, "cells", "free computational choice; swept in sensitivity.py"),
    ("num_strains", "M", 12, "-", "free computational choice; swept in sensitivity.py"),
    ("phi0", "phi_0", 0.70, "-", "free computational choice; swept in sensitivity.py"),
    ("N0", "N_0", 1.0, "nutrient units (dimensionless)", "free; initial condition only"),
    ("r_min", "r_min", 0.15, "generation^-1", "free computational choice"),
    ("r_max", "r_max", 0.45, "generation^-1", "free computational choice"),
    ("alpha", "alpha", 2.0, "nutrient units / (generation^-1)^exp", "free; swept in sensitivity.py"),
    ("tradeoff_exp", "k", 2.0, "-", "K_s = alpha r^k; k=2 in paper, k in {1,2,3} tested"),
    ("nutrient_inflow", "I_n", 0.10, "nutrient units generation^-1", "free; swept in sensitivity.py"),
    ("nutrient_max", "N_max", 3.0, "nutrient units", "free; swept in sensitivity.py"),
    ("delta_v", "delta_v", 0.8, "viral-load units", "free; swept in sensitivity.py"),
    ("lambda_v", "lambda_v", 0.98, "-", "free; swept in sensitivity.py"),
    ("rho_star", "rho*", 0.35, "fraction of window", "free; swept in sensitivity.py"),
    ("V_star", "V*", 1.0, "viral-load units", "free; swept in sensitivity.py"),
    ("V_ref", "V_ref", 2.0, "viral-load units", "free; swept in sensitivity.py"),
    ("p_max", "p_max", 0.95, "-", "free; swept in sensitivity.py"),
    ("p_d", "p_d", 0.30, "-", "free; swept in sensitivity.py"),
    ("T_s", "T_s", 3, "generations", "free; swept in sensitivity.py"),
    ("eta", "eta", 2, "cells", "free; swept in sensitivity.py"),
    ("mu", "mu", 0.0, "per replication", "0 in main runs; mu in {0,.001,.01,.05} tested"),
    ("p_fixed", "p_fixed", None, "per fed cell per generation", "CALIBRATED at run time (see 1A.py)"),
]


class PlanktonEcosystemSimulation:
    def __init__(self, grid_size=70, num_strains=12, mutation_rate=0.0,
                 mode='ktw', fixed_mort_rate=0.0, seed=0, dyn_seed=None,
                 alpha=2.0, tradeoff_exp=2.0, r_min=0.15, r_max=0.45,
                 n0=1.0, nutrient_inflow=0.10, nutrient_max=3.0,
                 delta_v=0.8, lambda_v=0.98, rho_star=0.35, v_star=1.0,
                 v_ref=2.0, p_max=0.95, p_d=0.30, t_s=3, eta=2, phi0=0.70):
        assert mode in ('ktw', 'fixed', 'none')
        self.N, self.M, self.mode = grid_size, num_strains, mode
        self.mu, self.p_fixed = mutation_rate, fixed_mort_rate
        self.nutrient_inflow, self.nutrient_max = nutrient_inflow, nutrient_max
        self.delta_v, self.lambda_v = delta_v, lambda_v
        self.rho_star, self.v_star, self.v_ref = rho_star, v_star, v_ref
        self.p_max, self.p_d, self.t_s, self.eta = p_max, p_d, t_s, int(eta)

        # Initial grid depends on `seed` ONLY -> the three conditions of a given
        # replicate start from the identical lattice (paired design).
        rng_init = np.random.default_rng(seed)
        probs = [phi0] + [(1 - phi0) / num_strains] * num_strains
        self.grid = rng_init.choice(np.arange(num_strains + 1), size=(self.N, self.N), p=probs)
        # Dynamics use a separate stream (differs by condition) -> independent noise.
        mode_id = {'ktw': 1, 'fixed': 2, 'none': 3}[mode]
        self.rng = np.random.default_rng([seed if dyn_seed is None else dyn_seed, mode_id, 7])

        self.nutrients = np.full((self.N, self.N), n0, dtype=float)
        self.viruses = np.zeros((num_strains + 1, self.N, self.N))
        self.starvation_counter = np.zeros((self.N, self.N), dtype=int)

        # Traits: fixed seed 42 => identical strain traits in every run/condition.
        rng_tr = np.random.default_rng(42)
        self.r = rng_tr.uniform(r_min, r_max, size=num_strains + 1)
        self.K = alpha * self.r ** tradeoff_exp

        self.kernel = np.array([[0.05, 0.10, 0.05], [0.10, 0.40, 0.10], [0.05, 0.10, 0.05]])
        # histories
        self.history_diversity, self.history_dominant, self.history_top3 = [], [], []
        self.history_vacancies, self.history_richness = [], []
        self.history_richness10, self.history_richness40 = [], []
        self.history_lysis, self.history_fed, self.history_occupied = [], [], []
        self.history_nutrient = []
        self.history_mutations = []
        self._lysis_events = 0
        self._fed_visits = 0
        self._occupied_visits = 0

    # ------------------------------------------------------------------
    def _local_density(self):
        """Density of each cell's OWN strain in its (2*eta+1)^2 periodic window.
        (Identical to the per-cell patch count of the original code: the grid is
        not modified inside a generation, so it can be computed once per step.)"""
        dens = np.zeros((self.N, self.N))
        size = 2 * self.eta + 1
        for s in np.unique(self.grid[self.grid > 0]):
            f = uniform_filter((self.grid == s).astype(float), size=size, mode='wrap')
            m = self.grid == s
            dens[m] = f[m]
        return dens

    def _record(self):
        counts = np.bincount(self.grid.ravel(), minlength=self.M + 1)[1:]
        occ = int(counts.sum())
        if occ > 0:
            p = counts / occ
            self.history_diversity.append(float(entropy(p[p > 0])))
            self.history_dominant.append(counts.max() / self.grid.size)
            self.history_top3.append(np.sort(counts)[::-1][:3].sum() / self.grid.size)
        else:
            self.history_diversity.append(0.0); self.history_dominant.append(0.0)
            self.history_top3.append(0.0)
        self.history_vacancies.append(self.grid.size - occ)
        self.history_richness.append(int((counts >= 20).sum()))     # "viable" = >=20 cells
        self.history_richness10.append(int((counts >= 10).sum()))
        self.history_richness40.append(int((counts >= 40).sum()))
        self.history_nutrient.append(float(self.nutrients.mean()))

    # ------------------------------------------------------------------
    def step(self):
        N, rng = self.N, self.rng
        next_grid = self.grid.copy()
        occ_coords = np.argwhere(self.grid > 0)
        occ_coords = occ_coords[rng.permutation(len(occ_coords))]   # random update order
        n = len(occ_coords)
        # one random number of each kind per occupied cell (each is used at most once)
        u_mort, u_grow, u_mut, u_disp = rng.random((4, n))
        d_r = rng.integers(-1, 2, size=n)
        d_c = rng.integers(-1, 2, size=n)
        mut_strain = rng.integers(1, self.M + 1, size=n)
        dens = self._local_density() if self.mode == 'ktw' else None
        lys = fed = muts = 0

        for i in range(n):
            r, c = occ_coords[i]
            s = self.grid[r, c]
            nut = self.nutrients[r, c]
            if nut >= self.K[s]:                              # Rule 1: cell feeds
                fed += 1
                self.nutrients[r, c] -= self.K[s]
                self.starvation_counter[r, c] = 0
                # Rule 2: mortality (fed cells only)
                if self.mode == 'ktw':
                    if dens[r, c] > self.rho_star:
                        self.viruses[s, r, c] += self.delta_v
                    v = self.viruses[s, r, c]
                    if v > self.v_star and u_mort[i] < min(self.p_max, v / self.v_ref):
                        next_grid[r, c] = 0
                        self.starvation_counter[r, c] = 0
                        lys += 1
                        continue
                elif self.mode == 'fixed':
                    if u_mort[i] < self.p_fixed:
                        next_grid[r, c] = 0
                        self.starvation_counter[r, c] = 0
                        lys += 1
                        continue
                # Rule 1 (cont.): replication with Monod probability
                if u_grow[i] < self.r[s] * nut / (self.K[s] + nut):
                    nr, nc = (r + d_r[i]) % N, (c + d_c[i]) % N
                    target = self.grid[nr, nc]
                    off = s
                    if u_mut[i] < self.mu:
                        off = mut_strain[i]; muts += 1
                    if target == 0 or (target != s and u_disp[i] < self.p_d):
                        next_grid[nr, nc] = off
                        self.starvation_counter[nr, nc] = 0
            else:                                              # starvation pathway
                self.starvation_counter[r, c] += 1
                if self.starvation_counter[r, c] >= self.t_s:
                    next_grid[r, c] = 0
                    self.starvation_counter[r, c] = 0

        self.grid = next_grid
        # Rule 3: diffusion + inflow; viral decay in place (no viral transport)
        self.nutrients = convolve2d(self.nutrients, self.kernel, mode='same', boundary='wrap')
        self.nutrients = np.clip(self.nutrients + self.nutrient_inflow, 0.0, self.nutrient_max)
        self.viruses *= self.lambda_v

        self._lysis_events += lys; self._fed_visits += fed; self._occupied_visits += n
        self.history_lysis.append(lys); self.history_fed.append(fed)
        self.history_occupied.append(n); self.history_mutations.append(muts)
        self._record()

    def run(self, generations):
        for _ in range(generations):
            self.step()
        return self

    # ------------------------------------------------------------------
    def mean_lysis_rate(self):
        """Realised deaths per FED cell per generation -- the same quantity that
        p_fixed sets in the 'fixed' condition, so the two are directly comparable."""
        return self._lysis_events / max(self._fed_visits, 1)

    def mean_lysis_rate_per_occupied(self):
        return self._lysis_events / max(self._occupied_visits, 1)


# ----------------------------------------------------------------------
# Analysis helpers shared by the scripts
# ----------------------------------------------------------------------
def calibrate_fixed_rate(n_pilot=5, pilot_gens=400, **kw):
    """Mean realised KtW lysis rate per fed cell, from short pilot runs, for the
    parameter set given in **kw (so it can be re-done for every sensitivity point)."""
    rates = []
    for rep in range(n_pilot):
        sim = PlanktonEcosystemSimulation(mode='ktw', seed=10_000 + rep, **kw).run(pilot_gens)
        rates.append(sim.mean_lysis_rate())
    return float(np.mean(rates))


def rank_abundance(grid, num_strains):
    counts = np.bincount(grid.ravel(), minlength=num_strains + 1)[1:]
    counts = np.sort(counts[counts > 0])[::-1].astype(float)
    return counts / counts.sum()


def fit_rank_abundance(p):
    """Fit exponential (ln p = a - b*rank) and power law (ln p = a - b*ln rank) by
    least squares; return R^2 and AIC for each (needs >=3 ranks)."""
    if len(p) < 3:
        return dict(n_ranks=len(p), aic_exp=np.nan, aic_pow=np.nan, r2_exp=np.nan, r2_pow=np.nan)
    y = np.log(p); k = np.arange(1, len(p) + 1)
    out = dict(n_ranks=len(p))
    for name, x in (('exp', k.astype(float)), ('pow', np.log(k))):
        A = np.vstack([np.ones_like(x), x]).T
        coef, res, *_ = np.linalg.lstsq(A, y, rcond=None)
        rss = float(((y - A @ coef) ** 2).sum())
        tss = float(((y - y.mean()) ** 2).sum())
        out['r2_' + name] = 1 - rss / tss if tss > 0 else np.nan
        out['aic_' + name] = len(y) * np.log(max(rss, 1e-12) / len(y)) + 2 * 2
    return out


def local_shannon(grid, num_strains, window=11):
    """Shannon diversity of strain shares in a sliding window x window (periodic)."""
    props = np.stack([uniform_filter((grid == s).astype(float), size=window, mode='wrap')
                      for s in range(1, num_strains + 1)])
    tot = props.sum(axis=0)
    with np.errstate(divide='ignore', invalid='ignore'):
        q = np.where(tot > 0, props / tot, 0.0)
        h = -np.nansum(np.where(q > 0, q * np.log(q), 0.0), axis=0)
    h[tot == 0] = np.nan
    return h


def patch_stats(grid, num_strains):
    """Mean size (cells) of 4-connected single-strain patches -- quantifies the
    'coherent territory' claim that Figure 3 only shows visually."""
    sizes = []
    for s in range(1, num_strains + 1):
        lab, k = label(grid == s)
        if k:
            sizes.extend(np.bincount(lab.ravel())[1:])
    return float(np.mean(sizes)) if sizes else 0.0


def isotropy(grid):
    """P(same strain | both occupied) at lag 1 along H, V and the two diagonals.
    Equal values (after accounting for distance) show the lattice has no built-in
    preferred direction, i.e. the diagonal streaks in the submitted Figure 3 were
    not a property of the model."""
    out = {}
    for name, (dx, dy) in dict(H=(0, 1), V=(1, 0), D_plus=(1, 1), D_minus=(1, -1)).items():
        b = np.roll(np.roll(grid, dx, 0), dy, 1)
        m = (grid > 0) & (b > 0)
        out[name] = float((grid[m] == b[m]).mean()) if m.any() else np.nan
    return out
