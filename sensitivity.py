import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
from scipy.stats import entropy


# =============================================================================
# CORE SIMULATION CLASS (aligned with main model)
# =============================================================================

class PlanktonEcosystemSimulation:
    """
    Three-rule plankton cellular automaton.
    Parameterised fully via constructor to support sensitivity sweeps.
    """

    def __init__(
        self,
        grid_size=70,
        num_strains=12,
        mutation_rate=0.0,
        ktw_enabled=True,
        crowding_threshold=0.35,   # rho*
        viral_increment=0.8,       # delta_v
        lysis_threshold=1.0,       # V*
        lysis_ref=2.0,             # V_ref
        max_lysis_prob=0.95,       # p_max
        displacement_prob=0.30,    # p_d
        starvation_steps=3,        # T_s
        alpha=2.0,                 # trade-off scaling
        nutrient_inflow=0.10,      # I_n
        nutrient_max=3.0,          # N_max
        seed=None,
    ):
        self.N = grid_size
        self.num_strains = num_strains
        self.mutation_rate = mutation_rate
        self.ktw_enabled = ktw_enabled

        # KtW parameters
        self.crowding_threshold = crowding_threshold
        self.viral_increment = viral_increment
        self.lysis_threshold = lysis_threshold
        self.lysis_ref = lysis_ref
        self.max_lysis_prob = max_lysis_prob

        # Competition / mortality parameters
        self.displacement_prob = displacement_prob
        self.starvation_steps = starvation_steps

        # Environmental parameters
        self.alpha = alpha
        self.nutrient_inflow = nutrient_inflow
        self.nutrient_max = nutrient_max

        # Starvation counter
        self.starvation_counter = np.zeros((self.N, self.N), dtype=int)

        # Grid initialisation
        if seed is not None:
            np.random.seed(seed)
        self.grid = np.random.choice(
            [0] + list(range(1, num_strains + 1)),
            size=(self.N, self.N),
            p=[0.7] + [0.3 / num_strains] * num_strains,
        )

        # Nutrient and viral layers
        self.nutrients = np.ones((self.N, self.N)) * 1.0
        self.viruses = np.zeros((num_strains + 1, self.N, self.N))

        # Strain trade-off profiles (fixed seed for reproducibility across runs)
        np.random.seed(42)
        self.r = np.random.uniform(0.15, 0.45, size=num_strains + 1)
        self.K = self.alpha * (self.r ** 2)

        # Diffusion kernel
        self.diffusion_kernel = np.array(
            [[0.05, 0.10, 0.05],
             [0.10, 0.40, 0.10],
             [0.05, 0.10, 0.05]]
        )

        # History
        self.history_diversity = []
        self.history_vacancies = []
        self.history_dominant = []
        self.history_richness = []

    # -------------------------------------------------------------------------
    def _apply_fluid_diffusion(self):
        self.nutrients = convolve2d(
            self.nutrients, self.diffusion_kernel, mode='same', boundary='wrap'
        )
        self.nutrients += self.nutrient_inflow
        self.nutrients = np.clip(self.nutrients, 0.0, self.nutrient_max)

        for s in range(1, self.num_strains + 1):
            self.viruses[s] = convolve2d(
                self.viruses[s], self.diffusion_kernel, mode='same', boundary='wrap'
            )
            self.viruses[s] *= 0.98

    # -------------------------------------------------------------------------
    def step(self):
        next_grid = self.grid.copy()
        coords = [(r, c) for r in range(self.N) for c in range(self.N)]
        np.random.shuffle(coords)

        for r, c in coords:
            strain = self.grid[r, c]
            if strain == 0:
                continue

            local_n = self.nutrients[r, c]
            monod_growth_prob = self.r[strain] * (local_n / (self.K[strain] + local_n))

            if local_n >= self.K[strain]:
                self.nutrients[r, c] -= self.K[strain]
                self.starvation_counter[r, c] = 0

                if self.ktw_enabled:
                    rows = [(r + dr) % self.N for dr in range(-2, 3)]
                    cols = [(c + dc) % self.N for dc in range(-2, 3)]
                    local_patch = self.grid[np.ix_(rows, cols)]
                    local_density = np.sum(local_patch == strain) / local_patch.size

                    if local_density > self.crowding_threshold:
                        self.viruses[strain, r, c] += self.viral_increment

                    viral_pressure = self.viruses[strain, r, c]
                    if viral_pressure > self.lysis_threshold:
                        infection_prob = min(
                            self.max_lysis_prob, viral_pressure / self.lysis_ref
                        )
                        if np.random.rand() < infection_prob:
                            next_grid[r, c] = 0
                            self.starvation_counter[r, c] = 0
                            continue

                if np.random.rand() < monod_growth_prob:
                    dr = np.random.choice([-1, 0, 1])
                    dc = np.random.choice([-1, 0, 1])
                    nr, nc = (r + dr) % self.N, (c + dc) % self.N
                    target = self.grid[nr, nc]

                    if target == 0:
                        next_grid[nr, nc] = strain
                        self.starvation_counter[nr, nc] = 0
                    elif target != strain:
                        if np.random.rand() < self.displacement_prob:
                            next_grid[nr, nc] = strain
                            self.starvation_counter[nr, nc] = 0
            else:
                self.starvation_counter[r, c] += 1
                if self.starvation_counter[r, c] >= self.starvation_steps:
                    next_grid[r, c] = 0
                    self.starvation_counter[r, c] = 0

        self.grid = next_grid
        self._apply_fluid_diffusion()

        counts = np.bincount(self.grid.flatten(), minlength=self.num_strains + 1)
        occupied = np.sum(counts[1:])
        if occupied > 0:
            p = counts[1:] / occupied
            shannon_H = entropy(p[p > 0])
            dominant_fraction = np.max(counts[1:]) / self.grid.size
            richness = np.sum(counts[1:] >= 20)
        else:
            shannon_H = 0.0
            dominant_fraction = 0.0
            richness = 0

        self.history_diversity.append(shannon_H)
        self.history_vacancies.append(self.grid.size - occupied)
        self.history_dominant.append(dominant_fraction)
        self.history_richness.append(richness)

    # -------------------------------------------------------------------------
    def run(self, generations):
        for _ in range(generations):
            self.step()


# =============================================================================
# SENSITIVITY ANALYSIS CLASS
# =============================================================================

class SensitivityAnalysis:
    """
    One-at-a-time (OAT) sensitivity analysis over Tier 1 KtW parameters:
        rho*       : crowding_threshold
        delta_v    : viral_increment
        V*         : lysis_threshold

    For each parameter, a range of values is swept while all others are held
    at their baseline. KtW-ON and KtW-OFF replicates are run per value to
    preserve the ablation framing of the main study.
    """

    # Baseline parameter values (aligned with main model)
    BASELINE = dict(
        crowding_threshold=0.35,
        viral_increment=0.8,
        lysis_threshold=1.0,
    )

    # Sweep ranges for each Tier 1 parameter
    SWEEPS = {
        'crowding_threshold': [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50],
        'viral_increment':    [0.2,  0.4,  0.6,  0.8,  1.0,  1.2,  1.4,  1.6],
        'lysis_threshold':    [0.4,  0.6,  0.8,  1.0,  1.2,  1.4,  1.6,  1.8],
    }

    PARAM_LABELS = {
        'crowding_threshold': r'Crowding Threshold $\rho^*$',
        'viral_increment':    r'Viral Increment $\delta_v$',
        'lysis_threshold':    r'Lysis Threshold $V^*$',
    }

    def __init__(self, generations=600, n_replicates=5, grid_size=70, num_strains=12):
        self.generations = generations
        self.n_replicates = n_replicates
        self.grid_size = grid_size
        self.num_strains = num_strains
        self.results = {}   # {param_name: {value: {'ktw': [...], 'abl': [...]}}}

    # -------------------------------------------------------------------------
    def _run_condition(self, ktw_enabled, param_name, param_value, rep):
        kwargs = dict(self.BASELINE)
        kwargs[param_name] = param_value
        seed = rep if ktw_enabled else rep + 100
        sim = PlanktonEcosystemSimulation(
            grid_size=self.grid_size,
            num_strains=self.num_strains,
            mutation_rate=0.0,
            ktw_enabled=ktw_enabled,
            seed=seed,
            **kwargs,
        )
        sim.run(self.generations)
        return sim

    # -------------------------------------------------------------------------
    def run(self):
        total = sum(len(v) for v in self.SWEEPS.values()) * 2 * self.n_replicates
        done = 0
        for param_name, values in self.SWEEPS.items():
            self.results[param_name] = {}
            for val in values:
                ktw_sims, abl_sims = [], []
                for rep in range(self.n_replicates):
                    done += 1
                    print(f"  [{done}/{total}] {param_name}={val:.2f}  "
                          f"KtW=ON  rep={rep+1}")
                    ktw_sims.append(
                        self._run_condition(True, param_name, val, rep)
                    )
                    done += 1
                    print(f"  [{done}/{total}] {param_name}={val:.2f}  "
                          f"KtW=OFF rep={rep+1}")
                    abl_sims.append(
                        self._run_condition(False, param_name, val, rep)
                    )
                self.results[param_name][val] = {
                    'ktw': ktw_sims,
                    'abl': abl_sims,
                }
        print("\nSensitivity analysis complete.")

    # -------------------------------------------------------------------------
    @staticmethod
    def _final_mean_std(sims, attr, last_n=50):
        """Mean and std of the last `last_n` generations across replicates."""
        vals = [np.mean(getattr(s, attr)[-last_n:]) for s in sims]
        return np.mean(vals), np.std(vals)


# =============================================================================
# PLOTTING CLASS
# =============================================================================

class SensitivityPlotter:
    """
    Publication-ready figures from a completed SensitivityAnalysis instance.

    Figure 1 (sensitivity_shannon.png):
        3-panel plot, one panel per Tier 1 parameter.
        Each panel shows final Shannon H' (mean +/- 1 SD) vs parameter value
        for KtW-ON and KtW-OFF conditions.

    Figure 2 (sensitivity_vacancies.png):
        Same layout for final vacant cell count.

    Figure 3 (sensitivity_dominance.png):
        Same layout for final dominant strain fraction (%).
    """

    COLORS = {'ktw': 'steelblue', 'abl': 'darkorange'}
    LABELS = {'ktw': 'KtW ON', 'abl': 'KtW OFF'}

    def __init__(self, analysis: SensitivityAnalysis):
        self.sa = analysis

    # -------------------------------------------------------------------------
    def _extract(self, param_name, attr, last_n=50):
        values = sorted(self.sa.results[param_name].keys())
        ktw_means, ktw_stds, abl_means, abl_stds = [], [], [], []
        for v in values:
            entry = self.sa.results[param_name][v]
            km, ks = SensitivityAnalysis._final_mean_std(entry['ktw'], attr, last_n)
            am, as_ = SensitivityAnalysis._final_mean_std(entry['abl'], attr, last_n)
            ktw_means.append(km); ktw_stds.append(ks)
            abl_means.append(am); abl_stds.append(as_)
        return (np.array(values),
                np.array(ktw_means), np.array(ktw_stds),
                np.array(abl_means), np.array(abl_stds))

    # -------------------------------------------------------------------------
    def _three_panel(self, attr, ylabel, scale=1.0, filename='fig.png', title=''):
        fig, axs = plt.subplots(1, 3, figsize=(14, 4), sharey=False)
        params = list(SensitivityAnalysis.SWEEPS.keys())

        for ax, param_name in zip(axs, params):
            xv, km, ks, am, as_ = self._extract(param_name, attr)
            km *= scale; ks *= scale; am *= scale; as_ *= scale

            ax.plot(xv, km, color=self.COLORS['ktw'], lw=2,
                    marker='o', ms=5, label=self.LABELS['ktw'])
            ax.fill_between(xv, km - ks, km + ks,
                            alpha=0.25, color=self.COLORS['ktw'])
            ax.plot(xv, am, color=self.COLORS['abl'], lw=2,
                    marker='s', ms=5, label=self.LABELS['abl'])
            ax.fill_between(xv, am - as_, am + as_,
                            alpha=0.25, color=self.COLORS['abl'])

            # Mark baseline
            baseline_val = SensitivityAnalysis.BASELINE[param_name]
            ax.axvline(baseline_val, color='gray', lw=1.2,
                       linestyle='--', label=f'Baseline ({baseline_val})')

            ax.set_xlabel(SensitivityAnalysis.PARAM_LABELS[param_name], fontsize=11)
            ax.set_ylabel(ylabel if ax == axs[0] else '', fontsize=11)
            ax.legend(fontsize=8)
            ax.grid(True, linestyle='--', alpha=0.4)

        fig.suptitle(title, fontsize=13, y=1.02)
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Saved: {filename}")

    # -------------------------------------------------------------------------
    def plot_shannon(self):
        self._three_panel(
            attr='history_diversity',
            ylabel="Shannon Diversity $H'$ (final mean)",
            scale=1.0,
            filename='sensitivity_shannon.png',
            title='Sensitivity Analysis: Shannon Diversity',
        )

    def plot_vacancies(self):
        self._three_panel(
            attr='history_vacancies',
            ylabel='Vacant Cells (final mean)',
            scale=1.0,
            filename='sensitivity_vacancies.png',
            title='Sensitivity Analysis: Vacant Cells',
        )

    def plot_dominance(self):
        self._three_panel(
            attr='history_dominant',
            ylabel='Dominant Strain Fraction (%, final mean)',
            scale=100.0,
            filename='sensitivity_dominance.png',
            title='Sensitivity Analysis: Dominant Strain Fraction',
        )

    def plot_all(self):
        self.plot_shannon()
        self.plot_vacancies()
        self.plot_dominance()


# =============================================================================
# SUMMARY PRINTER CLASS
# =============================================================================

class SensitivitySummary:
    """
    Prints a structured console summary of sensitivity results.
    Reports final Shannon H' separation (KtW-ON minus KtW-OFF) per parameter
    value, flagging values where separation collapses below a threshold.
    """

    def __init__(self, analysis: SensitivityAnalysis, collapse_threshold=0.05):
        self.sa = analysis
        self.collapse_threshold = collapse_threshold

    def print(self):
        print("\n" + "=" * 70)
        print("SENSITIVITY ANALYSIS SUMMARY")
        print(f"Generations: {self.sa.generations}  "
              f"Replicates: {self.sa.n_replicates}  "
              f"Grid: {self.sa.grid_size}x{self.sa.grid_size}  "
              f"Strains: {self.sa.num_strains}")
        print("=" * 70)

        for param_name, label in SensitivityAnalysis.PARAM_LABELS.items():
            print(f"\nParameter: {label}")
            print(f"  Baseline value: {SensitivityAnalysis.BASELINE[param_name]}")
            print(f"  {'Value':>8}  {'KtW-ON H\'':>12}  {'KtW-OFF H\'':>12}  "
                  f"{'Delta H\'':>10}  {'Status':>12}")
            print(f"  {'-'*8}  {'-'*12}  {'-'*12}  {'-'*10}  {'-'*12}")

            for val in sorted(self.sa.results[param_name].keys()):
                entry = self.sa.results[param_name][val]
                km, ks = SensitivityAnalysis._final_mean_std(
                    entry['ktw'], 'history_diversity'
                )
                am, as_ = SensitivityAnalysis._final_mean_std(
                    entry['abl'], 'history_diversity'
                )
                delta = km - am
                status = "ROBUST" if delta >= self.collapse_threshold else "COLLAPSED"
                baseline_marker = (
                    " <-- baseline"
                    if val == SensitivityAnalysis.BASELINE[param_name]
                    else ""
                )
                print(f"  {val:>8.2f}  {km:>10.3f}  {am:>10.3f}  "
                      f"{delta:>10.3f}  {status:>12}{baseline_marker}")

        print("\n" + "=" * 70)
        print("NOTE: COLLAPSED indicates KtW-ON and KtW-OFF Shannon diversity")
        print(f"separation < {self.collapse_threshold:.2f}, suggesting the coexistence")
        print("result is not robust at that parameter value.")
        print("=" * 70 + "\n")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == '__main__':

    print("Initialising sensitivity analysis...")
    print(f"Tier 1 parameters: crowding_threshold, viral_increment, lysis_threshold")
    print(f"Replicates per condition: 5  |  Generations: 600\n")

    sa = SensitivityAnalysis(
        generations=600,
        n_replicates=5,
        grid_size=70,
        num_strains=12,
    )

    sa.run()

    summary = SensitivitySummary(sa, collapse_threshold=0.05)
    summary.print()

    plotter = SensitivityPlotter(sa)
    plotter.plot_all()
