import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.signal import convolve2d
from scipy.stats import entropy


class PlanktonEcosystemSimulation:
    def __init__(self, grid_size=60, num_strains=10, mutation_rate=0.02):
        
        self.N = grid_size
        self.num_strains = num_strains
        self.mutation_rate = mutation_rate
        self.starvation_counter = np.zeros((self.N, self.N), dtype=int)
        self.ktw_enabled = True

        
        # --- Core Layer Initializations ---
        # 0 = Empty cell, 1 to num_strains = Plankton strains
        self.grid = np.random.choice(
            [0] + list(range(1, num_strains + 1)), 
            size=(self.N, self.N), 
            p=[0.7] + [0.3 / num_strains] * num_strains
        )
        
        # Environmental layers (Continuous floats)
        self.nutrients = np.ones((self.N, self.N)) * 1.0
        # 3D array: Each strain has its own localized virus density field
        self.viruses = np.zeros((num_strains + 1, self.N, self.N))
        
        # --- Rule 1: Generate Strict Mathematical Trade-Offs ---
        # Strains that grow fast (high r) MUST require more food to survive (high K)
        # Monod kinetics parameterization: K = alpha * r^2
        np.random.seed(42) # For reproducible strain profiles
        self.r = np.random.uniform(0.15, 0.45, size=num_strains + 1)
        alpha = 2.0
        self.K = alpha * (self.r ** 2)
        
        # --- Environmental Constants ---
        self.nutrient_inflow = 0.10
        self.diffusion_kernel = np.array([[0.05, 0.10, 0.05],
                                          [0.10, 0.40, 0.10],
                                          [0.05, 0.10, 0.05]]) # Must sum to 1.0

        # Data tracking for publication analysis
        self.history_diversity = []
        self.history_occupancy = []

        self.history_shannon = []
        self.history_dominant = []
        self.history_top3 = []
        self.history_vacancies = []
        self.history_richness = []
        
    def _apply_fluid_diffusion(self):
        """ Rule 3: Discrete Fluid Diffusion via 2D Convolution """
        # Diffuse nutrients
        self.nutrients = convolve2d(self.nutrients, self.diffusion_kernel, mode='same', boundary='wrap')
        self.nutrients += self.nutrient_inflow
        self.nutrients = np.clip(self.nutrients, 0.0, 3.0)
        
        # Diffuse all strain-specific virus fields independently
        for s in range(1, self.num_strains + 1):

            self.viruses[s] = convolve2d(
                self.viruses[s],
                self.diffusion_kernel,
                mode='same',
                boundary='wrap'
            )

            self.viruses[s] *= 0.98

    def step(self):
        """ Executes one full discrete generation of the 3-rule model """
        next_grid = self.grid.copy()
        
        # Generate a random sequence of coordinates to prevent directional grid bias
        coords = [(r, c) for r in range(self.N) for c in range(self.N)]
        np.random.shuffle(coords)
        
        for r, c in coords:
            strain = self.grid[r, c]
            
            if strain > 0:  # Cell is occupied by a plankton
                # --- RULE 1: Nutrient Uptake & Local Starvation ---
                local_n = self.nutrients[r, c]
                monod_growth_prob = self.r[strain] * (local_n / (self.K[strain] + local_n))
                
                if local_n >= self.K[strain]:
                    self.nutrients[r, c] -= self.K[strain] # Consume resource
                    self.starvation_counter[r, c] = 0
                    
                    # --- RULE 2: Kill-The-Winner Predation Trigger ---
                    # Calculate local density of its own strain in a 5x5 patch
                    rows = [(r + dr) % self.N for dr in range(-2, 3)]
                    cols = [(c + dc) % self.N for dc in range(-2, 3)]
                    local_patch = self.grid[np.ix_(rows, cols)]

                    local_density = np.sum(local_patch == strain) / local_patch.size

                                        
                    if self.ktw_enabled:
                        if local_density > 0.35:
                            self.viruses[strain, r, c] += 0.8

                        viral_pressure = self.viruses[strain, r, c]

                        if viral_pressure > 1.0:

                            infection_prob = min(0.95, viral_pressure / 2.0)

                            if np.random.rand() < infection_prob:
                                next_grid[r, c] = 0
                                self.starvation_counter[r, c] = 0
                                continue                        

                    
                    # --- Local Reproduction and Mutation ---
                    if np.random.rand() < monod_growth_prob:
                        # Choose random neighbor in 8-cell Moore neighborhood
                        dr, dc = np.random.choice([-1, 0, 1]), np.random.choice([-1, 0, 1])
                        nr, nc = (r + dr) % self.N, (c + dc) % self.N

                        target = self.grid[nr, nc]

                        # Empty-space colonization
                        if target == 0:

                            next_grid[nr, nc] = strain
                            self.starvation_counter[nr, nc] = 0

                        # Direct competition
                        elif target != strain:

                            if np.random.rand() < 0.30:

                                next_grid[nr, nc] = strain
                                self.starvation_counter[nr, nc] = 0
        
        


                else:
                    self.starvation_counter[r, c] += 1
                    if self.starvation_counter[r, c] >= 3:
                        next_grid[r, c] = 0
                        self.starvation_counter[r, c] = 0

        self.grid = next_grid
        
        # --- RULE 3: Environmental Fluid Step ---
        self._apply_fluid_diffusion()
        
        counts = np.bincount(self.grid.flatten(), minlength=self.num_strains + 1)
        occupied = np.sum(counts[1:])
        if occupied > 0:
            p = counts[1:] / occupied
            shannon_H = entropy(p[p > 0])
            dominant_fraction = np.max(counts[1:]) / self.grid.size
            sorted_counts = np.sort(counts[1:])[::-1]
            top3_fraction = np.sum(sorted_counts[:3]) / self.grid.size
            richness = np.sum(counts[1:] >= 20)
        else:
            shannon_H = 0
            dominant_fraction = 0
            top3_fraction = 0
            richness = 0
        self.history_diversity.append(shannon_H)
        self.history_dominant.append(dominant_fraction)
        self.history_top3.append(top3_fraction)
        self.history_vacancies.append(self.grid.size - occupied)
        self.history_richness.append(richness)

    def print_summary(self, generation):
        counts = np.bincount(self.grid.flatten(), minlength=self.num_strains + 1)
        
        # Occupancy
        total_cells = self.N * self.N
        occupied = np.sum(self.grid > 0)
        vacant = total_cells - occupied
        
        # Strain census
        active_strains = [(s, counts[s]) for s in range(1, self.num_strains + 1) if counts[s] >= 20]
        dominant_strain = max(range(1, self.num_strains + 1), key=lambda s: counts[s])
        dominant_fraction = counts[dominant_strain] / total_cells
        sorted_counts = sorted(counts[1:], reverse=True)
        top3_fraction = np.sum(sorted_counts[:3]) / total_cells
        
        # Nutrient state
        mean_nutrient = np.mean(self.nutrients)
        min_nutrient  = np.min(self.nutrients)
        
        # Viral state: max viral load across all strains and cells
        max_viral_load = np.max(self.viruses[1:])
        strains_under_viral_pressure = np.sum([
            np.any(self.viruses[s] > 0.5) for s in range(1, self.num_strains + 1)
        ])
        
        print(f"\n--- Generation {generation} ---")
        print(f"  Grid occupancy     : {occupied}/{total_cells} cells filled ({100*occupied/total_cells:.1f}%), {vacant} vacant")
        print(f"  Active strains     : {len(active_strains)} (threshold >= 20 cells)")
        print(f"  Strain abundances  : {[(s, n) for s, n in active_strains]}")
        print(f"  Dominant strain    : #{dominant_strain} at {100*dominant_fraction:.1f}% of grid")
        print(f"  Top-3 strains      : {100*top3_fraction:.1f}% of grid")
        occupied = counts[1:].sum()
        p = counts[1:] / occupied
        p = p[p > 0]
        shannon = -np.sum(p * np.log(p))
        print(f"  Shannon diversity  : {shannon:.3f}")
        print(f"  Nutrient field     : mean={mean_nutrient:.3f}, min={min_nutrient:.3f}")
        print(f"  Max viral load     : {max_viral_load:.3f}")
        print(f"  Strains under viral pressure (load > 0.5): {strains_under_viral_pressure}")
        

# --- KtW Ablation Study with Replication ---
N_REPLICATES = 10
GENERATIONS = 1200

all_ktw = []
all_abl = []

for rep in range(N_REPLICATES):
    print(f"\n=== Replicate {rep+1}/{N_REPLICATES} ===")

    np.random.seed(rep)
    sim_ktw = PlanktonEcosystemSimulation(grid_size=70, num_strains=12, mutation_rate=0.0)
    sim_ktw.ktw_enabled = True
    for gen in range(GENERATIONS):
        sim_ktw.step()
    all_ktw.append(sim_ktw)

    np.random.seed(rep + 100)
    sim_abl = PlanktonEcosystemSimulation(grid_size=70, num_strains=12, mutation_rate=0.0)
    sim_abl.ktw_enabled = False
    for gen in range(GENERATIONS):
        sim_abl.step()
    all_abl.append(sim_abl)

# --- Aggregate histories ---
def mean_std(sims, attr):
    arr = np.array([getattr(s, attr) for s in sims])
    return arr.mean(axis=0), arr.std(axis=0)

ktw_div_mean, ktw_div_std = mean_std(all_ktw, 'history_diversity')
abl_div_mean, abl_div_std = mean_std(all_abl, 'history_diversity')

ktw_rich_mean, ktw_rich_std = mean_std(all_ktw, 'history_richness')
abl_rich_mean, abl_rich_std = mean_std(all_abl, 'history_richness')

ktw_dom_mean, ktw_dom_std = mean_std(all_ktw, 'history_dominant')
abl_dom_mean, abl_dom_std = mean_std(all_abl, 'history_dominant')

ktw_top3_mean, ktw_top3_std = mean_std(all_ktw, 'history_top3')
abl_top3_mean, abl_top3_std = mean_std(all_abl, 'history_top3')

ktw_vac_mean, ktw_vac_std = mean_std(all_ktw, 'history_vacancies')
abl_vac_mean, abl_vac_std = mean_std(all_abl, 'history_vacancies')

gens = np.arange(GENERATIONS)

# --- Publication Metrics Plot ---
fig, axs = plt.subplots(2, 2, figsize=(12, 10))

for ax, ktw_m, ktw_s, abl_m, abl_s, title in [
    (axs[0,0], ktw_rich_mean, ktw_rich_std, abl_rich_mean, abl_rich_std, "Richness"),
    (axs[0,1], ktw_div_mean,  ktw_div_std,  abl_div_mean,  abl_div_std,  "Shannon Diversity"),
    (axs[1,0], ktw_dom_mean*100, ktw_dom_std*100, abl_dom_mean*100, abl_dom_std*100, "Dominant Strain (%)"),
    (axs[1,1], ktw_top3_mean*100, ktw_top3_std*100, abl_top3_mean*100, abl_top3_std*100, "Top-3 Strain Share (%)"),
]:
    ax.plot(gens, ktw_m, color='steelblue', label='KtW ON')
    ax.fill_between(gens, ktw_m - ktw_s, ktw_m + ktw_s, alpha=0.25, color='steelblue')
    ax.plot(gens, abl_m, color='orange', label='KtW OFF')
    ax.fill_between(gens, abl_m - abl_s, abl_m + abl_s, alpha=0.25, color='orange')
    ax.set_title(title)
    ax.legend()

plt.tight_layout()
plt.savefig('shannon-diversity.png', dpi=300)
plt.show()

# --- Vacancies Plot ---
plt.figure(figsize=(6, 4))
plt.plot(gens, ktw_vac_mean, color='steelblue', label='KtW ON')
plt.fill_between(gens, ktw_vac_mean - ktw_vac_std, ktw_vac_mean + ktw_vac_std, alpha=0.25, color='steelblue')
plt.plot(gens, abl_vac_mean, color='orange', label='KtW OFF')
plt.fill_between(gens, abl_vac_mean - abl_vac_std, abl_vac_mean + abl_vac_std, alpha=0.25, color='orange')
plt.ylabel("Vacant Cells")
plt.xlabel("Generation")
plt.legend()
plt.tight_layout()
plt.savefig('vacancies.png', dpi=300)
plt.show()

# --- Ablation Comparison Plot ---
fig_abl, ax_abl = plt.subplots(figsize=(8, 5))
ax_abl.plot(gens[:600], ktw_div_mean[:600], color='crimson', lw=2, label='KtW Active')
ax_abl.fill_between(gens[:600], (ktw_div_mean - ktw_div_std)[:600], (ktw_div_mean + ktw_div_std)[:600], alpha=0.25, color='crimson')
ax_abl.plot(gens[:600], abl_div_mean[:600], color='steelblue', lw=2, label='KtW Disabled (Ablation)')
ax_abl.fill_between(gens[:600], (abl_div_mean - abl_div_std)[:600], (abl_div_mean + abl_div_std)[:600], alpha=0.25, color='steelblue')
ax_abl.set_xlim(0, 600)
ax_abl.set_ylim(0, 3)
ax_abl.set_xlabel("Generations")
ax_abl.set_ylabel("Shannon Diversity (H')")
ax_abl.set_title("KtW Ablation: Diversity With vs Without Viral Predation")
ax_abl.legend()
ax_abl.grid(True, linestyle='--', alpha=0.6)
fig_abl.savefig("ablation_comparison.png", dpi=300, bbox_inches='tight')

# --- Spatial Grid Panels (last replicate) ---
fig_grid, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
cmap = plt.cm.get_cmap('tab20', 14)

ax1.imshow(all_ktw[-1].grid, cmap=cmap, vmin=0, vmax=13, interpolation='nearest')
ax1.set_title("Final Grid: KtW Active")
ax1.axis('off')

ax2.imshow(all_abl[-1].grid, cmap=cmap, vmin=0, vmax=13, interpolation='nearest')
ax2.set_title("Final Grid: KtW Disabled")
ax2.axis('off')

fig_grid.savefig("ablation_grids.png", dpi=300, bbox_inches='tight')
plt.show()



