# Kill-the-Winner Dynamics: a minimal cellular automaton

Code and data for *"A Minimal Cellular Automaton Addresses the Paradox of the Plankton via Kill-the-Winner Dynamics"*.

A three-rule, spatially explicit cellular automaton (70×70 periodic lattice, 12 strains, one diffusing nutrient) compares three conditions that differ only in the mortality rule:

| Condition | Mortality rule |
|---|---|
| **KtW-active** | density-dependent, strain-specific viral accumulation and lysis |
| **Fixed mortality** | constant death probability per *fed* cell, independent of density (calibrated to match KtW-active) |
| **No mortality** | viral rule absent |

All quantities are dimensionless (time is in generations).

## Files

| File | Purpose |
|---|---|
| `plankton_model.py` | The model (shared by both scripts), plus calibration and analysis helpers |
| `1A.py` | Main three-condition experiment, run-length check, spatial and rank–abundance analyses |
| `sensitivity.py` | One-at-a-time robustness sweeps (15 parameters, lattice size, mutation rate, trade-off shape) |

## Requirements

Python 3.9+ with `numpy`, `pandas`, `scipy`, `matplotlib`.

```
pip install numpy pandas scipy matplotlib
```

## Usage

```bash
# Quick smoke test (tiny runs, a few minutes at most)
QUICK=1 python 1A.py
QUICK=1 python sensitivity.py --analysis tier1

# Main experiment: 20 paired replicates x 1200 generations,
# plus a 2400-generation check (5 replicates)
python 1A.py

# Sensitivity analyses (settings used in the paper: 5 replicates x 600 generations)
python sensitivity.py --analysis all --reps 5 --gens 600 --workers 8
```

Available `--analysis` options: `tier1` (ρ\*, δ_v, V\*), `tier2` (all other parameters), `grid` (L = 50–140), `tradeoff` (α and exponent k), `tradeoff_matched` (k with α rescaled so the mean K_s is unchanged), `mutation` (μ = 0–0.05), or `all`.

Note: `sensitivity.py` defaults to 10 replicates; pass `--reps 5` to reproduce the tables in the Supplement.

## Design notes

- **Fixed-mortality control.** `p_fixed` is calibrated as deaths per *fed* cell per generation from KtW-active pilot runs (0.0202 at baseline) and is re-calibrated at every parameter value in the sweeps. Realised death rates of all conditions are reported.
- **Paired design.** For replicate *k*, all three conditions start from the same lattice (`seed = k`) and then use independent random streams. Strain traits are fixed across replicates and conditions.
- **Mutation.** The main runs use μ = 0. μ is a relabelling rate (an offspring becomes one of the 12 existing strains); no new trait values arise.
- **Same model everywhere.** `sensitivity.py` imports the model from `plankton_model.py`, so the main experiment and the sweeps use identical rules.

## Outputs

**`1A.py`** writes:

- `parameters.csv` – parameter values, units and status
- `simulation_data.csv` – per-generation data for every replicate
- `simulation_summary.csv` – final-state statistics (Table 2), 95% CIs, patch size, isotropy, rank–abundance fits
- `simulation_tests.csv` – Mann–Whitney tests on final Shannon diversity
- `extended_run_checkpoints.csv` – 2400-generation checkpoints (Table S3)
- Figures `fig1`–`fig7` and `figS1` (`.png` and `.pdf`)

**`sensitivity.py`** writes, for each analysis, `sensitivity_<analysis>_raw.csv`, `sensitivity_<analysis>_summary.csv` (with the ΔH′ separations and the 0.05 collapse flag) and `sensitivity_<analysis>.png`.

Figure mapping to the manuscript: `fig1`–`fig5` = Figures 1–5; `fig6` = Fig. S4 (rank–abundance); `fig7` = Fig. S5 (replicate convergence); `figS1` = Fig. S1; `sensitivity_tier1.png` = Fig. S2; the other sensitivity plots = Fig. S3.

## Citation

If you use this code, please cite the paper (details to be added on publication).

## Contact

Kishore Dutta, Department of Physics, Handique Girls' College, Guwahati 781 001, India – kdkishore77@gmail.com
