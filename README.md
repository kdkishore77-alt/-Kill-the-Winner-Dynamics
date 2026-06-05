# ACCESS INFORMATION

## 1. Licenses/restrictions placed on the data or code

CC0 1.0 Universal (CC0 1.0) Public Domain Dedication.

All code and data files in this repository are provided without restriction for use, modification, and redistribution.

## 2. Data derived from other sources

No external empirical datasets were used in this study.

All data files contained in this repository were generated directly from the simulation models included in the code archive.

## 3. Recommended citation for this data/code archive

Anonymous. Simulation code and generated data associated with the manuscript under review. Data and code repository submitted for peer review in *The American Naturalist*.

# DATA & CODE FILE OVERVIEW

This repository consists of two Python scripts, three generated data files, multiple figure files, and this README document.

## Data files and variables

### 1. raw_simulation_data.csv

Generation-level output from the Kill-the-Winner (KtW) ablation experiment implemented in `1A.py`.

Variables:

| Variable          | Description                                                             |
| ----------------- | ----------------------------------------------------------------------- |
| condition         | Experimental treatment (KtW_ON or KtW_OFF)                              |
| replicate         | Replicate identifier                                                    |
| generation        | Simulation generation (time step)                                       |
| shannon_diversity | Shannon diversity index (H') computed from strain abundances            |
| richness          | Number of strains exceeding the activity threshold (≥20 occupied cells) |
| dominant_fraction | Fraction of the lattice occupied by the dominant strain                 |
| top3_fraction     | Fraction of the lattice occupied by the three most abundant strains     |
| vacancies         | Number of unoccupied lattice cells                                      |

### 2. raw_sensitivity_data.csv

Generation-level output from the sensitivity analysis implemented in `sensitivity.py`.

Variables:

| Variable          | Description                                                             |
| ----------------- | ----------------------------------------------------------------------- |
| parameter         | Parameter varied during sensitivity analysis                            |
| parameter_value   | Value assigned to the parameter                                         |
| condition         | Experimental treatment (KtW_ON or KtW_OFF)                              |
| replicate         | Replicate identifier                                                    |
| generation        | Simulation generation (time step)                                       |
| shannon_diversity | Shannon diversity index (H')                                            |
| vacancies         | Number of unoccupied lattice cells                                      |
| dominant_fraction | Fraction of the lattice occupied by the dominant strain                 |
| richness          | Number of strains exceeding the activity threshold (≥20 occupied cells) |

### 3. sensitivity_summary_statistics.csv

Processed summary statistics calculated from the sensitivity-analysis simulations.

Variables:

| Variable        | Description                                                            |
| --------------- | ---------------------------------------------------------------------- |
| parameter       | Parameter varied during sensitivity analysis                           |
| parameter_value | Value assigned to the parameter                                        |
| condition       | Experimental treatment (KtW_ON or KtW_OFF)                             |
| mean_shannon    | Mean Shannon diversity over the final 50 generations across replicates |
| sd_shannon      | Standard deviation of Shannon diversity across replicates              |

## Code scripts and workflow

### 1. 1A.py

Primary simulation model of a spatial plankton ecosystem.

The model incorporates:

* Nutrient-dependent growth using Monod kinetics.
* Explicit growth–resource trade-offs among competing strains.
* Localized Kill-the-Winner (KtW) viral predation.
* Diffusion of nutrients and viral particles across the spatial lattice.
* Competition for space among plankton strains.

The script performs replicated simulations under two conditions:

* KtW active.
* KtW disabled (ablation condition).

Outputs:

* raw_simulation_data.csv
* shannon-diversity.png
* vacancies.png
* ablation_comparison.png
* ablation_grids.png

### 2. sensitivity.py

Sensitivity-analysis framework used to evaluate the robustness of model outcomes to variation in key KtW parameters.

Parameters evaluated:

* crowding_threshold (ρ*)
* viral_increment (δv)
* lysis_threshold (V*)

For each parameter value, replicated simulations are performed under both KtW-ON and KtW-OFF conditions.

Outputs:

* raw_sensitivity_data.csv
* sensitivity_summary_statistics.csv
* sensitivity_shannon.png
* sensitivity_vacancies.png
* sensitivity_dominance.png

## Workflow

To reproduce all analyses and figures:

1. Run `1A.py`.
2. Run `sensitivity.py`.
3. Generated CSV files contain the complete simulation outputs used for analysis.
4. Generated PNG files reproduce the figures reported in the manuscript.

# SOFTWARE VERSIONS

Analyses were conducted in Python 3.

Required packages:

* numpy
* scipy
* matplotlib
* pandas

Key imported modules include:

* numpy
* matplotlib.pyplot
* scipy.signal.convolve2d
* scipy.stats.entropy
* pandas

Users should install versions of these packages compatible with their Python distribution.

# REFERENCES

Monod, J. 1949. The growth of bacterial cultures. *Annual Review of Microbiology* 3:371–394.

Thingstad, T. F. 2000. Elements of a theory for the mechanisms controlling abundance, diversity, and biogeochemical role of lytic bacterial viruses in aquatic systems. *Limnology and Oceanography* 45:1320–1328.

Winter, C., T. Bouvier, M. G. Weinbauer, and T. F. Thingstad. 2010. Trade-offs between competition and defense specialists among unicellular planktonic organisms: the “Kill the Winner” hypothesis revisited. *Microbiology and Molecular Biology Reviews* 74:42–57.
