# spyglass-paper

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Code and materials for reproducing Figure 5 from the Spyglass paper, demonstrating neural decoding analysis pipelines using the [Spyglass](https://github.com/LorenFrankLab/spyglass) data analysis framework.

## Citation

If you use this code, please cite:

> Lee, K.H.\*, Denovellis, E.L.\*, Ly, R., Magland, J., Soules, J., Comrie, A.E., Gramling, D.P., Guidera, J.A., Nevers, R., Adenekan, P., Brozdowski, C., Bray, S., Monroe, E., Bak, J.H., Coulter, M.E., Sun, X., Broyles, E., Shin, D., Chiang, S., Holobetz, C., Tritt, A., Rübel, O., Nguyen, T., Yatsenko, D., Chu, J., Kemere, C., Garcia, S., Buccino, A., Frank, L.M., 2024. Spyglass: a data analysis framework for reproducible and shareable neuroscience research. bioRxiv. [10.1101/2024.01.25.577295](https://doi.org/10.1101/2024.01.25.577295).

*\* Equal contribution*

## Overview

This repository contains:

- **Figure 5 generation scripts**: Publication-quality figures showing neural decoding results from two datasets
- **Jupyter notebooks**: Example decoding analysis workflows using Spyglass
- **Plotting utilities**: Shared functions for consistent figure styling

### Datasets

The figures demonstrate two decoding approaches:

1. **Frank Lab (j1620210710)**: Clusterless decoding analysis
2. **Buzsaki Lab (MS2220180629)**: Sorted spikes decoding analysis

## Installation

### Prerequisites

- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Mamba](https://mamba.readthedocs.io/)
- A configured Spyglass database (see [Spyglass documentation](https://lorenfranklab.github.io/spyglass/))

### Setup

1. Clone the repository:

   ```bash
   git clone https://github.com/edeno/spyglass-paper.git
   cd spyglass-paper
   ```

1. Create and activate the conda environment:

   ```bash
   conda env create -f environment.yml
   conda activate spyglass-paper
   ```

For exact reproducibility of published figures, use the pinned environment:

```bash
conda env create -f pinned_paper_environment.yml
```

## Data

### Buzsaki Lab Dataset (MS2220180629)

Download from DANDI (<https://doi.org/10.48324/dandi.000059/0.230907.2101>):

```bash
dandi download DANDI:000059/0.230907.2101
mkdir -p data/nwb/raw/MS2220180629.nwb
mv sub-MS22_ses-Peter-MS22-180629-110319-concat_desc-processed_behavior+ecephys.nwb data/nwb/raw/MS2220180629.nwb/MS2220180629.nwb
```

### Frank Lab Dataset (j1620210710)

This dataset will be available on DANDI soon.

## Reproducing Figures

### Step 1: Run the decoding notebooks

First, execute the Jupyter notebooks to populate the Spyglass database with decoding results:

```bash
jupyter lab notebooks/figure5/
```

Run each notebook:

- `MS2220180629_sorted_decode.ipynb` (Buzsaki Lab data)
- `j1620210710_clusterless_decode.ipynb` (Frank Lab data)

### Step 2: Generate figures

After the notebooks have populated the database:

```bash
# Generate main Figure 5
python notebooks/figure5/figure5.py

# Generate supplemental figure (pipeline flowchart)
python notebooks/figure5-supplemental/figure5_supp.py
```

Output files (PDF and PNG) will be saved in the respective notebook directories.

## Repository Structure

```text
spyglass-paper/
├── src/
│   └── paper_plotting.py      # Shared plotting utilities
├── notebooks/
│   ├── figure5/               # Main figure generation
│   │   ├── figure5.py         # Figure 5 script
│   │   └── *.ipynb            # Decoding analysis notebooks
│   └── figure5-supplemental/  # Supplemental figure
│       └── figure5_supp.py    # Pipeline flowchart script
├── data/nwb/raw/              # NWB data files (not tracked)
├── environment.yml            # Conda environment
└── pinned_paper_environment.yml  # Exact versions for reproducibility
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
