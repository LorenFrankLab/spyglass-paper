# Spyglass-paper

Paper-specific code supporting:
 > Lee, K.H., Denovellis, E.L., Ly, R., Magland, J., Soules, J., Gramling, D.P., Guidera, J.A., Nevers, R., Adenekan, P., Bray, S., Monroe, E., Bak, J.H., Coulter, M., Sun, X., Rübel, O., Nguyen, T., Yatsenko, D., Chu, J., Kemere, C., Buccino, A., Garcia, Samuel, Frank, Loren M., 2024. **Spyglass: a data analysis framework for reproducible and shareable neuroscience research.** bioRxiv. <https://doi.org/10.1101/2024.01.25.577295>

### Installation

```bash
pip install spyglass-paper
```

Or

```bash
conda install -c franklab spyglass-paper
```

Or

```bash
git clone https://github.com/edeno/spyglass-paper.git
pip install .
```

### Usage

### Developer Installation

1. Install miniconda (or anaconda) if it isn't already installed.
2. git clone <https://github.com/edeno/spyglass-paper.git>
2. Setup editiable package with dependencies

```bash
cd <spyglass-paper>
conda env create -f environment.yml
conda activate spyglass-paper
pip install --editable . --no-deps
```
