# spyglass-paper

Paper-specific code supporting:
> Lee, K.H.\*, Denovellis, E.L.\*, Ly, R., Magland, J., Soules, J., Comrie, A.E., Gramling, D.P., Guidera, J.A., Nevers, R., Adenekan, P., Brozdowski, C., Bray, S., Monroe, E., Bak, J.H., Coulter, M.E., Sun, X., Tritt, A., Rübel, O., Nguyen, T., Yatsenko, D., Chu, J., Kemere, C., Garcia, S., Buccino, A., Frank, L.M., 2024. Spyglass: a data analysis framework for reproducible and shareable neuroscience research. bioRxiv. [10.1101/2024.01.25.577295](https://doi.org/10.1101/2024.01.25.577295 ).

*\* Equal contribution*

### Installation

1. Install miniconda (or mamba) if it isn't already installed.
2. git clone <https://github.com/edeno/spyglass-paper.git>
3. Setup editiable package with dependencies

```bash
cd spyglass-paper
conda env create -f environment.yml
conda activate spyglass-paper
```

### Data

For the MS2220180629 data <https://doi.org/10.48324/dandi.000059/0.230907.2101>, you can download the data from DANDI using the following command:

```bash
dandi download DANDI:000059/0.230907.2101
mv sub-MS22_ses-Peter-MS22-180629-110319-concat_desc-processed_behavior+ecephys.nwb spyglass-paper/data/nwb/raw/MS2220180629.nwb/MS2220180629.nwb # rename the file

```

The j1620210710 data will be available on DANDI soon.
