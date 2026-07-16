# Data Utilities

Data utilities handle small, explicit preprocessing tasks around arrays and tabular
data. They are intentionally separate from domains and constraints: load or scale
data here, then pass the resulting arrays into `DatasetDomain`, point-set
constraints, or model inference code.

- `CSVReader` reads CSV files with Polars and returns JAX arrays for numeric data.
- `scalers` provides immutable JAX-compatible scaling modules.
- `train_test_split_indices` and `kfold_indices` create deterministic case-index
  splits for `DatasetDomain`, `TrajectoryDatasetDomain`, and empirical constraints.
