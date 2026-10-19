# Splits

Index split helpers return JAX integer arrays. Use them with
`SupervisedDatasetTerm(indices=...)`,
`RaggedTimeSeriesDataTerm(case_indices=...)`, and
`TrajectoryCaseDataTerm(case_indices=...)`.

Expected standard output: none.

```python {.executable}
import jax.random as jr
import phydrax as phx

train_idx, test_idx = phx.data_utils.train_test_split_indices(
    10,
    test_fraction=0.2,
    key=jr.key(0),
)
folds = phx.data_utils.kfold_indices(10, 5, key=jr.key(1))
train_idx, calibration_idx, test_idx = (
    phx.data_utils.train_calibration_test_split_indices(
        10,
        calibration_fraction=0.2,
        test_fraction=0.2,
        key=jr.key(2),
    )
)
assert train_idx.shape == (6,)
assert calibration_idx.shape == (2,)
assert test_idx.shape == (2,)
assert len(folds) == 5
```

::: phydrax.data_utils.train_test_split_indices

---

::: phydrax.data_utils.train_calibration_test_split_indices

---

::: phydrax.data_utils.kfold_indices
