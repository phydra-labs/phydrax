# Splits

Index split helpers return JAX integer arrays. Use them with
`SupervisedDatasetConstraint(indices=...)`,
`RaggedTimeSeriesDataConstraint(case_indices=...)`, and
`TrajectoryCaseDataConstraint(case_indices=...)`.

```python
import jax.random as jr
import phydrax as phx

train_idx, test_idx = phx.data_utils.train_test_split_indices(
    10,
    test_fraction=0.2,
    key=jr.key(0),
)
folds = phx.data_utils.kfold_indices(10, 5, key=jr.key(1))
```

::: phydrax.data_utils.train_test_split_indices

---

::: phydrax.data_utils.kfold_indices
