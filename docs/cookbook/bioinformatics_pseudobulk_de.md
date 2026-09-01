# Pseudobulk negative-binomial differential expression

Eight technical cell rows are aggregated into four donor-by-condition experimental
units before a paired NB2 model is fitted.

```python
from phydrax.bioinformatics.omics import (
    CountAssay,
    build_experimental_design,
    fit_negative_binomial_glm,
    pairwise_condition_contrast,
    pseudobulk_counts,
    wald_test,
)

cells = CountAssay(
    [
        [10, 4], [12, 5],  # donor 0, condition 0
        [25, 7], [27, 8],  # donor 0, condition 1
        [8, 6], [9, 5],    # donor 1, condition 0
        [22, 9], [24, 8],  # donor 1, condition 1
    ]
)
# Unit rows: d0/c0, d0/c1, d1/c0, d1/c1.
bulk = pseudobulk_counts(
    cells,
    [0, 0, 1, 1, 2, 2, 3, 3],
    num_units=4,
)
assert bool(bulk.valid.all())

design = build_experimental_design(
    [0, 1, 0, 1],
    num_conditions=2,
    donor_indices=[0, 0, 1, 1],
    num_donors=2,
    paired=True,
)
assert bool(design.valid)

# Dispersion is an explicit model input; estimate it under a separately documented
# training/design procedure when it is not known.
fit = fit_negative_binomial_glm(
    bulk.assay,
    design,
    dispersion=[0.10, 0.10],
    maximum_steps=256,
)
contrast = pairwise_condition_contrast(design, 1, 0)
test = wald_test(fit, contrast)
print(test.log2_fold_change, test.p_value, test.valid)
```

The independent rows are the four donor-by-condition units, not the eight cells.
`pseudobulk_counts` preserves missing and structural-absence semantics and reports empty
units. The Wald p-values are asymptotic and valid only where the NB2 fit, design rank,
residual degrees of freedom, contrast estimability, and supplied dispersion are valid.
Apply a declared multiple-testing procedure across the intended feature family. Fit
dispersion trends, filters, and normalization without evaluation-set leakage.
