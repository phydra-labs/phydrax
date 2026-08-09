# Metrics and inspection

## Metrics and scorers

Every metric returns explicit value/status evidence through its documented result
contract. Exact label, order, rank, and cluster metrics are distinct from the
`smooth_*` probability, soft-order, and soft-assignment metrics. Output reduction,
averaging, gains, calibration norms, and empty/undefined policies are explicit.

::: phydrax.ml.metrics
    options:
        filters: ["!^_"]

## Model inspection

Gradient/Jacobian/Hessian sensitivity use the callable model's actual JAX program.
Partial dependence and permutation importance preserve case/sample geometry and
weights. Influence functions require the listed regularity of the fitted objective;
linear leverage and Cook's distance use exact model structure.

::: phydrax.ml.inspection
    options:
        filters: ["!^_"]
