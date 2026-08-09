# Ground costs

A ground cost maps two finite coordinate vectors to one real nonnegative scalar.
Transport solvers use `pairwise(left, right)` for one pair and `matrix(left, right)` for
all pairs. Custom costs subclass `AbstractGroundCost`; implementations must be JAX
transformable and must return finite nonnegative values on active support.

## Built-in costs

`SquaredEuclideanCost` is the unscaled baseline. With physical coordinates of mixed
units, use `WeightedSquaredEuclideanCost` and supply positive component scales rather
than relying on numerical magnitude. Each component is divided by its scale before
squaring.

`PeriodicSquaredEuclideanCost` applies the shortest wrapped displacement on every
component. Its period vector must match the feature size and contain finite positive
periods.

`PrecomputedCost` accepts an explicit source-by-target matrix. It is useful when costs
come from graph geodesics or another external solver, but it disables pointwise cost
evaluation and blockwise matrix-free execution because no coordinate-local cost rule
exists.

::: phydrax.transport.AbstractGroundCost

---

::: phydrax.transport.SquaredEuclideanCost

---

::: phydrax.transport.WeightedSquaredEuclideanCost

---

::: phydrax.transport.PeriodicSquaredEuclideanCost

---

::: phydrax.transport.PrecomputedCost

## Scaling rule

For a squared cost, multiplying every coordinate by a factor `s` multiplies cost by
`s**2`. A dimensionless regularization regime therefore requires multiplying
`epsilon` by the same factor. Prefer explicit physical nondimensionalization or
component scales over tuning `epsilon` around accidental units.
