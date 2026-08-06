# HyperRectangle

`HyperRectangle` is an analytic axis-aligned box in `R^d` represented as one
vector-valued domain factor. Use it when each point should be passed around as a
single dense vector, for example feature vectors, parameter boxes, and supervised
learning data with rows shaped `(d,)`.

This differs from a product of scalar intervals: `Interval1d(...) @ ... @
Interval1d(...)` creates multiple labeled scalar factors, while
`HyperRectangle(lower, upper, label="x")` creates one label whose value is a
vector.

!!! example
    ```python
    import jax.numpy as jnp
    import phydrax as phx

    features = phx.domain.HyperRectangle(
        lower=jnp.zeros(6),
        upper=jnp.ones(6),
        label="x",
    )

    points = jnp.array(
        [
            [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
            [0.5, 0.4, 0.3, 0.2, 0.1, 0.0],
        ]
    )
    values = jnp.sum(points, axis=1)

    @features.Function("x")
    def u(x):
        return jnp.sum(x)

    data = phx.constraints.DiscreteInteriorDataConstraint(
        "u",
        features,
        points=points,
        values=values,
    )
    ```

::: phydrax.domain.HyperRectangle
    options:
        members:
            - __init__
            - make_enforcement_gate
            - label
            - spatial_dim
            - bounds
            - volume
            - boundary_measure_value
            - sample_interior
            - sample_boundary
            - estimate_boundary_subset_measure
