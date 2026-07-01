# Ragged Series Dataset Domain

`RaggedSeriesDatasetDomain` represents finite empirical cases whose inputs include
aligned variable-length series. Use it for conditional regression or operator
learning where each row is shaped like:

```text
(static values, series values over a valid length) -> scalar/vector target
```

The domain samples cases, not individual series timesteps. The padded series axis,
time grid, boolean mask, and valid length are part of the sampled row payload.
This is different from `TrajectoryDatasetDomain`, which represents models of the
form `u(data, t)` and supervises outputs at sampled times.

```python
import jax.numpy as jnp
import phydrax as phx

static = jnp.asarray([[1.0, 0.0], [2.0, 1.0]])
series = jnp.asarray(
    [
        [[1.0, 2.0], [3.0, 4.0], [0.0, 0.0]],
        [[5.0, 6.0], [7.0, 8.0], [9.0, 10.0]],
    ]
)
lengths = jnp.asarray([2, 3])

domain = phx.domain.RaggedSeriesDatasetDomain(
    series,
    lengths,
    static=static,
    start=0.0,
    dt=0.1,
)

batch = domain.points_from_indices([0, 1])
payload = batch["data"]
```

The payload contains:

- `static`: optional static per-case data.
- `series`: padded series values shaped `(batch, Lmax, ...)`.
- `time`: the uniform time grid broadcast to `(batch, Lmax)`.
- `mask`: valid-entry mask shaped `(batch, Lmax)`.
- `length`: valid lengths shaped `(batch,)`.

For row-aligned targets, pair this domain with
`phydrax.constraints.RaggedSeriesSupervisedConstraint`. For neural models, wrap a
ragged-series encoder with `phydrax.nn.RaggedSeriesModel`; the built-in
`phydrax.nn.MaskedSeriesPoolingModel` provides a small masked-pooling baseline.

::: phydrax.domain.RaggedSeriesDatasetDomain
    options:
        members:
            - __init__
            - label
            - size
            - max_length
            - measure_mode
            - measure
            - time_axis
            - sample
            - sample_indices
            - input_rows
            - points_from_indices
            - equivalent
