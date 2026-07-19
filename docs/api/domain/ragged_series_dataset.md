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
targets = jnp.sum(static, axis=1)

batch = domain.points_from_indices([0, 1])
payload = batch["data"]
```

The payload contains:

- `static`: optional static per-case data.
- `series`: padded series values shaped `(batch, Lmax, ...)`.
- `time`: the uniform time grid broadcast to `(batch, Lmax)`.
- `mask`: valid-entry mask shaped `(batch, Lmax)`.
- `length`: valid lengths shaped `(batch,)`.
- `sample_index`: integer source indices for the returned series entries.
- `sample_scale`: `length / valid_sample_count`, useful for sampled sum estimates.

For row-aligned targets, pair this domain with
`phydrax.constraints.RaggedSeriesSupervisedConstraint`. For neural models, wrap a
ragged-series encoder with `phydrax.nn.RaggedSeriesModel`; the built-in
`phydrax.nn.MaskedSeriesPoolingModel` provides a small masked-pooling baseline.

## Long series

Full-row batches materialize `(batch, global_Lmax, ...)`, which is simple and
deterministic but can be expensive when one case is much longer than the others.
For training on long records, use sampled fixed-width views through
`RaggedSeriesSupervisedConstraint`:

```python
constraint = phx.constraints.RaggedSeriesSupervisedConstraint(
    "u",
    domain.component(),
    targets,
    num_cases=64,
    series_sampling="window_uniform",
    num_series_points=256,
)
```

Sampled modes materialize `(batch, num_series_points, ...)` regardless of the
global padded length. The domain stores a packed valid representation internally,
so sampled views gather only real observations.

Available sampled modes:

- `points_uniform`: random valid points per case.
- `window_uniform`: a random contiguous window per case.
- `prefix`: the first `num_series_points` entries.
- `suffix`: the last `num_series_points` entries.

Use full mode for small data or deterministic full-sequence evaluation. Use
sampled modes for large training sets and long series.

When you want full-sequence supervision but the global maximum length is much
larger than most records, bucket by length:

```python
constraints = phx.constraints.RaggedSeriesSupervisedConstraint.bucketed(
    "u",
    domain.component(),
    targets,
    num_cases=64,
    num_buckets=8,
)
```

This returns one constraint per non-empty length bucket. Each bucket samples from
its own case subset and materializes a prefix view whose width is that bucket's
maximum valid length, so every case in the bucket is covered without padding to
the global maximum length. The requested `num_cases` is split across buckets by
case count, and bucket losses are scaled so the combined estimator matches one
full padded constraint with the same reduction while avoiding global padding. For
many bucket shapes, call `solve(..., train_constraint_sample_size=1)` so JIT
compiles one bucket-shaped optimizer step at a time instead of one large graph
containing every bucket.

::: phydrax.domain.RaggedSeriesDatasetDomain
    options:
        members:
            - __init__
            - from_padded
            - from_sequences
            - label
            - size
            - max_length
            - measure_mode
            - measure
            - time_axis
            - total_observations
            - sample
            - sample_indices
            - input_rows
            - points_from_indices
            - sampled_input_rows
            - sampled_points_from_indices
            - equivalent
