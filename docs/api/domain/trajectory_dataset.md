# Trajectory Dataset Domains

`TrajectoryDatasetDomain` represents a finite dataset where each row owns its own
time horizon. Use it for models of the form `u(data, t)` when each dataset element is
a function, parameter vector, forcing, or latent descriptor and the associated time
series is ragged.

Use `TrajectoryDatasetDomain` when all rows share one uniform `dt`. Use
`IrregularTrajectoryDatasetDomain` when each row has explicit, strictly
increasing observation times and the spacing may differ across rows.

Unlike `DatasetDomain(...) @ TimeInterval(...)`, this domain samples `data` and `t`
as a coupled pair. `FixedEnd()` means the end time for the sampled dataset row, not a
single global end time.

Use a `SampleLayout` block containing both labels, even for `FixedStart()`,
`FixedEnd()`, or `Fixed(value)` time components. The time coordinate remains
conditional on the sampled dataset row.

Axis-based trajectory grids are intentionally unsupported. Use joint point
mini-batches for physics residuals and ragged data terms.

For exact branch-conditional data, pair this domain with
`phydrax.enforcement.enforce_ragged_time_series(...)`. The enforcer uses the row
index carried by trajectory batches, so the hard condition is tied to the
dataset element rather than inferred from branch values. Linear hard interpolation
supports first-order time residuals; `cubic_hermite` supports second-order time
residuals and optional selected-component enforcement.

If the trajectory data is an observed input rather than the learned field, use
`phydrax.terms.TrajectorySignal(...)` to expose it as a fixed `DomainFunction`
over `(data, t)`. If the target is attached to the dataset row itself, use
`phydrax.terms.TrajectoryCaseDataTerm(...)` so the target is supervised once per
case rather than repeated over time. Fixed signals and domain arrays remain
non-trainable solver state even though they are JAX arrays.

Measure modes:

- `case_time_probability`: default expectation-style weighting.
- `time_integral_average`: average, over dataset rows, of each row's time integral.
- `time_integral_sum`: sum, over dataset rows, of each row's time integral.

```python
import jax.numpy as jnp
import jax.random as jr
import phydrax as phx

inputs = jnp.asarray([[0.0], [1.0], [2.0]])
lengths = jnp.asarray([2, 4, 3])

domain = phx.domain.TrajectoryDatasetDomain(inputs, lengths, dt=0.5)
component = domain.component()
sampling = phx.domain.PointSampling(
    8,
    layout=phx.domain.SampleLayout((("data", "t"),)),
)
batch = component.sample(sampling, key=jr.key(0))

condition = phx.conditions.Residual(
    "u",
    component,
    lambda u: phx.operators.partial_t(u, var="t"),
)
source = phx.integration.per_step(
    phx.integration.mean_over(component),
    sampling,
)
physics = phx.terms.ResidualPenalty(condition, source)
```

## Uniform Time Grids

::: phydrax.domain.TrajectoryDatasetDomain
    options:
        members:
            - __init__
            - labels
            - data_label
            - time_label
            - measure_mode
            - sampling_mode
            - size
            - max_length
            - total_observations
            - durations
            - end_times
            - factor
            - same_support
            - input_rows
            - observation_times
            - points_from_case_time

## Irregular Time Grids

`IrregularTrajectoryDatasetDomain` has the same paired `(data, t)` sampling model,
but time values are gathered from a per-case padded time table instead of the
formula `start + dt * index`. It is the right domain when observations are
non-uniform in time or when cases use different time grids.

```python
import jax.numpy as jnp
import jax.random as jr
import phydrax as phx

inputs = jnp.asarray([[0.0], [1.0]])
times = jnp.asarray(
    [
        [0.0, 0.2, 0.7, 0.0],
        [0.0, 0.1, 0.4, 1.0],
    ]
)
lengths = jnp.asarray([3, 4])

domain = phx.domain.IrregularTrajectoryDatasetDomain(inputs, times, lengths)
component = domain.component()
sampling = phx.domain.PointSampling(
    8,
    layout=phx.domain.SampleLayout((("data", "t"),)),
)
batch = component.sample(sampling, key=jr.key(0))
```

::: phydrax.domain.IrregularTrajectoryDatasetDomain
    options:
        members:
            - __init__
            - labels
            - data_label
            - time_label
            - measure_mode
            - sampling_mode
            - size
            - max_length
            - total_observations
            - start_times
            - end_times
            - durations
            - node_widths
            - factor
            - same_support
            - input_rows
            - observation_times
            - lower_time_indices
            - nearest_time_indices
            - bracketing_time_indices
            - points_from_case_time
