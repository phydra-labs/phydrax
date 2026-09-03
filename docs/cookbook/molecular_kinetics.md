# Fit and validate slow molecular coordinates

Normalize observations as `TrajectoryData`, then fit the coordinate and state model
separately.

```python
import jax.numpy as jnp
import phydrax as phx

data = phx.dynamics.TrajectoryData(
    times,
    states,
    state_layout=phx.dynamics.StateLayout((feature_count,)),
    reset_mask=resets,
    weights=weights,
    source_id="unbiased-trajectories",
)
tica = phx.dynamics.identification.fit_tica(
    data,
    lag=10,
    n_modes=2,
    regularization=1.0e-6,
    weighting=phx.dynamics.identification.LaggedPairWeighting.SOURCE,
)
slow = tica.transform(data.states)
assignments = cluster_model(slow)
short = phx.dynamics.identification.fit_markov_state_model(
    data, assignments, state_count=4, lag=10, reversible=True
)
long = phx.dynamics.identification.fit_markov_state_model(
    data, assignments, state_count=4, lag=20, reversible=True
)
validation = phx.dynamics.analysis.validate_markov_models(short, long, 2)
if not bool(tica.valid & short.valid & long.valid & validation.valid):
    raise RuntimeError("Kinetic model did not qualify")
```

Split training and validation by complete trajectory or contiguous time block. Random
lagged-pair splitting leaks neighboring observations. Do not interpret configuration
weights from biased dynamics as kinetic path weights.
