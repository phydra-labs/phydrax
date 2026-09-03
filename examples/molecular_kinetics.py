import jax.numpy as jnp

import phydrax as phx


state = jnp.asarray([1.0, -0.4])
observations = []
for step in range(256):
    observations.append(state)
    state = jnp.asarray([0.98 * state[0], 0.55 * state[1]]) + 0.02 * jnp.asarray(
        [jnp.sin(0.31 * step), jnp.cos(0.47 * step)]
    )
states = jnp.stack(observations)
data = phx.dynamics.TrajectoryData(
    0.1 * jnp.arange(states.shape[0]),
    states,
    state_layout=phx.dynamics.StateLayout((2,), component_names=("slow", "fast")),
    source_id="molecular-kinetics-example",
)
tica = phx.dynamics.identification.fit_tica(data, lag=2, n_modes=1, regularization=1.0e-6)
coordinates = tica.transform(states)[:, 0]
assignments = (coordinates > jnp.median(coordinates)).astype(jnp.int32)
model = phx.dynamics.identification.fit_markov_state_model(
    data, assignments, state_count=2, lag=2, reversible=True
)
if not bool(tica.valid & model.valid):
    raise RuntimeError("Variational kinetics example did not qualify.")
print(float(tica.eigenvalues[0]), model.transition_matrix)
