"""Run one De Groote--Fregly 2016 compliant-tendon muscle transaction."""

import jax.numpy as jnp

from phydrax.applications.skeletal_muscle.musculotendon import (
    de_groote_fregly_2016_active_force_length,
    de_groote_fregly_2016_force_velocity,
    de_groote_fregly_2016_inverse_tendon_force_length,
    DeGrooteFregly2016ImplicitTendonForcePlan,
    DeGrooteFregly2016Parameters,
    DeGrooteFregly2016Plan,
    DeGrooteFregly2016State,
)


parameters = DeGrooteFregly2016Parameters(
    jnp.asarray([1800.0]),
    jnp.asarray([0.10]),
    jnp.asarray([0.24]),
    jnp.asarray([0.14]),
    jnp.asarray([1.0]),
)
activation = jnp.asarray([0.4])
cosine = jnp.cos(parameters.pennation_angle_at_optimum_rad)
normalized_force = (
    activation
    * de_groote_fregly_2016_active_force_length(parameters, jnp.ones((1,)))
    * de_groote_fregly_2016_force_velocity(parameters, jnp.zeros((1,)))
    * cosine
)
tendon_length = (
    de_groote_fregly_2016_inverse_tendon_force_length(parameters, normalized_force)
    * parameters.tendon_slack_length_m
)
musculotendon_length = tendon_length + parameters.optimal_fiber_length_m * cosine
state = DeGrooteFregly2016State(activation, normalized_force)
model = DeGrooteFregly2016Plan(parameters, ("soleus",)).prepare(state)

evaluation = model.evaluate(
    state,
    activation,
    musculotendon_length,
    jnp.zeros((1,)),
)
candidate = model.candidate(
    state,
    activation,
    musculotendon_length,
    jnp.zeros((1,)),
    jnp.asarray(1.0e-5),
)
accepted = model.commit(candidate)
implicit = DeGrooteFregly2016ImplicitTendonForcePlan(
    parameters, ("soleus",)
).prepare(state)
implicit_candidate = implicit.candidate(
    state,
    activation,
    musculotendon_length,
    jnp.zeros((1,)),
    jnp.asarray(1.0e-5),
)

print("tendon force [N]", evaluation.tendon_force_N)
print("force-equilibrium residual", evaluation.evidence.force_equilibrium_residual_normalized)
print("power-balance residual [W]", evaluation.evidence.power_balance_residual_W)
print("candidate accepted", bool(candidate.successful))
print("accepted normalized tendon force", accepted.normalized_tendon_force)
print("implicit S25 residual", implicit_candidate.evidence.algebraic_residual)
print("implicit candidate accepted", bool(implicit_candidate.successful))
