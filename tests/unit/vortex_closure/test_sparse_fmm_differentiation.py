from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp

import phydrax as phx


def test_sparse_vortex_fmm_strength_gradient_is_finite() -> None:
    position = jnp.asarray(((-0.7, -0.4), (-0.2, 0.5), (0.35, -0.3), (0.75, 0.45)))
    target_position = jnp.asarray(((-0.5, 0.2), (0.55, 0.1)))
    core = jnp.full((4,), 0.08)
    prepared = phx.operators.VortexFMMPlan(
        position,
        (-1.0, -1.0),
        (1.0, 1.0),
        depth=2,
        expansion_order=1,
        leaf_capacity=2,
        maximum_reference_displacement=0.5,
    ).prepare(
        source_capacity=4,
        target_capacity=2,
        target_topology="arbitrary-targets",
    )
    target = phx.discretization.VortexTargetState(target_position)

    def objective(strength):
        source = phx.discretization.VortexSourceState(
            position, strength, core_radius=core
        )
        velocity = prepared.evaluate(source, target).velocity
        return jnp.sum(velocity**2)

    strength = jnp.asarray((0.5, -0.3, 0.8, -0.4))
    gradient = jax.grad(objective)(strength)
    assert bool(jnp.all(jnp.isfinite(gradient)))
    assert float(jnp.linalg.norm(gradient)) > 0.0
    evaluation = eqx.filter_jit(prepared.evaluate)(
        phx.discretization.VortexSourceState(position, strength, core_radius=core),
        target,
    )
    assert bool(evaluation.successful)
