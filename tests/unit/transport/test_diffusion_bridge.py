import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

import phydrax as phx
from phydrax._probability import DiagonalNormalLaw
from phydrax.integration._api import IntegrationRealization
from phydrax.stochastic._state_space import (
    CallableTransitionKernel,
    StateSpaceStepContext,
)
from phydrax.transport.dynamic._diffusion_problem import (
    DiffusionBridgePlan,
    DiffusionBridgeProblem,
)
from phydrax.transport.dynamic._diffusion_solver import (
    prepare_diffusion_bridge,
    sample_diffusion_bridge,
    solve_diffusion_bridge,
)


def _brownian_transition():
    def sample(key, state, t0, t1, context):
        del context
        return state + jnp.sqrt(t1 - t0) * jr.normal(key, state.shape)

    def log_prob(next_state, state, t0, t1, context):
        del context
        variance = t1 - t0
        residual = next_state - state
        return -0.5 * jnp.sum(residual**2 / variance + jnp.log(2.0 * jnp.pi * variance))

    return CallableTransitionKernel(
        sample,
        state_shape=(1,),
        process_id="brownian-reference",
        approximation_id="exact-brownian-step",
        log_prob_fn=log_prob,
    )


def _proposal_realization(points):
    target = phx.integration.weighted(
        points,
        -jnp.log(jnp.asarray(points.shape[0], dtype=points.dtype))
        * jnp.ones((points.shape[0],), dtype=points.dtype),
        normalized=True,
        independent=False,
        sample_axes=0,
        provenance="bridge-proposal",
    )
    return IntegrationRealization(target, None, None, None)


def _problem_and_plan(capacity=9):
    grid = phx.dynamics.TimeGrid(
        jnp.asarray([0.0, 0.5, 1.0]),
        time_id="brownian-bridge-grid",
    )
    contexts = (
        StateSpaceStepContext.empty(step_index=0),
        StateSpaceStepContext.empty(step_index=1),
    )
    problem = DiffusionBridgeProblem(
        DiagonalNormalLaw(jnp.asarray([0.0]), jnp.asarray([0.7]), event_shape=(1,)),
        DiagonalNormalLaw(jnp.asarray([1.0]), jnp.asarray([0.7]), event_shape=(1,)),
        _brownian_transition(),
        grid,
        contexts,
        problem_id="finite-brownian-bridge",
    )
    points = jnp.linspace(-3.0, 4.0, 9)[:, None]
    proposals = tuple(_proposal_realization(points) for _ in range(3))
    plan = DiffusionBridgePlan(
        capacity,
        proposals,
        4,
        solver=phx.transport.dynamic.SchrodingerBridgeSolver(
            max_iterations=256,
            tolerance=1.0e-7,
        ),
        audit_capacity=32,
        minimum_ess=2.0,
        maximum_tail_error=2.0,
    )
    return problem, plan


def test_diffusion_bridge_prepares_solves_and_samples_finite_chain():
    problem, plan = _problem_and_plan()
    prepared = prepare_diffusion_bridge(problem, plan, key=jr.key(12))
    result = solve_diffusion_bridge(prepared)
    paths = sample_diffusion_bridge(result, jr.key(13), (16,))

    assert prepared.supports.shape == (3, 9, 1)
    assert prepared.log_transitions.shape == (2, 9, 9)
    assert result.doob_transitions.shape == (2, 9, 9)
    assert result.physical_marginals.shape == (3, 9)
    assert jnp.allclose(
        result.physical_marginals[0],
        prepared.endpoint_probabilities[0],
        atol=1.0e-5,
    )
    assert jnp.allclose(
        result.physical_marginals[-1],
        prepared.endpoint_probabilities[-1],
        atol=1.0e-5,
    )
    assert paths.shape == (16, 3, 1)
    assert result.approximation_kind == "exact-prepared-chain-diffusion-approximation"


def test_diffusion_bridge_prepared_solve_is_jittable():
    problem, plan = _problem_and_plan()
    prepared = prepare_diffusion_bridge(problem, plan, key=jr.key(14))
    marginals = jax.jit(lambda value: solve_diffusion_bridge(value).physical_marginals)(
        prepared
    )

    assert marginals.shape == (3, 9)
    assert jnp.all(jnp.isfinite(marginals))


def test_diffusion_bridge_fails_closed_on_support_capacity_mismatch():
    problem, plan = _problem_and_plan(capacity=8)
    with pytest.raises(ValueError, match="support_capacity"):
        prepare_diffusion_bridge(problem, plan, key=jr.key(15))
