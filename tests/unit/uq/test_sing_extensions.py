from typing import Any

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

import phydrax as phx
from phydrax.uq._sing_gp import SINGSparseGPDrift
from phydrax.uq._sing_transition import (
    _projected_sing_problem,
    _ProjectedEulerTransition,
    evaluate_sing_transition,
    sing_constrained_smoother,
    sing_objective,
    SINGSupportPlan,
    SINGTransitionPlan,
)


def test_sing_extensions_scenario_1() -> None:
    support = SINGSupportPlan(
        jnp.asarray([[0.0, 1.0]]),
        jnp.asarray([[1.0], [0.0]]),
        offset=jnp.asarray([2.0]),
        rank=1,
        support_id="horizontal-line",
    )

    assert support.reference == "hausdorff"
    assert support.rank == 1
    assert jnp.allclose(support.residual(jnp.asarray([3.0, 2.0])), 0.0)
    plan = SINGTransitionPlan(support=support)
    # ty: ignore[unresolved-attribute]
    assert plan.support.support_id == "horizontal-line"
    with pytest.raises(ValueError, match="tangent_basis"):
        SINGSupportPlan(
            jnp.asarray([[0.0, 1.0]]),
            jnp.asarray([[1.0], [1.0]]),
            rank=1,
            support_id="invalid",
        )
    problem = _affine_singular_problem()
    support = SINGSupportPlan(
        jnp.asarray([[0.0, 1.0]]),
        jnp.asarray([[1.0], [0.0]]),
        offset=jnp.asarray([2.0]),
        rank=1,
        support_id="horizontal-affine-support",
    )
    plan = SINGTransitionPlan(support=support)
    result = sing_constrained_smoother(
        problem,
        transition_plan=plan,
        key=jr.key(41),
        max_iterations=8,
    )
    objective = sing_objective(
        problem,
        result.state,
        transition_plan=plan,
    )

    assert result.reference_measure == "hausdorff"
    assert result.approximation_kind == "exact-affine-hausdorff-euler"
    assert jnp.all(jnp.abs(result.support_residuals) < 1.0e-8)
    assert len(result.solve_evidence) >= 3
    assert objective.objective_kind == "elbo"
    assert objective.transition_semantics == "affine-hausdorff"
    transition_evidence = evaluate_sing_transition(
        problem.model.transition,
        jnp.asarray([0.0, 2.0]),
        jnp.asarray([0.1, 2.0]),
        jnp.asarray(0.0),
        jnp.asarray(0.25),
        problem.step_context(0, 0),
        plan,
    )

    assert jnp.isfinite(objective.objective)
    assert transition_evidence.reference_measure == "hausdorff"
    assert transition_evidence.rank == 1
    assert transition_evidence.valid
    assert jnp.isfinite(transition_evidence.log_density)
    problem = _affine_singular_problem()
    bad_support = SINGSupportPlan(
        jnp.asarray([[1.0, 0.0]]),
        jnp.asarray([[0.0], [1.0]]),
        offset=jnp.asarray([0.0]),
        rank=1,
        support_id="vertical-incompatible-support",
    )
    with pytest.raises(ValueError, match="affine/tangent"):
        sing_constrained_smoother(
            problem,
            transition_plan=SINGTransitionPlan(support=bad_support),
            key=jr.key(42),
            max_iterations=2,
        )
    base = _affine_singular_problem()
    model = phx.stochastic.StateSpaceModel(
        phx.stochastic.CategoricalStatePrior(
            jnp.asarray([[0.0, 2.0], [1.0, 2.0]]), jnp.asarray([0.5, 0.5])
        ),
        base.model.transition,
        base.model.observation,
        model_id="affine-categorical-model",
    )
    problem = phx.stochastic.StateSpaceProblem(
        model,
        base.observations,
        initial_time=0.0,
        problem_id="affine-categorical-problem",
    )
    support = SINGSupportPlan(
        jnp.asarray([[0.0, 1.0]]),
        jnp.asarray([[1.0], [0.0]]),
        offset=jnp.asarray([2.0]),
        rank=1,
        support_id="horizontal-affine-support",
    )
    with pytest.raises(TypeError, match="GaussianStatePrior"):
        sing_constrained_smoother(
            problem,
            transition_plan=SINGTransitionPlan(support=support),
            key=jr.key(43),
            max_iterations=2,
        )
    points = jnp.asarray([[-1.0], [0.0], [1.0]])
    kernel = phx.kernels.SquaredExponentialKernel(length_scale=0.7)
    drift = SINGSparseGPDrift(
        points,
        (kernel,),
        jnp.zeros((1, 3)),
        jnp.eye(3)[None, ...],
        drift_id="zero-whitened-gp",
    )

    assert drift.approximation_kind == "fixed-inducing-fitc"
    assert drift.inducing_points.shape == (3, 1)
    assert jnp.allclose(drift(jnp.asarray([0.2])), jnp.asarray([0.0]))
    assert jnp.allclose(drift.kl_divergence(), 0.0)
    assert jnp.all(drift.fitc_variance(jnp.asarray([0.2])) >= 0.0)
    with pytest.raises(TypeError, match="surrogate_provider"):
        SINGTransitionPlan("local-linearization")


def _affine_singular_problem() -> Any:
    system = phx.dynamics.ContinuousSystem(
        lambda time, state, args: jnp.asarray([-0.25 * state[0], 0.0]),
        state_layout=phx.dynamics.StateLayout((2,)),
        system_id="affine-singular-system",
    )
    noise = phx.solver.WienerTerm(
        "affine-singular-noise",
        lambda time, state, args: jnp.asarray([[0.3], [0.0]]),
        (1,),
        structure="additive",
        basis_id="affine-singular-basis",
    )
    transition = phx.stochastic.EulerMaruyamaTransitionKernel(
        system,
        (noise,),
        state_shape=(2,),
        noise_shape=(1,),
        process_id="affine-singular-euler",
    )
    observation = phx.stochastic.LinearGaussianObservationModel(
        jnp.asarray([[1.0, 0.0]]),
        jnp.asarray([[0.1]]),
        state_shape=(2,),
        observation_shape=(1,),
        observation_id="affine-singular-observation",
    )
    sequence = phx.stochastic.ObservationSequence(
        jnp.asarray([0.25, 0.5, 0.75]),
        jnp.asarray([[0.2], [0.1], [-0.1]]),
        sequence_id="affine-singular-sequence",
    )
    model = phx.stochastic.StateSpaceModel(
        phx.stochastic.GaussianStatePrior(
            jnp.asarray([0.0, 2.0]),
            jnp.asarray([[0.8, 0.0], [0.0, 0.5]]),
            state_shape=(2,),
            prior_id="affine-singular-prior",
        ),
        transition,
        observation,
        model_id="affine-singular-model",
    )
    return phx.stochastic.StateSpaceProblem(
        model,
        sequence,
        initial_time=0.0,
        problem_id="affine-singular-problem",
    )


def test_support_drift_actions_match_dense_random_affine_reference() -> None:
    rng = np.random.default_rng(719)
    orthogonal, _ = np.linalg.qr(rng.normal(size=(5, 5)))
    basis, normal = orthogonal[:, :2], orthogonal[:, 2:].T
    offset = rng.normal(size=3)
    origin = normal.T @ offset
    matrix = rng.normal(size=(2, 2))
    forcing = rng.normal(size=2)
    noise = basis @ rng.normal(size=(2, 2))
    support = SINGSupportPlan(
        normal,
        basis,
        offset=offset,
        rank=2,
        support_id="random-affine-directional-support",
    )

    def drift(time: jax.Array, state: jax.Array, args: None) -> jax.Array:
        del time, args
        return jnp.asarray(basis) @ (
            jnp.asarray(matrix) @ (jnp.asarray(basis.T) @ (state - jnp.asarray(origin)))
            + jnp.asarray(forcing)
        )

    system = phx.dynamics.ContinuousSystem(
        drift,
        state_layout=phx.dynamics.StateLayout((5,)),
        system_id="random-support-system",
    )
    term = phx.solver.WienerTerm(
        "random-support-noise",
        lambda t, x, args: jnp.asarray(noise),
        (2,),
        structure="additive",
        basis_id="random-support-noise-basis",
    )
    transition = phx.stochastic.EulerMaruyamaTransitionKernel(
        system,
        (term,),
        state_shape=(5,),
        noise_shape=(2,),
        process_id="random-support-euler",
    )
    observation = phx.stochastic.LinearGaussianObservationModel(
        jnp.eye(5),
        0.1 * jnp.eye(5),
        state_shape=(5,),
        observation_shape=(5,),
        observation_id="random-support-observation",
    )
    model = phx.stochastic.StateSpaceModel(
        phx.stochastic.GaussianStatePrior(
            origin, np.eye(5), state_shape=(5,), prior_id="random-support-prior"
        ),
        transition,
        observation,
        model_id="random-support-model",
    )
    problem = phx.stochastic.StateSpaceProblem(
        model,
        phx.stochastic.ObservationSequence(
            jnp.asarray([0.1]),
            jnp.asarray(origin[None]),
            sequence_id="random-support-data",
        ),
        initial_time=0.0,
        problem_id="random-support-problem",
    )
    # The previous ambient Jacobian certifies the same affine support probes.
    ambient_jacobian = basis @ matrix @ basis.T
    np.testing.assert_allclose(ambient_jacobian @ basis, basis @ matrix, atol=2e-12)
    reduced, _ = _projected_sing_problem(problem, support)
    reduced_transition = reduced.model.transition
    assert isinstance(reduced_transition, _ProjectedEulerTransition)
    point = jnp.asarray(rng.normal(size=2))
    context = reduced.step_context(0, 0)

    def reduced_drift(q: jax.Array) -> jax.Array:
        return reduced_transition.drift(jnp.asarray(0.0), q, context)

    np.testing.assert_allclose(
        reduced_drift(point), matrix @ np.asarray(point) + forcing, atol=2e-12, rtol=2e-12
    )
    np.testing.assert_allclose(
        jax.jacfwd(reduced_drift)(point), matrix, atol=2e-12, rtol=2e-12
    )
