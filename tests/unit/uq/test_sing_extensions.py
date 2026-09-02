import jax.numpy as jnp
import jax.random as jr
import pytest

import phydrax as phx
from phydrax.uq._sing_gp import SINGSparseGPDrift
from phydrax.uq._sing_transition import (
    evaluate_sing_transition,
    sing_constrained_smoother,
    sing_objective,
    SINGSupportPlan,
    SINGTransitionPlan,
)


def test_affine_sing_support_is_hausdorff_and_constant_rank():
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
    assert plan.support.support_id == "horizontal-line"


def test_affine_sing_support_rejects_misaligned_or_rank_changing_basis():
    with pytest.raises(ValueError, match="tangent_basis"):
        SINGSupportPlan(
            jnp.asarray([[0.0, 1.0]]),
            jnp.asarray([[1.0], [1.0]]),
            rank=1,
            support_id="invalid",
        )


def test_sparse_gp_drift_has_fixed_inducing_topology_and_exact_whitened_kl():
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


def test_solver_backed_sing_requires_explicit_surrogate_provider():
    with pytest.raises(TypeError, match="surrogate_provider"):
        SINGTransitionPlan("local-linearization")


def _affine_singular_problem():
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


def test_affine_hausdorff_smoother_has_normalized_objective_and_solve_evidence():
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


def test_affine_hausdorff_smoother_rejects_nontangent_diffusion():
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
