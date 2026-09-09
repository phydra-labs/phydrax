#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax.numpy as jnp
import numpy as np

import phydrax as phx


opt = phx.optim


def _policy():
    return opt.StateAcceptancePolicy(
        state_relative_tolerance=0.0,
        state_absolute_tolerance=1e-6,
        adjoint_relative_tolerance=0.0,
        adjoint_absolute_tolerance=1e-6,
    )


def test_response_pytree_cotangent_includes_direct_and_implicit_design_paths():
    problem = opt.StateDesignProblem(
        lambda state, design, _: (
            state - jnp.array([design[0] ** 2, design[0] + design[1]])
        ),
        lambda state, design, _: jnp.sum(state) + jnp.sum(design),
        acceptance_policy=_policy(),
    )
    point = opt.prepare_state_design_linearization(
        problem, jnp.array([2.0, 3.0]), jnp.zeros(2)
    )
    response = lambda state, design, _: {
        "outputs": jnp.array([state[0] * state[1], design[1] ** 2]),
        "direct": design[0],
    }
    pull = eqx.filter_jit(opt.state_design_response_vjp)(
        point,
        response,
        {"outputs": jnp.array([1.0, -0.5]), "direct": jnp.array(0.25)},
    )
    assert bool(pull.accepted)
    np.testing.assert_allclose(pull.values["outputs"], [20.0, 9.0], atol=1e-6)
    np.testing.assert_allclose(pull.design_cotangent, [24.25, 1.0], atol=1e-6)


def test_design_only_response_remains_valid_at_singular_state_linearization():
    problem = opt.StateDesignProblem(
        lambda state, design, _: (state - design) ** 3,
        lambda state, design, _: jnp.sum(design**2),
        acceptance_policy=_policy(),
    )
    design = jnp.array([2.0, 3.0])
    point = opt.prepare_state_design_linearization(problem, design, design)
    pull = opt.state_design_response_vjp(point, depends_on_state=False)
    assert bool(pull.accepted)
    np.testing.assert_allclose(pull.design_cotangent, [4.0, 6.0], atol=1e-6)


def test_changed_realization_cannot_reuse_an_accepted_response_derivative():
    problem = opt.StateDesignProblem(
        lambda state, design, _: state - design,
        lambda state, design, _: jnp.sum(state**2),
        acceptance_policy=_policy(),
        state_realization=lambda state, design, valid: valid,
    )
    point = opt.prepare_state_design_linearization(
        problem, jnp.array([2.0]), jnp.zeros(1), args=jnp.asarray(True)
    )
    assert bool(opt.state_design_response_vjp(point).accepted)
    stale = eqx.tree_at(lambda value: value.args, point, jnp.asarray(False))
    assert not bool(opt.state_design_response_vjp(stale).accepted)


def test_blockwise_acceptance_cannot_hide_failed_small_scale_equation():
    policy = opt.StateAcceptancePolicy(
        state_relative_tolerance=0.0, state_absolute_tolerance=1e-8
    )
    failed = policy.state_evidence(
        jnp.array([1.0]),
        jnp.array([2e-8]),
        opt.OptimizationStatus.SUCCESS,
        reference_norm=1.0,
        admissible=True,
        realization_matches=True,
    )
    exact_policy = opt.StateAcceptancePolicy(
        state_relative_tolerance=0.0, state_absolute_tolerance=0.0
    )
    exact = exact_policy.state_evidence(
        jnp.array([1e8]),
        jnp.zeros(1),
        opt.OptimizationStatus.SUCCESS,
        reference_norm=1e8,
        admissible=True,
        realization_matches=True,
    )
    combined = opt.StateAcceptanceEvidence.from_blocks(
        ("small-equation", "large-equation"), (failed, exact)
    )
    assert not bool(combined.accepted)
    assert bool(combined.blocks[1].accepted)
    np.testing.assert_allclose(combined.normalized_residual, 2.0)


def test_all_at_once_success_requires_final_physical_admissibility():
    problem = opt.StateDesignProblem(
        lambda state, design, _: state - design,
        lambda state, design, _: jnp.sum((state - 1.0) ** 2) + 0.1 * jnp.sum(design**2),
        acceptance_policy=_policy(),
        state_admissibility=lambda state, design, _: jnp.all(state < 0.5),
    )
    compiled = opt.compile_structured_state_design(problem, jnp.zeros(1), jnp.zeros(1))
    result = opt.solve_structured_state_design(
        compiled,
        method=opt.PrimalDualInteriorPoint(mode="dense-filter", max_dense_dimension=16),
        termination=opt.OptimizationTermination(
            absolute_optimality=1e-6, relative_optimality=0.0, maximum_steps=80
        ),
    )
    assert bool(result.optimization.successful)
    assert not bool(result.successful)
    assert not bool(result.state_acceptance.admissible)
    assert int(result.status) == int(opt.OptimizationStatus.CERTIFICATION_FAILED)
