#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
from phydrax._score_field import StateTimeScoreField
from phydrax.applications.solid_mechanics._learned_design import (
    MechanicsPotentialGuidance,
    solve_learned_topology_design,
)
from phydrax.optim._state_design_parameterization import reparameterize_state_design
from tools.learned_physical_design_benchmarks import (
    _linear_policy,
    build_density_case,
    build_shape_case,
)


def _parameterization(problem, decoder=lambda latent: latent, *, template=None, **kwargs):
    template = jnp.asarray([0.5]) if template is None else template
    return reparameterize_state_design(
        problem,
        decoder,
        jnp.asarray([0.5]),
        template,
        decoder_id="frozen-test-decoder",
        realization_id="fixed-test-realization",
        latent_bounds=phx.optim.Bounds(-3.0, 3.0),
        **kwargs,
    )


def test_physical_response_vjp_composes_decoder_and_preserves_constraints():
    physical = phx.optim.StateDesignProblem(
        lambda state, design, args: state - args * design["gain"],
        lambda state, design, args: 0.5 * state**2 + 0.3 * design["gain"],
        design_bounds=phx.optim.Bounds(0.2, 2.0),
        constraints=(
            phx.optim.StateDesignConstraint(
                lambda state, design, args: state,
                upper=3.0,
                constraint_id="state-cap",
            ),
        ),
    )
    decoder = lambda latent: {"gain": jnp.exp(latent[0])}
    lowered = _parameterization(physical, decoder, template={"gain": jnp.asarray(1.0)})
    latent = jnp.asarray([0.1])
    response = lowered.response_vjp(latent, jnp.asarray(0.0), args=2.0)
    gain = jnp.exp(latent[0])
    assert response.accepted
    np.testing.assert_allclose(response.values, 2.0 * gain**2 + 0.3 * gain, rtol=1e-6)
    np.testing.assert_allclose(
        response.latent_cotangent, jnp.asarray([4.0 * gain**2 + 0.3 * gain]), rtol=1e-6
    )
    epsilon = 1.0e-4
    plus = lowered.response_vjp(latent + epsilon, jnp.asarray(0.0), args=2.0)
    minus = lowered.response_vjp(latent - epsilon, jnp.asarray(0.0), args=2.0)
    np.testing.assert_allclose(
        response.latent_cotangent[0],
        (plus.values - minus.values) / (2 * epsilon),
        rtol=2e-4,
    )
    state = lowered.problem.solve_state(latent, jnp.asarray(0.0), args=2.0)
    values = lowered.problem.constraint_values(state.state, latent, 2.0)
    np.testing.assert_allclose(values[0], 2.0 * gain)
    lower, upper = lowered.problem.constraints[1].bounds(values[1])
    np.testing.assert_allclose(lower, 0.2)
    np.testing.assert_allclose(upper, 2.0)


def test_latent_mma_cannot_escape_original_physical_bound():
    physical = phx.optim.StateDesignProblem(
        lambda state, design, args: state - design,
        lambda state, design, args: jnp.sum((state - 1.5) ** 2),
        design_bounds=phx.optim.Bounds(0.2, 0.6),
    )
    lowered = _parameterization(physical, jax.nn.sigmoid)
    result = phx.optim.solve_state_design(
        lowered.problem,
        jnp.zeros((1,)),
        jnp.zeros((1,)),
        method=phx.optim.ReducedMMA(),
        termination=phx.optim.OptimizationTermination(
            maximum_steps=100,
            absolute_optimality=2e-5,
            relative_optimality=0.0,
        ),
    )
    assert result.successful
    np.testing.assert_allclose(lowered.decode(result.design), 0.6, atol=5e-5)
    assert float(jnp.max(lowered.decode(result.design))) <= 0.60005


def test_lowering_preserves_physical_block_certification():
    policy = phx.optim.StateAcceptancePolicy()

    def certify(
        state, design, residual, status, *, reference_norm, args, solver_acceptance=None
    ):
        del args, solver_acceptance
        block = policy.state_evidence(
            state,
            residual,
            status,
            reference_norm=reference_norm,
            admissible=design["gain"] < 1.5,
            realization_matches=True,
        )
        return phx.optim.StateAcceptanceEvidence.from_blocks(
            ("physical-branch",), (block,)
        )

    physical = phx.optim.StateDesignProblem(
        lambda state, design, args: state - design["gain"],
        lambda state, design, args: state**2,
        state_certification=certify,
    )
    lowered = _parameterization(
        physical,
        lambda latent: {"gain": 2.0 * latent[0]},
        template={"gain": jnp.asarray(1.0)},
    )
    accepted = lowered.problem.solve_state(jnp.asarray([0.5]), jnp.asarray(0.0))
    rejected = lowered.problem.solve_state(jnp.asarray([1.0]), jnp.asarray(0.0))
    assert accepted.acceptance.accepted
    assert not rejected.acceptance.accepted
    assert rejected.acceptance.block_ids == ("physical-branch",)
    assert not rejected.acceptance.blocks[0].admissible


def test_decoder_rejects_schema_shape_dtype_and_invalid_geometry():
    problem = phx.optim.StateDesignProblem(
        lambda state, design, args: state - design,
        lambda state, design, args: jnp.sum(state**2),
    )
    with pytest.raises(ValueError):
        _parameterization(problem, lambda latent: latent[:, None])
    with pytest.raises(TypeError):
        _parameterization(problem, lambda latent: latent.astype(jnp.int32))
    lowered = _parameterization(
        problem, design_admissibility=lambda design: jnp.all(design > 0.0)
    )
    with pytest.raises(ValueError):
        lowered.decode(jnp.asarray([[0.5]]))
    with pytest.raises((ValueError, eqx.EquinoxRuntimeError)):
        lowered.problem.solve_state(jnp.asarray([-1.0]), jnp.zeros((1,)))
    geometry = phx.geometry.design
    first_schema = geometry.ParameterSchema(
        (
            geometry.ParameterSpec(
                geometry.ParameterId("a", "height"),
                (1,),
                str(jnp.asarray([0.5]).dtype),
                "geometry",
            ),
        )
    )
    other_schema = geometry.ParameterSchema(
        (
            geometry.ParameterSpec(
                geometry.ParameterId("b", "height"),
                (1,),
                str(jnp.asarray([0.5]).dtype),
                "geometry",
            ),
        )
    )
    with pytest.raises(ValueError):
        _parameterization(
            problem,
            lambda latent: geometry.DesignState(other_schema, (latent,)),
            template=geometry.DesignState(first_schema, (jnp.asarray([0.5]),)),
        )


@pytest.mark.parametrize("failure", ("state", "transpose"))
def test_native_guided_score_rejects_failed_physical_certification(failure):
    physical = phx.optim.StateDesignProblem(
        (lambda state, design, args: state - design)
        if failure == "state"
        else (lambda state, design, args: jnp.zeros_like(state)),
        lambda state, design, args: jnp.sum(state),
        state_admissibility=(lambda state, design, args: jnp.asarray(False))
        if failure == "state"
        else None,
    )
    lowered = _parameterization(physical)
    guidance = MechanicsPotentialGuidance(
        lowered, jnp.zeros((1,)), linear_policy=_linear_policy()
    )
    domain = phx.domain.HyperRectangle(
        jnp.asarray([-3.0]), jnp.asarray([3.0]), label="x"
    ) @ phx.domain.TimeInterval(0.0, 1.0)
    base = StateTimeScoreField(
        domain.Function("x", "t")(lambda state, time: -state),
        state_label="x",
        time_label="t",
    )
    guided = phx.transport.GuidedScoreField(base, (guidance,))
    score, evaluations, valid = guided.evaluate(
        jnp.asarray([0.5]), 0.2, phx.transport.ScoreContext({})
    )
    assert not valid
    assert not evaluations[0].valid
    assert not jnp.all(jnp.isfinite(score))


def test_native_fem_density_fixed_cells_and_guidance_pullback():
    learned, initial, _, mask, fixed = build_density_case(2)
    parameterization = learned.parameterization
    latent = jnp.asarray([-0.2, 0.1])
    raw = parameterization.decode(latent)
    physical = learned.topology_problem.physical_density(raw)
    np.testing.assert_array_equal(raw[~mask], fixed[~mask])
    np.testing.assert_array_equal(physical[~mask], fixed[~mask])
    derivative = jax.grad(
        lambda z: jnp.sum(
            learned.topology_problem.physical_density(parameterization.decode(z))[~mask]
        )
    )(latent)
    np.testing.assert_array_equal(derivative, jnp.zeros_like(latent))
    guidance = MechanicsPotentialGuidance(
        parameterization,
        initial,
        scale=0.2,
        linear_policy=_linear_policy(),
        denoise=lambda value, time, context: value / (1.0 + time),
    )
    evaluation = guidance.evaluate(latent, 0.25, phx.transport.ScoreContext({}))
    response = parameterization.response_vjp(
        latent / 1.25, initial, linear_policy=_linear_policy()
    )
    assert response.accepted and evaluation.valid
    assert evaluation.exactness == "heuristic"
    np.testing.assert_allclose(
        evaluation.correction,
        -0.2 * response.latent_cotangent / 1.25,
        rtol=1e-6,
        atol=1e-9,
    )
    direction = jnp.asarray([0.3, -0.2])
    epsilon = 1.0e-4
    plus = parameterization.response_vjp(
        latent / 1.25 + epsilon * direction, initial, linear_policy=_linear_policy()
    )
    minus = parameterization.response_vjp(
        latent / 1.25 - epsilon * direction, initial, linear_policy=_linear_policy()
    )
    np.testing.assert_allclose(
        phx.ein.contract("i,i->", response.latent_cotangent, direction),
        (plus.values - minus.values) / (2 * epsilon),
        rtol=2e-4,
        atol=1e-8,
    )


def test_native_fem_shape_schema_and_geometry_gradient():
    parameterization, initial = build_shape_case(2)
    latent = jnp.asarray([0.1, -0.2])
    response = parameterization.response_vjp(
        latent, initial, linear_policy=_linear_policy()
    )
    assert response.accepted
    assert isinstance(response.physical_design, phx.geometry.design.DesignState)
    epsilon = 1.0e-4
    direction = jnp.asarray([0.4, -0.3])
    plus = parameterization.response_vjp(
        latent + epsilon * direction, initial, linear_policy=_linear_policy()
    )
    minus = parameterization.response_vjp(
        latent - epsilon * direction, initial, linear_policy=_linear_policy()
    )
    np.testing.assert_allclose(
        phx.ein.contract("i,i->", response.latent_cotangent, direction),
        (plus.values - minus.values) / (2 * epsilon),
        rtol=2e-4,
        atol=1e-8,
    )


def test_native_fem_latent_solve_requires_reference_volume_feasibility():
    learned, initial, plan, mask, fixed = build_density_case(2)
    source = learned.topology_problem
    sm = phx.applications.solid_mechanics
    strict_reference = sm.TopologyMechanicsProblem(
        source.state_residual,
        source.load_cases,
        source.density_transform,
        source.material_interpolation,
        0.3,
        source.state_solver,
        aggregation=source.aggregation,
        acceptance_policy=source.acceptance_policy,
        branch_evaluator=source.branch_evaluator,
        state_realization=source.state_realization,
        problem_id="stricter-reference-volume",
    )
    strict_plan = eqx.tree_at(lambda item: item.reference_problem, plan, strict_reference)
    result = solve_learned_topology_design(
        learned,
        initial,
        jnp.zeros((2,)),
        strict_plan,
        initial,
        method=phx.optim.ReducedMMA(linear_policy=_linear_policy()),
        termination=phx.optim.OptimizationTermination(
            maximum_steps=100,
            absolute_optimality=2e-5,
            relative_optimality=0.0,
        ),
    )
    assert result.latent_result.successful
    assert result.reanalysis.evidence.mechanics.accepted
    assert result.reanalysis.evidence.transfer.accepted
    assert result.reference_volume_ratio > 0.3
    assert not result.reference_feasible
    assert not result.accepted
    np.testing.assert_array_equal(
        result.topology_result.physical_density[~mask], fixed[~mask]
    )
