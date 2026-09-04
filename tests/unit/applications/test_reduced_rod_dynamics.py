from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from phydrax.applications.solid_mechanics._rod_dynamics import prepare_rod, RodPlan
from phydrax.applications.solid_mechanics._rod_loads import RodLoad, RodLoadLedger
from phydrax.applications.solid_mechanics._rod_materials import (
    KelvinVoigtRodMaterialPlan,
    RodConstitutiveControl,
)
from phydrax.applications.solid_mechanics._rod_reduced_basis import (
    RodStrainBasisPlan,
)
from phydrax.applications.solid_mechanics._rod_reduced_dynamics import (
    prepare_reduced_rod_dynamics,
    ReducedRodDenseCholeskyPlan,
    ReducedRodDirectLoad,
    ReducedRodMaterialControl,
    ReducedRodMatrixFreeCGPlan,
)
from phydrax.applications.solid_mechanics._rod_reduced_kinematics import (
    lift_effort_pullback_operator,
)
from phydrax.applications.solid_mechanics._rod_reduction import (
    prepare_reduced_rod,
    ReducedRodPlan,
    ReducedRodState,
)
from phydrax.linalg import DenseLinearOperator, FunctionLinearOperator


def _spatial_reduction():
    dtype = jnp.float32
    rod = prepare_rod(
        RodPlan(
            jnp.asarray(((0, 1), (1, 2)), dtype=jnp.int32),
            jnp.asarray(
                ((0.0, 0.0, 0.0), (0.8, 0.0, 0.0), (1.9, 0.0, 0.0)),
                dtype=dtype,
            ),
            jnp.broadcast_to(jnp.eye(3, dtype=dtype), (2, 3, 3)),
            jnp.asarray((0.7, 1.3, 2.1), dtype=dtype),
            jnp.asarray(
                (
                    ((0.31, 0.02, 0.0), (0.02, 0.47, 0.01), (0.0, 0.01, 0.62)),
                    ((0.54, 0.01, 0.02), (0.01, 0.78, 0.03), (0.02, 0.03, 0.95)),
                ),
                dtype=dtype,
            ),
            jnp.asarray(
                (
                    ((90.0, 4.0, 0.0), (4.0, 55.0, 1.0), (0.0, 1.0, 43.0)),
                    ((115.0, 3.0, 2.0), (3.0, 72.0, 1.0), (2.0, 1.0, 61.0)),
                ),
                dtype=dtype,
            ),
            jnp.asarray(
                (((13.0, 0.5, 0.0), (0.5, 17.0, 0.3), (0.0, 0.3, 21.0)),),
                dtype=dtype,
            ),
        )
    )
    basis = RodStrainBasisPlan.shifted_legendre(
        0,
        dimension=3,
        component_scales=jnp.asarray((0.18, 0.23, 0.29, 0.34, 0.41, 0.47), dtype=dtype),
    )
    return prepare_reduced_rod(rod, ReducedRodPlan(basis))


def _kelvin_voigt_dynamics(*, gravity=None, plan=None):
    reduction = _spatial_reduction()
    dtype = reduction.rod.plan.rest_positions.dtype
    stretch_viscosity = jnp.broadcast_to(
        jnp.diag(jnp.asarray((0.9, 0.7, 0.5), dtype=dtype)),
        reduction.rod.plan.stretch_shear_stiffness.shape,
    )
    bend_viscosity = jnp.broadcast_to(
        jnp.diag(jnp.asarray((0.3, 0.4, 0.6), dtype=dtype)),
        reduction.rod.plan.bend_twist_stiffness.shape,
    )
    stretch = KelvinVoigtRodMaterialPlan(
        reduction.rod.plan.stretch_shear_stiffness,
        stretch_viscosity,
    ).prepare(reduction.rod.stretch_shear_workset)
    bend = KelvinVoigtRodMaterialPlan(
        reduction.rod.plan.bend_twist_stiffness,
        bend_viscosity,
    ).prepare(reduction.rod.bend_twist_workset)
    dynamics = prepare_reduced_rod_dynamics(
        reduction,
        ReducedRodDenseCholeskyPlan() if plan is None else plan,
        stretch_shear_material=stretch,
        bend_twist_material=bend,
        gravity=gravity,
    )
    return reduction, dynamics


def test_typed_intrinsic_strain_control_flows_through_material_force_ledger():
    reduction = _spatial_reduction()
    dynamics = prepare_reduced_rod_dynamics(reduction)
    passive = dynamics.initialize_material_control()
    intrinsic = (
        jnp.zeros_like(passive.stretch_shear_control.intrinsic_strain).at[:, 1].set(0.03)
    )
    controlled_stretch = RodConstitutiveControl(
        intrinsic,
        jnp.zeros_like(intrinsic),
        passive.stretch_shear_control.stiffness,
        jnp.zeros_like(passive.stretch_shear_control.stiffness),
        workset_id=reduction.rod.stretch_shear_workset.workset_id,
        material_id=passive.stretch_shear_control.material_id,
        control_id="test-controlled-stretch",
        intrinsic_owner_id="test-intrinsic-actuator",
    )
    control = ReducedRodMaterialControl(controlled_stretch, passive.bend_twist_control)
    zeros = jnp.zeros((6,), dtype=jnp.float32)
    evaluation = dynamics.evaluate(
        ReducedRodState(zeros, zeros),
        material_control=control,
        step_size=jnp.asarray(0.1, dtype=jnp.float32),
    )
    resultants = jnp.einsum(
        "sij,sj->si",
        reduction.rod.plan.stretch_shear_stiffness,
        -intrinsic,
    )
    expected = -jnp.einsum(
        "sdk,sd,s->k",
        reduction.stretch_shear_basis,
        resultants,
        reduction.rod.stretch_shear_measures,
    )

    assert jnp.linalg.norm(expected) > 0.0
    assert jnp.allclose(
        evaluation.forces.elastic_effort, expected, rtol=3.0e-5, atol=3.0e-6
    )
    assert jnp.allclose(evaluation.forces.effort_for_source("elastic"), expected)
    assert (
        evaluation.stretch_shear_material_result.control_id == "test-controlled-stretch"
    )


@pytest.mark.parametrize("coordinate", range(6))
def test_pure_extension_shear_bend_and_twist_use_native_material_quadrature(
    coordinate,
):
    reduction = _spatial_reduction()
    dynamics = prepare_reduced_rod_dynamics(reduction)
    coefficients = jnp.zeros((6,), dtype=jnp.float32).at[coordinate].set(0.08)
    state = ReducedRodState(coefficients, jnp.zeros_like(coefficients))
    evaluation = dynamics.evaluate(state, step_size=jnp.asarray(0.1, dtype=jnp.float32))

    stretch_increment = jnp.einsum(
        "sdk,k->sd", reduction.stretch_shear_basis, coefficients
    )
    bend_increment = jnp.einsum("sdk,k->sd", reduction.bend_twist_basis, coefficients)
    stretch_resultants = jnp.einsum(
        "sij,sj->si",
        reduction.rod.plan.stretch_shear_stiffness,
        stretch_increment,
    )
    bend_resultants = jnp.einsum(
        "sij,sj->si",
        reduction.rod.plan.bend_twist_stiffness,
        bend_increment,
    )
    expected = -jnp.einsum(
        "sdk,sd,s->k",
        reduction.stretch_shear_basis,
        stretch_resultants,
        reduction.rod.stretch_shear_measures,
    ) - jnp.einsum(
        "sdk,sd,s->k",
        reduction.bend_twist_basis,
        bend_resultants,
        reduction.rod.bend_twist_measures,
    )

    assert jnp.allclose(
        evaluation.forces.elastic_effort, expected, rtol=3.0e-5, atol=3.0e-6
    )
    assert jnp.allclose(evaluation.forces.kelvin_voigt_effort, 0.0, atol=2.0e-7)
    assert evaluation.stretch_shear_material_result.evidence.valid
    assert evaluation.bend_twist_material_result.evidence.valid


def test_dense_mass_maps_tangents_to_true_duals_and_inverse_roundtrips():
    reduction = _spatial_reduction()
    dynamics = prepare_reduced_rod_dynamics(reduction)
    coefficients = jnp.asarray(
        (0.03, -0.02, 0.01, 0.025, -0.015, 0.02), dtype=jnp.float32
    )
    first = jnp.asarray((0.4, -0.2, 0.3, 0.1, -0.5, 0.25), dtype=jnp.float32)
    second = jnp.asarray((-0.1, 0.35, 0.15, -0.3, 0.2, 0.45), dtype=jnp.float32)
    mass = dynamics.mass(coefficients)

    assert isinstance(mass.operator, DenseLinearOperator)
    assert mass.operator.source.compatible(reduction.coefficient_space)
    assert mass.operator.target.compatible(reduction.reduced_effort_space)
    assert not mass.operator.source.compatible(mass.operator.target)
    left = reduction.reduced_effort_space.pair(mass.operator.mv(first), second)
    right = reduction.reduced_effort_space.pair(mass.operator.mv(second), first)
    quadratic = reduction.reduced_effort_space.pair(mass.operator.mv(first), first)
    assert left == pytest.approx(right, rel=3.0e-5, abs=3.0e-6)
    assert quadratic > 0.0
    assert mass.evidence.symmetric
    assert mass.evidence.positive_definite
    assert mass.evidence.pivot_checked
    assert mass.evidence.pivot_valid
    assert mass.evidence.valid

    effort = jnp.asarray((1.2, -0.7, 0.9, -0.4, 0.6, 0.3), dtype=jnp.float32)
    inverse = dynamics.inverse_mass(coefficients, effort)
    assert inverse.inverse_mass_operator.source.compatible(reduction.reduced_effort_space)
    assert inverse.inverse_mass_operator.target.compatible(reduction.coefficient_space)
    assert jnp.allclose(
        mass.operator.mv(inverse.acceleration), effort, rtol=2.0e-5, atol=2.0e-6
    )
    assert inverse.solve_evidence.roundtrip_valid
    assert inverse.solve_evidence.valid


def test_fused_actions_match_dense_ad_authority_and_forward_inverse_roundtrip():
    reduction, dynamics = _kelvin_voigt_dynamics()
    state = ReducedRodState(
        jnp.asarray((0.04, -0.03, 0.02, 0.025, -0.015, 0.01), dtype=jnp.float32),
        jnp.asarray((-0.12, 0.09, -0.07, 0.05, -0.04, 0.08), dtype=jnp.float32),
    )
    step = jnp.asarray(0.05, dtype=jnp.float32)
    production = dynamics.evaluate(state, step_size=step)
    reference = dynamics.dense_reference(state, step_size=step)

    assert jnp.allclose(
        production.mass.operator.matrix, reference.mass_matrix, rtol=5.0e-5, atol=6.0e-6
    )
    assert jnp.allclose(
        production.bias.effort, reference.bias_effort, rtol=8.0e-5, atol=8.0e-6
    )
    assert jnp.allclose(
        production.forces.elastic_effort,
        reference.elastic_effort,
        rtol=5.0e-5,
        atol=6.0e-6,
    )
    assert jnp.allclose(
        production.forces.kelvin_voigt_effort,
        reference.kelvin_voigt_effort,
        rtol=5.0e-5,
        atol=6.0e-6,
    )
    assert production.energy.kinetic_energy == pytest.approx(
        reference.kinetic_energy, rel=4.0e-5, abs=4.0e-6
    )
    assert production.energy.stored_energy == pytest.approx(
        reference.stored_energy, rel=4.0e-5, abs=4.0e-6
    )
    assert reference.finite

    direct = ReducedRodDirectLoad(
        jnp.asarray((0.5, -0.1, 0.2, 0.05, -0.08, 0.12), dtype=jnp.float32),
        source_id="command",
        power_channel="actuation",
    )
    forward = dynamics.forward_dynamics(
        state,
        step_size=step,
        direct_reduced_loads=(direct,),
    )
    inverse = dynamics.inverse_dynamics(
        state,
        forward.acceleration,
        step_size=step,
        direct_reduced_loads=(direct,),
    )
    assert jnp.allclose(inverse.required_effort, 0.0, atol=2.0e-5)
    assert jnp.allclose(inverse.residual, 0.0, atol=2.0e-5)
    assert forward.solve_evidence.valid
    assert forward.valid
    assert inverse.valid


def test_gravity_native_and_direct_load_ledgers_preserve_effort_and_power():
    gravity_vector = jnp.asarray((0.0, -9.81, 0.4), dtype=jnp.float32)
    reduction, dynamics = _kelvin_voigt_dynamics(gravity=gravity_vector)
    state = ReducedRodState(
        jnp.asarray((0.02, -0.01, 0.015, 0.01, -0.008, 0.012), dtype=jnp.float32),
        jnp.asarray((0.16, -0.12, 0.07, -0.04, 0.09, 0.13), dtype=jnp.float32),
    )
    forces = (
        jnp.zeros_like(reduction.rod.plan.rest_positions)
        .at[2]
        .set(jnp.asarray((1.4, -0.6, 0.8), dtype=jnp.float32))
    )
    moments = jnp.asarray(((0.1, -0.2, 0.3), (-0.15, 0.25, 0.05)), dtype=jnp.float32)
    native = RodLoad(
        forces,
        moments,
        source_id="tip_wrench",
        power_channel="environment",
    )
    direct_value = jnp.asarray((0.3, -0.2, 0.1, 0.05, -0.04, 0.08), dtype=jnp.float32)
    direct = ReducedRodDirectLoad(
        direct_value,
        source_id="motor",
        power_channel="actuation",
    )
    evaluation = dynamics.evaluate(
        state,
        step_size=jnp.asarray(0.04, dtype=jnp.float32),
        native_loads=RodLoadLedger((native,)),
        direct_reduced_loads=(direct,),
    )
    ledger = evaluation.forces
    pullback = lift_effort_pullback_operator(reduction, state.coefficients)
    expected_native = pullback.mv(reduction.rod.effort_from_load(forces, moments))
    gravity_forces = reduction.rod.node_masses[:, None] * gravity_vector[None, :]
    expected_gravity = pullback.mv(
        reduction.rod.effort_from_load(gravity_forces, jnp.zeros_like(moments))
    )

    assert ledger.source_ids == (
        "elastic",
        "kelvin_voigt",
        "gravity",
        "tip_wrench",
        "motor",
    )
    assert ledger.source_channels == (
        "elastic",
        "kelvin_voigt",
        "gravity",
        "environment",
        "actuation",
    )
    assert jnp.allclose(ledger.effort_for_source("gravity"), expected_gravity)
    assert jnp.allclose(ledger.effort_for_source("tip_wrench"), expected_native)
    assert jnp.allclose(ledger.effort_for_source("motor"), direct_value)
    assert jnp.allclose(ledger.native_external_effort, expected_native)
    assert jnp.allclose(ledger.direct_reduced_effort, direct_value)
    expected_external_power = reduction.reduced_effort_space.pair(
        expected_native, state.coefficient_velocities
    )
    assert ledger.power_for_source("tip_wrench") == pytest.approx(
        expected_external_power, rel=3.0e-6, abs=3.0e-6
    )
    assert ledger.total_power == pytest.approx(
        ledger.paired_power, rel=3.0e-6, abs=3.0e-6
    )
    assert jnp.allclose(ledger.effort_for_channel("actuation"), direct_value, atol=2.0e-6)
    assert ledger.power_valid
    assert ledger.valid


def test_matrix_free_policy_uses_only_fused_actions_and_records_fixed_work(monkeypatch):
    plan = ReducedRodMatrixFreeCGPlan(
        relative_tolerance=1.0e-5,
        absolute_tolerance=1.0e-7,
        maximum_iterations=24,
        spectral_iterations=4,
        condition_limit=1.0e10,
    )
    reduction, dynamics = _kelvin_voigt_dynamics(plan=plan)
    state = ReducedRodState(
        jnp.asarray((0.025, -0.02, 0.01, 0.015, -0.012, 0.008), dtype=jnp.float32),
        jnp.asarray((-0.08, 0.06, -0.04, 0.03, -0.02, 0.05), dtype=jnp.float32),
    )

    def forbidden(*args, **kwargs):
        raise AssertionError(
            "production reduced dynamics materialized a global derivative"
        )

    monkeypatch.setattr(jax, "jacobian", forbidden)
    monkeypatch.setattr(jax, "hessian", forbidden)
    evaluation = dynamics.evaluate(state, step_size=jnp.asarray(0.1, dtype=jnp.float32))
    inverse = dynamics.inverse_mass(
        state.coefficients,
        jnp.asarray((0.7, -0.3, 0.4, -0.2, 0.1, 0.5), dtype=jnp.float32),
    )

    assert isinstance(evaluation.mass.operator, FunctionLinearOperator)
    assert evaluation.mass.operator.source.compatible(reduction.coefficient_space)
    assert evaluation.mass.operator.target.compatible(reduction.reduced_effort_space)
    assert evaluation.mass.evidence.solver == "matrix_free_cg"
    assert evaluation.mass.evidence.spectral_iterations == 4
    assert not evaluation.mass.evidence.pivot_checked
    assert inverse.solve_evidence.solver == "matrix_free_cg"
    assert inverse.solve_evidence.iterations <= plan.maximum_iterations
    assert inverse.solve_evidence.roundtrip_valid


def test_mass_evidence_fails_closed_for_condition_pivot_and_nonfinite_inputs():
    reduction = _spatial_reduction()
    condition_plan = ReducedRodDenseCholeskyPlan(
        condition_limit=1.000001,
        pivot_tolerance=1.0e6,
    )
    dynamics = prepare_reduced_rod_dynamics(reduction, condition_plan)
    coefficients = jnp.zeros((6,), dtype=jnp.float32)
    evidence = dynamics.mass(coefficients).evidence

    assert not evidence.conditioned
    assert not evidence.pivot_valid
    assert not evidence.valid

    nonfinite = coefficients.at[0].set(jnp.nan)
    nonfinite_evidence = dynamics.mass(nonfinite).evidence
    assert not nonfinite_evidence.finite
    assert not nonfinite_evidence.valid
