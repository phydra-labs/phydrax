from __future__ import annotations

import jax.numpy as jnp
import pytest

from phydrax.applications.solid_mechanics._rod_dynamics import (
    prepare_rod,
    RodPlan,
)
from phydrax.applications.solid_mechanics._rod_reduced_basis import (
    prepare_rod_strain_basis,
    RodStrainBasisPlan,
)
from phydrax.applications.solid_mechanics._rod_reduced_kinematics import (
    lift_configuration,
    lift_effort_pullback_operator,
    lift_velocity_operator,
)
from phydrax.applications.solid_mechanics._rod_reduction import (
    evaluate_reduced_rod,
    prepare_reduced_rod,
    ReducedRodPlan,
    ReducedRodState,
)


def _spatial_rod(*, scale: float = 1.0):
    dtype = jnp.float32
    return prepare_rod(
        RodPlan(
            jnp.asarray(((0, 1), (1, 2)), dtype=jnp.int32),
            jnp.asarray(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0)), dtype=dtype),
            jnp.broadcast_to(jnp.eye(3, dtype=dtype), (2, 3, 3)),
            jnp.asarray((1.0, 1.2, 0.9), dtype=dtype),
            jnp.broadcast_to(jnp.eye(3, dtype=dtype), (2, 3, 3)),
            scale
            * jnp.broadcast_to(
                jnp.diag(jnp.asarray((80.0, 50.0, 40.0), dtype=dtype)), (2, 3, 3)
            ),
            scale
            * jnp.broadcast_to(
                jnp.diag(jnp.asarray((7.0, 8.0, 9.0), dtype=dtype)), (1, 3, 3)
            ),
        )
    )


def _planar_rod():
    dtype = jnp.float32
    return prepare_rod(
        RodPlan(
            jnp.asarray(((0, 1), (1, 2)), dtype=jnp.int32),
            jnp.asarray(((0.0, 0.0), (1.0, 0.0), (2.0, 0.0)), dtype=dtype),
            jnp.broadcast_to(jnp.eye(2, dtype=dtype), (2, 2, 2)),
            jnp.asarray((1.0, 1.2, 0.9), dtype=dtype),
            jnp.asarray((0.2, 0.3), dtype=dtype),
            jnp.broadcast_to(jnp.diag(jnp.asarray((80.0, 40.0), dtype=dtype)), (2, 2, 2)),
            jnp.asarray((((6.0,),),), dtype=dtype),
        )
    )


def _spatial_basis(*, label: str | None = None):
    return RodStrainBasisPlan.shifted_legendre(
        0,
        dimension=3,
        component_scales=jnp.asarray((0.2, 0.3, 0.4, 0.5, 0.6, 0.7), dtype=jnp.float32),
        label=label,
    )


def _spatial_reduction(*, rod=None, plan=None):
    source = _spatial_rod() if rod is None else rod
    reduction_plan = ReducedRodPlan(_spatial_basis()) if plan is None else plan
    return prepare_reduced_rod(source, reduction_plan)


def test_basis_constructors_use_canonical_physical_component_scaling():
    scales = jnp.asarray((2.0, 3.0, 4.0, 5.0, 6.0, 7.0), dtype=jnp.float32)
    pcs = RodStrainBasisPlan.piecewise_constant(
        jnp.asarray((0.0, 0.5, 1.0), dtype=jnp.float32),
        dimension=3,
        components=("nu_y", "kappa_z"),
        component_scales=scales,
    )
    left = pcs.evaluate_normalized(jnp.asarray(0.25, dtype=jnp.float32))
    right = pcs.evaluate_normalized(jnp.asarray(0.75, dtype=jnp.float32))

    assert pcs.coordinate_count == 4
    assert left.shape == right.shape == (6, 4)
    assert left[1, 0] == pytest.approx(3.0)
    assert right[1, 1] == pytest.approx(3.0)
    assert left[5, 2] == pytest.approx(7.0)
    assert right[5, 3] == pytest.approx(7.0)
    assert jnp.count_nonzero(left) == 2
    assert jnp.count_nonzero(right) == 2

    gvs = RodStrainBasisPlan.shifted_legendre(
        (2,),
        dimension=3,
        components=("nu_x",),
        component_scales=jnp.ones((6,), dtype=jnp.float32),
    )
    endpoint_values = gvs.evaluate_normalized(jnp.asarray((0.0, 1.0), dtype=jnp.float32))
    assert jnp.allclose(endpoint_values[0, 0], jnp.asarray((1.0, -1.0, 1.0)))
    assert jnp.allclose(endpoint_values[1, 0], jnp.asarray((1.0, 1.0, 1.0)))

    coefficients = jnp.zeros((1, 6, 2, 2), dtype=jnp.float32)
    coefficients = coefficients.at[0, 0, 0, 0].set(1.0)
    coefficients = coefficients.at[0, 5, 1, 1].set(1.0)
    explicit = RodStrainBasisPlan.explicit(
        jnp.asarray((0.0, 1.0), dtype=jnp.float32),
        coefficients,
        dimension=3,
        components=("nu_x", "kappa_z"),
        component_scales=scales,
    )
    value = explicit.evaluate_normalized(jnp.asarray(0.25, dtype=jnp.float32))
    assert value[0, 0] == pytest.approx(2.0)
    assert value[5, 1] == pytest.approx(1.75)


def test_prepared_basis_has_physical_worksets_rank_condition_and_dtype_evidence():
    rod = _spatial_rod()
    basis = RodStrainBasisPlan.piecewise_constant(
        jnp.asarray((0.0, 0.5, 1.0), dtype=jnp.float32),
        dimension=3,
        components=("nu_x",),
        component_scales=jnp.ones((6,), dtype=jnp.float32),
    )
    prepared = prepare_rod_strain_basis(basis, rod)

    assert jnp.array_equal(prepared.breakpoints, jnp.asarray((0.0, 1.0, 2.0)))
    assert jnp.array_equal(prepared.stretch_arc_lengths, jnp.asarray((0.5, 1.5)))
    assert jnp.array_equal(prepared.bend_arc_lengths, jnp.asarray((1.0,)))
    assert jnp.array_equal(prepared.stretch_interval_ids, jnp.asarray((0, 1)))
    assert jnp.sum(prepared.quadrature_weights) == pytest.approx(2.0)
    assert prepared.method == "piecewise_constant"
    assert prepared.domain_start == pytest.approx(0.0)
    assert prepared.domain_end == pytest.approx(2.0)
    assert prepared.evidence.numerical_rank == 2
    assert prepared.evidence.full_column_rank
    assert prepared.evidence.condition_valid
    assert prepared.evidence.dtype_retained
    assert prepared.evidence.valid


def test_weighted_rank_and_condition_failures_reject_preparation():
    rod = _spatial_rod()
    duplicate = jnp.zeros((1, 6, 2, 1), dtype=jnp.float32)
    duplicate = duplicate.at[0, 0, :, 0].set(1.0)
    duplicate_plan = RodStrainBasisPlan.explicit(
        jnp.asarray((0.0, 1.0), dtype=jnp.float32),
        duplicate,
        dimension=3,
        components=("nu_x",),
    )
    with pytest.raises(ValueError, match="full column rank"):
        prepare_rod_strain_basis(duplicate_plan, rod)

    ill_conditioned = jnp.zeros((1, 6, 2, 1), dtype=jnp.float32)
    ill_conditioned = ill_conditioned.at[0, 0, 0, 0].set(1.0)
    ill_conditioned = ill_conditioned.at[0, 0, 1, 0].set(1.0)
    ill_conditioned = ill_conditioned.at[0, 1, 1, 0].set(1.0e-4)
    ill_conditioned_plan = RodStrainBasisPlan.explicit(
        jnp.asarray((0.0, 1.0), dtype=jnp.float32),
        ill_conditioned,
        dimension=3,
        components=("nu_x", "nu_y"),
        rank_tolerance=1.0e-9,
        maximum_condition_number=1.0e3,
    )
    with pytest.raises(ValueError, match="condition number"):
        prepare_rod_strain_basis(ill_conditioned_plan, rod)


def test_plan_and_prepared_ids_bind_content_not_display_labels():
    first_basis = _spatial_basis(label="first")
    renamed_basis = _spatial_basis(label="renamed")
    changed_basis = RodStrainBasisPlan.shifted_legendre(
        0,
        dimension=3,
        component_scales=jnp.asarray((0.21, 0.3, 0.4, 0.5, 0.6, 0.7), dtype=jnp.float32),
    )
    first_plan = ReducedRodPlan(first_basis, label="first")
    renamed_plan = ReducedRodPlan(renamed_basis, label="renamed")
    changed_reference = ReducedRodPlan(
        first_basis,
        reference_coefficients=jnp.asarray(
            (0.1, 0.0, 0.0, 0.0, 0.0, 0.0), dtype=jnp.float32
        ),
    )
    fixed = ReducedRodPlan(
        first_basis,
        base_policy="fixed",
        fixed_base_position=jnp.asarray((1.0, 2.0, 3.0), dtype=jnp.float32),
        fixed_base_orientation=jnp.asarray((1.0, 0.0, 0.0, 0.0), dtype=jnp.float32),
    )
    fixed_same_pose_sign = ReducedRodPlan(
        first_basis,
        base_policy="fixed",
        fixed_base_position=jnp.asarray((1.0, 2.0, 3.0), dtype=jnp.float32),
        fixed_base_orientation=jnp.asarray((-1.0, 0.0, 0.0, 0.0), dtype=jnp.float32),
    )

    assert first_basis.plan_id == renamed_basis.plan_id
    assert first_plan.plan_id == renamed_plan.plan_id
    assert changed_basis.plan_id != first_basis.plan_id
    assert changed_reference.plan_id != first_plan.plan_id
    assert fixed.plan_id == fixed_same_pose_sign.plan_id
    assert fixed.plan_id != first_plan.plan_id

    rod = _spatial_rod()
    same_rod = _spatial_rod()
    changed_rod = _spatial_rod(scale=2.0)
    first = prepare_reduced_rod(rod, first_plan)
    repeated = prepare_reduced_rod(same_rod, renamed_plan)
    assert first.prepared_id == repeated.prepared_id
    assert prepare_reduced_rod(changed_rod, first_plan).prepared_id != first.prepared_id


def test_only_explicit_fixed_or_native_reference_base_semantics_are_accepted():
    basis = _spatial_basis()
    with pytest.raises(ValueError, match="floating rods are unsupported"):
        ReducedRodPlan(basis, base_policy="floating")
    with pytest.raises(ValueError, match="requires both"):
        ReducedRodPlan(
            basis,
            base_policy="fixed",
            fixed_base_position=jnp.zeros((3,), dtype=jnp.float32),
        )
    with pytest.raises(ValueError, match="forbidden"):
        ReducedRodPlan(
            basis,
            base_policy="reference",
            fixed_base_position=jnp.zeros((3,), dtype=jnp.float32),
            fixed_base_orientation=jnp.asarray((1.0, 0.0, 0.0, 0.0), dtype=jnp.float32),
        )

    fixed_plan = ReducedRodPlan(
        basis,
        base_policy="fixed",
        fixed_base_position=jnp.asarray((1.0, -2.0, 0.5), dtype=jnp.float32),
        fixed_base_orientation=jnp.asarray((0.0, 0.0, 0.0, 1.0), dtype=jnp.float32),
    )
    fixed = prepare_reduced_rod(_spatial_rod(), fixed_plan)
    native = fixed.lift(fixed.rest_state())
    assert jnp.allclose(native.positions[0], fixed.base_position, atol=2.0e-6)
    assert jnp.allclose(native.orientations[0], fixed.base_orientation, atol=2.0e-6)
    assert jnp.allclose(native.velocities[0], 0.0, atol=2.0e-6)
    assert jnp.allclose(native.angular_velocities[0], 0.0, atol=2.0e-6)
    assert fixed.evaluate(fixed.rest_state()).lift_evidence.valid


@pytest.mark.parametrize("coordinate", range(6))
def test_pure_spatial_extension_shear_bend_and_twist_reconstruct_at_native_sites(
    coordinate,
):
    reduction = _spatial_reduction()
    coefficients = jnp.zeros((6,), dtype=jnp.float32).at[coordinate].set(0.2)
    state = ReducedRodState(coefficients, jnp.zeros_like(coefficients))
    evaluation = evaluate_reduced_rod(reduction, state)

    expected_stretch = jnp.einsum(
        "sdk,k->sd", reduction.stretch_shear_basis, coefficients
    )
    expected_bend = jnp.einsum("sdk,k->sd", reduction.bend_twist_basis, coefficients)
    assert jnp.allclose(
        evaluation.native_evaluation.stretch_shear_strain,
        expected_stretch,
        rtol=2.0e-5,
        atol=2.0e-6,
    )
    assert jnp.allclose(
        evaluation.native_evaluation.bend_twist_strain,
        expected_bend,
        rtol=2.0e-5,
        atol=2.0e-6,
    )
    assert jnp.allclose(
        jnp.linalg.norm(evaluation.native_state.orientations, axis=-1), 1.0, atol=2.0e-6
    )
    assert evaluation.native_evaluation.chart_valid
    assert evaluation.strain_evidence.valid


def test_mixed_spatial_strain_reconstructs_and_preserves_quaternion_charts():
    reduction = _spatial_reduction()
    coefficients = jnp.asarray((0.12, -0.08, 0.05, 0.11, -0.09, 0.07), dtype=jnp.float32)
    state = ReducedRodState(
        coefficients, jnp.asarray((-0.3, 0.2, -0.1, 0.15, -0.12, 0.09), dtype=jnp.float32)
    )
    evaluation = reduction.evaluate(state)

    assert evaluation.strain_evidence.maximum_stretch_shear_error <= 2.0e-6
    assert evaluation.strain_evidence.maximum_bend_twist_error <= 2.0e-6
    assert evaluation.native_evaluation.chart_valid
    assert evaluation.native_evaluation.orientation_valid
    assert evaluation.native_discrete_energy_valid
    assert evaluation.valid


def test_velocity_jvp_and_effort_vjp_use_native_spaces_and_preserve_power():
    reduction = _spatial_reduction()
    coefficients = jnp.asarray((0.08, -0.05, 0.04, 0.06, -0.03, 0.02), dtype=jnp.float32)
    rates = jnp.asarray((-0.13, 0.17, -0.11, 0.19, -0.07, 0.05), dtype=jnp.float32)
    velocity_operator = lift_velocity_operator(reduction, coefficients)
    pullback_operator = lift_effort_pullback_operator(reduction, coefficients)
    forces = jnp.asarray(
        ((0.4, -0.2, 0.7), (0.3, 0.6, -0.1), (-0.5, 0.2, 0.8)), dtype=jnp.float32
    )
    moments = jnp.asarray(((0.1, -0.3, 0.2), (0.5, 0.4, -0.2)), dtype=jnp.float32)
    effort = reduction.rod.effort_from_load(forces, moments)
    velocity = velocity_operator.mv(rates)
    generalized_effort = pullback_operator.mv(effort)

    assert velocity_operator.source.compatible(reduction.coefficient_space)
    assert velocity_operator.target.compatible(reduction.rod.velocity_space)
    assert pullback_operator.source.compatible(reduction.rod.effort_space)
    assert pullback_operator.target.compatible(reduction.reduced_effort_space)
    native_power = reduction.rod.effort_space.pair(effort, velocity)
    reduced_power = reduction.reduced_effort_space.pair(generalized_effort, rates)
    assert reduced_power == pytest.approx(native_power, rel=2.0e-6, abs=2.0e-6)


def test_planar_reduction_uses_the_same_basis_and_fixed_base_api():
    rod = _planar_rod()
    basis = RodStrainBasisPlan.shifted_legendre(
        0,
        dimension=2,
        component_scales=jnp.asarray((0.3, 0.2, 1.0, 1.0, 1.0, 0.4), dtype=jnp.float32),
    )
    reduction = prepare_reduced_rod(rod, ReducedRodPlan(basis))
    coefficients = jnp.asarray((0.15, -0.1, 0.12), dtype=jnp.float32)
    positions, orientations = lift_configuration(reduction, coefficients)
    evaluation = reduction.evaluate(
        ReducedRodState(coefficients, jnp.zeros_like(coefficients))
    )

    assert positions.shape == (3, 2)
    assert orientations.shape == (2,)
    assert evaluation.native_evaluation.stretch_shear_strain.shape == (2, 2)
    assert evaluation.native_evaluation.bend_twist_strain.shape == (1, 1)
    assert evaluation.strain_evidence.valid
    assert evaluation.valid
