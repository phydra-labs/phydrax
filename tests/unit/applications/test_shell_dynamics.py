from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import pytest

import phydrax as phx
from phydrax.applications.solid_mechanics._shell_dynamics import (
    ShellDynamicsPlan,
    ShellMaterialParameters,
    ShellRejectionReason,
    TriangularShellPlan,
)
from phydrax.discretization.particle._rigid_contact import RigidContactGeometry


TRIANGLES = jnp.asarray(((0, 1, 2), (0, 2, 3)), dtype=jnp.int32)
SQUARE = jnp.asarray(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0)))


def _material() -> ShellMaterialParameters:
    return ShellMaterialParameters.isotropic(young_modulus=20.0, poisson_ratio=0.25)


def _prepared_square(*, thickness: float = 0.1, fixed_mask=None):
    plan = TriangularShellPlan(
        TRIANGLES,
        _material(),
        thickness=thickness,
        density=2.0,
        fixed_mask=fixed_mask,
    )
    return plan.prepare(SQUARE)


def _folded_square(angle: float = 0.45):
    axis = (SQUARE[2] - SQUARE[0]) / jnp.linalg.norm(SQUARE[2] - SQUARE[0])
    relative = SQUARE[3] - SQUARE[0]
    rotated = (
        jnp.cos(angle) * relative
        + jnp.sin(angle) * jnp.cross(axis, relative)
        + (1.0 - jnp.cos(angle)) * jnp.sum(axis * relative) * axis
    )
    return SQUARE.at[3].set(SQUARE[0] + rotated)


def test_flat_reference_has_zero_energy_and_finite_geometry_evidence():
    prepared = _prepared_square()
    evaluation = prepared.evaluate(SQUARE)

    assert evaluation.stored_energy == pytest.approx(0.0, abs=1.0e-12)
    assert evaluation.membrane_energy == pytest.approx(0.0, abs=1.0e-12)
    assert evaluation.bending_energy == pytest.approx(0.0, abs=1.0e-12)
    assert jnp.allclose(evaluation.forces, 0.0, atol=1.0e-10)
    assert jnp.allclose(evaluation.geometry.area_ratio, 1.0)
    assert jnp.allclose(evaluation.geometry.orientation_ratio, 1.0)
    assert bool(evaluation.geometry.valid)
    assert bool(evaluation.finite)


def test_membrane_energy_matches_constant_green_strain():
    material = _material()
    reference = jnp.asarray(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)))
    plan = TriangularShellPlan(
        jnp.asarray(((0, 1, 2),), dtype=jnp.int32),
        material,
        thickness=0.2,
        density=1.0,
    )
    prepared = plan.prepare(reference)
    deformed = reference.at[:, 0].multiply(1.1).at[:, 1].multiply(0.9)
    evaluation = prepared.evaluate(deformed)

    strain = jnp.asarray((0.5 * (1.1**2 - 1.0), 0.5 * (0.9**2 - 1.0), 0.0))
    expected = 0.5 * 0.5 * 0.2 * (strain @ material.membrane_matrix @ strain)
    assert evaluation.membrane_energy == pytest.approx(float(expected), rel=1.0e-10)
    assert evaluation.bending_energy == pytest.approx(0.0, abs=1.0e-12)


def test_energy_and_force_are_invariant_under_proper_rigid_motion():
    prepared = _prepared_square()
    positions = _folded_square()
    original = prepared.evaluate(positions)
    axis = jnp.asarray((1.0, 2.0, -1.0))
    axis = axis / jnp.linalg.norm(axis)
    angle = 0.73
    skew = jnp.asarray(
        ((0.0, -axis[2], axis[1]), (axis[2], 0.0, -axis[0]), (-axis[1], axis[0], 0.0))
    )
    rotation = (
        jnp.eye(3) * jnp.cos(angle)
        + (1.0 - jnp.cos(angle)) * axis[:, None] * axis[None, :]
        + jnp.sin(angle) * skew
    )
    transformed_positions = positions @ rotation.T + jnp.asarray((3.0, -2.0, 0.4))
    transformed = prepared.evaluate(transformed_positions)

    assert transformed.stored_energy == pytest.approx(
        float(original.stored_energy), rel=1.0e-10
    )
    assert jnp.allclose(
        transformed.forces, original.forces @ rotation.T, rtol=1.0e-8, atol=1.0e-9
    )
    assert jnp.allclose(
        transformed.geometry.area_ratio, original.geometry.area_ratio, rtol=1.0e-10
    )
    assert jnp.allclose(
        transformed.geometry.orientation_ratio,
        original.geometry.orientation_ratio,
        rtol=1.0e-8,
        atol=1.0e-9,
    )
    assert bool(transformed.valid)


def test_isometric_cylindrical_fold_has_only_hinge_bending_response():
    prepared = _prepared_square()
    folded = _folded_square()
    evaluation = prepared.evaluate(folded)

    assert evaluation.membrane_energy == pytest.approx(0.0, abs=1.0e-10)
    assert evaluation.bending_energy > 0.0
    assert jnp.linalg.norm(evaluation.forces) > 0.0
    assert bool(evaluation.valid)


def test_membrane_and_bending_have_linear_and_cubic_thickness_scaling():
    thin = _prepared_square(thickness=0.1)
    thick = _prepared_square(thickness=0.2)

    affine = SQUARE.at[:, 0].multiply(1.05)
    thin_membrane = thin.evaluate(affine).membrane_energy
    thick_membrane = thick.evaluate(affine).membrane_energy
    assert thick_membrane / thin_membrane == pytest.approx(2.0, rel=1.0e-10)

    folded = _folded_square()
    thin_bending = thin.evaluate(folded).bending_energy
    thick_bending = thick.evaluate(folded).bending_energy
    assert thick_bending / thin_bending == pytest.approx(8.0, rel=1.0e-10)


def test_fixed_nodes_remain_at_reference_during_damped_explicit_step():
    fixed = jnp.asarray((True, True, False, False))
    shell_plan = TriangularShellPlan(
        TRIANGLES,
        _material(),
        thickness=0.1,
        density=2.0,
        fixed_mask=fixed,
    )
    dynamics = ShellDynamicsPlan(
        shell_plan,
        damping=0.2,
        maximum_step_size=0.1,
        maximum_displacement_ratio=0.5,
    ).prepare(SQUARE)
    state = dynamics.initialize_state()
    load = jnp.tile(jnp.asarray((0.0, 0.0, 0.5)), (4, 1))
    step_size = 0.1 * dynamics.stable_step_size
    result = dynamics.step(state, jnp.asarray(0.0), step_size, load)

    assert bool(result.successful)
    assert jnp.array_equal(result.accepted_state.positions[fixed], SQUARE[fixed])
    assert jnp.array_equal(result.accepted_state.velocities[fixed], jnp.zeros((2, 3)))
    assert jnp.any(result.accepted_state.positions[~fixed, 2] > 0.0)
    assert result.evaluation.kinetic_energy > 0.0


def test_degenerate_reference_and_inconsistent_orientation_are_rejected():
    material = _material()
    one_triangle = jnp.asarray(((0, 1, 2),), dtype=jnp.int32)
    collinear = jnp.asarray(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0)))
    plan = TriangularShellPlan(one_triangle, material, thickness=0.1, density=1.0)
    with pytest.raises(ValueError, match="degenerate"):
        plan.prepare(collinear)

    same_edge_direction = jnp.asarray(((0, 1, 2), (0, 1, 3)), dtype=jnp.int32)
    with pytest.raises(ValueError, match="opposite interior-edge orientations"):
        TriangularShellPlan(same_edge_direction, material, thickness=0.1, density=1.0)


def test_inverted_triangle_is_reported_and_rolls_back_dynamics():
    shell_plan = TriangularShellPlan(TRIANGLES, _material(), thickness=0.1, density=2.0)
    prepared = shell_plan.prepare(SQUARE)
    inverted = SQUARE.at[3].set(jnp.asarray((1.0, 0.0, 0.0)))
    evaluation = prepared.evaluate(inverted)
    assert bool(evaluation.geometry.inverted[1])
    assert not bool(evaluation.valid)

    dynamics = ShellDynamicsPlan(shell_plan).prepare(SQUARE)
    state = dynamics.initialize_state(inverted)
    result = dynamics.step(state, jnp.asarray(0.0), 0.1 * dynamics.stable_step_size)
    assert not bool(result.successful)
    assert int(result.rejection_reasons) & int(ShellRejectionReason.INVALID_STATE)
    assert jnp.array_equal(result.accepted_state.positions, state.positions)
    assert jnp.array_equal(result.accepted_state.velocities, state.velocities)


def test_fixed_capacity_self_contact_payload_matches_hard_contact_geometry():
    triangles = jnp.asarray(((0, 1, 2), (3, 4, 5)), dtype=jnp.int32)
    positions = jnp.asarray(
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 0.2),
            (1.0, 0.0, 0.2),
            (0.0, 1.0, 0.2),
        )
    )
    plan = TriangularShellPlan(
        triangles,
        _material(),
        thickness=0.1,
        density=1.0,
        self_contact_pairs=jnp.asarray(((0, 1), (-1, -1)), dtype=jnp.int32),
    )
    prepared = plan.prepare(positions)
    geometry = prepared.self_contact_geometry(positions)
    collision_surface = prepared.collision_surface()
    scene = phx.discretization.PreparedCollisionScene((collision_surface,))
    epoch = phx.discretization.SweepAndPruneContactSearchPlan(
        edge_vertex_capacity=0,
        edge_edge_capacity=16,
        face_vertex_capacity=16,
        activation_distance=0.3,
    ).build(scene, scene.positions(jnp.zeros_like(positions)))

    assert isinstance(geometry, RigidContactGeometry)
    assert geometry.normal.shape == (2, 3)
    assert geometry.gap.shape == (2,)
    assert geometry.contact_keys.shape == (2,)
    assert jnp.array_equal(geometry.valid, jnp.asarray((True, False)))
    assert geometry.gap[0] == pytest.approx(0.1)
    assert geometry.normal[0, 2] == pytest.approx(-1.0)
    assert bool(geometry.successful)
    assert geometry.as_contact_batch().normal.shape == (2, 3)
    assert collision_surface.plan.face_count == 2
    assert collision_surface.plan.minimum_separation == pytest.approx(0.1)
    assert bool(epoch.successful)
    assert epoch.candidate_count > 0


def test_evaluation_and_step_are_jittable_and_unstable_step_rolls_back():
    shell_plan = TriangularShellPlan(TRIANGLES, _material(), thickness=0.1, density=2.0)
    prepared = shell_plan.prepare(SQUARE)
    compiled_evaluation = eqx.filter_jit(prepared.evaluate)(
        SQUARE, jnp.zeros_like(SQUARE)
    )
    assert compiled_evaluation.stored_energy == pytest.approx(0.0, abs=1.0e-12)
    assert bool(compiled_evaluation.valid)

    dynamics = ShellDynamicsPlan(shell_plan, maximum_step_size=0.1).prepare(SQUARE)
    state = dynamics.initialize_state()
    compiled_result = eqx.filter_jit(
        lambda value: dynamics.step(
            value,
            jnp.asarray(0.0),
            0.1 * dynamics.stable_step_size,
        )
    )(state)
    assert bool(compiled_result.successful)

    rejected = dynamics.step(
        state,
        jnp.asarray(0.0),
        2.0 * dynamics.stable_step_size,
    )
    assert not bool(rejected.successful)
    assert int(rejected.rejection_reasons) & int(ShellRejectionReason.UNSTABLE_STEP)
    assert jnp.array_equal(rejected.accepted_state.positions, state.positions)
    assert jnp.array_equal(rejected.accepted_state.velocities, state.velocities)
