#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx


def _quad_plan(nx=3, ny=2):
    vertices = np.asarray(
        [(i / nx, j / ny) for j in range(ny + 1) for i in range(nx + 1)]
    )
    cells = []
    for j in range(ny):
        for i in range(nx):
            lower_left = j * (nx + 1) + i
            lower_right = lower_left + 1
            upper_left = lower_left + nx + 1
            upper_right = upper_left + 1
            cells.append((lower_left, lower_right, upper_right, upper_left))
    return phx.discretization.UnstructuredFiniteVolumePlan(
        vertices,
        quadrilaterals=np.asarray(cells),
        vertex_global_ids=np.arange(100, 100 + vertices.shape[0]),
        cell_global_ids=np.arange(500, 500 + len(cells)),
    )


def _initial_cell_volumes(plan):
    return plan.prepare().cell_volumes


def test_ssprk_ale_rigid_translation_and_legacy_diagnostic_are_consistent():
    plan = _quad_plan()
    velocity = jnp.asarray((0.3, -0.2))

    def translation(time, vertices, args):
        del args
        return vertices + time * velocity

    motion = phx.discretization.FixedConnectivityMotionPlan(
        plan, translation, mapping_id="rigid-translation"
    )
    step = motion.prepare_ssprk33_step(
        jnp.asarray(0.0),
        jnp.asarray(0.25),
        "translation-epoch",
        jnp.asarray(3, dtype=jnp.int32),
        jnp.asarray(7, dtype=jnp.int32),
        None,
        prior_effective_cell_volumes=_initial_cell_volumes(plan),
    )

    assert bool(step.passed)
    assert int(step.status) == int(
        phx.discretization.finite_volume.FiniteVolumeGeometryStatus.SUCCESS
    )
    for stage in (step.stage_1, step.stage_2, step.stage_3):
        np.testing.assert_allclose(
            stage.effective_cell_volumes,
            step.stage_1.coordinate_effective_cell_volumes,
            atol=2e-12,
        )
        np.testing.assert_allclose(stage.mesh_volume_rate, 0.0, atol=2e-12)
    np.testing.assert_allclose(
        step.stage_1_vertex_velocity,
        jnp.broadcast_to(velocity, step.stage_1_vertex_velocity.shape),
        atol=2e-12,
    )
    np.testing.assert_allclose(
        step.stage_1_face_mesh_volume_rate,
        jnp.sum(
            step.stage_1.face_blocks[0].quadrature_weights
            * step.stage_1.face_blocks[0].quadrature_grid_normal_velocity,
            axis=1,
        ),
        atol=2e-12,
    )
    legacy = motion.advance(jnp.asarray(0.0), jnp.asarray(0.25))
    np.testing.assert_allclose(legacy.cell_volume_change, 0.0, atol=2e-12)
    assert legacy.report.maximum_gcl_residual < 2e-12
    assert bool(legacy.report.passed)


def test_ssprk_ale_constant_geometry_has_zero_grid_rates():
    plan = _quad_plan(2, 2)

    def stationary(time, vertices, args):
        del time, args
        return vertices

    motion = phx.discretization.FixedConnectivityMotionPlan(
        plan, stationary, mapping_id="constant-geometry"
    )
    step = motion.prepare_ssprk33_step(
        0.4,
        0.3,
        "constant-epoch",
        0,
        0,
        None,
        prior_effective_cell_volumes=_initial_cell_volumes(plan),
    )

    assert bool(step.passed)
    for face_rate in (
        step.stage_1_face_mesh_volume_rate,
        step.stage_2_face_mesh_volume_rate,
        step.stage_3_face_mesh_volume_rate,
    ):
        np.testing.assert_array_equal(face_rate, jnp.zeros_like(face_rate))
    for cell_rate in (step.g1, step.g2, step.g3):
        np.testing.assert_array_equal(cell_rate, jnp.zeros_like(cell_rate))
    np.testing.assert_allclose(
        step.accepted_geometry.effective_cell_volumes,
        step.stage_1.effective_cell_volumes,
        rtol=0.0,
        atol=1e-15,
    )


def test_ssprk_ale_carries_accepted_volume_and_preserves_constant_state():
    plan = _quad_plan(2, 2)
    policy = phx.discretization.finite_volume.ALEGeometryConsistencyPolicy(
        absolute_tolerance=5.0e-2,
        relative_tolerance=5.0e-2,
    )

    def deformation(time, vertices, rate):
        return vertices.at[:, 0].set((1.0 + rate * time**5) * vertices[:, 0])

    motion = phx.discretization.FixedConnectivityMotionPlan(
        plan,
        deformation,
        mapping_id="accepted-volume-carry",
        consistency_policy=policy,
    )
    first = motion.prepare_ssprk33_step(
        0.0,
        0.4,
        "volume-carry-epoch",
        0,
        0,
        0.5,
        prior_effective_cell_volumes=_initial_cell_volumes(plan),
    )
    second = motion.prepare_ssprk33_step(
        0.4,
        0.2,
        "volume-carry-epoch",
        4,
        4,
        0.5,
        prior_effective_cell_volumes=(first.accepted_geometry.effective_cell_volumes),
    )

    assert bool(first.passed)
    assert bool(second.passed)
    np.testing.assert_array_equal(
        second.stage_1.effective_cell_volumes,
        first.accepted_geometry.effective_cell_volumes,
    )
    expected_defect = jnp.abs(
        first.accepted_geometry.effective_cell_volumes
        - second.stage_1.coordinate_effective_cell_volumes
    )
    assert jnp.max(expected_defect) > 1.0e-12
    np.testing.assert_allclose(
        second.stage_1.evidence.coordinate_effective_volume_defect,
        expected_defect,
        rtol=0.0,
        atol=1.0e-15,
    )
    constant_state = jnp.asarray((2.0, -0.5))
    accepted_content = (
        first.accepted_geometry.effective_cell_volumes[:, None] * constant_state[None, :]
    )
    np.testing.assert_allclose(
        accepted_content / second.stage_1.effective_cell_volumes[:, None],
        jnp.broadcast_to(constant_state, accepted_content.shape),
        rtol=0.0,
        atol=1.0e-15,
    )


def test_ssprk_ale_validates_and_certifies_prior_volume_mismatch():
    plan = _quad_plan(2, 2)

    def stationary(time, vertices, args):
        del time, args
        return vertices

    motion = phx.discretization.FixedConnectivityMotionPlan(
        plan,
        stationary,
        mapping_id="prior-volume-validation",
    )
    initial_volumes = _initial_cell_volumes(plan)
    with pytest.raises(ValueError, match="exact shape"):
        motion.prepare_ssprk33_step(
            0.0,
            0.1,
            "prior-validation-epoch",
            0,
            0,
            None,
            prior_effective_cell_volumes=initial_volumes[:-1],
        )
    with pytest.raises(
        (ValueError, eqx.EquinoxRuntimeError),
        match="prior_effective_cell_volumes",
    ):
        motion.prepare_ssprk33_step(
            0.0,
            0.1,
            "prior-validation-epoch",
            0,
            0,
            None,
            prior_effective_cell_volumes=initial_volumes.at[0].set(0.0),
        )

    mismatched = 1.1 * initial_volumes
    step = motion.prepare_ssprk33_step(
        0.0,
        0.1,
        "prior-validation-epoch",
        0,
        0,
        None,
        prior_effective_cell_volumes=mismatched,
    )
    assert not bool(step.passed)
    np.testing.assert_allclose(
        step.stage_1.evidence.coordinate_effective_volume_defect,
        jnp.abs(mismatched - initial_volumes),
    )


def test_ssprk_ale_negative_trial_volume_returns_order_aware_retry_evidence():
    plan = _quad_plan(2, 2)
    policy = phx.discretization.finite_volume.ALEGeometryConsistencyPolicy(
        absolute_tolerance=1.0e-10,
        relative_tolerance=1.0e-10,
        reduction_safety_factor=0.8,
        minimum_reduction_factor=1.0e-8,
    )

    def oscillatory_compression(time, vertices, amplitude):
        scale = 1.0 - amplitude * jnp.sin(jnp.pi * time)
        return vertices.at[:, 0].set(scale * vertices[:, 0])

    motion = phx.discretization.FixedConnectivityMotionPlan(
        plan,
        oscillatory_compression,
        mapping_id="negative-v1-retry",
        consistency_policy=policy,
    )
    step = motion.prepare_ssprk33_step(
        0.0,
        1.0,
        "negative-v1-epoch",
        0,
        0,
        0.8,
        prior_effective_cell_volumes=_initial_cell_volumes(plan),
    )

    assert not bool(step.passed)
    assert int(step.status) == int(
        phx.discretization.finite_volume.FiniteVolumeGeometryStatus.FAILED
    )
    assert not bool(step.stage_2.evidence.passed)
    evidence = step.stage_2.evidence
    ratio = jnp.max(
        jnp.concatenate(
            (
                evidence.coordinate_effective_volume_defect
                / evidence.coordinate_effective_volume_tolerance,
                evidence.face_closure_defect / evidence.face_closure_tolerance,
                evidence.gcl_identity_defect / evidence.gcl_identity_tolerance,
            )
        )
    )
    expected = jnp.maximum(
        policy.minimum_reduction_factor,
        policy.reduction_safety_factor
        * jnp.maximum(ratio, 1.0) ** (-1.0 / evidence.expected_order),
    )
    assert jnp.isfinite(step.proposed_reduction_factor)
    assert 0.0 < step.proposed_reduction_factor < 1.0
    np.testing.assert_allclose(
        evidence.proposed_reduction_factor,
        expected,
        rtol=2.0e-13,
    )
    np.testing.assert_allclose(
        step.proposed_reduction_factor,
        min(
            stage.evidence.proposed_reduction_factor
            for stage in (
                step.stage_1,
                step.stage_2,
                step.stage_3,
                step.accepted_geometry,
            )
        ),
    )


def test_ssprk_ale_exact_identities_still_reject_intermediate_coordinate_defect():
    plan = _quad_plan(2, 2)
    rate = 0.4
    policy = phx.discretization.finite_volume.ALEGeometryConsistencyPolicy(
        absolute_tolerance=1.0e-13,
        relative_tolerance=1.0e-13,
        reduction_safety_factor=0.8,
        minimum_reduction_factor=1.0e-8,
    )

    def expansion(time, vertices, args):
        del args
        return (1.0 + rate * time) * vertices

    motion = phx.discretization.FixedConnectivityMotionPlan(
        plan,
        expansion,
        mapping_id="linear-scale-nonlinear-volume",
        consistency_policy=policy,
    )
    step = motion.prepare_ssprk33_step(
        0.0,
        0.5,
        "expansion-epoch",
        0,
        0,
        None,
        prior_effective_cell_volumes=_initial_cell_volumes(plan),
    )

    assert not bool(step.passed)
    assert int(step.status) == int(
        phx.discretization.finite_volume.FiniteVolumeGeometryStatus.FAILED
    )
    assert bool(step.stage_1.evidence.passed)
    assert not bool(step.stage_2.evidence.passed)
    assert not bool(step.stage_3.evidence.passed)
    assert bool(step.accepted_geometry.evidence.passed)
    for evidence in (
        step.stage_1.evidence,
        step.stage_2.evidence,
        step.stage_3.evidence,
        step.accepted_geometry.evidence,
    ):
        np.testing.assert_allclose(evidence.gcl_identity_defect, 0.0, atol=2e-15)

    stage_evidence = step.stage_2.evidence
    stage_ratio = jnp.max(
        jnp.concatenate(
            (
                stage_evidence.coordinate_effective_volume_defect
                / stage_evidence.coordinate_effective_volume_tolerance,
                stage_evidence.face_closure_defect
                / stage_evidence.face_closure_tolerance,
                stage_evidence.gcl_identity_defect
                / stage_evidence.gcl_identity_tolerance,
            )
        )
    )
    expected_stage_factor = jnp.maximum(
        policy.minimum_reduction_factor,
        policy.reduction_safety_factor * stage_ratio ** (-0.5),
    )
    np.testing.assert_allclose(
        stage_evidence.proposed_reduction_factor,
        expected_stage_factor,
        rtol=2e-13,
    )


def test_ssprk_ale_final_defect_uses_fourth_order_reduction():
    policy = phx.discretization.finite_volume.ALEGeometryConsistencyPolicy(
        absolute_tolerance=1.0e-14,
        relative_tolerance=1.0e-14,
        reduction_safety_factor=0.75,
        minimum_reduction_factor=1.0e-8,
    )

    def fifth_order_deformation(time, vertices, args):
        scale = 1.0 + args * time**5
        return vertices.at[:, 0].set(scale * vertices[:, 0])

    motion = phx.discretization.FixedConnectivityMotionPlan(
        _quad_plan(2, 2),
        fifth_order_deformation,
        mapping_id="fifth-order-coordinate-deformation",
        consistency_policy=policy,
    )
    step = motion.prepare_ssprk33_step(
        0.0,
        0.5,
        "final-order-epoch",
        0,
        0,
        1.0,
        prior_effective_cell_volumes=_initial_cell_volumes(motion.base_plan),
    )
    evidence = step.accepted_geometry.evidence

    assert not bool(evidence.passed)
    assert int(evidence.expected_order) == 4
    final_ratio = jnp.max(
        jnp.concatenate(
            (
                evidence.coordinate_effective_volume_defect
                / evidence.coordinate_effective_volume_tolerance,
                evidence.face_closure_defect / evidence.face_closure_tolerance,
                evidence.gcl_identity_defect / evidence.gcl_identity_tolerance,
            )
        )
    )
    expected_final_factor = jnp.maximum(
        policy.minimum_reduction_factor,
        policy.reduction_safety_factor * final_ratio ** (-0.25),
    )
    np.testing.assert_allclose(
        evidence.proposed_reduction_factor,
        expected_final_factor,
        rtol=2e-13,
    )


def test_ssprk_ale_prepare_is_jittable_and_differentiable():
    def parameterized_expansion(time, vertices, rate):
        return (1.0 + rate * time) * vertices

    motion = phx.discretization.FixedConnectivityMotionPlan(
        _quad_plan(2, 2),
        parameterized_expansion,
        mapping_id="differentiable-expansion",
        consistency_policy=(
            phx.discretization.finite_volume.ALEGeometryConsistencyPolicy(
                absolute_tolerance=1.0,
                relative_tolerance=1.0,
            )
        ),
    )
    base_volumes = _initial_cell_volumes(motion.base_plan)

    @eqx.filter_jit
    def prepare(rate):
        return motion.prepare_ssprk33_step(
            jnp.asarray(0.1),
            jnp.asarray(0.05),
            "differentiable-epoch",
            jnp.asarray(4, dtype=jnp.int32),
            jnp.asarray(9, dtype=jnp.int32),
            rate,
            prior_effective_cell_volumes=((1.0 + 0.1 * rate) ** 2 * base_volumes),
        )

    compiled = prepare(jnp.asarray(0.2))
    assert bool(compiled.passed)
    derivative = jax.grad(
        lambda rate: jnp.sum(prepare(rate).accepted_geometry.effective_cell_volumes)
    )(jnp.asarray(0.2))
    assert jnp.isfinite(derivative)
    assert derivative > 0.0


def test_ssprk_ale_rejects_inverted_coordinate_geometry():
    def inversion(time, vertices, args):
        del args
        return vertices.at[:, 0].set((1.0 - 2.0 * time) * vertices[:, 0])

    motion = phx.discretization.FixedConnectivityMotionPlan(
        _quad_plan(2, 2), inversion, mapping_id="inverting-motion"
    )
    step = motion.prepare_ssprk33_step(
        0.0,
        0.75,
        "inversion-epoch",
        0,
        0,
        None,
        prior_effective_cell_volumes=_initial_cell_volumes(motion.base_plan),
    )

    assert not bool(step.passed)
    assert int(step.status) == int(
        phx.discretization.finite_volume.FiniteVolumeGeometryStatus.FAILED
    )
    assert jnp.isfinite(step.proposed_reduction_factor)
    assert 0.0 < step.proposed_reduction_factor < 1.0


def test_ssprk_ale_routes_topology_geometry_and_evidence_versions_exactly():
    def translation(time, vertices, velocity):
        return vertices + time * velocity

    motion = phx.discretization.FixedConnectivityMotionPlan(
        _quad_plan(),
        translation,
        mapping_id="versioned-translation",
    )
    step = motion.prepare_ssprk33_step(
        0.2,
        0.1,
        "runtime-epoch-17",
        jnp.asarray(11, dtype=jnp.int32),
        jnp.asarray(21, dtype=jnp.int32),
        jnp.asarray((0.1, -0.05)),
        prior_effective_cell_volumes=_initial_cell_volumes(motion.base_plan),
    )
    stages = (step.stage_1, step.stage_2, step.stage_3, step.accepted_geometry)

    assert step.topology_epoch_id == "runtime-epoch-17"
    assert step.motion_plan_id == motion.plan_id
    assert step.start_geometry.topology_id == motion.base_plan.topology_id
    assert step.end_geometry.topology_id == motion.base_plan.topology_id
    assert step.start_geometry.geometry_layout_id == motion.geometry_layout_id
    assert step.end_geometry.geometry_layout_id == motion.geometry_layout_id
    assert int(step.start_geometry.geometry_version) == 11
    assert int(step.end_geometry.geometry_version) == 14
    assert tuple(int(stage.geometry_version) for stage in stages) == (11, 12, 13, 14)
    assert tuple(int(stage.evidence.evidence_version) for stage in stages) == (
        21,
        22,
        23,
        24,
    )
    assert all(stage.topology_epoch_id == step.topology_epoch_id for stage in stages)
    assert all(stage.geometry_layout_id == motion.geometry_layout_id for stage in stages)
    assert len({stage.face_blocks[0].layout.block_id for stage in stages}) == 1
    assert all(
        stage.evidence.policy_id == motion.consistency_policy.policy_id
        for stage in stages
    )


def test_conservative_remap_identity_is_jittable_and_exact():
    discretization = _quad_plan().prepare()
    count = discretization.cell_count
    remap = phx.discretization.UnstructuredConservativeRemapPlan(
        discretization,
        discretization,
        np.arange(count + 1, dtype=np.int32),
        np.arange(count, dtype=np.int32),
        discretization.cell_volumes,
        method="identity-common-refinement",
        provenance="unit-test",
    )
    values = jnp.stack(
        (
            1.0 + discretization.cell_centers[:, 0],
            2.0 - discretization.cell_centers[:, 1],
        ),
        axis=-1,
    )
    transferred = eqx.filter_jit(remap.apply)(values)
    np.testing.assert_allclose(transferred, values)
    np.testing.assert_allclose(remap.conservation_defect(values, transferred), 0.0)
    gradient = jax.grad(lambda state: jnp.sum(remap.apply(state) ** 2))(values)
    np.testing.assert_allclose(gradient, 2.0 * values)

    with pytest.raises(ValueError, match="completely cover"):
        phx.discretization.UnstructuredConservativeRemapPlan(
            discretization,
            discretization,
            np.arange(count + 1, dtype=np.int32),
            np.arange(count, dtype=np.int32),
            0.5 * discretization.cell_volumes,
            method="incomplete",
            provenance="unit-test",
        )


def test_topology_changing_common_refinement_remap_preserves_integral():
    source_vertices = np.asarray(
        ((0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (0.0, 1.0), (1.0, 1.0), (2.0, 1.0))
    )
    source = phx.discretization.UnstructuredFiniteVolumePlan(
        source_vertices,
        quadrilaterals=np.asarray(((0, 1, 4, 3), (1, 2, 5, 4))),
        cell_global_ids=np.asarray((10, 11)),
    ).prepare()
    target_vertices = np.asarray(((0.0, 0.0), (2.0, 0.0), (2.0, 1.0), (0.0, 1.0)))
    target = phx.discretization.UnstructuredFiniteVolumePlan(
        target_vertices,
        quadrilaterals=np.asarray(((0, 1, 2, 3),)),
        cell_global_ids=np.asarray((20,)),
    ).prepare()
    remap = phx.discretization.UnstructuredConservativeRemapPlan(
        source,
        target,
        np.asarray((0, 2), dtype=np.int32),
        np.asarray((0, 1), dtype=np.int32),
        np.asarray((1.0, 1.0)),
        method="exact-rectangle-overlap",
        provenance="analytic",
    )
    source_values = jnp.asarray(((1.0, 2.0), (3.0, 4.0)))
    target_values = remap.apply(source_values)

    assert source.topology_id != target.topology_id
    np.testing.assert_allclose(target_values, ((2.0, 3.0),))
    np.testing.assert_allclose(
        remap.conservation_defect(source_values, target_values), 0.0, atol=1e-14
    )
    assert remap.report.maximum_target_coverage_defect < 1e-12
    assert remap.report.maximum_source_coverage_defect < 1e-12
