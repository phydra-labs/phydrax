#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
from phydrax.discretization.finite_volume import (
    _unstructured_dynamics as unstructured_dynamics,
)


def _grid_plan(system, nx=2, ny=2):
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
        quadrilaterals=np.asarray(cells, dtype=np.int32),
        vertex_global_ids=np.arange(100, 100 + vertices.shape[0]),
        cell_global_ids=np.arange(500, 500 + len(cells)),
        component_names=system.component_names,
    )


def _prepared_runtime(
    *,
    motion=None,
    mapping_id="test-motion",
    consistency_policy=None,
    wall_velocity_provider=None,
    step_policy=None,
    source=None,
    boundary_values=None,
):
    system = phx.equations.EulerSystem(2)
    plan = _grid_plan(system)
    discretization = plan.prepare()
    if boundary_values is None:
        if wall_velocity_provider is None:
            boundary_values = {
                name: phx.discretization.ExtrapolationBoundary()
                for name in discretization.boundary_patch_names
            }
        else:
            boundary_values = {
                name: phx.discretization.MovingSlipWallBoundary(
                    wall_velocity_provider,
                    wall_velocity_provider_id=f"{mapping_id}:{name}",
                )
                for name in discretization.boundary_patch_names
            }
    boundaries = phx.discretization.UnstructuredFiniteVolumeBoundarySet(
        discretization.boundary_patch_names,
        boundary_values,
    )
    method = phx.discretization.UnstructuredFiniteVolumeMethodPlan(
        phx.discretization.PiecewiseConstantReconstruction(),
        phx.discretization.RusanovFluxPlan(),
    )
    problem = phx.equations.ConservationProblemIR(
        f"ale-runtime:{mapping_id}",
        "state",
        system,
        boundaries,
        source=source,
    )
    coupling = None
    if motion is not None:
        motion_plan = phx.discretization.FixedConnectivityMotionPlan(
            plan,
            motion,
            mapping_id=mapping_id,
            consistency_policy=consistency_policy,
        )
        coupling = phx.discretization.UnstructuredFiniteVolumeCouplingPlan(
            motion=motion_plan
        )
    dynamics = phx.equations.compile_conservation_problem(
        problem,
        discretization,
        method,
        coupling=coupling,
    ).dynamics
    runtime = phx.solver.PreparedFiniteVolumeRuntime(
        dynamics,
        phx.discretization.FluxPositivityPlan(),
        step_policy,
    )
    return plan, discretization, system, runtime


def _uniform_conserved(system, discretization, velocity=(0.0, 0.0)):
    primitive = jnp.broadcast_to(
        jnp.asarray((1.0, velocity[0], velocity[1], 1.0)),
        discretization.state_shape,
    )
    return system.primitive_to_conserved(primitive)


def _stationary(time, vertices, args):
    del time, args
    return vertices


def _interior_linear_deformation(time, vertices, args):
    del args
    return vertices.at[4, 0].add(0.15 * time)


_DIRECT_COUPLED_SSPRK_ERROR = (
    "Coupled unstructured finite-volume dynamics require "
    "PreparedFiniteVolumeRuntime; UnsplitFiniteVolumeSSPRK3Plan supports only "
    "canonically uncoupled static unstructured dynamics."
)


def _with_prepared_coupling_marker(dynamics, component):
    coupling = eqx.tree_at(
        lambda value: getattr(value, component),
        dynamics.coupling,
        jnp.asarray(1),
        is_leaf=lambda value: value is None,
    )
    return eqx.tree_at(lambda value: value.coupling, dynamics, coupling)


@pytest.mark.parametrize(
    "component",
    (
        "motion",
        "embedded_boundary",
        "vof",
        "capillarity",
        "contact_angles",
        "amr",
        "overset",
        "sliding",
    ),
)
def test_direct_ssprk_rejects_every_prepared_unstructured_coupling_category(
    component,
):
    _, _, _, runtime = _prepared_runtime(mapping_id=f"direct-reject:{component}")
    coupled = _with_prepared_coupling_marker(runtime.dynamics, component)

    with pytest.raises(ValueError) as error:
        phx.solver.UnsplitFiniteVolumeSSPRK3Plan(coupled)

    assert str(error.value) == _DIRECT_COUPLED_SSPRK_ERROR


def test_direct_ssprk_rejects_prepared_topology_events():
    _, discretization, _, runtime = _prepared_runtime(mapping_id="direct-reject:events")
    event_coupling = phx.discretization.UnstructuredFiniteVolumeCouplingPlan(
        topology_event_capacity=1,
        topology_event_policy="accepted_step",
    ).prepare(discretization)
    coupled = eqx.tree_at(
        lambda value: value.coupling,
        runtime.dynamics,
        event_coupling,
    )

    with pytest.raises(ValueError) as error:
        phx.solver.UnsplitFiniteVolumeSSPRK3Plan(coupled)

    assert str(error.value) == _DIRECT_COUPLED_SSPRK_ERROR


def test_direct_ssprk_matches_canonically_uncoupled_unstructured_runtime():
    _, discretization, system, runtime = _prepared_runtime(
        mapping_id="direct-uncoupled-parity"
    )
    primitive = (
        jnp.broadcast_to(
            jnp.asarray((1.0, 0.08, -0.02, 1.0)),
            discretization.state_shape,
        )
        .at[:, 0]
        .add(0.02 * jnp.arange(discretization.cell_count))
    )
    initial_average = system.primitive_to_conserved(primitive)
    initial = runtime.initialize_state(initial_average, 0.0, 1.0e-3)

    direct = phx.solver.UnsplitFiniteVolumeSSPRK3Plan(runtime.dynamics).advance(
        initial.content_state.time,
        initial_average,
        initial.step_size,
    )
    prepared = runtime.advance(initial)

    assert bool(prepared.accepted)
    np.testing.assert_allclose(
        direct.state,
        prepared.runtime_state.cell_average(),
        rtol=2.0e-10,
        atol=2.0e-11,
    )
    np.testing.assert_allclose(
        direct.time,
        prepared.runtime_state.content_state.time,
    )


def test_moving_initialize_state_rejects_nonzero_time():
    _, discretization, system, runtime = _prepared_runtime(
        motion=_stationary,
        mapping_id="initialize-nonzero-time",
    )

    with pytest.raises(ValueError):
        runtime.initialize_state(
            _uniform_conserved(system, discretization),
            1.0e-3,
            1.0e-3,
        )


def test_moving_initialize_state_rejects_nonidentity_t0_motion():
    def displaced_at_t0(time, vertices, args):
        del time, args
        return vertices.at[4, 0].add(0.05)

    _, discretization, system, runtime = _prepared_runtime(
        motion=displaced_at_t0,
        mapping_id="initialize-displaced-t0",
    )

    with pytest.raises(ValueError):
        runtime.initialize_state(
            _uniform_conserved(system, discretization),
            0.0,
            1.0e-3,
        )


def test_moving_initialize_state_accepts_identity_t0_with_base_volumes():
    _, discretization, system, runtime = _prepared_runtime(
        motion=_stationary,
        mapping_id="initialize-identity-t0",
    )

    initial = runtime.initialize_state(
        _uniform_conserved(system, discretization),
        0.0,
        1.0e-3,
    )

    np.testing.assert_array_equal(
        initial.content_state.effective_cell_volumes,
        discretization.cell_volumes,
    )


def test_stationary_ale_and_static_runtime_publish_one_compatible_ledger():
    _, static_discretization, system, static_runtime = _prepared_runtime(
        mapping_id="static"
    )
    _, moving_discretization, _, moving_runtime = _prepared_runtime(
        motion=_stationary,
        mapping_id="stationary-ale",
    )
    primitive = (
        jnp.broadcast_to(
            jnp.asarray((1.0, 0.1, -0.03, 1.0)),
            static_discretization.state_shape,
        )
        .at[:, 0]
        .add(0.03 * jnp.arange(static_discretization.cell_count))
    )
    static_average = system.primitive_to_conserved(primitive)
    moving_average = jnp.reshape(static_average, moving_discretization.state_shape)
    static_initial = static_runtime.initialize_state(static_average, 0.0, 1.0e-3)
    moving_initial = moving_runtime.initialize_state(moving_average, 0.0, 1.0e-3)

    static_result = static_runtime.advance(static_initial)
    moving_result = moving_runtime.advance(moving_initial)

    assert bool(static_result.accepted)
    assert bool(moving_result.accepted)
    assert static_result.ale is None
    assert moving_result.ale is not None
    np.testing.assert_allclose(
        moving_result.runtime_state.cell_average(),
        static_result.runtime_state.cell_average(),
        rtol=2.0e-10,
        atol=2.0e-11,
    )
    static_ledger = static_result.accepted_flux_integrals
    moving_ledger = moving_result.accepted_flux_integrals
    assert moving_ledger.units == "content"
    assert tuple(block.route_id for block in moving_ledger.blocks) == tuple(
        block.route_id for block in static_ledger.blocks
    )
    np.testing.assert_allclose(
        moving_ledger.scatter_content_integral(),
        (
            moving_result.runtime_state.content_state.conservative_content
            - moving_initial.content_state.conservative_content
        ),
        rtol=2.0e-10,
        atol=2.0e-11,
    )
    assert "accepted_" + "integrated_fluxes" not in vars(moving_result)
    assert "flux_" + "integrals" not in vars(moving_result.ale)


def test_deforming_free_stream_preserves_every_stage_and_exact_content_volume_gcl():
    _, discretization, system, runtime = _prepared_runtime(
        motion=_interior_linear_deformation,
        mapping_id="interior-linear-deformation",
    )
    uniform = _uniform_conserved(system, discretization)
    initial = runtime.initialize_state(uniform, 0.0, 2.0e-2)
    result = runtime.advance(initial)

    assert bool(result.accepted)
    assert result.ale is not None
    ale = result.ale
    dt = result.accepted_step_size
    geometry = ale.geometry
    rate_1, rate_2, rate_3 = (
        ledger.scatter_content_rate() for ledger in ale.stage_rate_ledgers
    )
    qn = initial.content_state.conservative_content
    q1 = qn + dt * rate_1
    q2 = 0.75 * qn + 0.25 * (q1 + dt * rate_2)
    qnew = (1.0 / 3.0) * qn + (2.0 / 3.0) * (q2 + dt * rate_3)

    np.testing.assert_allclose(
        q1 / geometry.stage_2.effective_cell_volumes[:, None],
        uniform,
        rtol=3.0e-10,
        atol=3.0e-11,
    )
    np.testing.assert_allclose(
        q2 / geometry.stage_3.effective_cell_volumes[:, None],
        uniform,
        rtol=3.0e-10,
        atol=3.0e-11,
    )
    np.testing.assert_allclose(
        qnew / geometry.accepted_geometry.effective_cell_volumes[:, None],
        uniform,
        rtol=3.0e-10,
        atol=3.0e-11,
    )
    np.testing.assert_allclose(
        qnew,
        result.runtime_state.content_state.conservative_content,
        rtol=2.0e-12,
        atol=2.0e-12,
    )

    vn = geometry.stage_1.effective_cell_volumes
    v1 = vn + dt * geometry.g1
    v2 = 0.75 * vn + 0.25 * (v1 + dt * geometry.g2)
    vnew = (1.0 / 3.0) * vn + (2.0 / 3.0) * (v2 + dt * geometry.g3)
    np.testing.assert_allclose(v1, geometry.stage_2.effective_cell_volumes)
    np.testing.assert_allclose(v2, geometry.stage_3.effective_cell_volumes)
    np.testing.assert_allclose(vnew, geometry.accepted_geometry.effective_cell_volumes)

    accepted_ledger = result.accepted_flux_integrals
    for accepted_block, first, second, third in zip(
        accepted_ledger.blocks,
        ale.stage_rate_ledgers[0].blocks,
        ale.stage_rate_ledgers[1].blocks,
        ale.stage_rate_ledgers[2].blocks,
        strict=True,
    ):
        expected = dt * (
            (1.0 / 6.0) * first.flux_rate
            + (1.0 / 6.0) * second.flux_rate
            + (2.0 / 3.0) * third.flux_rate
        )
        np.testing.assert_allclose(accepted_block.flux_integral, expected)
    expected_source = dt * (
        (1.0 / 6.0) * ale.stage_rate_ledgers[0].source_rate
        + (1.0 / 6.0) * ale.stage_rate_ledgers[1].source_rate
        + (2.0 / 3.0) * ale.stage_rate_ledgers[2].source_rate
    )
    np.testing.assert_allclose(accepted_ledger.source_integral, expected_source)
    np.testing.assert_allclose(
        qn + accepted_ledger.scatter_content_integral(),
        qnew,
        rtol=3.0e-12,
        atol=3.0e-12,
    )
    second = runtime.advance(result.runtime_state)
    assert bool(second.accepted)
    assert second.ale is not None
    np.testing.assert_allclose(
        second.ale.geometry.stage_1.effective_cell_volumes,
        result.runtime_state.content_state.effective_cell_volumes,
    )
    assert int(second.runtime_state.content_state.geometry_version) == 6
    assert int(second.runtime_state.content_state.evidence_version) == 6
    assert second.runtime_state.topology_journal.current_epoch_id == (
        initial.topology_journal.current_epoch_id
    )


def test_translating_moving_wall_has_zero_mass_flux_and_uses_relative_cfl():
    velocity = jnp.asarray((0.23, -0.11))

    def translation(time, vertices, args):
        del args
        return vertices + time * velocity

    def wall_velocity(time, points, normal, args):
        del time, normal, args
        return jnp.broadcast_to(velocity, points.shape)

    _, discretization, system, runtime = _prepared_runtime(
        motion=translation,
        mapping_id="rigid-translation-wall",
        wall_velocity_provider=wall_velocity,
    )
    uniform = _uniform_conserved(system, discretization, tuple(velocity))
    initial = runtime.initialize_state(uniform, 0.0, 1.0e-2)
    result = runtime.advance(initial)

    assert bool(result.accepted)
    assert result.ale is not None
    np.testing.assert_allclose(
        result.runtime_state.cell_average(), uniform, rtol=2.0e-10, atol=2.0e-11
    )
    dt = result.accepted_step_size
    qn = initial.content_state.conservative_content
    rate_1, rate_2, rate_3 = (
        ledger.scatter_content_rate() for ledger in result.ale.stage_rate_ledgers
    )
    q1 = qn + dt * rate_1
    q2 = 0.75 * qn + 0.25 * (q1 + dt * rate_2)
    qnew = (1.0 / 3.0) * qn + (2.0 / 3.0) * (q2 + dt * rate_3)
    for content, metrics in (
        (q1, result.ale.geometry.stage_2),
        (q2, result.ale.geometry.stage_3),
        (qnew, result.ale.geometry.accepted_geometry),
    ):
        np.testing.assert_allclose(
            content / metrics.effective_cell_volumes[:, None],
            uniform,
            rtol=2.0e-10,
            atol=2.0e-11,
        )
    for block in result.accepted_flux_integrals.blocks:
        boundary = block.neighbour_cells < 0
        np.testing.assert_allclose(
            jnp.where(boundary, block.flux_integral[:, 0], 0.0),
            0.0,
            atol=3.0e-11,
        )
    assert result.accepted_step_size <= result.ale.relative_cfl_step
    np.testing.assert_allclose(
        result.ale.relative_cfl_step * result.ale.maximum_relative_rate,
        runtime.policy.cfl,
        rtol=2.0e-10,
    )


def test_ale_boundary_dispatch_passes_exact_patch_quadrature_contexts(
    monkeypatch,
):
    probe_system = phx.equations.EulerSystem(2)
    probe_discretization = _grid_plan(probe_system).prepare()

    def stationary_wall_velocity(time, points, normal, args):
        del time, normal, args
        return jnp.zeros_like(points)

    boundary_values = {
        name: (
            phx.discretization.MovingSlipWallBoundary(
                stationary_wall_velocity,
                wall_velocity_provider_id=f"patch-context:{name}",
            )
            if patch_id == 0
            else phx.discretization.ExtrapolationBoundary()
        )
        for patch_id, name in enumerate(probe_discretization.boundary_patch_names)
    }
    _, discretization, system, runtime = _prepared_runtime(
        motion=_stationary,
        mapping_id="patch-context",
        boundary_values=boundary_values,
    )
    uniform = _uniform_conserved(system, discretization)
    initial = runtime.initialize_state(uniform, 0.0, 1.0e-3)
    result = runtime.advance(initial)
    assert bool(result.accepted)
    assert result.ale is not None

    observed = []
    original_make_context = phx.discretization.MovingSlipWallBoundary.make_context
    original_static_context = unstructured_dynamics.ALEBoundaryContext

    def capture_moving_context(self, *args, **kwargs):
        context = original_make_context(self, *args, **kwargs)
        observed.append(("moving", context))
        return context

    def capture_static_context(**kwargs):
        context = original_static_context(**kwargs)
        observed.append(("static", context))
        return context

    monkeypatch.setattr(
        phx.discretization.MovingSlipWallBoundary,
        "make_context",
        capture_moving_context,
    )
    monkeypatch.setattr(
        unstructured_dynamics,
        "ALEBoundaryContext",
        capture_static_context,
    )

    metrics = result.ale.geometry.stage_1
    states = runtime.dynamics._stage_face_states(uniform, metrics, None)
    jax.block_until_ready(states)

    block = metrics.face_blocks[0]
    points = runtime.dynamics.precision.reconstruction(block.quadrature_points)
    normal = block.area_vectors / block.face_measures[:, None]
    normal = jnp.broadcast_to(normal[:, None, :], points.shape)
    grid_normal = runtime.dynamics.precision.reconstruction(
        block.quadrature_grid_normal_velocity
    )
    routed_indices = []
    expected_contexts = []
    for policy, indices in zip(
        runtime.dynamics.boundaries.boundaries,
        runtime.dynamics.boundary_face_indices,
        strict=True,
    ):
        if indices.shape[0] == 0:
            continue
        routed_indices.append(np.asarray(indices))
        expected_contexts.append(
            (
                "moving"
                if isinstance(policy, phx.discretization.MovingSlipWallBoundary)
                else "static",
                indices,
            )
        )

    assert len(observed) == len(expected_contexts)
    all_routed_indices = np.concatenate(routed_indices)
    assert np.unique(all_routed_indices).size == all_routed_indices.size
    for (branch, context), (expected_branch, indices) in zip(
        observed, expected_contexts, strict=True
    ):
        expected_points = points[indices]
        expected_normal = normal[indices]
        expected_grid_velocity = grid_normal[indices, :, None] * expected_normal
        assert branch == expected_branch
        assert context.face_point.shape == expected_points.shape
        assert context.outward_normal.shape == expected_points.shape
        assert context.quadrature_grid_velocity.shape == expected_points.shape
        assert context.face_point.dtype == expected_points.dtype
        assert context.outward_normal.dtype == expected_normal.dtype
        assert context.quadrature_grid_velocity.dtype == expected_grid_velocity.dtype
        np.testing.assert_array_equal(context.face_point, expected_points)
        np.testing.assert_array_equal(context.outward_normal, expected_normal)
        np.testing.assert_array_equal(
            context.quadrature_grid_velocity,
            expected_grid_velocity,
        )


def test_stage_source_is_integrated_against_each_target_geometry_volume_once():
    source_vector = jnp.asarray((0.0, 0.0, 0.0, 0.7))

    def source(time, state, coordinates, args):
        del time, coordinates, args
        return jnp.broadcast_to(source_vector, state.shape)

    _, discretization, system, runtime = _prepared_runtime(
        motion=_interior_linear_deformation,
        mapping_id="volume-source",
        source=source,
    )
    initial = runtime.initialize_state(
        _uniform_conserved(system, discretization), 0.0, 1.0e-3
    )
    result = runtime.advance(initial)

    assert bool(result.accepted)
    assert result.ale is not None
    for ledger, metrics in zip(
        result.ale.stage_rate_ledgers,
        (
            result.ale.geometry.stage_1,
            result.ale.geometry.stage_2,
            result.ale.geometry.stage_3,
        ),
        strict=True,
    ):
        np.testing.assert_allclose(
            ledger.source_rate,
            metrics.effective_cell_volumes[:, None] * source_vector,
        )
        assert ledger.units == "content/time"
    expected_source_integral = result.accepted_step_size * (
        (1.0 / 6.0) * result.ale.stage_rate_ledgers[0].source_rate
        + (1.0 / 6.0) * result.ale.stage_rate_ledgers[1].source_rate
        + (2.0 / 3.0) * result.ale.stage_rate_ledgers[2].source_rate
    )
    np.testing.assert_allclose(
        result.accepted_flux_integrals.source_integral,
        expected_source_integral,
    )
    source_sum, boundary_sum, net_cell_sum = (
        result.accepted_flux_integrals.conservation_sums()
    )
    np.testing.assert_allclose(
        source_sum - boundary_sum,
        net_cell_sum,
        rtol=2e-11,
        atol=2e-11,
    )


def test_geometry_evidence_uses_its_order_aware_factor_for_a_successful_retry():
    consistency = phx.discretization.finite_volume.ALEGeometryConsistencyPolicy(
        absolute_tolerance=3.0e-3,
        relative_tolerance=0.0,
        reduction_safety_factor=0.8,
        minimum_reduction_factor=0.1,
    )

    def nonlinear_deformation(time, vertices, args):
        del args
        return vertices.at[4, 0].add(0.8 * time**2)

    policy = phx.solver.FiniteVolumeStepPolicy(
        maximum_retries=3,
        reduction_factor=0.5,
    )
    _, discretization, system, runtime = _prepared_runtime(
        motion=nonlinear_deformation,
        mapping_id="retry-nonlinear-geometry",
        consistency_policy=consistency,
        step_policy=policy,
    )
    initial = runtime.initialize_state(
        _uniform_conserved(system, discretization), 0.0, 0.2
    )
    result = runtime.advance(initial)

    assert bool(result.accepted)
    assert int(result.retries) > 0
    assert result.accepted_step_size < result.attempted_step_size
    assert result.ale is not None
    assert result.ale.geometry_reduction_factor < 1.0
    assert bool(result.ale.geometry.passed)


def test_all_rejected_retries_publish_one_final_attempt_evidence_envelope():
    strict = phx.discretization.finite_volume.ALEGeometryConsistencyPolicy(
        absolute_tolerance=1.0e-16,
        relative_tolerance=1.0e-16,
        reduction_safety_factor=0.8,
        minimum_reduction_factor=0.1,
    )

    def nonlinear_deformation(time, vertices, args):
        del args
        return vertices.at[4, 0].add(0.8 * time**2)

    policy = phx.solver.FiniteVolumeStepPolicy(
        maximum_retries=2,
        reduction_factor=0.5,
    )
    _, discretization, system, runtime = _prepared_runtime(
        motion=nonlinear_deformation,
        mapping_id="rejected-nonlinear-geometry",
        consistency_policy=strict,
        step_policy=policy,
    )
    initial = runtime.initialize_state(
        _uniform_conserved(system, discretization), 0.0, 0.2
    )
    result = runtime.advance(initial)

    assert not bool(result.accepted)
    assert result.ale is not None
    assert result.ale.geometry_reduction_factor < 1.0
    geometry = result.ale.geometry
    assert not bool(geometry.passed)
    assert float(geometry.accepted_geometry.time) < float(
        initial.time + initial.step_size
    )
    np.testing.assert_array_equal(
        result.runtime_state.content_state.conservative_content,
        initial.content_state.conservative_content,
    )
    np.testing.assert_array_equal(
        result.runtime_state.content_state.effective_cell_volumes,
        initial.content_state.effective_cell_volumes,
    )
    assert int(result.runtime_state.content_state.geometry_version) == int(
        initial.content_state.geometry_version
    )
    assert int(result.runtime_state.content_state.evidence_version) == int(
        initial.content_state.evidence_version
    )
    assert result.runtime_state.topology_journal.current_epoch_id == (
        initial.topology_journal.current_epoch_id
    )
    assert int(result.runtime_state.topology_journal.count) == int(
        initial.topology_journal.count
    )
    rejected_ledger = result.accepted_flux_integrals
    stage_metrics = (
        geometry.stage_1,
        geometry.stage_2,
        geometry.stage_3,
    )
    stage_ledgers = result.ale.stage_rate_ledgers
    np.testing.assert_allclose(geometry.stage_1.time, initial.time)
    np.testing.assert_allclose(
        geometry.stage_2.time,
        geometry.accepted_geometry.time,
    )
    np.testing.assert_allclose(
        geometry.stage_3.time,
        initial.time + 0.5 * (geometry.accepted_geometry.time - initial.time),
    )
    np.testing.assert_allclose(rejected_ledger.start_time, initial.time)
    np.testing.assert_allclose(
        rejected_ledger.end_time,
        geometry.accepted_geometry.time,
    )
    assert int(rejected_ledger.start_geometry_version) == int(
        initial.content_state.geometry_version
    )
    assert int(rejected_ledger.start_evidence_version) == int(
        initial.content_state.evidence_version
    )
    assert int(rejected_ledger.end_geometry_version) == int(
        geometry.accepted_geometry.geometry_version
    )
    assert int(rejected_ledger.end_evidence_version) == int(
        geometry.accepted_geometry.evidence.evidence_version
    )
    assert tuple(
        int(value) for value in rejected_ledger.stage_geometry_versions
    ) == tuple(int(metrics.geometry_version) for metrics in stage_metrics)
    assert tuple(
        int(value) for value in rejected_ledger.stage_evidence_versions
    ) == tuple(int(metrics.evidence.evidence_version) for metrics in stage_metrics)
    for ledger, metrics in zip(stage_ledgers, stage_metrics, strict=True):
        assert int(ledger.geometry_version) == int(metrics.geometry_version)
        assert int(ledger.evidence_version) == int(metrics.evidence.evidence_version)
        assert ledger.geometry_layout_id == metrics.geometry_layout_id
        assert ledger.evidence_policy_id == metrics.evidence.policy_id
        assert ledger.topology_epoch_id == geometry.topology_epoch_id
    assert not bool(result.positivity.high_order_valid)
    assert not bool(result.positivity.fallback_valid)
    assert not bool(result.positivity.limited_state_valid)
    for integral_block, stage_block in zip(
        rejected_ledger.blocks,
        result.ale.stage_rate_ledgers[0].blocks,
        strict=True,
    ):
        assert integral_block.block_id == stage_block.block_id
        assert integral_block.route_id == stage_block.route_id
        np.testing.assert_array_equal(
            integral_block.flux_integral,
            jnp.zeros_like(integral_block.flux_integral),
        )
    np.testing.assert_array_equal(
        rejected_ledger.source_integral,
        jnp.zeros_like(rejected_ledger.source_integral),
    )


def test_stage_rate_positivity_blends_high_and_fallback_against_target_volumes():
    system = phx.equations.EulerSystem(1)
    primitive = jnp.asarray(((1.0, 0.0, 1.0), (1.0, 0.0, 1.0)))
    content = system.primitive_to_conserved(primitive)
    owner = jnp.asarray((0,), dtype=jnp.int32)
    neighbour = jnp.asarray((1,), dtype=jnp.int32)
    active_face = jnp.asarray((True,))
    high_block = phx.discretization.FiniteVolumeStageFluxRateBlock(
        jnp.asarray(((10.0, 0.0, 0.0),)),
        owner,
        neighbour,
        active_face,
        "positivity-face",
        "interior",
    )
    fallback_block = high_block.with_flux_rate(jnp.zeros_like(high_block.flux_rate))
    kwargs = dict(
        geometry_family_id="positivity-family",
        geometry_layout_id="positivity-layout",
        geometry_version=0,
        evidence_policy_id="positivity-evidence",
        evidence_version=0,
        topology_epoch_id="positivity-epoch",
    )
    high = phx.discretization.FiniteVolumeStageFluxRateLedger(
        (high_block,), jnp.zeros_like(content), jnp.ones(2, dtype=bool), **kwargs
    )
    fallback = phx.discretization.FiniteVolumeStageFluxRateLedger(
        (fallback_block,), jnp.zeros_like(content), jnp.ones(2, dtype=bool), **kwargs
    )

    limited = phx.discretization.FluxPositivityPlan().limit_stage_rate_ledgers(
        system,
        content,
        high,
        fallback,
        0.2,
        jnp.ones((2,)),
    )

    assert bool(limited.report.fallback_valid)
    assert bool(limited.report.limited_state_valid)
    assert bool(limited.report.activated)
    assert limited.report.blend_factor < 1.0
    assert jnp.all(system.admissible(limited.euler_cell_average))
    np.testing.assert_allclose(
        jnp.sum(limited.euler_content, axis=0),
        jnp.sum(content, axis=0),
    )
    assert limited.ledger.units == "content/time"


def test_mixed_polygon_geometry_closure_is_exact_under_jit():
    plan = phx.discretization.UnstructuredFiniteVolumePlan(
        np.asarray(
            (
                (0.0, 0.0),
                (1.0, 0.0),
                (0.0, 1.0),
                (2.0, 0.0),
                (3.0, 0.0),
                (3.0, 1.0),
                (2.0, 1.0),
            )
        ),
        triangles=np.asarray(((0, 1, 2),), dtype=np.int32),
        quadrilaterals=np.asarray(((3, 4, 5, 6),), dtype=np.int32),
    )
    discretization = plan.prepare()
    connectivity = phx.discretization.polygonal_connectivity(
        plan.triangles,
        plan.quadrilaterals,
        plan.vertices.shape[0],
    )
    arguments = (
        plan.vertices,
        plan.triangles,
        plan.quadrilaterals,
        plan.tetrahedra,
        connectivity,
        discretization.owner_cells,
        discretization.owner_signs,
    )

    eager = phx.discretization.evaluate_unstructured_fv_geometry(*arguments)
    compiled = eqx.filter_jit(phx.discretization.evaluate_unstructured_fv_geometry)(
        *arguments
    )

    np.testing.assert_array_equal(compiled[5], eager[5])
    np.testing.assert_array_equal(compiled[5], jnp.zeros((2, 2)))


def test_ale_advance_is_jittable_differentiable_and_checkpoint_versions_are_ready():
    _, discretization, system, runtime = _prepared_runtime(
        motion=_stationary,
        mapping_id="jit-grad-checkpoint",
    )
    initial = runtime.initialize_state(
        _uniform_conserved(system, discretization, (0.07, 0.0)),
        0.0,
        1.0e-3,
        accepted_step=9,
    )
    advance = eqx.filter_jit(runtime.advance)
    result = advance(initial)

    assert bool(result.accepted)
    assert result.ale is not None
    assert int(result.runtime_state.content_state.geometry_version) == 3
    assert int(result.runtime_state.content_state.evidence_version) == 3
    assert int(result.runtime_state.accepted_step) == 10
    assert int(result.runtime_state.topology_journal.count) == 0
    assert result.runtime_state.topology_journal.current_epoch_id == (
        initial.topology_journal.current_epoch_id
    )
    assert tuple(
        int(version) for version in result.accepted_flux_integrals.stage_geometry_versions
    ) == (0, 1, 2)
    assert int(result.accepted_flux_integrals.end_geometry_version) == 3

    def objective(speed):
        primitive = jnp.broadcast_to(
            jnp.stack((1.0, speed, jnp.asarray(0.0), jnp.asarray(1.0))),
            discretization.state_shape,
        )
        state = runtime.initialize_state(
            system.primitive_to_conserved(primitive), 0.0, 1.0e-3
        )
        advanced = advance(state)
        return jnp.sum(advanced.runtime_state.content_state.conservative_content)

    derivative = jax.grad(objective)(jnp.asarray(0.07))
    assert jnp.isfinite(derivative)
