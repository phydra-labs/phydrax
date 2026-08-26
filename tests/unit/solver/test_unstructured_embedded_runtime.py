#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

import phydrax as phx


def _strip_plan(system, nx=2):
    vertices = np.asarray(
        [(i / nx, j) for j in range(2) for i in range(nx + 1)],
        dtype=float,
    )
    cells = []
    for i in range(nx):
        lower_left = i
        lower_right = i + 1
        upper_left = nx + 1 + i
        upper_right = upper_left + 1
        cells.append((lower_left, lower_right, upper_right, upper_left))
    return phx.discretization.UnstructuredFiniteVolumePlan(
        vertices,
        quadrilaterals=np.asarray(cells, dtype=np.int32),
        vertex_global_ids=np.arange(100, 100 + vertices.shape[0]),
        cell_global_ids=np.arange(500, 500 + len(cells)),
        component_names=system.component_names,
    )


def _runtime(
    level_set,
    *,
    interface_solver=None,
    stabilization=None,
    source=None,
    step_policy=None,
    field_id="runtime-cut",
    nx=2,
):
    system = phx.equations.EulerSystem(2)
    plan = _strip_plan(system, nx=nx)
    discretization = plan.prepare()
    boundaries = phx.discretization.UnstructuredFiniteVolumeBoundarySet(
        discretization.boundary_patch_names,
        {
            name: phx.discretization.ExtrapolationBoundary()
            for name in discretization.boundary_patch_names
        },
    )
    method = phx.discretization.UnstructuredFiniteVolumeMethodPlan(
        phx.discretization.PiecewiseConstantReconstruction(),
        (
            phx.discretization.RusanovFluxPlan()
            if interface_solver is None
            else interface_solver
        ),
    )
    embedded = phx.discretization.EmbeddedBoundaryPlan(
        discretization,
        level_set,
        field_id=field_id,
        body_tag=7,
        stabilization_policy=stabilization,
    )
    cut_boundaries = phx.discretization.UnstructuredEmbeddedBoundarySet(
        {7: phx.discretization.SlipWallBoundary()}
    )
    coupling = phx.discretization.UnstructuredFiniteVolumeCouplingPlan(
        embedded_boundary=embedded,
        embedded_boundaries=cut_boundaries,
    )
    problem = phx.equations.ConservationProblemIR(
        f"embedded-runtime:{field_id}",
        "state",
        system,
        boundaries,
        source=source,
        source_id=None if source is None else f"{field_id}-source",
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
    return discretization, system, dynamics, runtime


def _uniform(system, discretization, *, velocity=(0.0, 0.0), pressure=1.0):
    primitive = jnp.broadcast_to(
        jnp.asarray((1.0, velocity[0], velocity[1], pressure)),
        discretization.state_shape,
    )
    return system.primitive_to_conserved(primitive)


def _block(ledger, kind):
    return next(block for block in ledger.blocks if block.block_kind == kind)


def test_nonzero_time_free_stream_uses_true_ssprk_times_and_zero_cut_mass_flux():
    discretization, system, _, runtime = _runtime(
        lambda points, args: points[:, 0] - 0.25,
        field_id="nonzero-time-free-stream",
    )
    initial_average = _uniform(system, discretization, velocity=(0.0, 0.2))
    initial = runtime.initialize_state(initial_average, 1.25, 1.0e-3)

    result = runtime.advance(initial)

    assert bool(result.accepted)
    assert result.ale is None
    assert result.embedded is not None
    np.testing.assert_allclose(
        result.runtime_state.cell_average(),
        initial_average,
        rtol=2.0e-6,
        atol=2.0e-7,
    )
    np.testing.assert_allclose(
        jnp.stack(tuple(stage.time for stage in result.embedded.stage_metrics)),
        jnp.asarray((1.25, 1.251, 1.2505)),
    )
    cut = _block(result.accepted_flux_integrals, "cut")
    np.testing.assert_allclose(cut.flux_integral[:, 0], 0.0, atol=2.0e-8)


def test_outer_cut_and_physical_source_close_the_public_content_budget():
    source_vector = jnp.asarray((0.0, 0.0, 0.0, 0.4))

    def source(time, state, centers, args):
        del time, centers, args
        return jnp.broadcast_to(source_vector, state.shape)

    discretization, system, _, runtime = _runtime(
        lambda points, args: points[:, 0] - 0.25,
        source=source,
        field_id="accepted-budget",
    )
    average = _uniform(system, discretization, velocity=(0.0, 0.15))
    initial = runtime.initialize_state(average, 0.3, 2.0e-3)

    result = runtime.advance(initial)

    assert bool(result.accepted)
    ledger = result.accepted_flux_integrals
    assert {block.block_kind for block in ledger.blocks} >= {"physical", "cut"}
    change = (
        result.runtime_state.content_state.conservative_content
        - initial.content_state.conservative_content
    )
    np.testing.assert_allclose(
        change,
        ledger.scatter_content_integral(),
        rtol=3.0e-6,
        atol=3.0e-8,
    )
    source_sum, boundary_sum, net_sum = ledger.conservation_sums()
    np.testing.assert_allclose(source_sum - boundary_sum, net_sum, atol=3.0e-8)
    np.testing.assert_allclose(jnp.sum(change, axis=0), net_sum, atol=3.0e-8)


def test_asymmetric_cut_source_uses_fluid_centroids_before_inactive_masking():
    def source(time, state, centers, args):
        del time
        scale = (
            jnp.asarray(1.0, dtype=state.dtype)
            if args is None
            else jnp.asarray(args, dtype=state.dtype)
        )
        inactive_poison = jnp.where(
            centers[:, 0] > 0.0,
            jnp.asarray(0.0, dtype=state.dtype),
            jnp.asarray(jnp.nan, dtype=state.dtype),
        )
        linear_density = (
            scale * (2.0 + 3.0 * centers[:, 0] - 2.0 * centers[:, 1]) + inactive_poison
        )
        return jnp.zeros_like(state).at[:, -1].set(linear_density)

    with jax.enable_x64(False):
        discretization, system, dynamics, runtime = _runtime(
            lambda points, args: points[:, 0] + 0.25 * points[:, 1] - 0.6,
            source=source,
            field_id="asymmetric-centroid-source",
            nx=3,
        )
    initial = runtime.initialize_state(
        _uniform(system, discretization).astype(jnp.float32),
        jnp.asarray(0.0, dtype=jnp.float32),
        jnp.asarray(1.0e-4, dtype=jnp.float32),
    )
    result = runtime.advance(initial)

    assert bool(result.accepted)
    stage = result.embedded.stage_metrics[0]
    embedded_metrics = dynamics.coupling.embedded_metrics
    assert embedded_metrics is not None
    assert stage.geometry_family_id == discretization.geometry_id
    assert embedded_metrics.fluid_cell_centers.dtype == jnp.float32
    assert embedded_metrics.evidence.volume_closure_defect.dtype == jnp.float32
    np.testing.assert_array_equal(stage.active_cell_mask, (False, True, True))
    np.testing.assert_array_equal(
        stage.cell_centers[0],
        jnp.zeros_like(stage.cell_centers[0]),
    )
    expected_cut_centroid = jnp.asarray(
        (769.0 / 1380.0, 14.0 / 23.0),
        dtype=stage.cell_centers.dtype,
    )
    np.testing.assert_allclose(
        stage.cell_centers[1],
        expected_cut_centroid,
        rtol=8.0e-7,
        atol=2.0e-7,
    )
    np.testing.assert_array_equal(
        stage.cell_centers[2],
        jnp.asarray(
            discretization.cell_centers[2],
            dtype=stage.cell_centers.dtype,
        ),
    )

    unit_scale = jnp.asarray(1.0, dtype=stage.cell_centers.dtype)

    def integrated_source(scale):
        evaluation = dynamics.evaluate_stage(
            initial.content_state,
            stage,
            scale,
            redistribution=runtime.embedded_redistribution,
        )
        return evaluation.ledger.source_rate[:, -1]

    expected_rate = jnp.asarray(
        (0.0, 1129.0 / 2400.0, 7.0 / 6.0),
        dtype=stage.cell_centers.dtype,
    )
    eager_rate = integrated_source(unit_scale)
    compiled_rate = eqx.filter_jit(integrated_source)(unit_scale)
    np.testing.assert_allclose(
        eager_rate,
        expected_rate,
        rtol=2.0e-6,
        atol=2.0e-7,
    )
    np.testing.assert_allclose(
        compiled_rate,
        expected_rate,
        rtol=2.0e-6,
        atol=2.0e-7,
    )
    assert jnp.all(jnp.isfinite(compiled_rate))

    full_fluid_density = (
        2.0
        + 3.0 * discretization.cell_centers[2, 0]
        - 2.0 * discretization.cell_centers[2, 1]
    )
    np.testing.assert_allclose(
        eager_rate[2],
        discretization.cell_volumes[2] * full_fluid_density,
        rtol=2.0e-6,
        atol=2.0e-7,
    )
    expected_integral = jnp.sum(expected_rate)
    source_gradient = eqx.filter_jit(
        jax.grad(lambda scale: jnp.sum(integrated_source(scale)))
    )(unit_scale)
    np.testing.assert_allclose(
        source_gradient,
        expected_integral,
        rtol=3.0e-6,
        atol=3.0e-7,
    )


def test_mixed_hllc_routes_and_content_never_assign_solid_ownership():
    discretization, system, _, runtime = _runtime(
        lambda points, args: points[:, 0] - 0.75,
        interface_solver=phx.discretization.HLLCFluxPlan(),
        field_id="mixed-hllc",
    )
    initial = runtime.initialize_state(
        _uniform(system, discretization, velocity=(0.0, 0.1)),
        0.0,
        1.0e-3,
    )

    result = runtime.advance(initial)

    assert bool(result.accepted)
    active = np.asarray(initial.content_state.active_cell_mask)
    assert active.tolist() == [False, True]
    np.testing.assert_array_equal(
        initial.content_state.conservative_content[~active],
        np.zeros((1, system.component_count)),
    )
    np.testing.assert_array_equal(
        result.runtime_state.content_state.conservative_content[~active],
        np.zeros((1, system.component_count)),
    )
    for stage in result.embedded.stage_metrics:
        for face in stage.face_blocks:
            owners = np.asarray(face.layout.owner_cells)
            neighbours = np.asarray(face.layout.neighbour_cells)
            assert np.all(active[owners])
            assert np.all(active[neighbours[neighbours >= 0]])


def test_sliver_redistribution_is_conservative_and_cfl_uses_stabilized_volume():
    stabilization = phx.discretization.EmbeddedBoundaryStabilizationPolicy(
        minimum_volume_fraction=0.2,
        maximum_recipients=1,
    )
    source_vector = jnp.asarray((0.0, 0.0, 0.0, 1.0))

    def source(time, state, centers, args):
        del time, centers, args
        return jnp.broadcast_to(source_vector, state.shape)

    discretization, system, dynamics, runtime = _runtime(
        lambda points, args: points[:, 0] - 0.49,
        stabilization=stabilization,
        source=source,
        field_id="sliver-redistribution",
    )
    initial = runtime.initialize_state(
        _uniform(system, discretization, velocity=(0.0, 0.1)),
        0.0,
        2.0e-4,
    )

    result = runtime.advance(initial)

    assert bool(result.accepted)
    assert int(result.embedded.redistribution.small_cell_count) == 1
    redistribution = _block(
        result.accepted_flux_integrals,
        "small-cell-redistribution",
    )
    assert jnp.any(jnp.abs(redistribution.flux_integral[..., -1]) > 0.0)
    redistribution_scatter = jnp.zeros_like(
        result.runtime_state.content_state.conservative_content
    )
    redistribution_scatter = redistribution_scatter.at[redistribution.owner_cells].add(
        -redistribution.flux_integral
    )
    redistribution_scatter = redistribution_scatter.at[
        redistribution.neighbour_cells
    ].add(redistribution.flux_integral)
    np.testing.assert_allclose(
        jnp.sum(redistribution_scatter, axis=0),
        0.0,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        result.runtime_state.content_state.conservative_content
        - initial.content_state.conservative_content,
        result.accepted_flux_integrals.scatter_content_integral(),
        rtol=4.0e-6,
        atol=4.0e-8,
    )

    metrics = result.embedded.stage_metrics[0]
    evaluation = dynamics.evaluate_stage(
        initial.content_state,
        metrics,
        cfl=runtime.policy.cfl,
        redistribution=runtime.embedded_redistribution,
    )
    speed_sum = jnp.zeros((discretization.cell_count,))
    for face, speeds in zip(metrics.face_blocks, evaluation.relative_signal_speeds):
        face_speed = jnp.sum(face.quadrature_weights * speeds, axis=1)
        owners = face.layout.owner_cells
        neighbours = face.layout.neighbour_cells
        speed_sum = speed_sum.at[owners].add(face_speed)
        speed_sum = speed_sum.at[jnp.maximum(neighbours, 0)].add(
            jnp.where(neighbours >= 0, face_speed, 0.0)
        )
    stabilized_volume = jnp.maximum(
        metrics.effective_cell_volumes,
        stabilization.minimum_volume_fraction * discretization.cell_volumes,
    )
    expected_rate = jnp.max(
        jnp.where(metrics.active_cell_mask, speed_sum / stabilized_volume, 0.0)
    )
    np.testing.assert_allclose(evaluation.maximum_relative_rate, expected_rate)
    sliver = int(
        np.flatnonzero(np.asarray(runtime.embedded_redistribution.small_cells))[0]
    )
    np.testing.assert_allclose(
        evaluation.cell_relative_rate[sliver],
        speed_sum[sliver] / stabilized_volume[sliver],
    )
    assert evaluation.cell_relative_rate[sliver] < (
        speed_sum[sliver] / metrics.effective_cell_volumes[sliver]
    )


def test_stage_positivity_blends_physical_cut_and_redistribution_on_active_cells():
    system = phx.equations.EulerSystem(1)
    active_state = system.primitive_to_conserved(
        jnp.asarray(((1.0, 0.0, 1.0), (1.0, 0.0, 1.0)))
    )
    content = jnp.concatenate((jnp.zeros((1, 3)), active_state), axis=0)
    active_cells = jnp.asarray((False, True, True))
    blocks = (
        phx.discretization.FiniteVolumeStageFluxRateBlock(
            jnp.asarray(((12.0, 0.0, 0.0),)),
            (1,),
            (-1,),
            (True,),
            "active-physical",
            "physical",
        ),
        phx.discretization.FiniteVolumeStageFluxRateBlock(
            jnp.asarray(((8.0, 0.0, 0.0),)),
            (1,),
            (-1,),
            (True,),
            "active-cut",
            "cut",
        ),
        phx.discretization.FiniteVolumeStageFluxRateBlock(
            jnp.asarray(((2.0, 0.0, 0.0),)),
            (1,),
            (2,),
            (True,),
            "active-redistribution",
            "small-cell-redistribution",
        ),
    )
    kwargs = dict(
        geometry_family_id="active-positivity-family",
        geometry_layout_id="active-positivity-layout",
        geometry_version=0,
        evidence_policy_id="active-positivity-evidence",
        evidence_version=0,
        topology_epoch_id="active-positivity-epoch",
    )
    high = phx.discretization.FiniteVolumeStageFluxRateLedger(
        blocks,
        jnp.zeros_like(content),
        active_cells,
        **kwargs,
    )
    fallback = phx.discretization.FiniteVolumeStageFluxRateLedger(
        tuple(block.with_flux_rate(jnp.zeros_like(block.flux_rate)) for block in blocks),
        jnp.zeros_like(content),
        active_cells,
        **kwargs,
    )

    limited = phx.discretization.FluxPositivityPlan().limit_stage_rate_ledgers(
        system,
        content,
        high,
        fallback,
        0.1,
        jnp.asarray((0.0, 1.0, 1.0)),
    )

    assert bool(limited.report.fallback_valid)
    assert bool(limited.report.limited_state_valid)
    assert bool(limited.report.activated)
    assert len(limited.face_blend_factors) == 3
    np.testing.assert_array_equal(limited.euler_content[0], jnp.zeros((3,)))
    assert jnp.all(system.admissible(limited.euler_cell_average[1:]))


def test_full_fluid_embedded_runtime_matches_static_runtime():
    discretization, system, _, embedded_runtime = _runtime(
        lambda points, args: jnp.ones((points.shape[0],)),
        field_id="full-fluid-parity",
    )
    boundaries = phx.discretization.UnstructuredFiniteVolumeBoundarySet(
        discretization.boundary_patch_names,
        {
            name: phx.discretization.ExtrapolationBoundary()
            for name in discretization.boundary_patch_names
        },
    )
    method = phx.discretization.UnstructuredFiniteVolumeMethodPlan(
        phx.discretization.PiecewiseConstantReconstruction(),
        phx.discretization.RusanovFluxPlan(),
    )
    problem = phx.equations.ConservationProblemIR(
        "full-fluid-static-parity",
        "state",
        system,
        boundaries,
    )
    static_dynamics = phx.equations.compile_conservation_problem(
        problem,
        discretization,
        method,
    ).dynamics
    static_runtime = phx.solver.PreparedFiniteVolumeRuntime(
        static_dynamics,
        phx.discretization.FluxPositivityPlan(),
    )
    primitive = jnp.asarray(((1.0, 0.1, -0.03, 1.0), (1.05, 0.08, -0.02, 1.02)))
    average = system.primitive_to_conserved(primitive)
    embedded_initial = embedded_runtime.initialize_state(average, 0.2, 2.0e-4)
    static_initial = static_runtime.initialize_state(average, 0.2, 2.0e-4)

    embedded = embedded_runtime.advance(embedded_initial)
    static = static_runtime.advance(static_initial)

    assert bool(embedded.accepted) and bool(static.accepted)
    np.testing.assert_allclose(
        embedded.runtime_state.cell_average(),
        static.runtime_state.cell_average(),
        rtol=3.0e-6,
        atol=3.0e-8,
    )
    physical = _block(embedded.accepted_flux_integrals, "physical")
    np.testing.assert_allclose(
        physical.flux_integral,
        static.accepted_flux_integrals.blocks[0].flux_integral,
        rtol=3.0e-6,
        atol=3.0e-8,
    )


def test_full_solid_hllc_skips_physics_and_advances_zero_content():
    def forbidden_source(time, state, centers, args):
        raise AssertionError("full-solid source must not be evaluated")

    discretization, system, _, runtime = _runtime(
        lambda points, args: -jnp.ones((points.shape[0],)),
        interface_solver=phx.discretization.HLLCFluxPlan(),
        source=forbidden_source,
        field_id="full-solid-hllc",
    )
    initial = runtime.initialize_state(
        _uniform(system, discretization, velocity=(0.4, -0.2)),
        0.7,
        1.0e-2,
    )

    result = runtime.advance(initial)

    assert bool(result.accepted)
    assert result.embedded is not None
    assert result.embedded.stage_metrics[0].face_blocks == ()
    assert result.accepted_flux_integrals.blocks == ()
    assert jnp.isinf(result.embedded.relative_cfl_step)
    np.testing.assert_array_equal(
        result.runtime_state.content_state.conservative_content,
        jnp.zeros_like(result.runtime_state.content_state.conservative_content),
    )
    np.testing.assert_array_equal(
        result.runtime_state.cell_average(),
        jnp.zeros_like(result.runtime_state.cell_average()),
    )


def test_cfl_rejection_preserves_content_journal_and_publishes_zero_ledger():
    policy = phx.solver.FiniteVolumeStepPolicy(
        cfl=0.45,
        maximum_retries=0,
    )
    discretization, system, _, runtime = _runtime(
        lambda points, args: points[:, 0] - 0.25,
        step_policy=policy,
        field_id="rejection-immutability",
    )
    initial = runtime.initialize_state(
        _uniform(system, discretization, velocity=(3.0, 0.0)),
        0.0,
        10.0,
        accepted_step=4,
    )

    result = runtime.advance(initial)

    assert not bool(result.accepted)
    assert int(result.runtime_state.accepted_step) == 4
    assert result.runtime_state.time == initial.time
    assert result.runtime_state.topology_journal.current_epoch_id == (
        initial.topology_journal.current_epoch_id
    )
    assert int(result.runtime_state.topology_journal.count) == int(
        initial.topology_journal.count
    )
    np.testing.assert_array_equal(
        result.runtime_state.content_state.conservative_content,
        initial.content_state.conservative_content,
    )
    np.testing.assert_array_equal(
        result.accepted_flux_integrals.scatter_content_integral(),
        jnp.zeros_like(initial.content_state.conservative_content),
    )


def test_embedded_advance_is_jittable_differentiable_and_result_identities_hold():
    discretization, system, _, runtime = _runtime(
        lambda points, args: points[:, 0] - 0.25,
        interface_solver=phx.discretization.HLLCFluxPlan(),
        field_id="jit-grad-identities",
    )
    advance = eqx.filter_jit(runtime.advance)
    initial = runtime.initialize_state(
        _uniform(system, discretization, velocity=(0.0, 0.12)),
        0.4,
        2.0e-4,
        accepted_step=8,
    )

    result = advance(initial)

    assert bool(result.accepted)
    assert result.embedded is not None and result.ale is None
    assert bool(result.embedded.accepted) == bool(result.accepted)
    assert result.embedded.accepted_metrics.time == result.runtime_state.time
    assert result.accepted_flux_integrals.units == "content"
    assert result.accepted_flux_integrals.accepted_step == 9
    assert tuple(
        block.block_kind for block in result.accepted_flux_integrals.blocks
    ) == tuple(block.block_kind for block in result.embedded.stage_rate_ledgers[0].blocks)
    assert "accepted_flux_integrals" not in vars(result.embedded)

    def objective(tangential_velocity):
        state = runtime.initialize_state(
            _uniform(
                system,
                discretization,
                velocity=(0.0, tangential_velocity),
            ),
            0.4,
            2.0e-4,
        )
        advanced = advance(state)
        return jnp.sum(advanced.runtime_state.content_state.conservative_content)

    derivative = jax.grad(objective)(jnp.asarray(0.12))
    assert jnp.isfinite(derivative)
