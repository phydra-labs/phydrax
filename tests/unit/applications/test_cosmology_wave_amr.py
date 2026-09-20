import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
from phydrax.applications.cosmology._wave_amr import (
    WaveAMRAdaptivityPlan,
    WaveAMRDiscretizationPlan,
    WaveAMRPhysicsPlan,
)
from phydrax.applications.cosmology._wave_boundaries import (
    AbsorbingWaveBoundaryPolicy,
    IsolatedPotentialGauge,
    IsolatedWaveBoundaryDescriptor,
)
from phydrax.discretization.amr._complex_field import complex_amr_fill_patch


def _hierarchy(*, fine_capacity=4, refined=False, periodic=True):
    grid = phx.discretization.TensorGridPlan(
        (phx.discretization.UniformCellAxisSpec(8, periodic=periodic),),
        axis_names=("x",),
    ).prepare(jnp.asarray([[0.0], [1.0]]))
    hierarchy = phx.discretization.BlockHierarchyPlan(
        grid,
        (
            phx.discretization.BlockLevelPlan(
                0, (4,), 2, halo_width=1, refinement_ratio=2
            ),
            phx.discretization.BlockLevelPlan(1, (4,), fine_capacity, halo_width=1),
        ),
    )
    runtime = phx.discretization.FDAMRHierarchyPlan(hierarchy).prepare()
    topology = runtime.initial_topology()
    if refined:
        tags = jnp.zeros((2, 4), dtype="bool").at[0, 1:3].set(True)
        compiled = runtime.compile_topology(topology, (tags,))
        assert compiled.status.successful
        topology = compiled.topology
    return grid, runtime, topology


def _prepared(*, topology=None, fd_plan=None, adaptivity=None):
    if topology is None or fd_plan is None:
        _, fd_plan, topology = _hierarchy(refined=True)
    plan = WaveAMRDiscretizationPlan(
        fd_plan,
        adaptivity=adaptivity,
        norm_relative_tolerance=2.0e-8,
        maximum_phase_radians=2.0,
    )
    return plan.prepare(
        WaveAMRPhysicsPlan(
            1.0,
            gravitational_constant=0.02,
            reduced_planck_constant=0.05,
        ),
        topology,
        phx.applications.cosmology.FLRWBackground(1.0, 1.0),
    )


def _field(prepared, *, wave_number=2.0 * np.pi, packet=True):
    arrays = []
    lower = float(np.asarray(prepared.topology.plan.grid.structured_axes[0].bounds[0]))
    for level_plan, metadata, spacing in zip(
        prepared.topology.plan.levels,
        prepared.topology.levels,
        prepared.topology.plan.level_spacings,
        strict=True,
    ):
        values = jnp.zeros(
            (level_plan.maximum_blocks,) + level_plan.block_shape,
            dtype=jnp.complex128,
        )
        logical = np.asarray(metadata.logical_indices)
        for slot in np.flatnonzero(np.asarray(metadata.active)):
            origin = logical[slot, 0] * level_plan.block_shape[0]
            coordinates = (
                lower
                + (origin + jnp.arange(level_plan.block_shape[0]) + 0.5) * spacing[0]
            )
            amplitude = (
                0.3 + jnp.exp(-(((coordinates - 0.45) / 0.18) ** 2))
                if packet
                else jnp.ones_like(coordinates)
            )
            values = values.at[slot].set(
                amplitude * jnp.exp(1j * wave_number * coordinates)
            )
        arrays.append(values)
    return prepared.initialize(tuple(arrays), 1.0)


def test_fixed_topology_packet_soliton_current_uses_global_composite_solves():
    prepared = _prepared()
    state = _field(prepared)
    result = prepared.step(state, 1.00005)

    assert bool(result.successful)
    assert result.kinetic_solve.provenance.method == "gmres"
    assert result.gravity.solve_result.provenance.method == "projected-pcg"
    assert bool(result.second_gravity.successful)
    assert bool(result.diagnostics.initial_poisson_closed)
    assert bool(result.diagnostics.final_poisson_closed)
    assert any(
        bool(jnp.any(first != second))
        for first, second in zip(
            result.gravity.source,
            result.second_gravity.source,
            strict=True,
        )
    )
    assert result.gravity.topology_id == prepared.topology.topology_id
    assert float(result.diagnostics.probability_relative_error) < 2.0e-8
    assert float(result.diagnostics.cayley_relative_residual) < 1.0e-8
    assert float(result.diagnostics.self_adjoint_residual) < 1.0e-10
    assert bool(jnp.all(jnp.isfinite(result.diagnostics.final_current)))
    assert result.diagnostics.initial_current[0] > 0.0
    assert float(result.diagnostics.current_relative_defect) < 0.05
    np.testing.assert_allclose(
        result.gravity.interface_flux_conservation_defect,
        0.0,
        atol=0.0,
    )
    assert result.state.psi.topology.epoch.epoch_id == state.psi.topology.epoch.epoch_id


def test_amr_cayley_solve_honors_configured_iteration_cap_and_rolls_back():
    _, fd_plan, topology = _hierarchy(refined=True)
    prepared = WaveAMRDiscretizationPlan(
        fd_plan,
        solve_relative_tolerance=0.0,
        solve_absolute_tolerance=0.0,
        maximum_solve_steps=1,
        maximum_phase_radians=2.0,
    ).prepare(
        WaveAMRPhysicsPlan(
            1.0,
            gravitational_constant=0.02,
            reduced_planck_constant=0.05,
        ),
        topology,
        phx.applications.cosmology.FLRWBackground(1.0, 1.0),
    )
    state = _field(prepared)

    result = prepared.step(state, 1.01)

    assert prepared.solve_policy.tolerance.max_steps == 1
    assert not bool(result.kinetic_solve.successful)
    assert int(result.kinetic_solve.diagnostics.iterations) <= 1
    assert not bool(result.successful)
    assert result.state.accepted_boundary == state.accepted_boundary
    for actual, expected in zip(
        result.state.psi.levels,
        state.psi.levels,
        strict=True,
    ):
        np.testing.assert_array_equal(actual.values, expected.values)


def test_amr_kinetic_dispersion_phase_gate_rolls_back():
    _, fd_plan, topology = _hierarchy(refined=True)
    prepared = WaveAMRDiscretizationPlan(
        fd_plan,
        maximum_phase_radians=1.0e-8,
    ).prepare(
        WaveAMRPhysicsPlan(
            1.0,
            gravitational_constant=0.02,
            reduced_planck_constant=0.05,
        ),
        topology,
        phx.applications.cosmology.FLRWBackground(1.0, 1.0),
    )
    state = _field(prepared, packet=False)

    result = prepared.step(state, 1.00005)

    assert not bool(result.successful)
    assert int(result.diagnostics.status) == 7
    assert result.diagnostics.maximum_kinetic_phase > 1.0e-8
    assert bool(result.diagnostics.initial_poisson_closed)
    assert bool(result.diagnostics.final_poisson_closed)
    assert result.state.scale_factor == state.scale_factor


def test_complex_fill_patch_is_globally_u1_equivariant():
    prepared = _prepared()
    state = _field(prepared)
    phase = jnp.exp(0.37j)
    rotated = phx.discretization.BlockHierarchyState(
        state.psi.topology,
        tuple(
            phx.discretization.BlockLevelState(
                level.plan,
                level.metadata,
                phase * level.values,
            )
            for level in state.psi.levels
        ),
    )

    original_fill = complex_amr_fill_patch(prepared.fd_hierarchy, state.psi)
    rotated_fill = complex_amr_fill_patch(prepared.fd_hierarchy, rotated)

    assert bool(original_fill.complete)
    assert bool(rotated_fill.complete)
    for original, transformed in zip(
        original_fill.workspaces,
        rotated_fill.workspaces,
        strict=True,
    ):
        np.testing.assert_allclose(
            transformed.values,
            phase * original.values,
            rtol=1.0e-12,
            atol=1.0e-12,
        )


def test_fixed_topology_derivative_keeps_the_epoch_frozen():
    prepared = _prepared()
    state = _field(prepared, packet=False)
    tangent = tuple(0.01 * level.values for level in state.psi.levels)

    action = prepared.fixed_topology_jvp(state, tangent, 1.00001)

    assert len(action) == len(state.psi.levels)
    assert all(
        value.shape == level.values.shape and bool(jnp.all(jnp.isfinite(value)))
        for value, level in zip(action, state.psi.levels, strict=True)
    )
    assert state.psi.topology.epoch.epoch_id == prepared.topology.epoch.epoch_id


def test_phase_aware_topology_transfer_preserves_mass_current_and_winding_evidence():
    _, fd_plan, topology = _hierarchy(refined=False)
    adaptivity = WaveAMRAdaptivityPlan(
        maximum_phase_change=0.01,
        current_relative_tolerance=0.5,
        phase_defect_tolerance=1.0,
    )
    prepared = _prepared(topology=topology, fd_plan=fd_plan, adaptivity=adaptivity)
    state = _field(prepared, packet=False)

    proposal = prepared.propose_topology(state)
    transitioned = prepared.transition(state, proposal)

    assert proposal.compilation.status.changed
    assert bool(transitioned.successful)
    assert bool(transitioned.evidence.probability_preserved)
    assert bool(transitioned.evidence.current_preserved)
    assert bool(transitioned.evidence.winding_preserved)
    assert float(transitioned.evidence.probability_relative_defect) < 1.0e-10
    assert float(transitioned.evidence.current_relative_defect) < 0.5
    np.testing.assert_allclose(transitioned.evidence.source_winding, (1.0,))
    np.testing.assert_allclose(transitioned.evidence.target_winding, (1.0,))
    assert float(transitioned.evidence.winding_absolute_defect) < 1.0e-8
    assert transitioned.state.psi.topology.epoch.index == topology.epoch.index + 1
    assert (
        transitioned.prepared.topology.topology_id
        == transitioned.state.psi.topology.topology_id
    )
    unaccepted = type(state)(
        state.psi,
        state.scale_factor,
        jnp.asarray(False),
    )
    with pytest.raises(ValueError, match="accepted-boundary"):
        prepared.transition(unaccepted, proposal)
    with pytest.raises(ValueError, match="does not bind"):
        transitioned.prepared.transition(transitioned.state, proposal)

    strict = _prepared(
        topology=topology,
        fd_plan=fd_plan,
        adaptivity=WaveAMRAdaptivityPlan(
            maximum_phase_change=0.01,
            current_relative_tolerance=0.01,
            phase_defect_tolerance=1.0,
        ),
    )
    strict_state = _field(strict, packet=False)
    rejected = strict.transition(
        strict_state,
        strict.propose_topology(strict_state),
    )
    assert not bool(rejected.successful)
    assert bool(rejected.rolled_back)
    assert rejected.state.psi.topology.epoch.epoch_id == topology.epoch.epoch_id
    assert rejected.candidate_state.psi.topology.epoch.index == topology.epoch.index + 1


def test_zero_current_transfer_uses_named_absolute_tolerance():
    _, fd_plan, topology = _hierarchy(refined=False)
    prepared = _prepared(
        topology=topology,
        fd_plan=fd_plan,
        adaptivity=WaveAMRAdaptivityPlan(
            maximum_density_contrast=0.01,
            current_relative_tolerance=0.0,
            current_absolute_tolerance=1.0e-12,
            phase_defect_tolerance=1.0,
        ),
    )
    state = _field(prepared, wave_number=0.0, packet=True)
    transitioned = prepared.transition(state, prepared.propose_topology(state))

    assert bool(transitioned.successful)
    assert float(transitioned.evidence.current_absolute_defect) <= 1.0e-12
    assert float(transitioned.evidence.current_relative_defect) == 0.0
    assert bool(transitioned.evidence.current_preserved)


def test_phase_aware_restriction_preserves_probability_current_and_winding():
    _, fd_plan, topology = _hierarchy(refined=True)
    prepared = _prepared(
        topology=topology,
        fd_plan=fd_plan,
        adaptivity=WaveAMRAdaptivityPlan(
            maximum_phase_change=3.0,
            maximum_density_contrast=100.0,
            maximum_quantum_potential_indicator=100.0,
            current_relative_tolerance=0.5,
            phase_defect_tolerance=1.0,
        ),
    )
    state = _field(prepared, packet=False)

    proposal = prepared.propose_topology(state)
    restricted = prepared.transition(state, proposal)

    assert proposal.compilation.status.changed
    assert bool(restricted.successful)
    assert bool(restricted.evidence.probability_preserved)
    assert bool(restricted.evidence.current_preserved)
    assert bool(restricted.evidence.winding_preserved)
    assert float(restricted.evidence.probability_relative_defect) < 1.0e-10
    assert restricted.state.psi.topology.epoch.index == topology.epoch.index + 1


def test_adaptive_capacity_failure_rolls_back_before_state_transition():
    _, fd_plan, topology = _hierarchy(fine_capacity=1, refined=False)
    adaptivity = WaveAMRAdaptivityPlan(maximum_phase_change=0.01)
    prepared = _prepared(topology=topology, fd_plan=fd_plan, adaptivity=adaptivity)
    state = _field(prepared, packet=False)

    proposal = prepared.propose_topology(state)

    assert not proposal.successful
    assert proposal.compilation.status.code == "capacity_exceeded"
    assert (
        proposal.compilation.topology.epoch.epoch_id == state.psi.topology.epoch.epoch_id
    )
    assert proposal.compilation.evidence.overflow_level == 1


def test_distributed_resource_admission_requires_mesh_for_multipart_execution(
    monkeypatch,
):
    _, fd_plan, topology = _hierarchy(refined=True)
    prepared = _prepared(topology=topology, fd_plan=fd_plan)
    state = _field(prepared)

    local = prepared.prepare_distributed(
        phx.discretization.BlockAMRPartitionPlan(topology.plan, 1),
        maximum_bytes=10_000_000,
    )
    assert local.admitted
    assert local.executable
    assert local.source_prepared_id == prepared.prepared_id
    assert local.topology_id == prepared.topology.topology_id
    assert local.partition_plan_id == local.hierarchy.partition.plan_id
    assert bool(prepared.distributed_step(local, state, 1.00001).successful)

    multipart = prepared.prepare_distributed(
        phx.discretization.BlockAMRPartitionPlan(topology.plan, 2),
        maximum_bytes=10_000_000,
    )
    assert multipart.admitted
    assert not multipart.executable
    with pytest.raises(ValueError, match="requires a real JAX ExecutionGroup mesh"):
        prepared.distributed_step(multipart, state, 1.00001)

    denied_partition = phx.discretization.BlockAMRPartitionPlan(topology.plan, 1)

    def forbidden_prepare(*args, **kwargs):
        del args, kwargs
        raise AssertionError("distributed routes were constructed before admission")

    with monkeypatch.context() as guard:
        guard.setattr(type(denied_partition), "prepare", forbidden_prepare)
        denied = prepared.prepare_distributed(
            denied_partition,
            maximum_bytes=0,
        )
    assert not denied.admitted
    assert denied.hierarchy is None
    assert denied.preflight_required_bytes == denied.required_bytes
    assert denied.required_bytes > denied.maximum_bytes
    assert "before layout or route construction" in denied.reason


def test_isolated_boundary_requires_distinct_finite_domain_gravity_owner():
    _, fd_plan, topology = _hierarchy(periodic=False)
    gauge = IsolatedPotentialGauge((0.5,), 2.0, multipole_order=4)
    absorbing = AbsorbingWaveBoundaryPolicy(0.2, 3.0)
    boundary = IsolatedWaveBoundaryDescriptor(gauge, absorbing=absorbing)
    plan = WaveAMRDiscretizationPlan(fd_plan, boundary=boundary)

    with pytest.raises(ValueError, match="finite-domain multipole Poisson owner"):
        plan.prepare(
            WaveAMRPhysicsPlan(1.0),
            topology,
            phx.applications.cosmology.FLRWBackground(1.0, 1.0),
        )

    points = jnp.asarray([[0.0], [0.5], [1.0]])
    attenuation = absorbing.attenuation(points, jnp.asarray([[0.0], [1.0]]), 0.1)
    assert attenuation[1] == 1.0
    assert attenuation[0] < 1.0
    assert attenuation[-1] < 1.0

    psi = jnp.ones((3,), dtype=jnp.complex128)
    accepted = absorbing.apply(
        psi,
        points,
        jnp.asarray([[0.0], [1.0]]),
        1.0e-3,
        jnp.ones((3,)),
    )
    assert bool(accepted.successful)
    assert accepted.probability_loss_fraction > 0.0
    rejected = absorbing.apply(
        psi,
        points,
        jnp.asarray([[0.0], [1.0]]),
        0.1,
        jnp.ones((3,)),
    )
    assert not bool(rejected.successful)
    np.testing.assert_array_equal(rejected.psi, psi)
