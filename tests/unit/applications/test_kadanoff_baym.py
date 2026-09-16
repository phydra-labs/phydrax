#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax._physical import DimensionalScaleContract, RelativityScaleContract
from phydrax.applications.cosmology._dark_sector_species import DarkSectorSpeciesPlan
from phydrax.applications.cosmology._quantum_dark_kinetics import QuantumKineticState
from phydrax.applications.curved_spacetime_qft._kadanoff_baym import (
    advance_kadanoff_baym,
    checkpoint_kadanoff_baym,
    continue_kadanoff_baym_epoch,
    initialize_kadanoff_baym_memory,
    KadanoffBaymTransportPlan,
    restore_kadanoff_baym,
    WignerGradientPlan,
)
from phydrax.applications.curved_spacetime_qft._off_shell_transport import (
    breit_wigner_off_shell_state,
    OffShellTransportPlan,
)
from phydrax.applications.relativistic_scattering._unit_contract import (
    LocalRelativisticFramePlan,
    RelativisticUnitContract,
)
from phydrax.metrix import (
    ADMGridGeometry,
    CoordinateChart,
    minkowski_metric,
    orthonormal_tetrad,
    RelativityConvention,
)
from phydrax.solver._dark_sector_epoch_runtime import (
    DarkSectorEpochPlan,
    empty_dark_sector_epoch_state,
)


def test_centered_first_gradient_poisson_bracket_converges_at_second_order():
    def error(node_count):
        coordinates = np.arange(node_count) * (2.0 * np.pi / node_count)
        x, p = np.meshgrid(coordinates, coordinates, indexing="ij")
        left = np.sin(x) * np.cos(p)
        right = np.cos(x) * np.sin(p)
        exact = np.cos(x) ** 2 * np.cos(p) ** 2 - np.sin(x) ** 2 * np.sin(p) ** 2
        spacing = 2.0 * np.pi / node_count
        plan = WignerGradientPlan((node_count, node_count), (spacing,), (spacing,))
        computed = plan.poisson_bracket(jnp.asarray(left), jnp.asarray(right))
        return np.sqrt(np.mean((np.asarray(computed) - exact) ** 2))

    coarse = error(16)
    fine = error(32)
    assert fine < coarse / 3.5


def _units_and_frame():
    scale = RelativityScaleContract(DimensionalScaleContract.si(), 1, 3, 2, 1)
    units = RelativisticUnitContract(
        scale, RelativityConvention(metric_signature="mostly_minus")
    )
    geometry = ADMGridGeometry(
        jnp.asarray(1.0),
        jnp.zeros((3,)),
        jnp.eye(3),
        jnp.eye(3),
        jnp.asarray(1.0),
        jnp.zeros((3, 3)),
        jnp.asarray(True),
        jnp.asarray(True),
        snapshot_token=jnp.asarray(3),
        chart_id="minkowski-cartesian",
        convention_id=units.convention.convention_id,
        scale_id=units.scale.scale_id,
        topology_id="kb-periodic-grid",
        geometry_lineage_id="flat",
    )
    chart = CoordinateChart("minkowski", ("t", "x", "y", "z"))
    tetrad = orthonormal_tetrad(
        minkowski_metric(chart, convention="mostly_minus"),
        jnp.eye(4),
        jnp.zeros((4,)),
        convention=units.convention,
        tolerance=1.0e-7,
        source_id="kb-test-observer",
    )
    return units, LocalRelativisticFramePlan(
        geometry,
        tetrad,
        units,
        jnp.zeros((4,)),
        jnp.asarray(0.0),
        jnp.asarray(1.0),
        observer_id="kb-test-observer",
        orientation_id="right-handed-future",
    )


def _plans():
    units, frame = _units_and_frame()
    species = (
        DarkSectorSpeciesPlan(
            "kb-dark-fermion",
            1.0,
            mass_unit=units.scale.dimensional_scale.mass_unit.unit_id,
            energy_unit=units.energy_unit.unit_id,
        ),
    )
    support = QuantumKineticState(
        jnp.full((1, 3, 3), 0.2),
        None,
        species=species,
        statistics=jnp.asarray((-1,), dtype=jnp.int8),
        spatial_active=jnp.asarray((True, True, True)),
        momentum_active=jnp.asarray((True, True, True)),
        units=units,
        frame=frame,
    )
    nodes = np.linspace(-4.0, 6.0, 101)
    weights = np.full(nodes.shape, nodes[1] - nodes[0])
    weights[[0, -1]] *= 0.5
    off_shell = OffShellTransportPlan(
        support,
        nodes,
        weights,
        jnp.ones((3,)),
        jnp.ones((3,)),
        energy_quadrature_id="kb-energy-trapezoid",
        momentum_quadrature_id="kb-three-mode",
        spectral_tolerance=0.2,
        dyson_tolerance=1.0e-10,
        kms_tolerance=1.0e-10,
    )
    epoch = DarkSectorEpochPlan(
        packet_capacity=2,
        event_capacity=2,
        product_capacity=2,
        radiation_capacity=2,
        work_capacity=2,
        frontier_capacity=2,
        packet_width=2,
        event_width=2,
        product_width=2,
        radiation_width=2,
        work_width=2,
        frontier_width=2,
        species_revision_id="1" * 64,
        topology_revision_id="2" * 64,
    )
    gradient = WignerGradientPlan((3, 3), (1.0,), (1.0,))
    return off_shell, KadanoffBaymTransportPlan(
        off_shell,
        gradient,
        epoch,
        time_step=0.05,
        memory_depth=2,
    )


def test_fixed_depth_memory_is_solvable_reports_initial_tail_and_restarts_exactly():
    off_shell, plan = _plans()
    runtime = empty_dark_sector_epoch_state(plan.epoch, epoch_sequence=0)
    state = breit_wigner_off_shell_state(
        off_shell,
        jnp.full((3, 1), 1.0),
        0.3,
        inverse_temperature=0.5,
        chemical_potentials=jnp.asarray((0.0,)),
    )
    memory = initialize_kadanoff_baym_memory(plan, runtime)
    zeros = jnp.zeros(off_shell.spectral_shape)
    kernel = jnp.full(off_shell.spectral_shape, 0.02)
    initial = jnp.full(off_shell.spectral_shape, 0.001)
    compiled_step = eqx.filter_jit(
        lambda current_state, current_memory: advance_kadanoff_baym(
            plan,
            current_state,
            current_memory,
            runtime,
            collision_source=zeros,
            statistical_self_energy=zeros,
            real_retarded_propagator=zeros,
            memory_kernel=kernel,
            initial_correlation=initial,
        )
    )

    for _ in range(3):
        result = compiled_step(state, memory)
        assert bool(result.evidence.accepted)
        state, memory = result.accepted_state, result.accepted_memory

    assert int(memory.committed_samples) == 3
    assert int(jnp.sum(memory.valid)) == plan.memory_depth
    assert float(memory.discarded_tail_bound) > 0.0
    assert float(result.evidence.memory.initial_correlation_norm) > 0.0

    manifest_id = "a" * 64
    checkpoint = checkpoint_kadanoff_baym(
        plan, state, memory, epoch_manifest_id=manifest_id
    )
    restored_state, restored_memory = restore_kadanoff_baym(plan, checkpoint, runtime)
    np.testing.assert_array_equal(restored_state.occupation, state.occupation)
    np.testing.assert_array_equal(restored_memory.source_history, memory.source_history)
    np.testing.assert_array_equal(checkpoint.frame_token, off_shell.frame.frame_token)
    np.testing.assert_array_equal(checkpoint.frame_time, off_shell.frame.time)
    np.testing.assert_array_equal(
        checkpoint.frame_scale_factor, off_shell.frame.scale_factor
    )
    assert checkpoint.frame_realization_id == off_shell.frame.realization_id()
    tampered = eqx.tree_at(
        lambda value: value.frame_token,
        checkpoint,
        checkpoint.frame_token + 1,
    )
    with pytest.raises(ValueError, match="frame realization"):
        restore_kadanoff_baym(plan, tampered, runtime)

    next_runtime = empty_dark_sector_epoch_state(
        plan.epoch,
        epoch_sequence=1,
        parent_epoch_manifest_id=manifest_id,
    )
    continued_state, continued_memory = continue_kadanoff_baym_epoch(
        plan, checkpoint, next_runtime
    )
    np.testing.assert_array_equal(continued_state.occupation, state.occupation)
    np.testing.assert_array_equal(continued_memory.source_history, memory.source_history)
    assert int(continued_memory.epoch_sequence) == 1


def test_invalid_kb_candidate_rolls_back_state_and_memory_transactionally():
    off_shell, plan = _plans()
    runtime = empty_dark_sector_epoch_state(plan.epoch, epoch_sequence=0)
    state = breit_wigner_off_shell_state(
        off_shell,
        jnp.full((3, 1), 1.0),
        0.3,
        inverse_temperature=0.5,
        chemical_potentials=jnp.asarray((0.0,)),
    )
    memory = initialize_kadanoff_baym_memory(plan, runtime)
    result = advance_kadanoff_baym(
        plan,
        state,
        memory,
        runtime,
        collision_source=jnp.full(off_shell.spectral_shape, -100.0),
        statistical_self_energy=jnp.zeros(off_shell.spectral_shape),
        real_retarded_propagator=jnp.zeros(off_shell.spectral_shape),
        memory_kernel=jnp.zeros(off_shell.spectral_shape),
        initial_correlation=jnp.zeros(off_shell.spectral_shape),
    )

    assert bool(result.evidence.rolled_back)
    np.testing.assert_array_equal(result.accepted_state.occupation, state.occupation)
    np.testing.assert_array_equal(result.accepted_memory.valid, memory.valid)
