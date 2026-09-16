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
from phydrax.applications.curved_spacetime_qft._coherent_transport import (
    advance_coherent_transport,
    coherent_density_matrix_state,
    CoherentTransportPlan,
    LocalKrausCollisionMap,
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


def _units_and_frame():
    scale = RelativityScaleContract(DimensionalScaleContract.si(), 1, 3, 2, 1)
    units = RelativisticUnitContract(
        scale,
        RelativityConvention(metric_signature="mostly_minus"),
        spin_normalization="density-matrix-explicit",
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
        snapshot_token=jnp.asarray(1),
        chart_id="minkowski-cartesian",
        convention_id=units.convention.convention_id,
        scale_id=units.scale.scale_id,
        topology_id="coherent-test-cell",
        geometry_lineage_id="flat",
    )
    chart = CoordinateChart("minkowski", ("t", "x", "y", "z"))
    tetrad = orthonormal_tetrad(
        minkowski_metric(chart, convention="mostly_minus"),
        jnp.eye(4),
        jnp.zeros((4,)),
        convention=units.convention,
        tolerance=1.0e-7,
        source_id="coherent-test-observer",
    )
    return units, LocalRelativisticFramePlan(
        geometry,
        tetrad,
        units,
        jnp.zeros((4,)),
        jnp.asarray(0.0),
        jnp.asarray(1.0),
        observer_id="coherent-test-observer",
        orientation_id="right-handed-future",
    )


def _plan():
    units, frame = _units_and_frame()
    species = (
        DarkSectorSpeciesPlan(
            "oscillating-dark-fermion",
            1.0,
            charge_names=("dark-charge",),
            charges=(1.0,),
            mass_unit=units.scale.dimensional_scale.mass_unit.unit_id,
            energy_unit=units.energy_unit.unit_id,
        ),
    )
    support = QuantumKineticState(
        jnp.asarray([[[1.0]]]),
        None,
        species=species,
        statistics=jnp.asarray((-1,), dtype=jnp.int8),
        spatial_active=jnp.asarray((True,)),
        momentum_active=jnp.asarray((True,)),
        units=units,
        frame=frame,
    )
    return CoherentTransportPlan(
        support,
        jnp.asarray((1.0,)),
        jnp.asarray((1.0,)),
        momentum_quadrature_id="single-on-shell-mode",
        internal_dimension=2,
        trace_tolerance=2.0e-10,
    )


def _zero_hamiltonian():
    return jnp.zeros((2, 2), dtype=jnp.complex128)


def test_two_level_cayley_oscillation_preserves_psd_hermiticity_trace_and_charge():
    plan = _plan()
    density = jnp.asarray([[[[[1.0, 0.0], [0.0, 0.0]]]]], dtype=jnp.complex128)
    state = coherent_density_matrix_state(plan, density, time=0.0)
    omega = 1.7
    dt = 0.2
    hamiltonian = 0.5 * omega * jnp.asarray([[0.0, 1.0], [1.0, 0.0]])

    step = eqx.filter_jit(
        lambda current: advance_coherent_transport(
            plan,
            current,
            time_step=dt,
            vacuum_hamiltonian=hamiltonian,
            mean_field_hamiltonian=_zero_hamiltonian(),
            gravity_hamiltonian=_zero_hamiltonian(),
            gauge_hamiltonian=_zero_hamiltonian(),
        )
    )(state)

    cayley_angle = 2.0 * np.arctan(dt * omega / 4.0)
    expected_excited = np.sin(cayley_angle) ** 2
    np.testing.assert_allclose(
        step.state.density_matrix[0, 0, 0, 1, 1].real,
        expected_excited,
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    assert bool(step.accepted)
    assert bool(step.state.evidence.hermitian)
    assert bool(step.state.evidence.positive_semidefinite)
    np.testing.assert_allclose(step.state.evidence.trace_density, 1.0, atol=1.0e-12)
    np.testing.assert_allclose(step.state.evidence.charge_by_cell, [[1.0]], atol=1.0e-12)
    assert step.state.frame_id == plan.frame.frame_id
    assert step.state.unit_contract_id == plan.units.contract_id
    np.testing.assert_array_equal(step.state.frame_token, plan.frame.frame_token)
    np.testing.assert_array_equal(step.state.frame_time, plan.frame.time)
    np.testing.assert_array_equal(step.state.frame_scale_factor, plan.frame.scale_factor)
    assert step.state.frame_realization_id == plan.frame.realization_id()


def test_local_kraus_dephasing_is_completely_positive_and_trace_preserving():
    plan = _plan()
    plus = 0.5 * jnp.ones((2, 2), dtype=jnp.complex128)
    state = coherent_density_matrix_state(plan, plus[None, None, None], time=0.0)
    probability = 0.2
    operators = jnp.stack(
        (
            jnp.sqrt(1.0 - probability) * jnp.eye(2),
            jnp.sqrt(probability) * jnp.diag(jnp.asarray((1.0, -1.0))),
        )
    ).astype(jnp.complex128)
    collision = LocalKrausCollisionMap(
        operators,
        jnp.asarray((True, True)),
        internal_dimension=2,
        map_id="exact-dephasing-channel",
    )

    step = advance_coherent_transport(
        plan,
        state,
        time_step=0.1,
        vacuum_hamiltonian=_zero_hamiltonian(),
        mean_field_hamiltonian=_zero_hamiltonian(),
        gravity_hamiltonian=_zero_hamiltonian(),
        gauge_hamiltonian=_zero_hamiltonian(),
        collision=collision,
    )

    np.testing.assert_allclose(
        step.state.density_matrix[0, 0, 0, 0, 1],
        0.5 * (1.0 - 2.0 * probability),
        atol=1.0e-12,
    )
    np.testing.assert_allclose(step.state.evidence.trace_density, 1.0, atol=1.0e-12)
    assert bool(step.state.evidence.positive_semidefinite)
    assert bool(step.accepted)


def test_coherent_transaction_rolls_back_a_candidate_outside_plan_conservation_policy():
    plan = _plan()
    state = coherent_density_matrix_state(
        plan,
        jnp.asarray([[[[[1.0, 0.0], [0.0, 0.0]]]]]),
        time=0.0,
    )
    gain = LocalKrausCollisionMap(
        jnp.asarray([jnp.sqrt(1.0 + 1.0e-8) * jnp.eye(2)]),
        jnp.asarray((True,)),
        internal_dimension=2,
        trace_tolerance=1.0e-6,
        map_id="relaxed-provider-policy",
    )
    step = advance_coherent_transport(
        plan,
        state,
        time_step=0.1,
        vacuum_hamiltonian=_zero_hamiltonian(),
        mean_field_hamiltonian=_zero_hamiltonian(),
        gravity_hamiltonian=_zero_hamiltonian(),
        gauge_hamiltonian=_zero_hamiltonian(),
        collision=gain,
    )

    assert bool(step.rolled_back)
    assert not bool(step.accepted)
    np.testing.assert_array_equal(step.state.density_matrix, state.density_matrix)
    assert not np.array_equal(
        np.asarray(step.candidate_state.density_matrix),
        np.asarray(state.density_matrix),
    )


def test_non_psd_state_and_non_trace_preserving_collision_fail_without_projection():
    plan = _plan()
    with pytest.raises(Exception, match="positive semidefinite"):
        coherent_density_matrix_state(
            plan,
            jnp.asarray([[[[[1.1, 0.0], [0.0, -0.1]]]]]),
            time=0.0,
        )
    with pytest.raises(Exception, match="trace preserving"):
        LocalKrausCollisionMap(
            jnp.asarray([2.0 * jnp.eye(2)]),
            jnp.asarray((True,)),
            internal_dimension=2,
            map_id="invalid-gain-map",
        )
