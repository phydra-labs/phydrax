#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax.numpy as jnp
import numpy as np

from phydrax._physical import DimensionalScaleContract, RelativityScaleContract
from phydrax.applications.cosmology._dark_sector_species import DarkSectorSpeciesPlan
from phydrax.applications.cosmology._quantum_dark_kinetics import (
    CondensateCouplingPlan,
    equilibrium_occupancy,
    QuantumDarkKineticsPlan,
    QuantumKineticStatus,
)
from phydrax.applications.relativistic_scattering._unit_contract import (
    LocalRelativisticFramePlan,
    RelativisticUnitContract,
)
from phydrax.equations._uehling_uhlenbeck import UehlingUhlenbeckPlan
from phydrax.metrix import (
    ADMGridGeometry,
    CoordinateChart,
    minkowski_metric,
    orthonormal_tetrad,
    RelativityConvention,
)


def _units_and_frame():
    scale = RelativityScaleContract(DimensionalScaleContract.si(), 1, 1, 1, 1)
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
        snapshot_token=jnp.asarray(1),
        chart_id="minkowski-cartesian",
        convention_id=units.convention.convention_id,
        scale_id=units.scale.scale_id,
        topology_id="single-cell",
        geometry_lineage_id="flat",
    )
    chart = CoordinateChart("minkowski", ("t", "x", "y", "z"))
    tetrad = orthonormal_tetrad(
        minkowski_metric(chart, convention="mostly_minus"),
        jnp.eye(4),
        jnp.zeros((4,)),
        convention=units.convention,
        tolerance=1.0e-7,
        source_id="quantum-test-observer",
    )
    frame = LocalRelativisticFramePlan(
        geometry,
        tetrad,
        units,
        jnp.zeros((4,)),
        jnp.asarray(0.0),
        jnp.asarray(1.0),
        observer_id="quantum-test-observer",
        orientation_id="right-handed-future",
    )
    return units, frame


def _collision(time_unit_id, statistics=(1,)):
    return UehlingUhlenbeckPlan(
        jnp.asarray(statistics, dtype=jnp.int8),
        jnp.ones((len(statistics), 1)),
        jnp.zeros((1, 4), dtype=jnp.int32),
        jnp.zeros((1, 4), dtype=jnp.int32),
        jnp.zeros((1,)),
        jnp.broadcast_to(jnp.asarray((1.0, 0.0, 0.0, 0.0)), (len(statistics), 1, 4)),
        jnp.zeros((len(statistics), 0)),
        time_unit_id=time_unit_id,
        event_active=jnp.asarray((False,)),
    )


def _plan(*, coupling=True, threshold=0.5):
    units, frame = _units_and_frame()
    species = (
        DarkSectorSpeciesPlan(
            "dark-boson",
            1.0,
            mass_unit=units.scale.dimensional_scale.mass_unit.unit_id,
            energy_unit=units.energy_unit.unit_id,
        ),
    )
    condensate = (
        CondensateCouplingPlan(
            0,
            0,
            0.5,
            2.0,
            qualification_id="condensate-reference-evidence",
            model_id="single-mode-number-conserving-relaxation",
        )
        if coupling
        else None
    )
    return QuantumDarkKineticsPlan(
        species,
        units,
        frame,
        _collision(units.scale.dimensional_scale.time_unit.unit_id),
        condensate_coupling=condensate,
        condensation_threshold=threshold,
    )


def test_state_has_explicit_support_units_frame_and_separate_condensate():
    plan = _plan()
    state = plan.initialize(
        jnp.asarray([[[0.4], [0.0]]]),
        spatial_active=jnp.asarray((True, False)),
        momentum_active=jnp.asarray((True,)),
        condensate_density=jnp.asarray(((0.2, 0.0),)),
    )

    assert state.occupancy.shape == (1, 2, 1)
    assert state.condensate_density.shape == (1, 2)
    assert state.unit_contract_id == plan.units.contract_id
    assert state.frame_id == plan.frame.frame_id
    assert state.frame_realization_id == plan.frame.realization_id()
    np.testing.assert_array_equal(state.frame_token, plan.frame.frame_token)
    assert state.frame_scope == "homogeneous-frame-broadcast"
    assert state.species_plan_ids == (plan.species[0].species_plan_id,)


def test_qualified_condensate_transition_preserves_number_and_is_reported():
    plan = _plan()
    state = plan.initialize(
        jnp.asarray([[[0.8]]]),
        spatial_active=jnp.asarray((True,)),
        momentum_active=jnp.asarray((True,)),
        condensate_density=jnp.asarray(((0.0,),)),
    )
    result = eqx.filter_jit(plan.advance)(state, 1.0)

    assert bool(result.successful)
    assert int(result.evidence.status) == int(QuantumKineticStatus.SUCCESS)
    assert float(result.accepted_state.occupancy[0, 0, 0]) < 0.8
    assert float(result.accepted_state.condensate_density[0, 0]) > 0.0
    np.testing.assert_allclose(result.evidence.number_defect, 0.0, atol=1.0e-7)
    np.testing.assert_array_equal(result.accepted_state.time, plan.frame.time)
    assert result.evidence.qualification_id == "condensate-reference-evidence"


def test_condensation_without_qualified_coupling_is_refused_and_rolled_back():
    plan = _plan(coupling=False)
    state = plan.initialize(
        jnp.asarray([[[0.8]]]),
        spatial_active=jnp.asarray((True,)),
        momentum_active=jnp.asarray((True,)),
    )
    result = plan.advance(state, 0.1)

    assert not bool(result.successful)
    assert bool(result.evidence.condensate_required)
    assert int(result.evidence.status) == int(QuantumKineticStatus.CONDENSATE_REQUIRED)
    np.testing.assert_array_equal(result.accepted_state.occupancy, state.occupancy)


def test_equilibrium_helper_keeps_be_fd_and_classical_conventions_distinct():
    units, _ = _units_and_frame()
    energy = jnp.asarray(((2.0,), (2.0,), (2.0,)))
    values = equilibrium_occupancy(
        energy,
        1.0,
        jnp.asarray((0.0, 0.0, 0.0)),
        jnp.asarray((1, -1, 0), dtype=jnp.int8),
        units,
    )

    np.testing.assert_allclose(values[0], 1.0 / np.expm1(2.0))
    np.testing.assert_allclose(values[1], 1.0 / (np.exp(2.0) + 1.0))
    np.testing.assert_allclose(values[2], np.exp(-2.0))
    scaled_units = RelativisticUnitContract(
        RelativityScaleContract(DimensionalScaleContract.si(), 1, 1, 1, 2),
        RelativityConvention(metric_signature="mostly_minus"),
    )
    scaled = equilibrium_occupancy(
        jnp.asarray(((2.0,),)),
        1.0,
        jnp.asarray((0.0,)),
        jnp.asarray((0,), dtype=jnp.int8),
        scaled_units,
    )
    np.testing.assert_allclose(scaled, np.exp(-1.0))
