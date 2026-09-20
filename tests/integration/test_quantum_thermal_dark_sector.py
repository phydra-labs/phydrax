#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np

from phydrax._physical import DimensionalScaleContract, RelativityScaleContract
from phydrax.applications.cosmology._dark_sector_species import DarkSectorSpeciesPlan
from phydrax.applications.cosmology._quantum_dark_kinetics import (
    QuantumDarkKineticsPlan,
)
from phydrax.applications.cosmology._thermal_dark_rates import (
    HTLPolarizationPlan,
    LPMIntegralPlan,
    ThermalDarkRatePlan,
    ThermalKernelArtifact,
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


def _context():
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
        snapshot_token=jnp.asarray(3),
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
        source_id="integrated-thermal-observer",
    )
    frame = LocalRelativisticFramePlan(
        geometry,
        tetrad,
        units,
        jnp.zeros((4,)),
        jnp.asarray(0.0),
        jnp.asarray(1.0),
        observer_id="integrated-thermal-observer",
        orientation_id="right-handed-future",
    )
    return units, frame


def _thermal_artifact(units, frame, species_plan_ids):
    temperature = jnp.asarray((1.0, 2.0, 3.0, 4.0))
    momentum = jnp.asarray((0.0, 1.0))
    frequency = jnp.asarray((-1.0, 0.0, 1.0))
    spectral_shape = (4, 4, 2, 3)
    pressure = temperature**2
    entropy = 2.0 * temperature
    return ThermalKernelArtifact(
        temperature,
        momentum,
        frequency,
        jnp.ones((4, 4)),
        0.1 * jnp.ones((4, 4)),
        0.1j * jnp.ones(spectral_shape),
        jnp.ones(spectral_shape),
        jnp.ones((4, 2, 3), dtype="complex128"),
        jnp.ones((4, 2, 3), dtype="complex128"),
        temperature[None, :],
        pressure,
        temperature * entropy - pressure,
        entropy,
        jnp.eye(12),
        units,
        frame,
        species_plan_ids=species_plan_ids,
        rate_channel_ids=("integrated-2to2-rate",),
        source_kind="native-analytic",
        thermodynamic_tolerance=1.0e-6,
    )


def test_quantum_collision_thermal_screening_and_lpm_share_exact_units_and_frame():
    units, frame = _context()
    species = tuple(
        DarkSectorSpeciesPlan(
            name,
            np.sqrt(3.0),
            charge_names=("dark",),
            charges=(charge,),
            mass_unit=units.scale.dimensional_scale.mass_unit.unit_id,
            energy_unit=units.energy_unit.unit_id,
        )
        for name, charge in (("a", 1.0), ("b", -1.0), ("c", 1.0), ("d", -1.0))
    )
    thermal = _thermal_artifact(
        units,
        frame,
        tuple(value.species_plan_id for value in species),
    )
    rate_state = ThermalDarkRatePlan(thermal).evaluate(2.0)
    kernel = float(rate_state.rates[0])
    collision = UehlingUhlenbeckPlan(
        jnp.asarray((-1, -1, -1, -1), dtype=jnp.int8),
        jnp.ones((4, 1)),
        jnp.asarray(((0, 1, 2, 3),), dtype=jnp.int32),
        jnp.zeros((1, 4), dtype=jnp.int32),
        jnp.asarray((kernel,)),
        jnp.asarray(
            (
                ((2.0, 1.0, 0.0, 0.0),),
                ((2.0, -1.0, 0.0, 0.0),),
                ((2.0, 0.0, 1.0, 0.0),),
                ((2.0, 0.0, -1.0, 0.0),),
            )
        ),
        jnp.asarray(((1.0,), (-1.0,), (1.0,), (-1.0,))),
        time_unit_id=units.scale.dimensional_scale.time_unit.unit_id,
        invariant_tolerance=1.0e-6,
        entropy_tolerance=1.0e-6,
    )
    kinetics = QuantumDarkKineticsPlan(species, units, frame, collision)
    state = kinetics.initialize(
        jnp.asarray((0.8, 0.7, 0.1, 0.2)).reshape((4, 1, 1)),
        spatial_active=jnp.asarray((True,)),
        momentum_active=jnp.asarray((True,)),
    )
    advanced = kinetics.advance(state, 0.02)
    htl = HTLPolarizationPlan(2.0, units, frame).evaluate(0.5, 1.0)
    lpm = LPMIntegralPlan(
        jnp.asarray((0.0,)),
        jnp.asarray((1.0,)),
        jnp.asarray(((2.0,),)),
        jnp.asarray((1.0,)),
        jnp.asarray((1.0,)),
        units,
        frame,
        rate_prefactor=1.0,
    ).solve()

    assert thermal.unit_contract_id == state.unit_contract_id == units.contract_id
    assert thermal.frame_id == state.frame_id == frame.frame_id
    assert (
        thermal.frame_realization_id
        == state.frame_realization_id
        == frame.realization_id()
    )
    np.testing.assert_array_equal(thermal.frame_token, state.frame_token)
    assert thermal.species_plan_ids == state.species_plan_ids
    assert bool(rate_state.valid)
    assert bool(advanced.successful)
    assert bool(htl.evidence.valid)
    assert bool(htl.evidence.landau_damping_support)
    assert bool(lpm.successful)
    np.testing.assert_allclose(
        advanced.evidence.collision.charge_defect, 0.0, atol=2.0e-7
    )
    np.testing.assert_allclose(
        advanced.evidence.collision.four_momentum_defect, 0.0, atol=2.0e-7
    )
    assert float(jnp.min(advanced.evidence.collision.entropy_production)) >= -1.0e-7
