#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np

from phydrax._physical import DimensionalScaleContract, RelativityScaleContract
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
from phydrax.particle_physics._bound_states import (
    DarkBoundStateLevel,
    DarkBoundStateSpectrum,
    evaluate_radiative_capture_balance,
    evaluate_thermal_bound_state_balance,
    RadiativeCapturePlan,
)
from phydrax.particle_physics._identity import ParticleCatalogReference
from phydrax.particle_physics._species import ParticleSpeciesTable
from phydrax.solver._dark_sector_epoch_runtime import DarkSectorEpochPlan
from phydrax.units import COULOMB


def _capture_plan():
    units = RelativisticUnitContract(
        RelativityScaleContract(DimensionalScaleContract.si(), 1, 1, 1, 1),
        RelativityConvention(metric_signature="mostly_minus"),
    )
    geometry = ADMGridGeometry(
        jnp.asarray(1.0),
        jnp.zeros(3),
        jnp.eye(3),
        jnp.eye(3),
        jnp.asarray(1.0),
        jnp.zeros((3, 3)),
        jnp.asarray(True),
        jnp.asarray(True),
        snapshot_token=jnp.asarray(3),
        chart_id="flat",
        convention_id=units.convention.convention_id,
        scale_id=units.scale.scale_id,
        topology_id="one-cell",
        geometry_lineage_id="flat",
    )
    chart = CoordinateChart("minkowski", ("t", "x", "y", "z"))
    tetrad = orthonormal_tetrad(
        minkowski_metric(chart, convention="mostly_minus"),
        jnp.eye(4),
        jnp.zeros(4),
        convention=units.convention,
        source_id="observer",
    )
    frame = LocalRelativisticFramePlan(
        geometry,
        tetrad,
        units,
        jnp.zeros(4),
        jnp.asarray(0.0),
        jnp.asarray(1.0),
        observer_id="observer",
        orientation_id="future-right-handed",
    )
    runtime = DarkSectorEpochPlan(
        packet_capacity=1,
        event_capacity=2,
        product_capacity=4,
        radiation_capacity=2,
        work_capacity=4,
        frontier_capacity=4,
        packet_width=1,
        event_width=8,
        product_width=4,
        radiation_width=4,
        work_width=8,
        frontier_width=8,
        species_revision_id="5" * 64,
        topology_revision_id="6" * 64,
    )
    catalog = ParticleCatalogReference(
        source_id="bound-test",
        provider_release="test",
        checksum="checksum",
        citation_url="https://example.test/bound",
    )
    species = ParticleSpeciesTable(
        jnp.asarray((10, -10)),
        jnp.asarray((5.0, 5.0)),
        jnp.asarray((1.0, -1.0)),
        catalog=catalog,
        energy_unit=units.energy_unit,
        charge_unit=COULOMB,
    )
    level = DarkBoundStateLevel(
        20,
        (10, -10),
        rest_energy=9.0,
        charge=0.0,
        degeneracy=1,
        radial_quantum_number=1,
        orbital_angular_momentum=0,
        spin_twice=0,
        level_label="dark-1s",
    )
    spectrum = DarkBoundStateSpectrum(
        runtime,
        species,
        units,
        frame,
        (level,),
        model_id="dark-coulomb",
        model_revision_id="dark-coulomb-r1",
        spectrum_source_id="analytic-control",
        production_evidence_ids=("spectrum-control",),
    )
    return RadiativeCapturePlan(
        spectrum,
        20,
        capture_coefficient=0.25,
        emitted_radiation_degeneracy=2,
        constituent_degeneracies=(2, 2),
        multipole_order=1,
        cross_section_unit_id="energy^-2",
        coefficient_source_id="dipole-control",
    )


def test_radiative_capture_and_photo_dissociation_obey_pointwise_detailed_balance():
    plan = _capture_plan()
    result = evaluate_radiative_capture_balance(plan, jnp.asarray((0.2, 1.0, 2.0)))
    assert jnp.all(result.kinematically_open)
    assert jnp.all(result.finite)
    assert jnp.all(result.capture_cross_section > 0.0)
    assert jnp.all(result.photo_dissociation_cross_section > 0.0)
    np.testing.assert_allclose(result.detailed_balance_residual, 0.0, atol=1e-12)
    np.testing.assert_allclose(
        result.photo_dissociation_cross_section,
        result.detailed_balance_factor * result.capture_cross_section,
        rtol=1e-7,
    )


def test_thermal_forward_reverse_rates_share_one_saha_equilibrium_ratio():
    plan = _capture_plan()
    result = evaluate_thermal_bound_state_balance(plan, jnp.asarray((0.1, 0.5, 1.0)))
    assert jnp.all(result.finite)
    assert jnp.all(result.equilibrium_ratio > 0.0)
    np.testing.assert_allclose(result.detailed_balance_residual, 0.0, atol=1e-12)
    np.testing.assert_allclose(
        result.photo_dissociation_rate,
        result.equilibrium_ratio * result.capture_rate_coefficient,
        rtol=1e-7,
    )
