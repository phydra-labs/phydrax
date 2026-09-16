#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np

from phydrax._physical import DimensionalScaleContract, RelativityScaleContract
from phydrax.applications.cosmology._dark_sector_species import DarkSectorSpeciesPlan
from phydrax.applications.cosmology._quantum_dark_kinetics import QuantumKineticState
from phydrax.applications.curved_spacetime_qft._coherent_transport import (
    coherent_density_matrix_state,
    coherent_transport_observables,
    CoherentTransportPlan,
)
from phydrax.applications.curved_spacetime_qft._off_shell_transport import (
    breit_wigner_off_shell_state,
    off_shell_moments,
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


def _support(occupancy):
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
        snapshot_token=jnp.asarray(5),
        chart_id="minkowski-cartesian",
        convention_id=units.convention.convention_id,
        scale_id=units.scale.scale_id,
        topology_id="coherent-off-shell-cell",
        geometry_lineage_id="flat",
    )
    chart = CoordinateChart("minkowski", ("t", "x", "y", "z"))
    tetrad = orthonormal_tetrad(
        minkowski_metric(chart, convention="mostly_minus"),
        jnp.eye(4),
        jnp.zeros((4,)),
        convention=units.convention,
        tolerance=1.0e-7,
        source_id="coherent-off-shell-observer",
    )
    frame = LocalRelativisticFramePlan(
        geometry,
        tetrad,
        units,
        jnp.zeros((4,)),
        jnp.asarray(0.0),
        jnp.asarray(1.0),
        observer_id="coherent-off-shell-observer",
        orientation_id="right-handed-future",
    )
    species = (
        DarkSectorSpeciesPlan(
            "coherent-off-shell-fermion",
            1.0,
            charge_names=("dark-charge",),
            charges=(2.0,),
            mass_unit=units.scale.dimensional_scale.mass_unit.unit_id,
            energy_unit=units.energy_unit.unit_id,
        ),
    )
    return QuantumKineticState(
        jnp.asarray([[[occupancy]]]),
        None,
        species=species,
        statistics=jnp.asarray((-1,), dtype=jnp.int8),
        spatial_active=jnp.asarray((True,)),
        momentum_active=jnp.asarray((True,)),
        units=units,
        frame=frame,
    )


def test_narrow_width_off_shell_observable_matches_coherent_on_shell_trace():
    target_occupancy = 0.3
    support = _support(target_occupancy)
    coherent_plan = CoherentTransportPlan(
        support,
        jnp.asarray((1.0,)),
        jnp.asarray((1.0,)),
        momentum_quadrature_id="single-mode",
        internal_dimension=2,
    )
    coherent_density = jnp.asarray([[[[[0.2, 0.05], [0.05, 0.1]]]]], dtype=jnp.complex128)
    coherent = coherent_density_matrix_state(coherent_plan, coherent_density, time=0.0)
    four_momentum = jnp.asarray([[[2.0, 0.25, 0.0, 0.0]]])
    coherent_observables = coherent_transport_observables(
        coherent_plan, coherent, four_momentum
    )

    energy_nodes = np.linspace(-18.0, 22.0, 8001)
    spacing = energy_nodes[1] - energy_nodes[0]
    energy_weights = np.full(energy_nodes.shape, spacing)
    energy_weights[[0, -1]] *= 0.5
    off_shell_plan = OffShellTransportPlan(
        support,
        energy_nodes,
        energy_weights,
        jnp.asarray((1.0,)),
        jnp.asarray((1.0,)),
        energy_quadrature_id="narrow-width-trapezoid",
        momentum_quadrature_id="single-mode",
        spectral_tolerance=2.0e-3,
    )
    beta = 0.5
    chemical_potential = 2.0 - np.log(1.0 / target_occupancy - 1.0) / beta
    off_shell = breit_wigner_off_shell_state(
        off_shell_plan,
        jnp.asarray([[2.0]]),
        0.025,
        inverse_temperature=beta,
        chemical_potentials=jnp.asarray((chemical_potential,)),
    )
    off_shell_observables = off_shell_moments(off_shell_plan, off_shell, four_momentum)

    np.testing.assert_allclose(
        off_shell_observables.number_density,
        coherent_observables.number_density,
        atol=1.0e-3,
    )
    np.testing.assert_allclose(
        off_shell_observables.charge_density,
        coherent_observables.charge_density,
        atol=2.0e-3,
    )
    np.testing.assert_allclose(
        off_shell_observables.stress_energy,
        coherent_observables.stress_energy,
        atol=2.0e-3,
    )
    assert off_shell.frame_id == coherent.frame_id == support.frame_id
    assert (
        off_shell.unit_contract_id
        == coherent.unit_contract_id
        == support.unit_contract_id
    )
