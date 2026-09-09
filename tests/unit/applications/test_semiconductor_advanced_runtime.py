# Copyright © 2026 PHYDRA, Inc. All rights reserved.

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.applications import semiconductor as sc
from phydrax.meshing import MeshZone, MeshZoneRole


jax.config.update("jax_enable_x64", True)

Q = sc.ELEMENTARY_CHARGE_SI
EPS0 = sc.VACUUM_PERMITTIVITY_SI
REFERENCE = "synthetic common electronic datum"


def _bands(name, ec, ev, nc, nv, *, temperature_range=(200.0, 500.0)):
    return sc.BandThermodynamics(
        name,
        conduction_band_edge=ec * Q,
        valence_band_edge=ev * Q,
        conduction_density_of_states=nc,
        valence_density_of_states=nv,
        reference_temperature=300.0,
        temperature_range=temperature_range,
        energy_reference=REFERENCE,
        provenance="Independent synthetic aligned-band regression data",
    )


def _capacity(temperature_range=(200.0, 500.0)):
    return sc.ConstantLatticeHeatCapacity(
        1.6e6,
        reference_temperature=300.0,
        temperature_range=temperature_range,
        provenance="Synthetic constant volumetric heat capacity",
    )


def _material(name, bands, *, permittivity=11.7, carrier_energy=False):
    options = {}
    if carrier_energy:
        options = dict(
            electron_energy_transport=sc.CarrierEnergyTransport(
                bands,
                "electron",
                0.02,
                temperature_range=bands.temperature_range,
                provenance="Synthetic electron energy moment closure",
            ),
            hole_energy_transport=sc.CarrierEnergyTransport(
                bands,
                "hole",
                0.01,
                temperature_range=bands.temperature_range,
                provenance="Synthetic hole energy moment closure",
            ),
            electron_energy_relaxation=sc.CarrierEnergyRelaxation(
                bands,
                "electron",
                2e-13,
                temperature_range=bands.temperature_range,
                provenance="Synthetic electron-lattice relaxation",
            ),
            hole_energy_relaxation=sc.CarrierEnergyRelaxation(
                bands,
                "hole",
                3e-13,
                temperature_range=bands.temperature_range,
                provenance="Synthetic hole-lattice relaxation",
            ),
        )
    return sc.SemiconductorMaterial(
        name,
        permittivity=permittivity * EPS0,
        thermodynamics=bands,
        electron_mobility=0.12,
        hole_mobility=0.045,
        lattice_heat_capacity=_capacity(bands.temperature_range),
        lattice_thermal_conductivity=90.0,
        provenance="Synthetic material for conservation regressions",
        **options,
    )


def _contacts(support):
    return (
        sc.OhmicContact("left", support.boundary_patch("left", side="lower")),
        sc.OhmicContact("right", support.boundary_patch("right", side="upper")),
    )


def test_aligned_heterojunction_equilibrium_preserves_density_and_displacement_jumps():
    left = _material("left", _bands("left-bands", 0.56, -0.56, 2.8e25, 1.04e25))
    right = _material(
        "right", _bands("right-bands", 0.72, -0.38, 1.7e25, 8e24), permittivity=9.5
    )
    support = sc.TransportSupport.interval(np.linspace(0.0, 1e-6, 8), area=1e-12)
    zones = (
        MeshZone("left-zone", MeshZoneRole.MATERIAL, support.node_scope([0, 1, 2, 3])),
        MeshZone("right-zone", MeshZoneRole.MATERIAL, support.node_scope([4, 5, 6, 7])),
    )
    interface = sc.MaterialInterface(
        3, fraction=0.35, sheet_charge=3e-19, potential_jump=0.012
    )
    plan = sc.DevicePlan(
        support,
        materials=(
            sc.MaterialBinding(zones[0], left),
            sc.MaterialBinding(zones[1], right),
        ),
        contacts=_contacts(support),
        interfaces=(interface,),
    )
    device = sc.PreparedSemiconductorDevice(plan)
    point = device.equilibrium()
    electron, hole = device.densities(point.coordinates)
    electron_flux, hole_flux = device.edge_fluxes(point.coordinates)
    state = device.interface_states(point.coordinates)[0]
    assert bool(point.successful)
    np.testing.assert_array_equal(electron_flux, 0.0)
    np.testing.assert_array_equal(hole_flux, 0.0)
    assert not np.isclose(float(electron[3]), float(electron[4]), rtol=0.1)
    assert not np.isclose(float(hole[3]), float(hole[4]), rtol=0.1)
    np.testing.assert_allclose(
        state.potential_right - state.potential_left, interface.potential_jump, atol=2e-15
    )
    np.testing.assert_allclose(
        state.displacement_right - state.displacement_left,
        interface.sheet_charge,
        rtol=2e-12,
        atol=1e-31,
    )
    assert float(point.evidence.charge_balance_relative_error) < 1e-10


def test_thermionic_interface_adds_only_algebraic_traces_and_balances_at_equilibrium():
    left = _material("left", _bands("left-bands", 0.56, -0.56, 2.8e25, 1.04e25))
    right = _material("right", _bands("right-bands", 0.72, -0.38, 1.7e25, 8e24))
    support = sc.TransportSupport.interval(np.linspace(0.0, 1e-6, 8), area=1e-12)
    zones = (
        MeshZone("left-zone", MeshZoneRole.MATERIAL, support.node_scope(range(4))),
        MeshZone("right-zone", MeshZoneRole.MATERIAL, support.node_scope(range(4, 8))),
    )
    emission = sc.ThermionicInterface(
        2e23,
        temperature_range=(200.0, 500.0),
        energy_reference=REFERENCE,
        provenance="Synthetic reciprocal MB transmitting spectrum",
    )
    plan = sc.DevicePlan(
        support,
        materials=(
            sc.MaterialBinding(zones[0], left),
            sc.MaterialBinding(zones[1], right),
        ),
        contacts=_contacts(support),
        interfaces=(
            sc.MaterialInterface(
                3, fraction=0.5, electron_law=emission, hole_law=emission
            ),
        ),
    )
    device = sc.PreparedSemiconductorDevice(plan)
    point = device.equilibrium()
    assert plan.layout.shape == (3 * 8 + 4,)
    assert not bool(
        jnp.any(
            device.differential_mask[plan.layout.indices("interface_0_electron_left")]
        )
    )
    assert bool(point.successful)
    np.testing.assert_array_equal(device.edge_fluxes(point.coordinates)[0][3], 0.0)
    np.testing.assert_array_equal(device.edge_fluxes(point.coordinates)[1][3], 0.0)


def test_carrier_and_lattice_energy_storage_round_trip_uses_extensive_inventories():
    bands = _bands("energy-bands", 0.56, -0.56, 2.8e25, 1.04e25)
    material = _material("energy-material", bands, carrier_energy=True)
    support = sc.TransportSupport.interval(np.linspace(0.0, 1e-6, 7), area=1e-12)
    zone = MeshZone("material", MeshZoneRole.MATERIAL, support.node_scope())
    plan = sc.DevicePlan(
        support,
        materials=(sc.MaterialBinding(zone, material),),
        donor_density=1e20,
        contacts=_contacts(support),
        electrothermal=True,
        carrier_energy=True,
    )
    device = sc.PreparedSemiconductorDevice(plan)
    state = plan.equilibrium_coordinates()
    state = plan.layout.set(state, "lattice_energy", np.log(330.0 / 300.0))
    state = plan.layout.set(state, "electron_energy", np.log(360.0 / 300.0))
    state = plan.layout.set(state, "hole_energy", np.log(315.0 / 300.0))
    state = plan.layout.set(state, "electron", device.field(state, "electron") + 0.2)
    stored = device.storage_coordinates(state)
    recovered = device.coordinates_from_storage(stored)
    original_density = device.densities(state)
    recovered_density = device.densities(recovered)
    np.testing.assert_allclose(recovered_density[0], original_density[0], rtol=3e-11)
    np.testing.assert_allclose(recovered_density[1], original_density[1], rtol=3e-11)
    for original, restored in zip(
        device.temperatures(state), device.temperatures(recovered), strict=True
    ):
        np.testing.assert_allclose(restored, original, rtol=3e-11)
    physical = device.physical_storage(state)
    np.testing.assert_allclose(
        device.field(physical, "electron"),
        support.volumes * original_density[0],
        rtol=2e-14,
    )
    assert np.all(np.asarray(device.field(physical, "lattice_energy")) != 330.0)
    equilibrium = device.equilibrium()
    assert bool(equilibrium.successful)
    np.testing.assert_allclose(equilibrium.terminal_heat_flows, 0.0, atol=1e-24)
    assert float(equilibrium.evidence.energy_balance_relative_error) < 1e-10


def test_declared_quasi_fermi_high_field_law_reduces_only_nonequilibrium_flux():
    saturation = sc.LocalVelocitySaturation(
        1e5,
        2.0,
        reference_temperature=300.0,
        temperature_exponent=0.0,
        maximum_force=1e10,
        temperature_range=(250.0, 400.0),
        driving_force=sc.HighFieldDrivingForce.QUASI_FERMI_GRADIENT,
        orientation="synthetic longitudinal edge",
        provenance="Synthetic local velocity-saturation regression",
    )

    def prepare(electron_saturation):
        material = sc.SemiconductorMaterial(
            "high-field-material",
            permittivity=11.7 * EPS0,
            intrinsic_density=1e16,
            electron_mobility=0.12,
            hole_mobility=0.045,
            electron_saturation=electron_saturation,
            provenance="Synthetic high-field integration material",
        )
        support = sc.TransportSupport.interval(np.linspace(0.0, 1e-7, 7), area=1e-12)
        zone = MeshZone("material", MeshZoneRole.MATERIAL, support.node_scope())
        plan = sc.DevicePlan(
            support,
            materials=(sc.MaterialBinding(zone, material),),
            contacts=_contacts(support),
        )
        return sc.PreparedSemiconductorDevice(plan)

    low_field, high_field = prepare(None), prepare(saturation)
    equilibrium = high_field.plan.equilibrium_coordinates()
    np.testing.assert_array_equal(high_field.edge_fluxes(equilibrium)[0], 0.0)
    driven = high_field.layout.set(
        equilibrium, "electron", jnp.linspace(0.0, -50.0, high_field.num_nodes)
    )
    reference = low_field.layout.set(
        low_field.plan.equilibrium_coordinates(),
        "electron",
        jnp.linspace(0.0, -50.0, low_field.num_nodes),
    )
    saturated_flux = high_field.edge_fluxes(driven)[0]
    reference_flux = low_field.edge_fluxes(reference)[0]
    assert np.all(np.abs(np.asarray(saturated_flux)) < np.abs(np.asarray(reference_flux)))


def test_equilibrium_incomplete_ionization_is_rejected_by_dynamic_analyses():
    bands = _bands(
        "freezeout-bands", 0.56, -0.56, 2.8e25, 1.04e25, temperature_range=(50.0, 500.0)
    )
    ionization = sc.IncompleteIonization(
        donor_binding_energy=0.045 * Q,
        acceptor_binding_energy=0.057 * Q,
        donor_degeneracy=2.0,
        acceptor_degeneracy=4.0,
        provenance="Synthetic shallow-level equilibrium closure",
    )
    material = sc.SemiconductorMaterial(
        "freezeout-material",
        permittivity=11.7 * EPS0,
        thermodynamics=bands,
        incomplete_ionization=ionization,
        electron_mobility=0.12,
        hole_mobility=0.045,
        provenance="Synthetic freeze-out material",
    )
    support = sc.TransportSupport.interval(np.linspace(0.0, 1e-6, 7), area=1e-12)
    zone = MeshZone("material", MeshZoneRole.MATERIAL, support.node_scope())
    plan = sc.DevicePlan(
        support,
        materials=(sc.MaterialBinding(zone, material),),
        donor_density=1e21,
        contacts=_contacts(support),
        temperature=80.0,
    )
    device = sc.PreparedSemiconductorDevice(plan)
    point = device.equilibrium()
    donors, acceptors = device.ionized_dopants(point.coordinates)
    assert bool(point.successful)
    assert np.all(np.asarray(donors) < 1e21)
    np.testing.assert_array_equal(acceptors, 0.0)
    with pytest.raises(ValueError, match="explicit dynamic impurity populations"):
        sc.semiconductor_small_signal(device, point, [0.0])
