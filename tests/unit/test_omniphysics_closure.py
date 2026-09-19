#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np

import phydrax as phx


def test_closure_taxonomy_source_ledger_and_gap_matrix() -> None:
    catalog = phx.qualification.builtin_capability_catalog()
    ledger = phx.qualification.builtin_source_absorption_ledger()
    matrices = phx.qualification.builtin_omniphysics_closure_matrices(catalog)

    assert ledger.sources
    assert ledger.ledger_id
    assert matrices
    assert all(not matrix.unclassified_requirement_ids for matrix in matrices)
    assert all(matrix.matrix_id for matrix in matrices)


def test_material_state_history_and_homogenization() -> None:
    record = phx.materials.MaterialRecord.create("steel", "1", {"Fe": 0.98, "C": 0.02})
    state = phx.materials.MaterialState(300.0, 101325.0, jnp.asarray((0.7, 0.3)))
    history = phx.materials.MaterialHistory(
        (0.0, 1.0), (300.0, 400.0), ((1.0, 0.0), (0.5, 0.5))
    )

    assert record.record_id
    assert bool(state.admissible)
    assert history.history_id
    assert jnp.isclose(
        phx.materials.homogenize_scalar(
            jnp.asarray((100.0, 200.0)), jnp.asarray((0.5, 0.5)), "voigt"
        ),
        150.0,
    )
    assert jnp.isclose(phx.materials.jmak_fraction(0.0, 1.0, 2.0), 0.0)


def test_manufacturing_schedule_activation_source_and_history() -> None:
    event = phx.manufacturing.ToolpathEvent(
        "move-1", "move", 0.0, 1.0, "machine", (0.0, 0.0), (1.0, 0.0), power_w=100.0
    )
    schedule = phx.manufacturing.ProcessSchedule.create((event,))
    activation = phx.manufacturing.MaterialActivationState(
        jnp.asarray((True, False)), jnp.asarray((0.0, -1.0))
    ).activate(jnp.asarray((False, True)), 1.0)
    source = phx.manufacturing.GaussianMovingSource(100.0, 0.5, 0.1)
    history = phx.manufacturing.ProcessHistory(
        (0.0, 1.0), (0.0, 0.1), (0.0, 50.0), (1.0, 2.0)
    )

    assert schedule.schedule_id
    assert jnp.all(activation.active)
    assert source.evaluate(jnp.asarray(((0.0, 0.0),)), jnp.asarray((0.0, 0.0)))[0] > 0.0
    assert history.history_id


def test_frequency_network_passivity() -> None:
    axis = phx.frequency.FrequencyAxis((1.0, 2.0))
    network = phx.frequency.ScatteringMatrix(
        jnp.zeros((2, 2, 2), dtype=complex), jnp.asarray((50.0, 50.0))
    )

    assert jnp.allclose(axis.angular_frequency_rad_s, 2.0 * jnp.pi * axis.frequency_hz)
    assert bool(network.passive)
    assert jnp.isclose(network.reciprocal_error, 0.0)


def test_population_balance_growth_and_moments() -> None:
    plan = phx.population_balance.SectionalPopulationPlan((0.0, 1.0, 2.0, 3.0))
    density = jnp.asarray((1.0, 2.0, 1.0))

    assert bool(plan.realizable(density))
    assert plan.moments(density, jnp.asarray((0.0, 1.0))).shape == (2,)
    assert plan.growth_rate(density, jnp.ones_like(density)).shape == density.shape


def test_acausal_connections_and_structural_incidence() -> None:
    kind = phx.system_modeling.ConnectorType.create(
        "electrical",
        (
            phx.system_modeling.ConnectorVariable("voltage", "across", "V"),
            phx.system_modeling.ConnectorVariable("current", "through", "A"),
        ),
    )
    system = phx.system_modeling.AcausalSystem.create(
        (
            phx.system_modeling.Connector("a", kind),
            phx.system_modeling.Connector("b", kind),
        ),
        (phx.system_modeling.ConnectionSet.create(("a", "b")),),
    )
    residual = system.connection_residual(jnp.asarray(((1.0, 2.0), (1.0, 2.0))))
    _, rank = phx.system_modeling.structural_incidence(jnp.asarray(((1, 0), (1, 1))))

    assert jnp.allclose(residual, 0.0)
    assert rank == 2


def test_rheology_conformation_roundtrip_and_relaxation() -> None:
    conformation = jnp.asarray(((2.0, 0.1), (0.1, 1.5)))
    law = phx.rheology.ViscoelasticLaw("giesekus", 1.0, 2.0, mobility_factor=0.1)

    np.testing.assert_allclose(
        phx.rheology.exp_conformation(phx.rheology.log_conformation(conformation)),
        conformation,
        atol=1.0e-12,
    )
    assert law.stress(conformation).shape == conformation.shape
    assert law.free_energy_density(conformation) > 0.0


def test_interfacial_transport_controls() -> None:
    law = phx.interfacial_transport.LangmuirSurfactantLaw(0.072, 8.314, 300.0, 1.0e-6)
    wetting = phx.interfacial_transport.CoxVoinovWettingLaw(1.0, 1.0e-9, 1.0e-3)

    assert law.surface_tension(0.5e-6) < law.clean_surface_tension_n_m
    assert wetting.dynamic_angle(0.01) > wetting.equilibrium_angle_rad


def test_structural_and_correlation_controls() -> None:
    response = phx.structural_dynamics.harmonic_response(
        jnp.asarray(((2.0,),)),
        jnp.asarray(((1.0,),)),
        jnp.asarray(((0.1,),)),
        jnp.asarray((1.0,)),
        0.0,
    )
    mac = phx.correlation.modal_assurance_criterion(jnp.eye(2), jnp.eye(2))

    assert jnp.allclose(response, 0.5)
    assert jnp.allclose(mac, jnp.eye(2))


def test_ehd_phoresis_smart_and_chemo_couplings() -> None:
    field = jnp.asarray((2.0, 0.0))
    stress = phx.electrohydrodynamics.maxwell_stress(field, 3.0)
    velocity = phx.phoresis.smoluchowski_electrophoretic_velocity(field, 2.0, -0.1, 1.0)
    strain = phx.smart_materials.cubic_magnetostrictive_strain(
        jnp.asarray((1.0, 0.0, 0.0)), 1.0e-5
    )
    chemical = phx.chemo_mechanics.isotropic_chemical_strain(2.0, 1.0, 3.0e-6, 3)

    assert stress.shape == (2, 2)
    assert velocity[0] < 0.0
    assert jnp.isclose(jnp.trace(strain), 0.0, atol=1.0e-12)
    assert jnp.isclose(jnp.trace(chemical), 3.0e-6)


def test_tribology_thermal_membrane_surface_and_optical_controls() -> None:
    pressure = phx.tribology.reynolds_1d_pressure(
        jnp.asarray((2.0e-5, 1.5e-5, 1.0e-5)), 1.0e-3, 0.1, 1.0
    )
    radiosity, net = phx.thermal_systems.enclosure_radiosity(
        jnp.asarray((300.0, 300.0)),
        jnp.asarray((1.0, 1.0)),
        jnp.asarray(((0.0, 1.0), (1.0, 0.0))),
    )
    membrane = phx.membranes.reverse_osmosis_flux(1.0e-12, 1.0e6, 0.5e6)
    effectiveness = phx.surface_chemistry.spherical_pellet_effectiveness_factor(0.0)
    index = phx.optomechanics.thermo_optic_index(1.5, 310.0, 300.0, 1.0e-4)

    assert jnp.all(pressure >= 0.0)
    assert jnp.allclose(net, 0.0, atol=1.0e-8)
    assert membrane > 0.0
    assert jnp.isclose(effectiveness, 1.0)
    assert jnp.isclose(index, 1.501)


def test_acoustics_electrochemistry_and_process_controls() -> None:
    medium = phx.acoustics.AcousticMedium(1.2, 343.0)
    pressure = phx.acoustics.monopole_pressure(1.0e-6, 1.0, 1000.0, medium)
    current = phx.electrochemistry.butler_volmer_current_density(1.0, 0.01, 298.15)
    flash = phx.process_systems.isothermal_flash(
        jnp.asarray((0.5, 0.5)), jnp.asarray((2.0, 0.5))
    )

    assert jnp.isfinite(pressure)
    assert current > 0.0
    assert flash.material_balance_error < 1.0e-10
