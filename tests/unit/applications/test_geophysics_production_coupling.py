#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np

import phydrax as phx
from phydrax.applications import geophysics as geo, porous_media as porous
from phydrax.uq import ParameterSpace, TemporalDifferencePrior


def _geometry():
    return phx.discretization.UnstructuredFiniteVolumePlan(
        np.asarray(
            ((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), (0, 0, -1)),
            dtype=float,
        ),
        tetrahedra=np.asarray(((0, 1, 2, 3), (0, 2, 1, 4))),
    ).prepare()


def test_multiphase_flash_hysteresis_freezing_and_well_controls_are_physical():
    geometry = _geometry()
    plan = porous.MultiphaseConservationPlan(
        geometry, ("liquid", "gas"), ("water", "air")
    )
    saturation = jnp.asarray(((0.7, 0.3), (0.8, 0.2)))
    density = jnp.asarray(((1000.0, 1.0), (1000.0, 1.0)))
    composition = jnp.asarray((((1.0, 0.0), (0.0, 1.0)), ((1.0, 0.0), (0.0, 1.0))))
    state = plan.state(
        saturation,
        density,
        composition,
        jnp.asarray(((1e5, 2e5), (1e5, 2e5))),
        1e6,
        0.3,
    )
    flux = plan.fluxes(
        jnp.zeros((geometry.face_measures.size, 2)),
        jnp.ones((geometry.face_measures.size, 2)),
        jnp.broadcast_to(
            jnp.asarray(((1.0, 0.0), (0.0, 1.0))),
            (geometry.face_measures.size, 2, 2),
        ),
        jnp.zeros((geometry.face_measures.size, 2)),
    )
    residual = plan.residual(state, state, 1.0, flux)
    np.testing.assert_allclose(residual.component_kg_s, 0.0)
    np.testing.assert_allclose(residual.energy_W, 0.0)

    flash = porous.RachfordRiceFlashPlan(("light", "heavy"))
    flashed = flash.solve([0.5, 0.5], [2.0, 0.5])
    assert flashed.successful
    assert 0 < flashed.vapor_fraction < 1
    np.testing.assert_allclose(jnp.sum(flashed.liquid_composition), 1.0)
    np.testing.assert_allclose(jnp.sum(flashed.vapor_composition), 1.0)

    drainage = porous.BrooksCoreyRetention(1000.0, 2.0, 0.1, 0.05)
    imbibition = porous.BrooksCoreyRetention(500.0, 2.5, 0.1, 0.05)
    hysteresis = porous.HystereticRetentionPlan(drainage, imbibition)
    path = hysteresis.initialize(1000.0, branch="drainage")
    reversed_path = hysteresis.update(hysteresis.update(path, 2000.0), 1500.0)
    assert not reversed_path.derivative_available
    assert 0 <= reversed_path.saturation <= 1

    freezing = porous.FreezeThawMaterial(mushy_width_K=2.0)
    assert freezing.liquid_fraction(270.0) == 0
    assert freezing.liquid_fraction(276.0) == 1
    vapor = porous.VaporEquilibrium()
    assert vapor.saturation_pressure(280.0) > vapor.saturation_pressure(270.0)

    well = porous.WellCompletionPlan(
        [0, 1],
        [1e-12, 2e-12],
        [[1.0, 0.0], [1.0, 0.0]],
        [1000.0, 1000.0],
        [1e5, 1e5],
        2,
    )
    well_result = well.evaluate(
        [1e5, 1.1e5], [1e-6, 1e-6], porous.WellControl("rate", 1e-8)
    )
    assert well_result.successful
    np.testing.assert_allclose(jnp.sum(well_result.completion_volume_rate_m3_s), 1e-8)


def test_surface_flow_biot_fault_and_geodetic_observations_close_balances():
    surface = porous.UnstructuredShallowWaterPlan(
        [1.0, 1.0],
        [0.0, 0.0],
        [0],
        [1],
        [[1.0, 0.0]],
        [1.0],
    )
    state = surface.initial_state([0.2, 0.2])
    stepped = surface.step(state, 0.01)
    np.testing.assert_allclose(stepped.state.water_volume_m3, state.water_volume_m3)
    np.testing.assert_allclose(stepped.volume_balance_m3, 0.0, atol=1e-14)

    displacement = phx.linalg.ArraySpace((1,))
    pressure = phx.linalg.ArraySpace((1,))
    elasticity = phx.linalg.DenseLinearOperator(
        [[2.0]], source=displacement, target=displacement
    )
    coupling = phx.linalg.DenseLinearOperator(
        [[1.0]], source=displacement, target=pressure
    )
    storage = phx.linalg.DenseLinearOperator([[1.0]], source=pressure, target=pressure)
    flow = phx.linalg.DenseLinearOperator([[1.0]], source=pressure, target=pressure)
    biot = geo.MixedBiotPoromechanicsPlan(elasticity, coupling, storage, flow, 0.5)
    biot_result = biot.step(
        biot.initial_state(), 1.0, mechanical_load=[1.0], fluid_source=[0.2]
    )
    assert biot_result.successful
    assert biot_result.residual_norm < 1e-8

    fault = geo.RateStateFaultLaw(0.6, 0.01, 0.015, 0.01, 1e-6)
    fault_result = fault.step(fault.initialize(), 1e-6, 20e6, 5e6, 1.0)
    assert fault_result.dissipated_power_W_m2 >= 0
    contact = geo.CoulombContactLaw(1e9, 1e9, 0.6).evaluate(-1e-3, [2e-3])
    assert contact.active_contact
    assert jnp.abs(contact.shear_traction_Pa[0]) <= 0.6 * contact.normal_traction_Pa
    steady_fault = fault.step(fault.initialize(), 1e-6, 1e6, 0.0, 1e-6)
    cycle = geo.EarthquakeCyclePlan(
        [[1e6]],
        [0.0],
        [1e6],
        [1e6],
        fault,
    )
    cycle_state = cycle.initialize(
        steady_fault.shear_traction_Pa + 1.0,
        slip_rate_m_s=1e-6,
    )
    cycle_step = cycle.step(cycle_state, 0.1)
    assert cycle_step.successful
    assert jnp.max(jnp.abs(cycle_step.residual)) < 1e-6

    forward = geo.GeodeticDeformationObservationPlan(
        np.eye(3), np.asarray(((1.0, 0.0, 0.0),)), np.zeros((0, 3)), np.zeros((0, 3)), 3
    )
    prediction = forward.predict([1.0, 2.0, 3.0])
    np.testing.assert_allclose(prediction.gnss_displacement_m, [[1.0, 2.0, 3.0]])
    np.testing.assert_allclose(prediction.insar_los_displacement_m, [1.0])


def test_advanced_chemistry_and_fracture_network_are_conservative():
    sit = porous.SITActivityModel([1.0, -1.0], [[0.0, 0.01], [0.01, 0.0]])
    pitzer = porous.PitzerInteractionModel(
        [1.0, -1.0],
        [[0.0, 0.1], [0.1, 0.0]],
        [[0.0, 0.05], [0.05, 0.0]],
        [[0.0, 0.001], [0.001, 0.0]],
    )
    molality = jnp.asarray((1.0, 1.0))
    assert jnp.all(sit.activity_coefficients(molality) > 0)
    assert jnp.all(pitzer.activity_coefficients(molality) > 0)
    redox = porous.RedoxEquilibrium(1.0, 0.2)
    assert jnp.isfinite(redox.potential(298.15, 2.0, 1.0))
    assert porous.HenryGasEquilibrium(1e-5).dissolved_concentration(1e5) == 1.0

    network = porous.MixedDimensionalFractureNetworkPlan(
        [0.1, 0.1],
        [0.05],
        [[0, 1]],
        [1e-8],
        [[0, 0], [1, 0]],
        [1e-8, 1e-8],
        [0, 1],
        [1e-8, 1e-8],
        2,
        1,
    )
    fracture_state = network.initial_state(
        jnp.asarray(((1.0,), (1.0,))),
        jnp.asarray(((0.5,),)),
        jnp.asarray((1e5, 1e5)),
        jnp.asarray((0.5e5,)),
    )
    result = network.step(
        fracture_state,
        1.0,
        [1e5, 0.9e5],
        [0.95e5],
        [1.1e5, 0.8e5],
        [[10.0], [10.0]],
        [[10.0]],
        [[10.0], [10.0]],
        [1e6, 1e6],
        [1e6],
        [1e6, 1e6],
    )
    assert result.successful
    np.testing.assert_allclose(result.component_balance_kg, 0.0, atol=1e-12)
    np.testing.assert_allclose(result.energy_balance_J, 0.0, atol=1e-8)


def test_joint_petrophysical_geological_and_time_lapse_workflows_are_explicit():
    surface = geo.SurfaceConductionConductivity(1.0, 2.0, 2.0, 0.1, 1.0, 0.1)
    assert surface.predict(0.3, 0.8, 0.2).mean > 0
    crim = geo.CRIMPermittivity(4.0, 80.0, 1.0, 0.2)
    assert crim.predict(0.3, 0.8).mean > 1
    geology = geo.LevelSetInterfacePlan([[1.0, 0.0], [1.0, 1.0]], smoothing_width=0.1)
    fractions = geology.phase_fraction([0.0, 1.0])
    assert jnp.all((fractions > 0) & (fractions < 1))

    parameter_space = ParameterSpace(
        jnp.asarray((0.0,)), log_prior=lambda value: -0.5 * value[0] ** 2
    )
    term = geo.IndependentModalityTerm(
        "toy",
        lambda value: -0.5 * (value[0] - 1.0) ** 2,
        lambda value: value,
        "toy-data",
        likelihood_id="toy-gaussian-v1",
        prediction_id="toy-identity-v1",
    )
    joint = geo.MultimodalJointInferencePlan(parameter_space, (term,))
    assert joint.posterior().log_density(jnp.asarray((0.5,))) < 0

    temporal = geo.TimeLapseParameterization(
        3, 1, TemporalDifferencePrior([0.0, 1.0, 2.0], 1.0)
    )
    history = temporal.reconstruct([2.0], [[0.5], [-0.2]])
    np.testing.assert_allclose(history[:, 0], [2.0, 2.5, 2.3])
    assert jnp.isfinite(temporal.log_prior(history))


def test_planetary_potential_and_radial_thermal_energy_are_finite():
    body = phx.interchange.ReferenceBodyContract("body", 4e14, 6.4e6, 6.4e6, 0.0, 0.0)
    model = geo.RadialBodyModel(
        [0.0, 3.2e6, 6.4e6],
        [5000.0, 4000.0, 3000.0],
        [8000.0, 7000.0, 6000.0],
        [4000.0, 3500.0, 3000.0],
        [4000.0, 2000.0, 300.0],
        [1000.0, 1000.0, 1000.0],
        [4.0, 4.0, 4.0],
        body,
    )
    potential = geo.PlanetaryPotentialPlan(body).potential([6.4e6, 0.0, 0.0])
    np.testing.assert_allclose(potential, body.gravitational_parameter_m3_s2 / 6.4e6)
    thermal = geo.RadialThermalConductionPlan(model)
    result = thermal.step(model.temperature_K, 1.0, surface_temperature_K=300.0)
    assert result.successful
    assert jnp.isfinite(result.energy_residual_J)
