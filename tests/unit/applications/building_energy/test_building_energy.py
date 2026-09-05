# Copyright © 2026 PHYDRA, Inc. All rights reserved.
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax._physical import SpatialCoordinateContract
from phydrax.applications.building_energy import (
    Adjacency,
    Aperture,
    BuildingBoundary,
    BuildingExperiment,
    BuildingSource,
    calibrate_building,
    compile_building,
    Construction,
    energyplus_reference_weather,
    enrich_building_geometry,
    import_radiance_matrix,
    optimize_hvac,
    parse_epw,
    radiative_heat_gains,
    RadiativeBasis,
    RadiativeComposition,
    RadiativeOperator,
    replay_building,
    retrofit_building,
    Surface,
    surface_tag_labels,
    SurfaceRole,
    VentilationExchange,
    Zone,
)
from phydrax.applications.thermofluids import ResistiveHeatingLaw
from phydrax.dynamics import ContinuousSystem, DifferentialAlgebraicSystem
from phydrax.geometry.surface import SurfaceMetadata, SurfaceModel
from phydrax.optim import OptimizationTermination
from phydrax.units import derived_unit, JOULE, METER, ONE, SECOND


def one_zone(capacity=10000.0, conductance=10.0):
    return BuildingSource(
        (Zone("room", capacity),),
        adjacencies=(Adjacency("outdoor", "room", None, conductance),),
        source_id="one-zone",
    )


def test_one_zone_exact_affine_and_physical_parameter_derivative():
    model = compile_building(one_zone())
    assert isinstance(model.system, ContinuousSystem)
    result = model.step(jnp.array([300.0]), 280.0, jnp.array([50.0]), 500.0)
    expected = 285.0 + 15.0 * np.exp(-0.5)
    np.testing.assert_allclose(result.temperature, [expected], rtol=1e-8, atol=1e-8)
    assert bool(result.successful)
    derivative = jax.grad(
        lambda c: (
            compile_building(one_zone(c))
            .step(jnp.array([300.0]), 280.0, jnp.array([50.0]), 500.0)
            .temperature[0]
        )
    )(10000.0)
    np.testing.assert_allclose(derivative, 15 * np.exp(-0.5) * 5000 / 10000**2, rtol=1e-5)
    identity = model.step(jnp.array([300.0]), 280.0, jnp.array([50.0]), 0.0)
    np.testing.assert_array_equal(identity.temperature, [300.0])


def test_two_zone_internal_energy_conservation_and_balance_detection():
    source = BuildingSource(
        (Zone("a", 1000.0), Zone("b", 1000.0)),
        adjacencies=(Adjacency("wall", "a", "b", 10.0),),
        source_id="two-zone",
    )
    model = compile_building(source)
    result = model.step(jnp.array([300.0, 280.0]), 0.0, jnp.zeros(2), 100.0)
    expected = np.array([290 + 10 * np.exp(-2), 290 - 10 * np.exp(-2)])
    np.testing.assert_allclose(result.temperature, expected, atol=1e-7)
    np.testing.assert_allclose(
        jnp.sum(model.capacity * result.temperature), 580000, atol=1e-6
    )
    observation = model.observe(
        jnp.array([300.0, 280.0]),
        0.0,
        jnp.zeros(2),
        temperature_rate=jnp.array([-0.1, 0.1]),
    )
    np.testing.assert_allclose(observation.edge_heat_flow, [200.0])
    np.testing.assert_allclose(observation.balance_residual, [100.0, -100.0])


def test_massless_dae_and_series_resistance_equivalence():
    source = BuildingSource(
        (Zone("air", 10000.0), Zone("junction", 0, massless=True)),
        adjacencies=(
            Adjacency("inside", "air", "junction", 20.0),
            Adjacency("outside", "junction", None, 20.0),
        ),
        source_id="massless",
    )
    model = compile_building(source)
    assert isinstance(model.system, DifferentialAlgebraicSystem)
    result = model.step(jnp.array([300.0, -999.0]), 280.0, jnp.zeros(2), 500.0)
    expected = 280.0 + 20.0 * np.exp(-0.5)
    np.testing.assert_allclose(
        result.temperature, [expected, (expected + 280) / 2], atol=1e-7
    )
    rate = jnp.array([(280 - expected) * 10 / 10000, 0.0])
    residual = model.system.evaluate(
        500.0, result.temperature, rate, inputs=jnp.array([280.0, 0.0, 0.0])
    )
    np.testing.assert_allclose(residual, 0, atol=1e-7)
    with pytest.raises(ValueError, match="Unanchored"):
        compile_building(
            BuildingSource(
                (Zone("air", 1), Zone("floating", 0, massless=True)), source_id="bad"
            )
        )


def test_geometry_area_aperture_retrofit_and_stale_revision():
    metadata = SurfaceMetadata(
        source_id="wall",
        source_revision="measured",
        coordinate_contract=SpatialCoordinateContract.si(),
        provenance=("authored-fixture",),
        cell_tags=("wall", "wall"),
    )
    model = SurfaceModel.from_triangles(
        [[0, 0, 0], [2, 0, 0], [2, 3, 0], [0, 3, 0]], [[0, 1, 2], [0, 2, 3]], metadata
    )
    label = surface_tag_labels(model)[0]
    construction = Construction("wall", 2.0)
    source = enrich_building_geometry(
        (Zone("room", 10000),),
        model,
        (SurfaceRole(label, "room", construction),),
        source_id="geometry-building",
    )
    np.testing.assert_allclose(source.surfaces[0].area, 6)
    np.testing.assert_allclose(
        compile_building(source)
        .observe(jnp.array([300.0]), 280.0, jnp.zeros(1))
        .net_heat,
        [-60],
    )
    changed = retrofit_building(
        source,
        source_id="insulated",
        construction_replacements={"wall": Construction("insulation", 4.0)},
    )
    np.testing.assert_allclose(
        compile_building(changed)
        .observe(jnp.array([300.0]), 280.0, jnp.zeros(1))
        .net_heat,
        [-30],
    )
    np.testing.assert_allclose(
        compile_building(source)
        .observe(jnp.array([300.0]), 280.0, jnp.zeros(1))
        .net_heat,
        [-60],
    )
    with pytest.raises((ValueError, eqx.EquinoxRuntimeError)):
        Surface("invalid", "room", 2, construction, apertures=(Aperture("window", 3, 1),))
    refreshed = SurfaceModel(
        model.mesh.with_coordinates(model.mesh.coordinates * 2, numeric_version="new"),
        metadata,
    )
    with pytest.raises(ValueError, match="stale"):
        enrich_building_geometry(
            source.zones,
            refreshed,
            (SurfaceRole(label, "room", construction),),
            source_id="stale",
        )


def test_ground_ambient_supply_and_adiabatic_boundaries_remain_distinct():
    source = BuildingSource(
        (Zone("room", 1000),),
        boundaries=(
            BuildingBoundary("soil", kind="ground"),
            BuildingBoundary("air"),
            BuildingBoundary("supply", kind="fixed"),
        ),
        surfaces=(
            Surface("floor", "room", 2, Construction("slab", 1), boundary_id="soil"),
            Surface(
                "insulated", "room", 10, Construction("no-flux", 0.1), adiabatic=True
            ),
        ),
        adjacencies=(Adjacency("ambient", "room", None, 1, boundary_id="air"),),
        ventilation=(
            VentilationExchange("fan", "room", 4, boundary_id="supply"),
            VentilationExchange(
                "leak", "room", 0.5, boundary_id="air", kind="infiltration"
            ),
        ),
        source_id="multi-boundary",
    )
    model = compile_building(source)
    boundary = jnp.array([270.0, 290.0, 295.0])
    observation = model.observe(jnp.array([300.0]), boundary, jnp.zeros(1))
    np.testing.assert_allclose(observation.net_heat, [-95])
    np.testing.assert_allclose(observation.edge_heat_flow, [60, 10, 20, 5])
    np.testing.assert_allclose(
        model.system.evaluate(
            0.0, jnp.array([300.0]), inputs=jnp.array([270.0, 290.0, 295.0, 0.0])
        ),
        [-0.095],
    )
    result = replay_building(
        model,
        jnp.array([300.0]),
        jnp.array([0.0, 10.0]),
        boundary[None],
        jnp.zeros((1, 1)),
    )
    equilibrium = 2155 / 7.5
    np.testing.assert_allclose(
        result.temperature[-1],
        [equilibrium + (300 - equilibrium) * np.exp(-0.075)],
        atol=1e-7,
    )
    with pytest.raises(ValueError, match="boundary_ids"):
        model.step(jnp.array([300.0]), 290.0, jnp.zeros(1), 10.0)


def test_epw_standard_time_interval_energy_and_missing_flags():
    text = energyplus_reference_weather().decode()
    rows = text.splitlines()
    record = rows[8].split(",")
    record[6], record[13] = "99.9", "100"
    rows[8] = ",".join(record)
    weather = parse_epw("\n".join(rows))
    temperature = weather.quantity("dry_bulb_temperature")
    solar = weather.quantity("global_horizontal_energy")
    assert not bool(temperature.samples.sample_valid[0])
    np.testing.assert_allclose(temperature.samples.values[1:], 293.15)
    np.testing.assert_allclose(solar.samples.values[0], 360000)
    np.testing.assert_allclose(
        solar.samples.support.coordinates[jnp.array([0, -1])], [0, 86400]
    )
    assert solar.samples.alignment == "edge"
    assert temperature.samples.alignment == "node"
    assert weather.uncertainty_flags[0] == "?" * 29
    assert weather.location.standard_utc_offset == 0
    with pytest.raises(ValueError, match="missing"):
        parse_epw("\n".join(rows[:9] + rows[10:]))


def test_radiative_factorized_import_and_basis_rejection():
    sky = RadiativeBasis(("sky",), basis_id="sky", measure="coefficient", weights=(1,))
    window = RadiativeBasis(
        ("window",), basis_id="window", measure="coefficient", weights=(1,)
    )
    sensor = RadiativeBasis(
        ("sensor",), basis_id="sensor", measure="coefficient", weights=(1,)
    )
    imported = import_radiance_matrix(
        "NROWS=1\nNCOLS=1\nNCOMP=3\nFORMAT=ascii\n\n2 3 4\n",
        sky,
        window,
        input_unit=ONE,
        output_unit=ONE,
    )
    second = RadiativeOperator(
        jnp.array([[[0.5, 0.25, 0.125]]]), window, sensor, input_unit=ONE, output_unit=ONE
    )
    composition = RadiativeComposition((imported, second))
    expected = [[1, 0.75, 0.5]]
    np.testing.assert_allclose(composition.apply(jnp.ones((1, 3))), expected)
    np.testing.assert_allclose(
        composition.materialize().apply(jnp.ones((1, 3))), expected
    )
    with pytest.raises(ValueError, match="basis"):
        RadiativeComposition((second, imported))
    with pytest.raises(ValueError, match="truncated"):
        import_radiance_matrix(
            "NROWS=1\nNCOLS=1\nNCOMP=3\nFORMAT=ascii\n\n2 3\n",
            sky,
            window,
            input_unit=ONE,
            output_unit=ONE,
        )


def test_explicit_radiative_response_conserves_nodal_heat():
    basis = RadiativeBasis(
        ("window",),
        basis_id="bands",
        measure="surface-average",
        weights=(1,),
        channels=("visible", "near-infrared"),
    )
    unit = derived_unit("W/m²", ((JOULE, 1), (SECOND, -1), (METER, -2)))
    heat = radiative_heat_gains(
        jnp.array([[100.0, 200.0]]),
        basis,
        unit=unit,
        spectral_weights=jnp.ones(2),
        receiving_area=jnp.array([2.0]),
        absorption_fraction=jnp.array([[0.5, 0.25]]),
        heat_distribution=jnp.array([[0.25], [0.75]]),
    )
    np.testing.assert_allclose(heat, [50, 150])
    model = compile_building(
        BuildingSource((Zone("a", 1000), Zone("b", 2000)), source_id="solar")
    )
    step = model.step(jnp.array([290.0, 290.0]), 280.0, heat, 100.0)
    np.testing.assert_allclose(step.temperature, [295.0, 297.5], atol=1e-7)


def test_native_hvac_optimization_replays_bounded_controls():
    model = compile_building(one_zone(capacity=10000, conductance=10))
    times = jnp.array([0.0, 300.0, 900.0, 1200.0, 2400.0])
    result = optimize_hvac(
        model,
        jnp.array([293.15]),
        times,
        jnp.full((4,), 283.15),
        jnp.zeros((4, 1)),
        293.15,
        heat_distribution=jnp.ones((1, 1)),
        conversion_law=ResistiveHeatingLaw(),
        supply_temperature=313.15,
        power_upper=200,
        initial_power=jnp.full((4, 1), 50.0),
        power_scale=100,
        termination=OptimizationTermination(maximum_steps=60),
    )
    assert bool(result.successful)
    np.testing.assert_allclose(result.electrical_power, 100, atol=0.02)
    np.testing.assert_allclose(result.replay.temperature, 293.15, atol=1e-3)
    replay = replay_building(
        model, jnp.array([293.15]), times, jnp.full((4,), 283.15), result.delivered_heat
    )
    np.testing.assert_allclose(replay.temperature, result.replay.temperature, atol=1e-8)
    np.testing.assert_allclose(
        result.optimization.trajectory.states + result.state_reference_temperature,
        result.replay.temperature,
        atol=1e-5,
    )


def test_identifiable_calibration_predicts_unseen_forcing():
    time = jnp.arange(6) * 300.0
    truth = compile_building(one_zone(capacity=10000, conductance=10))
    heat = jnp.array([[0], [100], [20], [80], [0]], dtype=float)
    outside = jnp.full((5,), 280.0)
    target = replay_building(truth, jnp.array([300.0]), time, outside, heat).temperature[
        1:
    ]
    held_heat = jnp.full((5, 1), 40.0)
    held_outside = jnp.full((5,), 275.0)
    held_target = replay_building(
        truth, jnp.array([295.0]), time, held_outside, held_heat
    ).temperature[1:]
    training = BuildingExperiment(
        [300.0], time, outside, heat, target, experiment_id="training"
    )
    heldout = BuildingExperiment(
        [295.0], time, held_outside, held_heat, held_target, experiment_id="heldout"
    )
    result = calibrate_building(
        lambda p: one_zone(capacity=10000 * jnp.exp(p[0]), conductance=10),
        jnp.array([0.3]),
        training,
        heldout,
        observation_nodes=(0,),
        termination=OptimizationTermination(maximum_steps=30),
    )
    assert bool(result.successful)
    assert bool(result.identifiable)
    np.testing.assert_allclose(result.parameters, 0, atol=1e-5)
    assert float(result.heldout_rmse) < 1e-4
