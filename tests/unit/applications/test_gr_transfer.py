from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax._fingerprint import canonical_fingerprint
from phydrax._physical import ElectromagneticScaleContract, RelativityScaleContract
from phydrax.applications.astrophysics._gr_medium import FastLightSnapshot
from phydrax.applications.astrophysics._gr_microphysics import (
    invariant_emission_coefficients,
)
from phydrax.applications.astrophysics._gr_rays import GRRayPlan
from phydrax.applications.astrophysics._gr_screens import GRObserverScreenPlan
from phydrax.applications.astrophysics._gr_transfer import (
    InvariantScalarTransferPlan,
    InvariantTransferUnitContract,
    PolarizedInvariantTransferPlan,
    PolarizedRayPath,
)
from phydrax.applications.astrophysics._gr_worldtube import (
    MonotoneSlowLightWorldtube,
)
from phydrax.electromagnetics import ThermalFreeFreeModel, ThermalSynchrotronModel
from phydrax.metrix._chart import CoordinateChart
from phydrax.metrix._lorentzian import minkowski_metric
from phydrax.metrix._metric import LorentzianMetric
from phydrax.metrix._spacetime_conventions import RelativityConvention
from phydrax.units import CENTIMETER, COULOMB, SOLAR_MASS


jax.config.update("jax_enable_x64", True)


def _snapshot(
    time: Any,
    *,
    temperature_offset: Any = 0.0,
    chart_id: Any = "cartesian-inertial",
    scale: Any = None,
    convention: Any = None,
) -> Any:
    axes = tuple(jnp.asarray([0.0, 1.0]) for _ in range(3))
    x, y, z = jnp.meshgrid(*axes, indexing="ij")
    scalar = 1.0 + x + 2.0 * y + 3.0 * z
    velocity = jnp.broadcast_to(jnp.asarray([1.0, 0.0, 0.0, 0.0]), scalar.shape + (4,))
    magnetic = jnp.broadcast_to(jnp.asarray([0.0, 1.0, 0.0, 0.0]), scalar.shape + (4,))
    return FastLightSnapshot(
        axes,
        time,
        scalar,
        10.0 * scalar,
        1.0e10 + temperature_offset + scalar,
        velocity,
        magnetic,
        RelativityScaleContract.si() if scale is None else scale,
        RelativityConvention() if convention is None else convention,
        chart_id=chart_id,
        source_id=f"snapshot-{time}",
    )


def test_gr_transfer_scenario_1() -> None:
    snapshot = _snapshot(0.0)
    coordinates = jnp.asarray([[0.2, 0.3, 0.4], [1.2, 0.3, 0.4]])
    plan = snapshot.prepare_sampling(coordinates)
    assert plan.stencil.indices.shape == (2, 8)

    sampled = eqx.filter_jit(snapshot.sample)(plan)
    np.testing.assert_allclose(sampled.rest_mass_density[0], 3.0, rtol=1.0e-6)
    assert bool(sampled.evidence.qualified[0])
    assert bool(sampled.evidence.derivative_valid[0])
    assert not bool(sampled.evidence.in_support[1])
    assert not bool(sampled.evidence.qualified[1])
    dynamic = eqx.filter_jit(snapshot.evaluate)(coordinates)
    np.testing.assert_allclose(dynamic.rest_mass_density, sampled.rest_mass_density)
    spatial_derivative = jax.grad(
        lambda first_coordinate: (
            snapshot.evaluate(
                jnp.stack(
                    (
                        first_coordinate,
                        jnp.asarray(0.3),
                        jnp.asarray(0.4),
                    )
                )
            ).rest_mass_density
        )
    )(jnp.asarray(0.2))
    np.testing.assert_allclose(spatial_derivative, 1.0, rtol=1.0e-6)
    first = _snapshot(0.0)
    second = _snapshot(1.0)
    worldtube = MonotoneSlowLightWorldtube(
        (first, second), worldtube_id="stationary-two-slice"
    )
    xyz = jnp.asarray([[0.2, 0.3, 0.4], [0.7, 0.6, 0.1]])
    events = jnp.concatenate((jnp.full((2, 1), 0.4), xyz), axis=-1)

    fast = first.sample(first.prepare_sampling(xyz))
    slow_plan = worldtube.prepare_sampling(events)
    assert slow_plan.stencil.indices.shape == (2, 16)
    slow = eqx.filter_jit(worldtube.sample)(slow_plan)
    np.testing.assert_allclose(slow.rest_mass_density, fast.rest_mass_density)
    np.testing.assert_allclose(slow.electron_temperature, fast.electron_temperature)
    assert bool(jnp.all(slow.evidence.qualified))

    with pytest.raises(ValueError, match="strictly increasing"):
        MonotoneSlowLightWorldtube((first, _snapshot(0.0)), worldtube_id="nonmonotone")
    units = InvariantTransferUnitContract.si_affine_length()
    plan = InvariantScalarTransferPlan(jnp.asarray([2.0]), units, path_id="slab")

    vacuum = plan.evaluate(jnp.asarray([0.0]), jnp.asarray([0.0]), 3.0)
    np.testing.assert_allclose(vacuum.invariant_intensity, 3.0)
    assert bool(vacuum.evidence.qualified)

    slab = eqx.filter_jit(plan.evaluate)(
        jnp.asarray([4.0]), jnp.asarray([0.5]), jnp.asarray(1.0)
    )
    expected = np.exp(-1.0) + 8.0 * (1.0 - np.exp(-1.0))
    np.testing.assert_allclose(slab.invariant_intensity, expected, rtol=2.0e-6)
    np.testing.assert_allclose(slab.optical_depth, 1.0)
    padded_plan = InvariantScalarTransferPlan(
        jnp.asarray([2.0, 7.0, 11.0]),
        units,
        active=jnp.asarray([True, False, False]),
        path_id="padded-slab",
    )
    padded = eqx.filter_jit(padded_plan.evaluate)(
        jnp.asarray([4.0, jnp.nan, jnp.nan]),
        jnp.asarray([0.5, jnp.nan, jnp.nan]),
        jnp.asarray(1.0),
    )
    np.testing.assert_allclose(padded.invariant_intensity, expected, rtol=2.0e-6)
    assert bool(padded.evidence.finite)

    thin_plan = InvariantScalarTransferPlan(jnp.asarray([1.0e-3]), units, path_id="thin")
    thin = thin_plan.evaluate(jnp.asarray([2.0]), jnp.asarray([1.0e-4]), 1.0)
    np.testing.assert_allclose(thin.invariant_intensity, 1.002, rtol=2.0e-6)

    thick_plan = InvariantScalarTransferPlan(jnp.asarray([2.0]), units, path_id="thick")
    thick = thick_plan.evaluate(jnp.asarray([500.0]), jnp.asarray([100.0]), 0.0)
    np.testing.assert_allclose(thick.invariant_intensity, 5.0, rtol=2.0e-6)

    derivative = jax.grad(
        lambda source: (
            plan.evaluate(
                jnp.asarray([source]), jnp.asarray([0.5]), 1.0
            ).invariant_intensity
        )
    )(jnp.asarray(4.0))
    np.testing.assert_allclose(derivative, 2.0 * (1.0 - np.exp(-1.0)), rtol=2.0e-6)


class _OpaqueMinkowskiMetric:
    def __call__(self, coordinates: Any) -> Any:
        return jnp.diag(jnp.asarray((-1.0, 1.0, 1.0, 1.0), dtype=coordinates.dtype))


def _capture_after_affine_03(affine: Any, point: Any, tangent: Any) -> Any:
    return 0.3 - affine


def _flat_polarized_ray(
    affine_parameter: Any,
    *,
    capture_margin: Any = None,
    scale: Any = None,
    convention: Any = None,
    coordinate_unit: Any = None,
    affine_parameter_unit: Any = None,
    metric: Any = None,
    metric_semantic_id: Any = None,
    metric_numeric_id: Any = None,
) -> Any:
    metric = (
        minkowski_metric(CoordinateChart("transfer-cartesian", ("t", "x", "y", "z")))
        if metric is None
        else metric
    )
    scale = RelativityScaleContract.si() if scale is None else scale
    convention = RelativityConvention() if convention is None else convention
    coordinate_unit = (
        scale.dimensional_scale.length_unit
        if coordinate_unit is None
        else coordinate_unit
    )
    affine_parameter_unit = (
        coordinate_unit if affine_parameter_unit is None else affine_parameter_unit
    )
    screen = GRObserverScreenPlan(
        metric,
        jnp.zeros(4),
        jnp.asarray([1.0, 0.0, 0.0, 0.0]),
        jnp.asarray([0.0, 0.0, 0.0, 1.0]),
        jnp.asarray([0.0, 0.0, 1.0, 0.0]),
        jnp.asarray([[0.0, 0.0]]),
        scale=scale,
        convention=convention,
        coordinate_unit=coordinate_unit,
        affine_parameter_unit=affine_parameter_unit,
        metric_semantic_id=metric_semantic_id,
        metric_numeric_id=metric_numeric_id,
        temporal_direction="future",
    ).initialize()
    ray_plan = GRRayPlan.from_screen(
        metric,
        screen,
        affine_parameter,
        scale=scale,
        convention=convention,
        coordinate_unit=coordinate_unit,
        affine_parameter_unit=affine_parameter_unit,
        metric_semantic_id=metric_semantic_id,
        metric_numeric_id=metric_numeric_id,
        transport_screen_basis=True,
        affine_budget_is_event=False,
        capture_margin=capture_margin,
    )
    return metric, ray_plan.trace()


def _pure_faraday_matrix(rotation: Any) -> Any:
    return jnp.asarray(
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, rotation, 0.0],
            [0.0, -rotation, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ]
    )


def test_polarized_contracts() -> None:
    metric, ray = _flat_polarized_ray(jnp.asarray([0.0, 0.5]))
    path = PolarizedRayPath(ray, metric, ray_index=0, basis_tolerance=1.0e-7)
    plan = PolarizedInvariantTransferPlan(
        path, InvariantTransferUnitContract.si_affine_length()
    )
    rotation_rate = 0.8
    matrices = (
        jnp.zeros((plan.capacity, 4, 4)).at[0].set(_pure_faraday_matrix(rotation_rate))
    )
    result = eqx.filter_jit(plan.evaluate)(
        jnp.zeros((plan.capacity, 4)),
        matrices,
        jnp.asarray([2.0, 1.0, 0.0, 0.0]),
        jnp.zeros((plan.capacity,)),
    )
    angle = rotation_rate * 0.5
    np.testing.assert_allclose(result.invariant_stokes[0], 2.0, atol=2.0e-6)
    np.testing.assert_allclose(
        result.invariant_stokes[1:3],
        jnp.asarray([jnp.cos(angle), jnp.sin(angle)]),
        atol=2.0e-5,
    )
    np.testing.assert_allclose(
        jnp.linalg.norm(result.invariant_stokes[1:]), 1.0, atol=2.0e-5
    )
    assert bool(path.evidence.qualified)
    assert bool(result.evidence.qualified)
    assert result.evidence.stokes_cone_margin > 0.99

    emission = jnp.zeros((plan.capacity, 4)).at[0].set(jnp.asarray([2.0, 1.0, 0.0, 0.0]))
    rebased_emission = plan.evaluate(
        emission,
        jnp.zeros((plan.capacity, 4, 4)),
        jnp.zeros((4,)),
        jnp.zeros((plan.capacity,)).at[0].set(0.25 * jnp.pi),
    )
    np.testing.assert_allclose(
        rebased_emission.invariant_stokes,
        jnp.asarray([1.0, 0.0, -0.5, 0.0]),
        atol=2.0e-5,
    )

    outside_cone = plan.evaluate(
        jnp.zeros((plan.capacity, 4)),
        matrices,
        jnp.asarray([0.5, 1.0, 0.0, 0.0]),
        jnp.zeros((plan.capacity,)),
    )
    assert not bool(outside_cone.evidence.physically_valid)
    np.testing.assert_allclose(
        jnp.linalg.norm(outside_cone.invariant_stokes[1:]), 1.0, atol=2.0e-5
    )

    unsupported = plan.evaluate(
        jnp.zeros((plan.capacity, 4)),
        matrices,
        jnp.asarray([2.0, 1.0, 0.0, 0.0]),
        jnp.zeros((plan.capacity,)),
        support=jnp.zeros((plan.capacity,), dtype="bool"),
    )
    assert not bool(unsupported.evidence.in_support)
    assert not bool(unsupported.evidence.qualified)

    assert ray.transported_screen_basis is not None
    spoofed_basis = ray.transported_screen_basis.at[0, :, 0, :].set(
        ray.transported_screen_basis[0, :, 1, :]
    )
    spoofed_ray = eqx.tree_at(
        lambda value: value.transported_screen_basis, ray, spoofed_basis
    )
    spoofed_path = PolarizedRayPath(
        spoofed_ray, metric, ray_index=0, basis_tolerance=1.0e-7
    )
    spoofed_plan = PolarizedInvariantTransferPlan(
        spoofed_path, InvariantTransferUnitContract.si_affine_length()
    )
    rejected = spoofed_plan.evaluate(
        jnp.zeros((spoofed_plan.capacity, 4)),
        jnp.zeros((spoofed_plan.capacity, 4, 4)),
        jnp.asarray([2.0, 1.0, 0.0, 0.0]),
        jnp.zeros((spoofed_plan.capacity,)),
    )
    assert not bool(spoofed_path.evidence.basis_transported)
    assert not bool(rejected.evidence.basis_transported)
    assert not bool(rejected.evidence.qualified)
    metric, ray = _flat_polarized_ray(jnp.asarray([0.0, 0.25]))
    plan = PolarizedInvariantTransferPlan(
        PolarizedRayPath(ray, metric, ray_index=0),
        InvariantTransferUnitContract.si_affine_length(),
    )
    emission = (
        jnp.zeros((plan.capacity, 4))
        .at[0]
        .set(jnp.asarray([1.0e-41, -5.0e-42, 0.0, 0.0]))
    )
    result = eqx.filter_jit(plan.evaluate)(
        emission,
        jnp.zeros((plan.capacity, 4, 4)),
        jnp.zeros((4,)),
        jnp.zeros((plan.capacity,)),
    )
    np.testing.assert_allclose(
        result.invariant_stokes,
        emission[0] * 0.25,
        rtol=2.0e-12,
        atol=0.0,
    )
    assert bool(result.invariant_stokes[0] > 0.0)
    assert bool(result.evidence.qualified)
    metric, ray = _flat_polarized_ray(
        jnp.asarray([0.0, 0.5, 1.0]),
        capture_margin=_capture_after_affine_03,
    )
    assert bool(ray.event_ledger.recorded[0])
    path = PolarizedRayPath(ray, metric, ray_index=0, basis_tolerance=1.0e-7)
    plan = PolarizedInvariantTransferPlan(
        path, InvariantTransferUnitContract.si_affine_length()
    )
    assert plan.active_count == 1
    np.testing.assert_allclose(
        plan.segment_lengths[0], ray.event_ledger.affine_parameter[0], atol=2.0e-7
    )
    np.testing.assert_allclose(
        path.coordinates[1], ray.event_ledger.coordinates[0], atol=2.0e-7
    )
    np.testing.assert_allclose(path.tangents[1], ray.event_ledger.tangent[0], atol=2.0e-7)


def test_gr_transfer_scenario_2() -> None:
    metric, ray = _flat_polarized_ray(jnp.asarray([0.0, 0.5]))
    path = PolarizedRayPath(ray, metric, ray_index=0)
    wrong_metric = minkowski_metric(
        CoordinateChart("other-transfer-chart", ("t", "x", "y", "z"))
    )
    with pytest.raises(ValueError, match="exactly match"):
        PolarizedRayPath(ray, wrong_metric, ray_index=0)

    compatible = _snapshot(0.0, chart_id=path.chart_id)
    sampled = compatible.sample(compatible.prepare_path_sampling(path))
    assert bool(sampled.evidence.qualified[0])
    assert not bool(sampled.evidence.in_support[-1])
    with pytest.raises(ValueError, match="share exact"):
        _snapshot(0.0).prepare_path_sampling(path)
    wrong_convention = RelativityConvention(metric_signature="mostly_minus")
    with pytest.raises(ValueError, match="share exact"):
        _snapshot(
            0.0,
            chart_id=path.chart_id,
            convention=wrong_convention,
        ).prepare_path_sampling(path)
    wrong_scale = RelativityScaleContract.geometric(SOLAR_MASS)
    with pytest.raises(ValueError, match="share exact"):
        _snapshot(
            0.0,
            chart_id=path.chart_id,
            scale=wrong_scale,
        ).prepare_path_sampling(path)
    si_length = RelativityScaleContract.si().dimensional_scale.length_unit
    unit_metric, unit_ray = _flat_polarized_ray(
        jnp.asarray([0.0, 0.5]),
        coordinate_unit=CENTIMETER,
        affine_parameter_unit=si_length,
    )
    unit_path = PolarizedRayPath(unit_ray, unit_metric, ray_index=0)
    with pytest.raises(ValueError, match="share exact"):
        _snapshot(0.0, chart_id=unit_path.chart_id).prepare_path_sampling(unit_path)

    si_units = InvariantTransferUnitContract.si_affine_length()
    centimeter_affine = InvariantTransferUnitContract(
        CENTIMETER, si_units.invariant_stokes_unit
    )
    with pytest.raises(ValueError, match="path-parameter unit"):
        PolarizedInvariantTransferPlan(path, centimeter_affine)

    worldtube = MonotoneSlowLightWorldtube(
        (
            _snapshot(0.0, chart_id=path.chart_id),
            _snapshot(1.0, chart_id=path.chart_id),
        ),
        worldtube_id="ray-bound-worldtube",
    )
    slow = worldtube.sample(worldtube.prepare_path_sampling(path))
    assert bool(slow.evidence.qualified[0])
    assert not bool(slow.evidence.in_support[-1])
    metric = LorentzianMetric(
        _OpaqueMinkowskiMetric(),
        chart=CoordinateChart("opaque-transfer-cartesian", ("t", "x", "y", "z")),
    )
    semantic_id = canonical_fingerprint(
        {"kind": "test-metric-semantics", "law": "minkowski-mostly-plus"}
    )
    numeric_id = canonical_fingerprint(
        {"kind": "test-metric-numeric", "diagonal": [-1.0, 1.0, 1.0, 1.0]}
    )
    metric, ray = _flat_polarized_ray(
        jnp.asarray([0.0, 0.5]),
        metric=metric,
        metric_semantic_id=semantic_id,
        metric_numeric_id=numeric_id,
    )
    path = PolarizedRayPath(
        ray,
        metric,
        ray_index=0,
        metric_semantic_id=semantic_id,
        metric_numeric_id=numeric_id,
    )
    plan = PolarizedInvariantTransferPlan(
        path, InvariantTransferUnitContract.si_affine_length()
    )
    result = plan.evaluate(
        jnp.zeros((plan.capacity, 4)),
        jnp.zeros((plan.capacity, 4, 4)),
        jnp.asarray([1.0, 0.0, 0.0, 0.0]),
        jnp.zeros((plan.capacity,)),
    )
    assert bool(path.evidence.qualified)
    assert bool(result.evidence.qualified)

    with pytest.raises(ValueError, match="exactly match"):
        PolarizedRayPath(
            ray,
            metric,
            ray_index=0,
            metric_semantic_id=semantic_id,
            metric_numeric_id=canonical_fingerprint({"wrong": "metric-numeric-id"}),
        )


def test_invariant_emission_coefficients_convert_synchrotron_and_free_free() -> None:
    frequency = jnp.asarray([1.0e11, 3.0e11])
    synchrotron = ThermalSynchrotronModel().evaluate(1.0e6, 4.0e10, 1.0, frequency, 0.5)
    invariant = invariant_emission_coefficients(synchrotron, frequency)
    np.testing.assert_allclose(
        invariant.emission, synchrotron.emission / frequency[:, None] ** 2
    )
    np.testing.assert_allclose(
        invariant.propagation_matrix,
        synchrotron.propagation_matrix * frequency[:, None, None],
    )
    assert not bool(jnp.any(invariant.qualified))
    assert bool(jnp.all(invariant.finite))

    free_free = ThermalFreeFreeModel(ElectromagneticScaleContract.si()).evaluate(
        1.0e12, 1.0e12, 1.0e7, 2.0 * np.pi * frequency
    )
    converted = invariant_emission_coefficients(free_free, frequency)
    # j_nu = 2 pi j_omega; J = j_nu / nu^2 and K = nu alpha.
    np.testing.assert_allclose(
        converted.emission[:, 0],
        2.0 * np.pi * free_free.emission[:, 0] / frequency**2,
    )
    np.testing.assert_allclose(
        converted.propagation_matrix[:, 0, 0], frequency * free_free.absorption
    )
    assert bool(jnp.all(converted.qualified))

    mismatched = invariant_emission_coefficients(free_free, 2.0 * frequency)
    assert not bool(jnp.any(mismatched.physically_valid))
    assert bool(jnp.all(jnp.isnan(mismatched.emission)))

    code_scale = ElectromagneticScaleContract.code_units(
        ElectromagneticScaleContract.si().relativity.dimensional_scale,
        COULOMB,
        gravitational_constant=1,
        speed_of_light=1,
        reduced_planck_constant=1,
        boltzmann_constant=1,
        elementary_charge=1,
        electron_mass=1,
        vacuum_permittivity=1,
        constant_set_id="unit-test",
    )
    code = ThermalFreeFreeModel(code_scale).evaluate(1.0, 1.0, 1.0, 1.0)
    with pytest.raises(ValueError, match="SI scale"):
        invariant_emission_coefficients(code, jnp.asarray(1.0))
