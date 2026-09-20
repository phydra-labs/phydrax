import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax._fingerprint import canonical_fingerprint
from phydrax._physical import RelativityScaleContract
from phydrax.applications.astrophysics._gr_medium import FastLightSnapshot
from phydrax.applications.astrophysics._gr_microphysics import (
    _validated_log_bessel_k2,
    invariant_synchrotron_coefficients,
    ThermalSynchrotronModel,
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
from phydrax.metrix._chart import CoordinateChart
from phydrax.metrix._lorentzian import minkowski_metric
from phydrax.metrix._metric import LorentzianMetric
from phydrax.metrix._spacetime_conventions import RelativityConvention
from phydrax.units import CENTIMETER, SOLAR_MASS


jax.config.update("jax_enable_x64", True)


def _snapshot(
    time,
    *,
    temperature_offset=0.0,
    chart_id="cartesian-inertial",
    scale=None,
    convention=None,
):
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


def test_fast_light_fixed_stencil_reports_support_and_is_jittable():
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


def test_monotone_slow_light_stationary_limit_matches_fast_light():
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


def test_invariant_scalar_transfer_vacuum_slab_thin_and_thick_limits():
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
    def __call__(self, coordinates):
        return jnp.diag(jnp.asarray((-1.0, 1.0, 1.0, 1.0), dtype=coordinates.dtype))


def _flat_polarized_ray(
    affine_parameter,
    *,
    capture_margin=None,
    scale=None,
    convention=None,
    coordinate_unit=None,
    affine_parameter_unit=None,
    metric=None,
    metric_semantic_id=None,
    metric_numeric_id=None,
):
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


def _pure_faraday_matrix(rotation):
    return jnp.asarray(
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, rotation, 0.0],
            [0.0, -rotation, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ]
    )


def test_polarized_transfer_binds_ray_basis_and_preserves_faraday_cone():
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


def test_polarized_ray_path_retains_exact_terminal_partial_segment():
    metric, ray = _flat_polarized_ray(
        jnp.asarray([0.0, 0.5, 1.0]),
        capture_margin=lambda affine, point, tangent: 0.3 - affine,
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


def test_ray_metric_and_medium_chart_identities_are_exactly_bound():
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


def test_opaque_metric_identity_flows_through_polarized_transfer():
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


def test_polarized_transfer_preserves_tiny_physical_emission_scale():
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


def test_validated_log_bessel_k2_matches_high_precision_reference_values():
    arguments = jnp.asarray([1.0e-3, 1.0e-2, 1.0e-1, 1.0, 10.0, 100.0, 1000.0])
    trusted_log_values = jnp.asarray(
        [
            14.508657488524674,
            9.903462555643179,
            5.295834109025258,
            0.4854086715656462,
            -10.74700112206937,
            -102.05813713541278,
            -1003.2262122239944,
        ]
    )
    evaluated = eqx.filter_jit(_validated_log_bessel_k2)(arguments)
    np.testing.assert_allclose(evaluated, trusted_log_values, atol=1.0e-9, rtol=0.0)


def test_thermal_synchrotron_attribution_and_scoped_qualification():
    model = ThermalSynchrotronModel()
    coefficients = eqx.filter_jit(model.evaluate)(
        jnp.asarray(1.0e6),
        jnp.asarray(4.0e10),
        jnp.asarray(1.0),
        jnp.asarray(1.0e11),
        jnp.asarray(0.5),
    )
    assert model.units.number_density_unit.symbol == "m^-3"
    assert model.units.frequency_unit.symbol == "Hz"
    assert model.reference.doi == "10.1086/177422"
    assert model.reference.equation == "31"
    assert model.reference.maximum_shape_relative_error == 0.027
    assert model.reference.authors == (
        "Rohan Mahadevan",
        "Ramesh Narayan",
        "Insu Yi",
    )
    assert model.reference.polarization_status == (
        "unqualified-independent-approximation"
    )
    assert bool(coefficients.evidence.k2_approximation_valid)
    assert bool(coefficients.evidence.emission_reference_valid)
    assert bool(coefficients.evidence.emission_derivative_valid)
    assert not bool(coefficients.evidence.polarization_reference_valid)
    assert not bool(coefficients.evidence.faraday_reference_valid)
    assert not bool(coefficients.evidence.qualified)
    assert not bool(coefficients.evidence.derivative_valid)
    assert coefficients.emission[0] > 0.0
    assert coefficients.emission[1] < 0.0
    assert coefficients.absorption_i > 0.0
    assert coefficients.faraday_rotation > 0.0
    assert coefficients.faraday_conversion > 0.0

    invariant = invariant_synchrotron_coefficients(coefficients, jnp.asarray(1.0e11))
    assert not bool(invariant.qualified)
    assert bool(jnp.all(jnp.isfinite(invariant.emission)))
    assert bool(jnp.all(jnp.isfinite(invariant.propagation_matrix)))

    log_temperature_derivative = jax.grad(
        lambda log_temperature: jnp.log(
            model.evaluate(
                1.0e6,
                jnp.exp(log_temperature),
                1.0,
                1.0e11,
                0.5,
            ).emission[0]
        )
    )(jnp.log(jnp.asarray(4.0e10)))
    assert bool(jnp.isfinite(log_temperature_derivative))

    below_published_support = model.evaluate(1.0e6, 1.0e10, 1.0, 1.0e11, 0.5)
    assert bool(below_published_support.evidence.physically_valid)
    assert not bool(below_published_support.evidence.in_domain)
    assert not bool(below_published_support.evidence.emission_reference_valid)

    vacuum = model.evaluate(0.0, 1.0e10, 1.0, 1.0e10, 0.5)
    np.testing.assert_allclose(vacuum.emission, 0.0)
    np.testing.assert_allclose(vacuum.propagation_matrix, 0.0)
    assert bool(vacuum.evidence.qualified)
    assert not bool(vacuum.evidence.derivative_valid)

    invalid = model.evaluate(1.0e6, 4.0e10, 1.0, 1.0e11, 1.1)
    assert not bool(invalid.evidence.physically_valid)
    assert not bool(invalid.evidence.qualified)
    assert bool(jnp.all(jnp.isnan(invalid.emission)))
