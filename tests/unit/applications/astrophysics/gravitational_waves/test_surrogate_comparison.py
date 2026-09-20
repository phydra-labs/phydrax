import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx


def _surrogate_plan(
    *,
    resource_policy=None,
    time_origin_id="test:analytic-grid-origin",
):
    gw = phx.applications.astrophysics.gravitational_waves
    provenance = phx.applications.astrophysics.ObservationDataProvenance.native(
        "nr-polynomial-surrogate-test"
    )
    reconstruction = jnp.asarray([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]])
    coefficients = jnp.asarray([[1.0, 0.1], [2.0, 0.2]])
    orders = jnp.asarray(
        [
            [[0, 0, 0], [1, 0, 0]],
            [[0, 0, 0], [0, 1, 0]],
        ],
        dtype=jnp.int32,
    )
    active = jnp.ones((2, 2), dtype="bool")
    real = gw.PolynomialEmpiricalField(
        reconstruction,
        coefficients,
        orders,
        active,
        field_id="real-22",
        resource_policy=resource_policy,
    )
    imaginary = gw.PolynomialEmpiricalField(
        reconstruction,
        jnp.zeros((2, 1)),
        jnp.zeros((2, 1, 3), dtype=jnp.int32),
        jnp.ones((2, 1), dtype="bool"),
        field_id="imaginary-22",
        resource_policy=resource_policy,
    )
    artifact = gw.AlignedNRSurrogateArtifact(
        jnp.asarray([-1.0, 0.0, 1.0]),
        ((2, 2),),
        (real,),
        (imaginary,),
        jnp.asarray([0.0, -1.0, -1.0]),
        jnp.asarray([3.0, 1.0, 1.0]),
        provenance,
        maximum_mass_ratio=8.0,
        maximum_spin_magnitude=0.8,
        frame_id="test-inertial-frame",
        time_origin_id=time_origin_id,
        mode_normalization="s2fft-orthonormal-condon-shortley",
        artifact_id="analytic-polynomial-mode",
        resource_policy=resource_policy,
    )
    precision = phx.discretization.spectral.SpectralPrecisionPolicy(
        jnp.complex128, output_dtype=jnp.complex128
    )
    angular = phx.discretization.spectral.SphericalSpectralPlan(
        3,
        spin=-2,
        reality=False,
        precision=precision,
    ).prepare()
    return gw.AlignedNRSurrogatePlan(
        artifact,
        angular,
        phx.RelativityScaleContract.si(),
    )


def _parameters(mass_ratio=2.0):
    return {
        "mass_ratio": jnp.asarray(mass_ratio),
        "primary_spin": jnp.asarray(0.2),
        "secondary_spin": jnp.asarray(-0.1),
    }


def test_nr_surrogate_reconstructs_polynomial_nodes_and_mode_symmetry():
    plan = _surrogate_plan()
    result = eqx.filter_jit(plan.evaluate_modes)(_parameters())
    coordinates = phx.applications.astrophysics.gravitational_waves.aligned_spin_surrogate_coordinates(
        2.0, 0.2, -0.1
    )
    node_values = jnp.asarray([1.0 + 0.1 * coordinates[0], 2.0 + 0.2 * coordinates[1]])
    expected = jnp.asarray(
        [node_values[0], 0.5 * (node_values[0] + node_values[1]), node_values[1]]
    )
    center = plan.angular.layout.bandlimit - 1

    np.testing.assert_allclose(result.coefficients[2, center + 2], expected)
    np.testing.assert_allclose(result.coefficients[2, center - 2], jnp.conj(expected))
    assert bool(result.valid)
    assert float(result.symmetry_defect) == 0.0
    assert bool(result.intrinsic_derivative_valid)
    assert not plan.artifact.differentiation.higher_order
    assert plan.artifact.normalization_report.valid
    assert not bool(result.qualified)
    assert not plan.artifact.source_authenticated
    assert plan.artifact.normalization_report.status.name == "DECLARED_LOSS"
    integer_phase = plan.evaluate_modes(_parameters(), reference_phase=7)
    floating_phase = plan.evaluate_modes(_parameters(), reference_phase=7.0)
    np.testing.assert_allclose(integer_phase.coefficients, floating_phase.coefficients)
    huge_phase = plan.evaluate_modes(_parameters(), reference_phase=1.0e308)
    assert bool(huge_phase.valid)
    assert bool(jnp.all(jnp.isfinite(huge_phase.coefficients)))

    derivative = jax.grad(
        lambda ratio: jnp.real(
            plan.evaluate_modes(_parameters(ratio)).coefficients[2, center + 2]
        ).sum()
    )(jnp.asarray(2.0))
    with pytest.raises(TypeError, match="real numeric arrays"):
        phx.applications.astrophysics.gravitational_waves.aligned_spin_surrogate_coordinates(
            True, 0.0, 0.0
        )
    assert np.isfinite(float(derivative))
    assert float(derivative) != 0.0


def test_nr_surrogate_reports_time_support_and_physical_distance_scaling():
    plan = _surrogate_plan()
    geometric = plan.evaluate_geometric(
        jnp.asarray([-2.0, -0.5, 0.5, 2.0]),
        _parameters(),
        inclination=jnp.asarray(0.6),
    )
    np.testing.assert_array_equal(geometric.support, [False, True, True, False])
    np.testing.assert_array_equal(geometric.valid, [False, True, True, False])
    np.testing.assert_array_equal(
        geometric.intrinsic_derivative_valid, [False, True, True, False]
    )
    np.testing.assert_array_equal(
        geometric.extrinsic_derivative_valid, [False, True, True, False]
    )
    np.testing.assert_array_equal(
        geometric.time_derivative_valid, [False, True, True, False]
    )
    np.testing.assert_array_equal(
        geometric.mass_scaling_derivative_valid, [False, False, False, False]
    )
    np.testing.assert_allclose(geometric.values[:, (0, 3)], 0.0)
    periodic_angles = plan.evaluate_geometric(
        jnp.asarray([-0.5, 0.5]),
        _parameters(),
        inclination=jnp.asarray(0.6),
        azimuth=jnp.asarray(1.0e308),
        frame_angle=jnp.asarray(-1.0e308),
    )
    assert bool(jnp.all(periodic_angles.valid))
    assert bool(jnp.all(jnp.isfinite(periodic_angles.values)))

    mass = jnp.asarray(60.0 * 1.988409870698051e30)
    mass_time = plan.scale.mass_to_geometric_time(mass)
    seconds = jnp.asarray([-0.5, 0.0, 0.5]) * mass_time
    near = plan.evaluate_physical(
        seconds,
        _parameters(),
        detector_frame_total_mass_kg=mass,
        luminosity_distance_m=jnp.asarray(1.0e24),
        inclination=jnp.asarray(0.6),
    )
    far = plan.evaluate_physical(
        seconds,
        _parameters(),
        detector_frame_total_mass_kg=mass,
        luminosity_distance_m=jnp.asarray(2.0e24),
        inclination=jnp.asarray(0.6),
    )
    np.testing.assert_allclose(near.values, 2.0 * far.values)
    assert bool(jnp.all(near.valid))
    np.testing.assert_array_equal(near.time_derivative_valid, [True, False, True])
    overflow = plan.evaluate_physical(
        jnp.asarray([0.0]),
        _parameters(),
        detector_frame_total_mass_kg=jnp.asarray(1.0e308),
        luminosity_distance_m=jnp.asarray(1.0e-200),
        inclination=jnp.asarray(0.6),
    )
    assert not bool(jnp.any(overflow.valid))
    assert bool(jnp.all(overflow.values == 0.0))
    assert int(overflow.status[0]) == int(
        phx.applications.astrophysics.gravitational_waves.GravitationalWaveStatus.NONFINITE_WAVEFORM
    )
    np.testing.assert_array_equal(near.extrinsic_derivative_valid, [True, True, True])
    np.testing.assert_array_equal(near.mass_scaling_derivative_valid, [True, False, True])

    unsupported = plan.evaluate_modes(_parameters(9.0))
    assert not bool(unsupported.valid)
    extreme_spin = plan.evaluate_modes(
        {
            **_parameters(),
            "primary_spin": jnp.asarray(1.0e308),
        }
    )
    assert not bool(extreme_spin.valid)
    assert int(extreme_spin.status) == int(
        phx.applications.astrophysics.gravitational_waves.GravitationalWaveStatus.OUTSIDE_WAVEFORM_SUPPORT
    )
    assert int(unsupported.status) == int(
        phx.applications.astrophysics.gravitational_waves.GravitationalWaveStatus.OUTSIDE_WAVEFORM_SUPPORT
    )


def test_nr_surrogate_keeps_caller_provenance_unqualified_and_enforces_caps():
    gw = phx.applications.astrophysics.gravitational_waves
    template = _surrogate_plan()
    native = template.artifact
    caller_provenance = phx.applications.astrophysics.ObservationDataProvenance(
        producer="external-producer",
        producer_version="release-1",
        source_id="source:external-surrogate",
        checksum="0" * 64,
        license_id="CC-BY-4.0",
        differentiation="native-parameter",
    )
    caller_artifact = gw.AlignedNRSurrogateArtifact(
        native.geometric_time,
        native.modes,
        native.real_fields,
        native.imaginary_fields,
        native.fit_coordinate_lower,
        native.fit_coordinate_upper,
        caller_provenance,
        maximum_mass_ratio=8.0,
        maximum_spin_magnitude=0.8,
        frame_id=native.frame_id,
        time_origin_id=native.time_origin_id,
        mode_normalization=native.mode_normalization,
        artifact_id="caller-asserted-surrogate",
    )
    caller_plan = gw.AlignedNRSurrogatePlan(
        caller_artifact,
        template.angular,
        phx.RelativityScaleContract.si(),
    )
    caller_result = caller_plan.evaluate_modes(_parameters())
    assert not caller_artifact.source_authenticated
    assert bool(caller_result.valid)
    assert not bool(caller_result.qualified)

    normalized_byte_policy = gw.NRSurrogateResourcePolicy(maximum_normalized_bytes=100)
    with pytest.raises(ValueError, match="resource policy"):
        gw.PolynomialEmpiricalField(
            jnp.ones((3, 2), dtype=jnp.float16),
            jnp.ones((2, 2), dtype=jnp.float16),
            jnp.zeros((2, 2, 3), dtype=jnp.int8),
            jnp.ones((2, 2), dtype="bool"),
            field_id="normalized-byte-cap",
            resource_policy=normalized_byte_policy,
        )

    output_policy = gw.NRSurrogateResourcePolicy(maximum_modal_entries=50)
    output_bounded = _surrogate_plan(resource_policy=output_policy)
    with pytest.raises(ValueError, match="output-sample capacity"):
        output_bounded.evaluate_geometric(
            jnp.asarray([-0.75, -0.25, 0.25, 0.75]),
            _parameters(),
            inclination=jnp.asarray(0.6),
        )

    tiny = gw.NRSurrogateResourcePolicy(maximum_time_samples=2)
    with pytest.raises(ValueError, match="resource policy"):
        gw.PolynomialEmpiricalField(
            native.real_fields[0].reconstruction_matrix,
            native.real_fields[0].coefficients,
            native.real_fields[0].orders,
            native.real_fields[0].active_terms,
            field_id="over-capacity-field",
            resource_policy=tiny,
        )


def test_waveform_match_recovers_phase_and_sample_aligned_time_shift():
    gw = phx.applications.astrophysics.gravitational_waves
    sample_count = 64
    sample_interval = 1.0 / 64.0
    frequency = jnp.fft.rfftfreq(sample_count, sample_interval)
    provenance = phx.applications.astrophysics.ObservationDataProvenance.native(
        "waveform-match-test"
    )
    psd = gw.OneSidedPowerSpectralDensity(
        frequency,
        jnp.ones_like(frequency),
        provenance,
        sample_count=sample_count,
        sample_interval=sample_interval,
    )
    plan = gw.WaveformMatchPlan(psd)
    first = jnp.exp(-(((frequency - 10.0) / 4.0) ** 2)).astype("complex128")
    lag = 5 * sample_interval
    second = first * jnp.exp(-2.0j * jnp.pi * frequency * lag + 0.7j)

    fixed = plan.overlap(first, second, time_shift_seconds=lag)
    matched = plan.match(first, second)
    np.testing.assert_allclose(fixed.overlap, 1.0, atol=1.0e-12)
    periodic = plan.overlap(
        first,
        second,
        time_shift_seconds=lag + 1_000_000 * psd.duration,
    )
    np.testing.assert_allclose(periodic.overlap, 1.0, atol=2.0e-9)
    assert periodic.time_shift_seconds.dtype == psd.frequency.dtype
    np.testing.assert_allclose(matched.match, 1.0, atol=1.0e-12)
    np.testing.assert_allclose(matched.lag_seconds, lag, atol=1.0e-12)
    assert int(matched.lag_index) == 5
    expected_norm = (
        4.0
        / psd.duration
        * jnp.sum(jnp.where(psd.active, jnp.abs(first) ** 2 / psd.values, 0.0))
    )
    np.testing.assert_allclose(matched.first_norm_squared, expected_norm)
    assert bool(matched.valid)

    zero = plan.match(jnp.zeros_like(first), second)
    assert not bool(zero.valid)
    assert int(zero.status) == int(gw.WaveformComparisonStatus.ZERO_NORM_WAVEFORM)
    assert bool(jnp.isnan(zero.match))
    assert int(zero.lag_index) == -1
    with pytest.raises(TypeError, match="real numeric scalar"):
        plan.overlap(first, second, time_shift_seconds=True)
