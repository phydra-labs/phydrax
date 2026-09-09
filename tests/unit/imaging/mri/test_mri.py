import numpy as np

import phydrax as phx


def test_cartesian_encoding_adjoint_and_cg_sense_recover_image():
    coils = phx.imaging.mri.CoilSensitivityField(
        np.ones((1, 4, 4), dtype=np.complex64),
        ("coil-0",),
        field_id="uniform-coil",
    )
    plan = phx.imaging.mri.CartesianMRIEncodingPlan(coils)
    image = np.zeros((4, 4), dtype=np.complex64)
    image[1, 2] = 1.0 + 0.5j
    data, evidence = plan.forward(image)
    np.testing.assert_allclose(plan.adjoint(data), image, atol=1e-6)
    assert bool(evidence.successful)
    reconstruction = phx.imaging.mri.CGSensePlan(plan, 3).reconstruct(data)
    np.testing.assert_allclose(reconstruction.image, image, atol=1e-5)
    assert bool(reconstruction.successful)


def test_nufft_encoding_and_adjoint_obey_complex_inner_product():
    coils = phx.imaging.mri.CoilSensitivityField(
        np.ones((1, 4, 4), dtype=np.complex64),
        ("coil-0",),
        field_id="nufft-coil",
    )
    coordinates = np.asarray(((-1.0, -0.5), (0.0, 0.0), (0.7, 1.2)))
    support = phx.imaging.mri.KSpaceSupport(
        coordinates, ("coil-0",), "trajectory", "scanner"
    )
    plan = phx.imaging.mri.NUFFTMRIEncodingPlan(coils, support, tolerance=1e-5)
    image = np.arange(16, dtype=np.float32).reshape((4, 4)).astype(np.complex64)
    data = np.asarray(((1.0 + 1.0j,), (2.0 - 0.5j,), (-1.0 + 0.2j,)))
    encoded, evidence = plan.forward(image)
    left = np.vdot(encoded, data)
    right = np.vdot(image, np.asarray(plan.adjoint(data)))
    np.testing.assert_allclose(left, right, rtol=2e-4, atol=2e-4)
    assert bool(evidence.successful)


def test_coil_prewhitening_regularization_and_timed_off_resonance():
    covariance = phx.imaging.mri.CoilNoiseCovariance(
        np.asarray(((4.0, 0.0), (0.0, 1.0))),
        ("coil-0", "coil-1"),
    )
    whitened = covariance.whiten(np.asarray((2.0, 1.0)))
    np.testing.assert_allclose(whitened, (1.0, 1.0))

    coils = phx.imaging.mri.CoilSensitivityField(
        np.ones((1, 4, 4), dtype=np.complex64),
        ("coil-0",),
        field_id="timed-coil",
    )
    time = phx.measurement.SampleTimeAxis(
        "readout",
        np.asarray((0.0, 0.1, 0.2)),
        phx.units.SECOND,
    )
    support = phx.imaging.mri.KSpaceSupport(
        np.asarray(((-1.0, 0.0), (0.0, 0.0), (1.0, 0.0))),
        ("coil-0",),
        "timed-trajectory",
        "scanner",
        time,
    )
    encoding = phx.imaging.mri.NUFFTMRIEncodingPlan(coils, support)
    off_resonance = phx.imaging.mri.OffResonanceMRIEncodingPlan(
        encoding, np.zeros((4, 4))
    )
    encoded = off_resonance.forward(np.ones((4, 4), dtype=np.complex64))
    assert encoded.shape == support.sample_shape
    assert np.all(np.isfinite(np.asarray(encoded)))

    cartesian = phx.imaging.mri.CartesianMRIEncodingPlan(coils)
    data, _ = cartesian.forward(np.eye(4, dtype=np.complex64))
    regularized = phx.imaging.mri.RegularizedMRIPlan(
        phx.imaging.mri.CGSensePlan(cartesian, 2),
        shrinkage=0.01,
        outer_iterations=2,
    ).reconstruct(data)
    assert bool(regularized.successful)


def test_phase_contrast_quantitative_and_bloch_models_keep_physics_explicit():
    phase = phx.imaging.mri.PhaseContrastMRIPlan(2.0)
    velocity = np.asarray((0.5, -0.5))
    signal = phase.encode(np.ones(2), velocity)
    np.testing.assert_allclose(phase.decode(signal), velocity)

    quantitative = phx.imaging.mri.QuantitativeMRIPlan(1.0, 0.1)
    value = quantitative.signal(np.ones(2), np.ones(2), np.ones(2))
    assert np.all(np.asarray(np.abs(value)) > 0.0)

    bloch = phx.imaging.mri.BlochSequencePlan(0.1, 1.0).simulate(
        np.asarray((1.0, 0.0, 0.0)),
        np.zeros((3, 3)),
        1.0,
        0.5,
    )
    assert bool(bloch.successful)
    assert bloch.magnetization[0] < 1.0
    assert bloch.magnetization[2] > 0.0
