import numpy as np
import pytest

from phydrax import signal, units
from phydrax.chemistry.periodic._kubo import (
    FiniteFrequencyKuboResponse,
    KuboRawTransitions,
    KuboResponseEvidence,
)
from phydrax.chemistry.spectroscopy._instrument import (
    apply_spectral_instrument,
    prepare_spectral_instrument,
    SpectralInstrumentKind,
    SpectralInstrumentPlan,
)
from phydrax.chemistry.spectroscopy._optical import OpticalDielectricPlan
from phydrax.chemistry.spectroscopy._periodic_vibrational import (
    periodic_vibrational_lines,
    PeriodicSpectroscopyEvidence,
    PeriodicSpectroscopyRequest,
    PeriodicSpectroscopyTensorResult,
    PeriodicVibrationalSpectroscopyPlan,
)
from phydrax.chemistry.spectroscopy._profile import SpectralLineShape, SpectralProfilePlan
from phydrax.chemistry.spectroscopy._response import (
    SpectralResponseEvidence,
    SpectralResponseProduct,
    SpectralResponseRepresentation,
)


def _evidence():
    return SpectralResponseEvidence(0.0, 0.0, 0.0, 0.0, True)


def test_raw_lines_and_instrument_product_are_distinct_and_preserve_area():
    lines = SpectralResponseProduct(
        [1.0, 2.0],
        [[2.0, 3.0]],
        [True, True],
        units.ELECTRONVOLT,
        units.ONE,
        ("xx",),
        SpectralResponseRepresentation.LINES,
        "analytic-lines",
        "source",
        _evidence(),
    )
    profile = SpectralProfilePlan(
        SpectralLineShape.GAUSSIAN,
        0.0,
        3.0,
        grid_size=3001,
        fwhm=0.05,
        area_tolerance=1.0e-8,
    )
    plan = SpectralInstrumentPlan(
        SpectralInstrumentKind.LINE_PROFILE,
        "analytic-lines",
        1,
        3001,
        profile=profile,
        area_tolerance=1.0e-8,
    )
    result = apply_spectral_instrument(prepare_spectral_instrument(plan), lines)

    assert result.raw_response is lines
    assert type(result) is not type(lines)
    assert result.theory.values.shape == (3001,)
    np.testing.assert_allclose(
        np.trapezoid(result.convolved_values[0], result.coordinates),
        5.0,
        rtol=1.0e-8,
    )
    assert bool(result.evidence.successful)


def test_stationary_instrument_refuses_line_strengths_instead_of_reinterpreting_them():
    lines = SpectralResponseProduct(
        [1.0, 2.0],
        [[1.0, 0.0]],
        [True, True],
        units.ELECTRONVOLT,
        units.ONE,
        ("line",),
        SpectralResponseRepresentation.LINES,
        "line-source",
        "source",
        _evidence(),
    )
    plan = SpectralInstrumentPlan(
        SpectralInstrumentKind.STATIONARY_KERNEL,
        "line-source",
        1,
        2,
        kernel_capacity=1,
    )
    with pytest.raises(TypeError):
        apply_spectral_instrument(prepare_spectral_instrument(plan, [1.0]), lines)


def test_fourier_positive_exponent_places_negative_phase_tone_at_positive_frequency():
    count = 64
    frequency = 5.0 / count
    time = np.arange(count)
    samples = np.exp(-2.0j * np.pi * frequency * time)
    result = signal.FourierSpectrumPlan(
        count,
        1.0,
        exponent_sign=1,
        shifted=True,
        parseval_tolerance=1.0e-12,
    ).evaluate(samples)

    peak = int(np.argmax(np.abs(result.spectrum)))
    np.testing.assert_allclose(result.frequencies[peak], frequency, atol=1.0e-12)
    assert float(result.parseval_residual) < 1.0e-12
    assert bool(result.successful)


def test_optical_frontend_keeps_drude_separate_and_uses_retarded_dielectric_sign():
    raw = KuboRawTransitions(
        np.zeros((1, 1)),
        np.zeros((1, 1)),
        np.zeros((1, 1, 1)),
        np.zeros((1, 1, 1)),
        np.zeros((1, 1, 1), dtype="bool"),
        np.zeros((1, 1, 1)),
        np.zeros((1, 1, 1, 3, 3)),
        np.eye(3) * 7.0,
        np.eye(3),
        np.eye(3),
        np.eye(3),
        0.0,
        0.0,
        True,
        "kubo-plan",
        "diamagnetic-source",
    )
    kubo_evidence = KuboResponseEvidence(
        1.0,
        0.0,
        True,
        True,
        True,
        "physical-linewidth",
        False,
        False,
    )
    omega = np.asarray([1.0e14, 2.0e14])
    conductivity = np.broadcast_to(np.eye(3), (omega.size, 3, 3)).astype("complex128")
    kubo = FiniteFrequencyKuboResponse(
        omega,
        conductivity,
        raw,
        kubo_evidence,
        True,
        "kubo-plan",
    )
    result = OpticalDielectricPlan(
        np.eye(3),
        ("xx", "yy", "zz"),
    ).evaluate(kubo, units.HERTZ, units.SIEMENS)

    assert bool(result.evidence.successful)
    dielectric_diagonal = np.diagonal(
        np.asarray(result.dielectric_tensor), axis1=1, axis2=2
    )
    assert np.all(np.imag(dielectric_diagonal) > 0.0)
    np.testing.assert_allclose(result.drude_weight, np.eye(3) * 7.0)


def test_periodic_provider_tensors_retain_selection_and_stokes_balance_evidence():
    class Phonons:
        fractional_qpoints = np.zeros((1, 3))
        angular_frequencies = np.asarray([[0.0, 0.0, 0.0, 1.0, 2.0, 3.0]])
        eigenvectors = np.eye(6)[None, :, :]
        acoustic_mask = np.asarray([[True, True, True, False, False, False]])
        imaginary_mask = np.zeros((1, 6), dtype="bool")
        result_id = "gamma-phonons"
        successful = np.asarray(True)

    request = PeriodicSpectroscopyRequest(
        "two-atom-cell",
        Phonons.result_id,
        ("A", "B"),
    )
    raman = np.zeros((6, 3, 3))
    raman[3:] = np.eye(3)
    tensors = PeriodicSpectroscopyTensorResult(
        np.asarray([np.eye(3), -np.eye(3)]),
        raman,
        request,
        PeriodicSpectroscopyEvidence(0.0, 0.0, True),
        "response-provider",
        ("sha256:tensors",),
    )
    result = periodic_vibrational_lines(
        PeriodicVibrationalSpectroscopyPlan(
            [1.0, 1.0],
            [[1.0, 0.0, 0.0]],
            [[1.0, 0.0, 0.0]],
            ("xx",),
            temperature=4.0,
            laser_angular_frequency=20.0,
        ),
        Phonons(),
        tensors,
        units.HERTZ,
        units.ONE,
    )

    assert bool(result.successful)
    assert float(result.ir_lines.values[:, 3:].sum()) > 0.0
    assert float(result.raman_lines.values[0, 3:].sum()) > 0.0
    assert float(result.detailed_balance_residual) < 1.0e-10
