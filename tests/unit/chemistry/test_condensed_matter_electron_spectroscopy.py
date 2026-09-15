import numpy as np
import pytest

from phydrax import units
from phydrax.chemistry.spectroscopy._electron import (
    ARPESPlan,
    PhotoemissionMatrixElementRequest,
    PhotoemissionMatrixElementResult,
    TersoffHamannPlan,
    VacuumLDOSRequest,
    VacuumLDOSResult,
)
from phydrax.chemistry.spectroscopy._loss import (
    ElectronEnergyLossPlan,
    MacroscopicDielectricResult,
)


def test_arpes_requires_provider_matrix_elements_and_preserves_forbidden_channel():
    energy = np.linspace(-5.0, 5.0, 2001)
    gaussian = np.exp(-0.5 * (energy / 0.25) ** 2)
    gaussian /= np.trapezoid(gaussian, energy)
    spectral = np.broadcast_to(gaussian, (2, 2, energy.size)).copy()
    request = PhotoemissionMatrixElementRequest(
        [[0.0, 0.0], [0.5, 0.0]],
        21.2,
        [1.0, 0.0, 0.0],
        "spectral-function",
        "fixed-cut",
    )
    elements = PhotoemissionMatrixElementResult(
        [[1.0, 0.0], [0.0, 0.0]],
        request,
        "matrix-provider",
        ("sha256:matrix",),
        True,
    )
    result = ARPESPlan(
        energy,
        chemical_potential=0.0,
        temperature=0.05,
        kpoint_capacity=2,
        band_capacity=2,
        moment_tolerance=1.0e-6,
    ).evaluate(spectral, "spectral-function", elements, units.ELECTRONVOLT, units.ONE)

    assert bool(result.evidence.successful)
    np.testing.assert_allclose(result.raw_response.values[1], 0.0, atol=0.0)
    assert (
        result.raw_response.values[0, -1]
        < result.raw_response.values[0, energy.size // 2]
    )
    with pytest.raises(ValueError):
        PhotoemissionMatrixElementResult(
            [], request, "matrix-provider", ("sha256:matrix",), True
        )


def test_ters_off_hamann_refuses_negative_ldos_and_closes_current_integral():
    energy = np.linspace(-2.0, 2.0, 2001)
    bias = np.linspace(-1.0, 1.0, 401)
    request = VacuumLDOSRequest([[0.0, 0.0, 5.0]], energy, "surface-green")
    with pytest.raises(ValueError):
        VacuumLDOSResult(
            -np.ones((1, energy.size)),
            request,
            "vacuum-provider",
            ("sha256:ldos",),
            0.0,
            True,
        )
    ldos = VacuumLDOSResult(
        np.ones((1, energy.size)),
        request,
        "vacuum-provider",
        ("sha256:ldos",),
        0.0,
        True,
    )
    result = TersoffHamannPlan(
        bias,
        temperature=0.05,
        current_scale=2.0,
        position_capacity=1,
        closure_tolerance=2.0e-3,
    ).evaluate(ldos, units.VOLT, units.ONE)

    assert bool(result.evidence.successful)
    zero = bias.size // 2
    np.testing.assert_allclose(result.current[0, zero], 0.0, atol=1.0e-14)
    np.testing.assert_allclose(result.raw_didv.values, 2.0, rtol=2.0e-8, atol=2.0e-8)


def test_macroscopic_valence_eels_is_passive_and_refuses_q_zero():
    energy = np.linspace(0.01, 4.0, 2001)
    loss = 0.3 / ((energy - 1.2) ** 2 + 0.3**2)
    dielectric = 1.0 / (1.0 - 1.0j * loss)
    target = np.trapezoid(energy * loss, energy)
    with pytest.raises(ValueError):
        MacroscopicDielectricResult(
            [0.0],
            energy,
            dielectric[None, :],
            [target],
            0.0,
            "longitudinal-response",
            "dielectric-provider",
            ("sha256:dielectric",),
            True,
        )
    supplied = MacroscopicDielectricResult(
        [0.5],
        energy,
        dielectric[None, :],
        [target],
        0.0,
        "longitudinal-response",
        "dielectric-provider",
        ("sha256:dielectric",),
        True,
    )
    result = ElectronEnergyLossPlan(
        temperature=0.2,
        q_capacity=1,
        energy_capacity=energy.size,
        residual_tolerance=1.0e-10,
    ).evaluate(supplied, units.ELECTRONVOLT, units.ONE)

    assert bool(result.evidence.successful)
    assert np.all(np.asarray(result.loss_response.values) >= 0.0)
    assert np.all(np.asarray(result.dynamic_structure_response.values) >= 0.0)
    assert float(result.evidence.fsum_residual) < 1.0e-12
