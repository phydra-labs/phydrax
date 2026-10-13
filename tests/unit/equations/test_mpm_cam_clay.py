#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array
from numpy.typing import NDArray
from scipy.optimize import root_scalar

import phydrax as phx
from phydrax.equations import MPMConstitutiveResponse


pytestmark = pytest.mark.strict_jax


@dataclass(frozen=True)
class _Reference:
    """Host-only Borja–Tamagnini (1998), Sections 3–4, equation data."""

    mu0: float = 60.0
    alpha0: float = 0.0
    slope: float = 1.2
    lambda_raw: float = 0.12
    kappa_raw: float = 0.02
    e0: float = 0.8
    p0: float = 1.0

    @property
    def swelling(self) -> float:
        return self.kappa_raw / (1.0 + self.e0)

    @property
    def hardening(self) -> float:
        return (self.lambda_raw - self.kappa_raw) / (1.0 + self.e0)

    def parameters(self) -> phx.applications.solid_mechanics.ModifiedCamClayParameters:
        return phx.applications.solid_mechanics.ModifiedCamClayParameters(
            *(
                jnp.asarray(value, dtype=jnp.float64)
                for value in (
                    self.mu0,
                    self.alpha0,
                    self.slope,
                    self.lambda_raw,
                    self.kappa_raw,
                )
            )
        )


def _principal_elasticity(
    strain: NDArray[np.float64], reference: _Reference
) -> tuple[NDArray[np.float64], NDArray[np.float64], float]:
    """Independent energy gradient/Hessian; no implicit return-map equations."""
    volume = np.sum(strain)
    deviator = strain - volume / 3.0
    base = reference.p0 * np.exp(-volume / reference.swelling)
    shear = reference.mu0 + reference.alpha0 * base
    squared = np.dot(deviator, deviator)
    pressure = base * (1.0 + reference.alpha0 * squared / reference.swelling)
    stress = -pressure + 2.0 * shear * deviator
    ones = np.ones((3, 3), dtype=np.float64)
    hessian = (
        pressure / reference.swelling * ones
        - 2.0
        * reference.alpha0
        * base
        / reference.swelling
        * (deviator[:, None] + deviator[None, :])
        + 2.0 * shear * (np.eye(3, dtype=np.float64) - ones / 3.0)
    )
    energy = base * reference.swelling + shear * squared
    return stress, hessian, float(energy)


def _rate(
    history: NDArray[np.float64], reference: _Reference, drained: bool
) -> NDArray[np.float64]:
    """Coaxial associated MCC consistency, integrated independently with RK4.

    History is elastic principal strain, pc, plastic log volume, total lateral
    log strain. The monotonic NC paths remain active (gamma_dot >= 0).
    """
    stress, elasticity, _ = _principal_elasticity(history[:3], reference)
    pressure = -np.mean(stress)
    deviator = stress + pressure
    pc = history[3]
    normal = 3.0 * deviator / reference.slope**2 - (2.0 * pressure - pc) / 3.0
    elastic_normal = elasticity @ normal
    denominator = (
        normal @ elastic_normal
        + pressure * pc * (2.0 * pressure - pc) / reference.hardening
    )
    assert denominator > 0.0, "Rate reference encountered singular consistency."
    lateral = 0.5
    if drained:
        elastoplastic = (
            elasticity - np.outer(elastic_normal, elastic_normal) / denominator
        )
        lateral = elastoplastic[1, 0] / (elastoplastic[1, 1] + elastoplastic[1, 2])
    total_rate = np.asarray((-1.0, lateral, lateral), dtype=np.float64)
    gamma = float(normal @ elasticity @ total_rate / denominator)
    assert gamma >= -1e-12, "Reference path left its active loading branch."
    plastic = gamma * normal
    return np.concatenate(
        (
            total_rate - plastic,
            np.asarray(
                (-pc * np.sum(plastic) / reference.hardening, np.sum(plastic), lateral),
                dtype=np.float64,
            ),
        )
    )


def _rate_path(
    reference: _Reference, strain: float, steps: int, *, drained: bool
) -> NDArray[np.float64]:
    history = np.asarray((0.0, 0.0, 0.0, reference.p0, 0.0, 0.0), dtype=np.float64)
    path = np.empty((steps + 1, 5), dtype=np.float64)
    step = strain / steps
    for index in range(steps + 1):
        stress, _, _ = _principal_elasticity(history[:3], reference)
        pressure = -np.mean(stress)
        equivalent = np.sqrt(1.5 * np.dot(stress + pressure, stress + pressure))
        path[index] = (pressure, equivalent, history[3], history[4], history[5])
        if index == steps:
            break
        k1 = _rate(history, reference, drained)
        k2 = _rate(history + step * k1 / 2.0, reference, drained)
        k3 = _rate(history + step * k2 / 2.0, reference, drained)
        k4 = _rate(history + step * k3, reference, drained)
        history = history + step * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
    return path


def _metric(history: Array) -> NDArray[np.float64]:
    packed = np.asarray(history, dtype=np.float64)
    return np.asarray(
        (
            (packed[0], packed[5], packed[4]),
            (packed[5], packed[1], packed[3]),
            (packed[4], packed[3], packed[2]),
        ),
        dtype=np.float64,
    )


def _spectral_exp(matrix: NDArray[np.float64]) -> NDArray[np.float64]:
    values, vectors = np.linalg.eigh(matrix)
    return (vectors * np.exp(values)[None, :]) @ vectors.T


def _prestress(
    plan: phx.applications.solid_mechanics.ModifiedCamClayMPMConstitutivePlan,
    reference: _Reference,
    *,
    ocr: float = 1.0,
) -> Array:
    initialized = plan.initialize_state_from_stress(
        -reference.p0 * jnp.eye(3, dtype=jnp.float64),
        reference.parameters(),
        preconsolidation_pressure=ocr * reference.p0,
        initial_void_ratio=reference.e0,
    )
    assert bool(initialized.admissible)
    return initialized.state


def _discrete_path(
    reference: _Reference, strain: float, steps: int, *, drained: bool
) -> NDArray[np.float64]:
    """Consumer driver: prescribed axial log strain and Kirchhoff confinement."""
    plan = phx.applications.solid_mechanics.ModifiedCamClayMPMConstitutivePlan()
    parameters = reference.parameters()
    committed = _prestress(plan, reference)
    step = strain / steps

    @jax.jit
    def response(deformation: Array, history: Array) -> MPMConstitutiveResponse:
        return plan.evaluate(deformation, history, 1.0, parameters, 0.0, step)

    path = np.empty((steps + 1, 5), dtype=np.float64)
    path[0] = (reference.p0, 0.0, reference.p0, 0.0, 0.0)
    lateral = 0.0
    initial_plastic_volume = -0.5 * np.linalg.slogdet(_metric(committed))[1]
    for index in range(1, steps + 1):
        axial = -index * step

        def deformation(lateral_strain: float) -> Array:
            return jnp.diag(
                jnp.exp(
                    jnp.asarray(
                        (axial, lateral_strain, lateral_strain), dtype=jnp.float64
                    )
                )
            )

        if drained:

            def confinement(lateral_strain: float) -> float:
                gradient = deformation(lateral_strain)
                candidate = response(gradient, committed)
                assert bool(candidate.successful)
                tau = candidate.first_piola @ gradient.T
                return float(tau[1, 1]) + reference.p0

            root = root_scalar(
                confinement,
                bracket=(lateral - step, lateral + step),
                method="brentq",
                xtol=1e-12,
            )
            assert root.converged
            lateral = root.root
        else:
            lateral = -axial / 2.0
        gradient = deformation(lateral)
        result = response(gradient, committed)
        assert bool(result.successful), f"Constitutive failure at strain step {index}."
        assert bool(result.diagnostics["root_successful"])
        pressure = float(result.diagnostics["mean_pressure"])
        equivalent = float(result.diagnostics["equivalent_stress"])
        pc = float(result.diagnostics["preconsolidation_pressure"])
        assert (
            abs(equivalent**2 / reference.slope**2 + pressure * (pressure - pc))
            / max(reference.p0**2, pc**2)
            < 2e-7
        )
        assert float(result.diagnostics["plastic_multiplier"]) >= 0.0
        committed = result.trial_state
        plastic_volume = -0.5 * np.linalg.slogdet(_metric(committed))[1]
        np.testing.assert_allclose(
            pc / reference.p0,
            np.exp(-(plastic_volume - initial_plastic_volume) / reference.hardening),
            rtol=2e-8,
        )
        if drained:
            np.testing.assert_allclose(
                (result.first_piola @ gradient.T)[1:, 1:],
                -reference.p0 * np.eye(2, dtype=np.float64),
                atol=2e-8,
            )
        path[index] = (
            pressure,
            equivalent,
            pc,
            plastic_volume - initial_plastic_volume,
            lateral,
        )
    return path


def test_numpy_rate_undrained_nc_matches_closed_form() -> None:
    # Scale elastic/plastic compliance by 1/10 and shear stiffness by 10:
    # the same dimensionless critical-state path now lies at small strain.
    reference = _Reference(mu0=600.0, lambda_raw=0.012, kappa_raw=0.002)
    coarse = _rate_path(reference, 0.012, 960, drained=False)
    fine = _rate_path(reference, 0.012, 1920, drained=False)
    # The alpha=0, fixed-principal-axis, constant-reference-volume limit obeys
    # kappa d ln p + (lambda-kappa) d ln pc=0 independently of shear stiffness.
    eta = fine[:, 1] / fine[:, 0]
    exponent = (reference.lambda_raw - reference.kappa_raw) / reference.lambda_raw
    closed = (reference.slope**2 / (reference.slope**2 + eta**2)) ** exponent
    np.testing.assert_allclose(fine[:, 0] / reference.p0, closed, rtol=2e-8, atol=2e-10)
    np.testing.assert_allclose(coarse, fine[::2], rtol=2e-8, atol=2e-10)
    assert eta[-1] > 0.98 * reference.slope


def test_discrete_undrained_nc_converges_to_independent_rate_path() -> None:
    reference = _Reference(mu0=600.0, lambda_raw=0.012, kappa_raw=0.002)
    oracle = _rate_path(reference, 0.012, 1920, drained=False)
    coarse = _discrete_path(reference, 0.012, 96, drained=False)
    fine = _discrete_path(reference, 0.012, 192, drained=False)
    coarse_error = np.max(np.abs(coarse[:, :3] - oracle[::20, :3])) / reference.p0
    fine_error = np.max(np.abs(fine[:, :3] - oracle[::10, :3])) / reference.p0
    # Backward-Euler normal flow is first order: halving Delta epsilon must
    # reduce trajectory error, not merely satisfy a broad terminal tolerance.
    assert fine_error < 0.7 * coarse_error
    assert fine_error < 0.012
    np.testing.assert_allclose(fine[:, 3:], oracle[::10, 3:], atol=5e-4, rtol=0.0)
    np.testing.assert_allclose(fine[-1, 1] / fine[-1, 0], reference.slope, rtol=0.02)


def test_drained_nc_constant_kirchhoff_confinement_reaches_critical_state() -> None:
    reference = _Reference(mu0=600.0, lambda_raw=0.012, kappa_raw=0.002)
    # This spans the unscaled material's 0.6 axial-log-strain trajectory,
    # approaching the critical limit without leaving the small-strain regime.
    coarse_oracle = _rate_path(reference, 0.06, 3072, drained=True)
    oracle = _rate_path(reference, 0.06, 6144, drained=True)
    np.testing.assert_allclose(coarse_oracle, oracle[::2], rtol=2e-7, atol=2e-9)
    coarse = _discrete_path(reference, 0.06, 96, drained=True)
    fine = _discrete_path(reference, 0.06, 192, drained=True)
    coarse_error = np.max(np.abs(coarse[:, :3] - oracle[::64, :3])) / reference.p0
    fine_error = np.max(np.abs(fine[:, :3] - oracle[::32, :3])) / reference.p0
    assert fine_error < 0.75 * coarse_error
    # Delta epsilon=3.125e-4; the first-order implicit/rate discrepancy is
    # bounded over the entire p,q,pc trajectory, not only at critical state.
    assert fine_error < 0.04
    np.testing.assert_allclose(fine[:, 3:], oracle[::32, 3:], atol=2e-3, rtol=0.0)
    p, q, pc = fine[-1, :3]
    np.testing.assert_allclose(q / p, reference.slope, rtol=0.015)
    np.testing.assert_allclose(pc / p, 2.0, rtol=0.015)
    np.testing.assert_allclose(
        p, reference.p0 / (1.0 - reference.slope / 3.0), rtol=0.015
    )
    np.testing.assert_allclose(fine[-1, 3], fine[-2, 3], atol=2e-5, rtol=0.0)


@pytest.mark.parametrize(
    ("mu0", "alpha0"),
    ((12.0, 0.0), (12.0, 2.0), (0.0, 2.0)),
    ids=("constant-shear", "coupled-shear", "pressure-only-shear"),
)
def test_elastic_energy_gradient_matches_kirchhoff_stress(
    mu0: float, alpha0: float
) -> None:
    reference = _Reference(mu0=mu0, alpha0=alpha0)
    plan = phx.applications.solid_mechanics.ModifiedCamClayMPMConstitutivePlan()
    history = _prestress(plan, reference, ocr=4.0)
    strain = np.asarray(
        ((-0.006, 0.002, 0.001), (0.002, 0.001, -0.001), (0.001, -0.001, -0.002)),
        dtype=np.float64,
    )
    gradient = jnp.asarray(_spectral_exp(strain), dtype=jnp.float64)
    response = plan.evaluate(gradient, history, 1.0, reference.parameters(), 0.0, 0.01)
    assert bool(response.successful)
    assert response.diagnostics["plastic_multiplier"] == 0.0
    assert plan.capabilities.has_free_energy
    eigenvalues = np.linalg.eigvalsh(strain)
    _, _, energy = _principal_elasticity(eigenvalues, reference)
    np.testing.assert_allclose(response.reference_energy_density, energy, rtol=2e-10)
    finite_difference = np.empty((3, 3), dtype=np.float64)
    step = 1e-7
    for row in range(3):
        for column in range(row, 3):
            direction = np.zeros((3, 3), dtype=np.float64)
            direction[row, column] += 0.5
            direction[column, row] += 0.5
            plus = _principal_elasticity(
                np.linalg.eigvalsh(strain + step * direction), reference
            )[2]
            minus = _principal_elasticity(
                np.linalg.eigvalsh(strain - step * direction), reference
            )[2]
            finite_difference[row, column] = finite_difference[column, row] = (
                plus - minus
            ) / (2.0 * step)
    np.testing.assert_allclose(
        response.first_piola @ gradient.T, finite_difference, rtol=2e-7, atol=2e-8
    )


@pytest.mark.parametrize(
    ("mu0", "alpha0"),
    ((12.0, 0.0), (12.0, 2.0), (0.0, 2.0)),
    ids=("constant-shear", "coupled-shear", "pressure-only-shear"),
)
def test_anisotropic_batched_prestress_inverse_round_trip(
    mu0: float, alpha0: float
) -> None:
    reference = _Reference(mu0=mu0, alpha0=alpha0, p0=2.0)
    plan = phx.applications.solid_mechanics.ModifiedCamClayMPMConstitutivePlan()
    stress = jnp.asarray(
        ((-2.06, 0.01, 0.0), (0.01, -1.98, 0.0), (0.0, 0.0, -1.96)), dtype=jnp.float64
    )
    batch = jnp.stack((stress, 1.5 * stress))
    initialized = plan.initialize_state_from_stress(
        batch,
        reference.parameters(),
        preconsolidation_pressure=jnp.asarray((4.0, 6.0), dtype=jnp.float64),
        initial_void_ratio=reference.e0,
    )
    assert bool(jnp.all(initialized.admissible))
    assert initialized.state.shape == (2, 9)
    identity = jnp.broadcast_to(jnp.eye(3, dtype=jnp.float64), (2, 3, 3))
    response = plan.evaluate(
        identity,
        initialized.state,
        jnp.ones((2,), dtype=jnp.float64),
        reference.parameters(),
        0.0,
        0.01,
    )
    assert bool(jnp.all(response.successful))
    np.testing.assert_allclose(response.first_piola, batch, rtol=2e-9, atol=2e-10)
    np.testing.assert_allclose(
        response.trial_state, initialized.state, rtol=2e-9, atol=2e-10
    )
    np.testing.assert_allclose(initialized.state[:, 7], (2.0, 3.0), rtol=2e-12)
    np.testing.assert_allclose(initialized.state[:, 8], reference.e0, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(response.diagnostics["reference_pressure"], (2.0, 3.0))
    np.testing.assert_allclose(response.diagnostics["initial_void_ratio"], reference.e0)


@pytest.mark.parametrize("plastic", (False, True), ids=("elastic", "plastic"))
def test_cam_clay_objectivity(plastic: bool) -> None:
    reference = _Reference(alpha0=2.0)
    plan = phx.applications.solid_mechanics.ModifiedCamClayMPMConstitutivePlan()
    history = _prestress(plan, reference, ocr=1.0 if plastic else 4.0)
    gradient = jnp.asarray(
        ((0.98, 0.02, 0.0), (0.0, 1.005, 0.01), (0.0, 0.0, 1.002)), dtype=jnp.float64
    )
    if not plastic:
        identity = jnp.eye(3, dtype=jnp.float64)
        gradient = identity + 0.2 * (gradient - identity)
    angle = 0.37
    rotation = jnp.asarray(
        (
            (np.cos(angle), -np.sin(angle), 0.0),
            (np.sin(angle), np.cos(angle), 0.0),
            (0.0, 0.0, 1.0),
        ),
        dtype=jnp.float64,
    )
    original = plan.evaluate(gradient, history, 1.0, reference.parameters(), 0.0, 0.01)
    rotated = plan.evaluate(
        rotation @ gradient, history, 1.0, reference.parameters(), 0.0, 0.01
    )
    assert bool(original.successful) and bool(rotated.successful)
    assert bool(original.diagnostics["plastic_multiplier"] > 0.0) == plastic
    np.testing.assert_allclose(
        rotated.first_piola, rotation @ original.first_piola, rtol=2e-8, atol=2e-9
    )
    np.testing.assert_allclose(
        rotated.trial_state, original.trial_state, rtol=2e-8, atol=2e-9
    )
    np.testing.assert_allclose(
        rotated.reference_energy_density, original.reference_energy_density, rtol=2e-8
    )


@pytest.mark.parametrize(
    "branch", ("elastic-identity", "plastic-hydrostatic", "plastic-distinct")
)
def test_strict_branch_tangent_jvp_vjp_matches_finite_difference(branch: str) -> None:
    reference = _Reference(alpha0=2.0)
    plan = phx.applications.solid_mechanics.ModifiedCamClayMPMConstitutivePlan()
    parameters = reference.parameters()
    history = _prestress(
        plan, reference, ocr=3.0 if branch == "elastic-identity" else 1.0
    )
    if branch == "elastic-identity":
        gradient = jnp.eye(3, dtype=jnp.float64)
    elif branch == "plastic-hydrostatic":
        gradient = np.exp(-0.01) * jnp.eye(3, dtype=jnp.float64)
    else:
        gradient = jnp.asarray(
            ((0.97, 0.015, 0.0), (0.0, 1.006, 0.005), (0.0, 0.0, 1.002)),
            dtype=jnp.float64,
        )
    direction = jnp.asarray(
        ((0.2, 0.1, -0.05), (-0.03, -0.3, 0.1), (0.04, -0.02, 0.4)), dtype=jnp.float64
    )
    cotangent = jnp.asarray(
        ((0.3, -0.2, 0.1), (0.07, 0.4, -0.1), (-0.03, 0.2, -0.5)), dtype=jnp.float64
    )

    def stress(deformation: Array) -> Array:
        return plan.evaluate(deformation, history, 1.0, parameters, 0.0, 0.01).first_piola

    linearized = plan.evaluate_linearized(gradient, history, 1.0, parameters, 0.0, 0.01)
    assert bool(linearized.tangent_successful)
    assert bool(jnp.all(jnp.isfinite(linearized.algorithmic_tangent)))
    response = plan.evaluate(gradient, history, 1.0, parameters, 0.0, 0.01)
    assert bool(response.successful)
    assert bool(response.diagnostics["plastic_multiplier"] > 0.0) == (
        branch != "elastic-identity"
    )
    step = 2e-6
    plus = plan.evaluate(gradient + step * direction, history, 1.0, parameters, 0.0, 0.01)
    minus = plan.evaluate(
        gradient - step * direction, history, 1.0, parameters, 0.0, 0.01
    )
    assert bool(plus.successful) and bool(minus.successful)
    fd = (plus.first_piola - minus.first_piola) / (2.0 * step)
    _, jvp = jax.jvp(stress, (gradient,), (direction,))
    _, pullback = jax.vjp(stress, gradient)
    vjp = pullback(cotangent)[0]
    action = (
        linearized.algorithmic_tangent.reshape((9, 9)) @ direction.reshape((9,))
    ).reshape((3, 3))
    np.testing.assert_allclose(action, fd, rtol=2e-5, atol=2e-6)
    np.testing.assert_allclose(jvp, fd, rtol=2e-5, atol=2e-6)
    np.testing.assert_allclose(
        jnp.sum(vjp * direction), jnp.sum(cotangent * fd), rtol=2e-5, atol=2e-6
    )
    np.testing.assert_allclose(
        jnp.sum(vjp * direction), jnp.sum(cotangent * jvp), rtol=2e-9, atol=2e-9
    )


def test_mixed_elastic_plastic_batch_preserves_reverse_tangents() -> None:
    reference = _Reference(alpha0=2.0)
    plan = phx.applications.solid_mechanics.ModifiedCamClayMPMConstitutivePlan()
    parameters = reference.parameters()
    history = jnp.broadcast_to(_prestress(plan, reference, ocr=3.0), (2, 9))
    identity = jnp.eye(3, dtype=jnp.float64)
    gradient = jnp.stack((identity, np.exp(-0.01) * identity))
    direction = jnp.broadcast_to(
        jnp.asarray(
            ((0.2, 0.1, -0.05), (-0.03, -0.3, 0.1), (0.04, -0.02, 0.4)),
            dtype=jnp.float64,
        ),
        gradient.shape,
    )
    cotangent = jnp.broadcast_to(
        jnp.asarray(
            ((0.3, -0.2, 0.1), (0.07, 0.4, -0.1), (-0.03, 0.2, -0.5)),
            dtype=jnp.float64,
        ),
        gradient.shape,
    )
    response = plan.evaluate(gradient, history, 1.0, parameters, 0.0, 0.01)
    assert bool(jnp.all(response.successful))
    assert bool(jnp.all(response.diagnostics["tangent_successful"]))
    np.testing.assert_array_equal(response.diagnostics["branch_code"], (0, 1))

    def stress(deformation: Array) -> Array:
        return plan.evaluate(deformation, history, 1.0, parameters, 0.0, 0.01).first_piola

    step = 2e-6
    plus = plan.evaluate(gradient + step * direction, history, 1.0, parameters, 0.0, 0.01)
    minus = plan.evaluate(
        gradient - step * direction, history, 1.0, parameters, 0.0, 0.01
    )
    assert bool(jnp.all(plus.successful & minus.successful))
    fd = (plus.first_piola - minus.first_piola) / (2.0 * step)
    _, jvp = jax.jvp(stress, (gradient,), (direction,))
    vjp = jax.vjp(stress, gradient)[1](cotangent)[0]
    assert bool(jnp.all(jnp.isfinite(vjp)))
    np.testing.assert_allclose(jvp, fd, rtol=2e-5, atol=2e-6)
    np.testing.assert_allclose(
        jnp.sum(vjp * direction, axis=(-2, -1)),
        jnp.sum(cotangent * fd, axis=(-2, -1)),
        rtol=2e-5,
        atol=2e-6,
    )


def test_plastic_yield_hardening_and_metric_determinant() -> None:
    reference = _Reference(alpha0=2.0)
    plan = phx.applications.solid_mechanics.ModifiedCamClayMPMConstitutivePlan()
    history = _prestress(plan, reference)
    gradient = jnp.diag(jnp.exp(jnp.asarray((-0.03, 0.005, 0.002), dtype=jnp.float64)))
    result = plan.evaluate(gradient, history, 1.0, reference.parameters(), 0.0, 0.01)
    assert bool(result.successful) and bool(result.diagnostics["root_successful"])
    pressure = float(result.diagnostics["mean_pressure"])
    equivalent = float(result.diagnostics["equivalent_stress"])
    pc = float(result.diagnostics["preconsolidation_pressure"])
    gamma = float(result.diagnostics["plastic_multiplier"])
    residual = equivalent**2 / reference.slope**2 + pressure * (pressure - pc)
    assert gamma > 0.0
    assert abs(residual) / pc**2 < 2e-8
    np.testing.assert_allclose(result.diagnostics["yield_residual"], residual, atol=2e-9)
    metric = _metric(result.trial_state)
    assert np.min(np.linalg.eigvalsh(metric)) > 0.0
    volume_increment = -0.5 * (
        np.linalg.slogdet(metric)[1] - np.linalg.slogdet(_metric(history))[1]
    )
    np.testing.assert_allclose(
        volume_increment, -gamma * (2.0 * pressure - pc), rtol=2e-8
    )
    np.testing.assert_allclose(
        pc, reference.p0 * np.exp(-volume_increment / reference.hardening), rtol=2e-8
    )
    np.testing.assert_allclose(
        result.diagnostics["elastic_volume_strain"],
        np.linalg.slogdet(np.asarray(gradient))[1] - volume_increment,
        atol=2e-10,
    )
    elastic_volume = float(result.diagnostics["elastic_volume_strain"])
    elastic_shear = float(result.diagnostics["elastic_shear_strain"])
    base = reference.p0 * np.exp(-elastic_volume / reference.swelling)
    np.testing.assert_allclose(
        equivalent,
        3.0 * (reference.mu0 + reference.alpha0 * base) * elastic_shear,
        rtol=2e-9,
    )
    np.testing.assert_allclose(
        pressure,
        base * (1.0 + 1.5 * reference.alpha0 * elastic_shear**2 / reference.swelling),
        rtol=2e-9,
    )
    np.testing.assert_allclose(result.trial_state[7:], history[7:], rtol=0.0, atol=0.0)
    assert result.dissipation_increment >= 0.0


@pytest.mark.parametrize("pressure", (0.0, -1.0), ids=("zero-pressure", "tension"))
def test_noncompressive_prestress_is_inadmissible(pressure: float) -> None:
    reference = _Reference()
    plan = phx.applications.solid_mechanics.ModifiedCamClayMPMConstitutivePlan()
    initialized = plan.initialize_state_from_stress(
        -pressure * jnp.eye(3, dtype=jnp.float64),
        reference.parameters(),
        preconsolidation_pressure=2.0,
        initial_void_ratio=reference.e0,
    )
    assert not bool(initialized.admissible)


def test_uninitialized_history_rejects_and_retains_committed_state() -> None:
    reference = _Reference()
    plan = phx.applications.solid_mechanics.ModifiedCamClayMPMConstitutivePlan()
    history = plan.initialize_state((2,), jnp.float64)
    expected = np.broadcast_to(
        np.asarray((1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0), dtype=np.float64),
        (2, 9),
    )
    np.testing.assert_array_equal(history, expected)
    gradient = jnp.broadcast_to(jnp.eye(3, dtype=jnp.float64), (2, 3, 3))
    response = plan.evaluate(
        gradient,
        history,
        jnp.ones((2,), dtype=jnp.float64),
        reference.parameters(),
        0.0,
        0.01,
    )
    assert not bool(jnp.any(response.successful))
    assert not bool(jnp.any(response.admissible))
    np.testing.assert_array_equal(response.trial_state, history)
    linearized = plan.evaluate_linearized(
        gradient,
        history,
        jnp.ones((2,), dtype=jnp.float64),
        reference.parameters(),
        0.0,
        0.01,
    )
    assert not bool(jnp.any(linearized.tangent_successful))


def test_invalid_deformation_retains_prestressed_history() -> None:
    reference = _Reference()
    plan = phx.applications.solid_mechanics.ModifiedCamClayMPMConstitutivePlan()
    history = _prestress(plan, reference, ocr=2.0)
    result = plan.evaluate(
        jnp.diag(jnp.asarray((-1.0, 1.0, 1.0), dtype=jnp.float64)),
        history,
        1.0,
        reference.parameters(),
        0.0,
        0.01,
    )
    assert not bool(result.successful)
    assert not bool(result.admissible)
    np.testing.assert_array_equal(result.trial_state, history)


def test_implicit_mpm_prestressed_rest_succeeds() -> None:
    reference = _Reference()
    material = phx.applications.solid_mechanics.ModifiedCamClayMPMConstitutivePlan()
    # Uniform periodic particles integrate a constant prestress exactly; an
    # isolated cloud would have an unbalanced pressure boundary and is not rest.
    count_per_axis = 4
    coordinates = (jnp.arange(count_per_axis, dtype=jnp.float64) + 0.5) / count_per_axis
    position = jnp.stack(
        jnp.meshgrid(coordinates, coordinates, coordinates, indexing="ij"), axis=-1
    ).reshape((-1, 3))
    count = position.shape[0]
    volume = jnp.full((count,), 1.0 / count, dtype=jnp.float64)
    bounds = jnp.asarray(((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)), dtype=jnp.float64)
    grid = phx.discretization.TensorGridPlan(
        tuple(
            phx.discretization.UniformAxisSpec(
                count_per_axis, periodic=True, endpoint=False
            )
            for _ in range(3)
        ),
        axis_names=("x", "y", "z"),
    ).prepare(bounds)
    particles = phx.discretization.ParticleSetPlan(
        jnp.arange(count, dtype=jnp.int64),
        volume,
        ambient_dimension=3,
    ).prepare()
    splat = phx.discretization.ParticleGridSplatPlan(
        grid,
        assignment=phx.discretization.TensorBSplineSplatAssignment(2),
    ).prepare(particles)
    compiled = phx.equations.compile_material_point_problem(
        phx.equations.MaterialPointProblemIR("cam-clay-prestressed-rest", material),
        particles,
        splat,
        phx.discretization.ExplicitMPMMethodPlan(),
        phx.discretization.MPMParticleDomainPlan(
            bounds, periodic=(True, True, True), support_margin=0.0
        ),
    )
    arguments = phx.equations.MaterialPointArguments(reference.parameters())
    history = jnp.broadcast_to(_prestress(material, reference, ocr=2.0), (count, 9))
    initial = compiled.initialize_state(
        position, jnp.zeros_like(position), volume, arguments, material_state=history
    )
    detail = phx.solver.PreparedImplicitMPMDynamics(compiled.dynamics).step_detailed(
        initial, 0.0005, arguments
    )
    assert bool(detail.successful)
    assert bool(detail.diagnostics.tangent_successful)
    assert detail.diagnostics.residual_norm < 1e-8
    np.testing.assert_allclose(
        detail.accepted_state.particles.position, position, atol=1e-11, rtol=0.0
    )
    np.testing.assert_allclose(
        detail.accepted_state.particles.velocity, 0.0, atol=1e-10, rtol=0.0
    )
    np.testing.assert_allclose(
        detail.accepted_state.particles.material_state, history, atol=1e-10, rtol=0.0
    )


@pytest.mark.parametrize(
    "values",
    (
        (-1.0, 2.0, 1.2, 0.12, 0.02),
        (12.0, -1.0, 1.2, 0.12, 0.02),
        (0.0, 0.0, 1.2, 0.12, 0.02),
        (12.0, 2.0, 0.0, 0.12, 0.02),
        (12.0, 2.0, -1.0, 0.12, 0.02),
        (12.0, 2.0, 1.2, 0.02, 0.02),
        (12.0, 2.0, 1.2, 0.01, 0.02),
        (12.0, 2.0, 1.2, 0.12, 0.0),
        (12.0, 2.0, 1.2, 0.12, -0.01),
        (np.nan, 2.0, 1.2, 0.12, 0.02),
        (12.0, 2.0, 1.2, np.inf, 0.02),
    ),
    ids=(
        "negative-mu0",
        "negative-alpha0",
        "zero-total-shear",
        "zero-M",
        "negative-M",
        "equal-slopes",
        "reversed-slopes",
        "zero-kappa",
        "negative-kappa",
        "nan",
        "infinity",
    ),
)
def test_parameter_domain_refuses_invalid_values(
    values: tuple[float, float, float, float, float],
) -> None:
    with pytest.raises(ValueError):
        phx.applications.solid_mechanics.ModifiedCamClayParameters(
            *(jnp.asarray(value, dtype=jnp.float64) for value in values)
        )


def test_zero_initial_void_ratio_is_admissible() -> None:
    reference = _Reference(e0=0.0)
    plan = phx.applications.solid_mechanics.ModifiedCamClayMPMConstitutivePlan()
    history = _prestress(plan, reference, ocr=2.0)
    response = plan.evaluate(
        jnp.eye(3, dtype=jnp.float64),
        history,
        1.0,
        reference.parameters(),
        0.0,
        0.01,
    )
    assert bool(response.successful)
    np.testing.assert_allclose(
        response.first_piola, -np.eye(3, dtype=np.float64), atol=1e-12
    )
    assert response.diagnostics["initial_void_ratio"] == 0.0
    assert response.trial_state[8] == 0.0


def test_indefinite_coupled_elasticity_rejects_and_retains_history() -> None:
    reference = _Reference(mu0=0.0, alpha0=2.0)
    plan = phx.applications.solid_mechanics.ModifiedCamClayMPMConstitutivePlan()
    history = _prestress(plan, reference, ocr=100.0)
    strain = np.asarray((-0.2, 0.1, 0.1), dtype=np.float64)
    stress, hessian, _ = _principal_elasticity(strain, reference)
    pressure = -np.mean(stress)
    equivalent_squared = 1.5 * np.dot(stress + pressure, stress + pressure)
    # The trial point is inside the ellipse; failure belongs to the elastic
    # energy domain, not yielding or a failed plastic root.
    assert equivalent_squared / reference.slope**2 + pressure * (pressure - 100.0) < 0.0
    assert np.min(np.linalg.eigvalsh(hessian)) < 0.0
    gradient = jnp.diag(jnp.exp(jnp.asarray(strain, dtype=jnp.float64)))
    result = plan.evaluate(
        gradient,
        history,
        1.0,
        reference.parameters(),
        0.0,
        0.01,
    )
    assert not bool(result.successful)
    assert not bool(result.admissible)
    np.testing.assert_array_equal(result.trial_state, history)
    linearized = plan.evaluate_linearized(
        gradient,
        history,
        1.0,
        reference.parameters(),
        0.0,
        0.01,
    )
    assert not bool(linearized.tangent_successful)


def test_cam_clay_yield_boundary_tangent_uncertified() -> None:
    reference = _Reference(alpha0=2.0)
    plan = phx.applications.solid_mechanics.ModifiedCamClayMPMConstitutivePlan()
    history = _prestress(plan, reference, ocr=1.0)
    result = plan.evaluate_linearized(
        jnp.eye(3, dtype=jnp.float64), history, 1.0, reference.parameters(), 0.0, 0.01
    )
    assert bool(result.response.successful)
    assert not bool(result.tangent_successful)
