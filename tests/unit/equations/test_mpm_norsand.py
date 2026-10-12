# Copyright © 2026 PHYDRA, Inc. All rights reserved.
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.typing import NDArray
from scipy.integrate import solve_ivp
from scipy.optimize import root_scalar

import phydrax as phx
from phydrax.equations import MPMConstitutiveResponse


pytestmark = pytest.mark.strict_jax


@dataclass(frozen=True)
class _Reference:
    shear: float = 20000.0
    pressure: float = 100.0
    exponent: float = 0.4
    poisson: float = 0.2
    intercept: float = 1.0
    slope: float = 0.02
    friction: float = 1.3
    coupling: float = 0.3
    chi: float = 3.0
    h0: float = 250.0
    hy: float = 250.0

    def parameters(self) -> phx.applications.solid_mechanics.NorSandParameters:
        return phx.applications.solid_mechanics.NorSandParameters(
            *(
                jnp.asarray(v, dtype=jnp.float64)
                for v in (
                    self.shear,
                    self.pressure,
                    self.exponent,
                    self.poisson,
                    self.intercept,
                    self.slope,
                    self.friction,
                    self.coupling,
                    self.chi,
                    self.h0,
                    self.hy,
                )
            )
        )

    def initial_void(self, psi: float) -> float:
        return self.intercept - self.slope * np.log(self.pressure) + psi


def _rate_reference(
    reference: _Reference,
    psi: float,
    strain: NDArray[np.float64],
    drained: bool,
) -> NDArray[np.float64]:
    """Independent NumPy rate law, consistency differential and DOP853 ODE.

    No JAX return-map kernels or principal spectral functions are used. TC
    work conjugacy gives m_a=(D+3)/3 and m_r=(2D-3)/6. Total axial rate is 1.
    State is p,q,e,log(pi), integrated with adaptive steps below 2e-5 strain.
    """
    r = reference
    ratio = r.slope * r.chi / r.friction
    chi = 1.25 * r.chi if ratio > 0.2 else r.chi / (1 - ratio)

    def geometry(y: NDArray[np.float64]) -> tuple[float, float, float]:
        p, q, e, ell = y
        pi = np.exp(ell)
        state = e - r.intercept + r.slope * np.log(p)
        image_state = e - r.intercept + r.slope * ell
        mi = r.friction - r.coupling * chi * abs(image_state)
        hardening = (
            (r.h0 - r.hy * state)
            * (p / pi) ** 2
            * (np.exp(-chi * image_state / mi) - pi / p)
        )
        return float(mi), float(hardening), float(q / (p * mi) - 1 + np.log(p / pi))

    def rate(_: float, y: NDArray[np.float64]) -> NDArray[np.float64]:
        p, q, e, _ = y
        mi, hardening, yield_value = geometry(y)
        g = r.shear * (p / r.pressure) ** r.exponent
        k = 2 * g * (1 + r.poisson) / (3 * (1 - 2 * r.poisson))
        d = mi - q / p
        gradient = np.empty(4, dtype=np.float64)
        for i in range(4):
            step = 1e-5 * max(abs(y[i]), 1.0)
            direction = np.zeros(4, dtype=np.float64)
            direction[i] = step
            gradient[i] = (geometry(y + direction)[2] - geometry(y - direction)[2]) / (
                2 * step
            )

        def rates(radial: float, gamma: float) -> NDArray[np.float64]:
            return np.asarray(
                (
                    k * (1 + 2 * radial - d * gamma),
                    2 * g * (1 - radial) - 3 * g * gamma,
                    -(1 + e) * (1 + 2 * radial),
                    hardening * gamma,
                ),
                dtype=np.float64,
            )

        if drained:
            base = rates(0, 0)
            lateral = rates(1, 0) - base
            plastic = rates(0, 1) - base

            def confinement(v: NDArray[np.float64]) -> float:
                return float(v[0] - v[1] / 3)

            radial = -confinement(base) / confinement(lateral)
            gamma = 0.0
            if yield_value >= -1e-8:
                radial, gamma = np.linalg.solve(
                    np.asarray(
                        (
                            (confinement(lateral), confinement(plastic)),
                            (gradient @ lateral, gradient @ plastic),
                        ),
                        dtype=np.float64,
                    ),
                    np.asarray((-confinement(base), -gradient @ base), dtype=np.float64),
                )
        else:
            radial = -0.5
            base = rates(radial, 0)
            plastic = rates(radial, 1) - base
            gamma = (
                -float(gradient @ base) / float(gradient @ plastic)
                if yield_value >= -1e-8
                else 0.0
            )
        if gamma < 0:
            gamma = 0.0
            radial = -(1 - 2 * r.poisson) / (2 * (1 + r.poisson)) if drained else -0.5
        return rates(float(radial), float(gamma))

    initial = np.asarray(
        (r.pressure, 0, r.initial_void(psi), np.log(r.pressure) - 1), dtype=np.float64
    )
    solution = solve_ivp(
        rate,
        (0.0, float(strain[-1])),
        initial,
        t_eval=strain,
        method="DOP853",
        rtol=1e-8,
        atol=1e-9,
        max_step=2e-5,
    )
    assert solution.success
    return np.asarray(solution.y.T, dtype=np.float64)


def _triaxial(
    reference: _Reference,
    psi: float,
    strain: NDArray[np.float64],
    drained: bool,
    *,
    initial_lateral_ratio: float = 1.0,
    overconsolidation_ratio: float = 1.0,
) -> NDArray[np.float64]:
    """Single-point boundary controller using only public material APIs.

    Axial Hencky compression is prescribed; drained holds radial Kirchhoff
    stress, undrained holds det F=1 exactly. The reference uses small strain.
    """
    plan = phx.applications.solid_mechanics.NorSandMPMConstitutivePlan()
    par = reference.parameters()
    axial_pressure = 3 * reference.pressure / (1 + 2 * initial_lateral_ratio)
    radial_pressure = initial_lateral_ratio * axial_pressure
    init = plan.initialize_state_from_stress(
        -jnp.diag(
            jnp.asarray(
                (axial_pressure, radial_pressure, radial_pressure), dtype=jnp.float64
            )
        ),
        par,
        initial_void_ratio=reference.initial_void(psi),
        overconsolidation_ratio=overconsolidation_ratio,
    )
    assert bool(init.admissible)
    state = init.state
    evaluate = jax.jit(
        lambda f, h: plan.evaluate(
            f, h, jnp.asarray(1800.0, dtype=jnp.float64), par, 0.0, 1.0
        )
    )
    rows = [
        np.asarray(
            (
                reference.pressure,
                axial_pressure - radial_pressure,
                reference.initial_void(psi),
                np.log(float(state[22])),
            ),
            dtype=np.float64,
        )
    ]
    lateral = 0.0
    for axial in strain[1:]:

        def response(radial: float) -> MPMConstitutiveResponse:
            f = jnp.diag(
                jnp.exp(jnp.asarray((-axial, radial, radial), dtype=jnp.float64))
            )
            return evaluate(f, state)

        if drained:

            def residual(radial: float) -> float:
                out = response(radial)
                f = np.diag(
                    np.exp(np.asarray((-axial, radial, radial), dtype=np.float64))
                )
                tau = np.asarray(out.first_piola) @ f.T
                return float(tau[1, 1] + radial_pressure)

            root = root_scalar(residual, x0=lateral, x1=lateral + 1e-5, xtol=1e-11)
            assert root.converged
            lateral = float(root.root)
        else:
            lateral = float(axial / 2)
        out = response(lateral)
        assert bool(out.successful), (axial, out.diagnostics)
        state = out.trial_state
        f = np.diag(np.exp(np.asarray((-axial, lateral, lateral), dtype=np.float64)))
        tau = np.asarray(out.first_piola) @ f.T
        p = -np.trace(tau) / 3
        rows.append(
            np.asarray(
                (p, tau[1, 1] - tau[0, 0], float(state[21]), np.log(float(state[22]))),
                dtype=np.float64,
            )
        )
    return np.stack(rows)


@pytest.mark.parametrize("drained", [True, False], ids=["drained", "undrained"])
@pytest.mark.parametrize("psi", [-0.15, 0.04], ids=["dense", "loose"])
def test_triaxial_against_independent_rate_reference(psi: float, drained: bool) -> None:
    reference = _Reference()
    # Dense drained dilation starts after 1% for these parameters; retain the
    # same increment while integrating through the actual phase change.
    strain = (
        np.linspace(0, 0.02, 129, dtype=np.float64)
        if psi < 0 and drained
        else np.linspace(0, 0.01, 65, dtype=np.float64)
    )
    expected = _rate_reference(reference, psi, strain, drained)
    actual = _triaxial(reference, psi, strain, drained)
    error = np.max(np.abs(actual[:, :2] - expected[:, :2])) / reference.pressure
    print(
        f"NorSand rate comparison psi={psi} drained={drained}: max |(p,q) error|/p0={error:.8g}"
    )
    # Endpoint backward-Euler error against independently integrated rates.
    assert error < 0.025
    np.testing.assert_allclose(actual[:, 2], expected[:, 2], atol=5e-4, rtol=0)
    if not drained:
        np.testing.assert_allclose(actual[:, 2], reference.initial_void(psi), atol=2e-12)
    elif psi < 0:
        assert actual[-1, 2] > actual[-5, 2], "Dense sand must dilate after phase change."
    else:
        assert actual[-1, 2] < actual[-5, 2], "Loose sand must contract."


def test_prestress_objectivity_finite_tangent_and_failure_rollback() -> None:
    reference = _Reference()
    plan = phx.applications.solid_mechanics.NorSandMPMConstitutivePlan()
    par = reference.parameters()
    identity = jnp.eye(3, dtype=jnp.float64)
    initial = plan.initialize_state_from_stress(
        -reference.pressure * identity,
        par,
        initial_void_ratio=reference.initial_void(-0.1),
        overconsolidation_ratio=2.0,
    )
    assert bool(initial.admissible)
    assert not plan.capabilities.has_free_energy
    response = plan.evaluate_linearized(identity, initial.state, 1800.0, par, 0.0, 1.0)
    assert bool(response.tangent_successful)
    assert np.all(np.isfinite(response.algorithmic_tangent))
    np.testing.assert_allclose(
        response.response.first_piola, -reference.pressure * identity, atol=1e-10
    )
    deformation = jnp.diag(jnp.asarray((0.9995, 1.0001, 1.0001), dtype=jnp.float64))
    angle = 0.47
    rotation = jnp.asarray(
        (
            (np.cos(angle), -np.sin(angle), 0),
            (np.sin(angle), np.cos(angle), 0),
            (0, 0, 1),
        ),
        dtype=jnp.float64,
    )
    plain = plan.evaluate(deformation, initial.state, 1800.0, par, 0.0, 1.0)
    rotated = plan.evaluate(rotation @ deformation, initial.state, 1800.0, par, 0.0, 1.0)
    assert bool(plain.successful) and bool(rotated.successful)
    np.testing.assert_allclose(
        rotated.first_piola, rotation @ plain.first_piola, atol=2e-9
    )
    np.testing.assert_allclose(rotated.trial_state[:6], plain.trial_state[:6], atol=2e-11)
    failed = plan.evaluate(
        jnp.zeros((3, 3), dtype=jnp.float64), initial.state, 1800.0, par, 0.0, 1.0
    )
    assert not bool(failed.successful) and int(failed.branch_code) == -1
    np.testing.assert_array_equal(failed.trial_state, initial.state)


@pytest.mark.parametrize("pressure", [0.0, -100.0])
def test_nonpositive_pressure_inadmissible(pressure: float) -> None:
    reference = _Reference()
    plan = phx.applications.solid_mechanics.NorSandMPMConstitutivePlan()
    initialized = plan.initialize_state_from_stress(
        -pressure * jnp.eye(3, dtype=jnp.float64),
        reference.parameters(),
        initial_void_ratio=0.8,
        overconsolidation_ratio=1.0,
    )
    assert not bool(initialized.admissible)
    assert np.all(np.isfinite(initialized.state))
    response = plan.evaluate(
        jnp.eye(3, dtype=jnp.float64),
        initialized.state,
        1800.0,
        reference.parameters(),
        0.0,
        1.0,
    )
    assert not bool(response.successful)
    np.testing.assert_array_equal(response.trial_state, initialized.state)


def test_mandatory_initialization_and_parameter_domain() -> None:
    with pytest.raises(ValueError, match="initialize_state_from_stress"):
        phx.applications.solid_mechanics.NorSandMPMConstitutivePlan().initialize_state(
            (), jnp.float64
        )
    with pytest.raises(ValueError, match="inadmissible"):
        phx.applications.solid_mechanics.NorSandParameters(
            20000, 0, 0.4, 0.2, 1, 0.02, 1.3, 0.3, 3, 250, 250
        )


def test_licensed_dataset_variant_evidence() -> None:
    """Workbook diagnostics cannot establish exact frozen-equation parity.

    Keep measured evidence, rather than relaxing physics acceptance to hide
    the model-version discrepancy. Full attribution travels with the fixture.
    """
    path = Path(__file__).parents[2] / "fixtures/geomechanics/norsand4ai.json"
    payload = json.loads(path.read_text())
    assert payload["license"] == "CC BY 4.0"
    assert len(payload["curves"]) == 4
    row = np.asarray(payload["curves"][0]["data"][1], dtype=np.float64)
    attr = payload["curves"][0]["attributes"]
    # Modern frozen TC image strength is never larger than Mtc; the workbook's
    # recorded rate fields are demonstrably not the same endpoint quantities.
    assert row[8] + row[9] > attr["Mtc"]


@pytest.mark.parametrize("drained", [True, False], ids=["drained", "undrained"])
def test_dense_norsand4ai_small_strain_comparison(drained: bool) -> None:
    """A separate measured comparison envelope, not exact workbook parity."""
    path = Path(__file__).parents[2] / "fixtures/geomechanics/norsand4ai.json"
    payload = json.loads(path.read_text())
    kind = "Drained" if drained else "Undrained"
    curve = next(
        c
        for c in payload["curves"]
        if c["attributes"]["Type"] == kind and c["file"].endswith("Par_0_2.h5")
    )
    attr = curve["attributes"]
    reference = _Reference(
        shear=1000 * attr["Gmax_p0"],
        pressure=attr["p0"],
        exponent=attr["G_exp"],
        poisson=attr["nu"],
        intercept=attr["Gamma"],
        slope=attr["lambda"],
        friction=attr["Mtc"],
        coupling=attr["N"],
        chi=attr["Xtc"],
        h0=attr["H0"],
        hy=attr["Hy"],
    )
    target = np.asarray(curve["data"], dtype=np.float64)
    target = target[target[:, 0] <= 1.0]
    strain = np.linspace(0, 0.01, 65, dtype=np.float64)
    actual = _triaxial(
        reference,
        attr["Psi_0"],
        strain,
        drained,
        initial_lateral_ratio=attr["K0"],
        overconsolidation_ratio=attr["OCR"],
    )
    sampled = np.column_stack(
        [np.interp(target[:, 0] / 100, strain, actual[:, i]) for i in range(2)]
    )
    errors = np.max(np.abs(sampled - target[:, 2:4]), axis=0) / reference.pressure
    print(f"NorSand4AI dense {kind}: max |p,q error|/p0={errors.tolist()}")
    # This envelope includes the documented workbook current-psi convention,
    # endpoint integration and strain-driver differences; it is not a claim
    # that those differences arise solely from a finite-strain embedding.
    assert np.max(errors) < 0.025


@pytest.mark.parametrize("compression", [True, False], ids=["TC", "TE"])
def test_plastic_lode_endpoints_tangent_and_objectivity(compression: bool) -> None:
    reference = _Reference()
    plan = phx.applications.solid_mechanics.NorSandMPMConstitutivePlan()
    par = reference.parameters()
    identity = jnp.eye(3, dtype=jnp.float64)
    initial = plan.initialize_state_from_stress(
        -100 * identity,
        par,
        initial_void_ratio=reference.initial_void(-0.1),
        overconsolidation_ratio=1.0,
    )
    sign = -1.0 if compression else 1.0
    deformation = jnp.diag(
        jnp.exp(
            jnp.asarray(
                (sign * 0.0005, -sign * 0.00025, -sign * 0.00025), dtype=jnp.float64
            )
        )
    )
    linearized = plan.evaluate_linearized(
        deformation, initial.state, 1800.0, par, 0.0, 1.0
    )
    response = linearized.response
    assert bool(response.successful) and int(response.branch_code) == 1
    assert bool(linearized.tangent_successful)
    assert float(response.dissipation_increment) >= 0
    np.testing.assert_allclose(
        response.diagnostics["lode_factor"],
        1.0 if compression else 3 / (3 + reference.friction),
        atol=1e-12,
    )
    evaluate = jax.jit(lambda f: plan.evaluate(f, initial.state, 1800.0, par, 0.0, 1.0))
    permutation = jnp.asarray(((0, 1, 0), (0, 0, 1), (1, 0, 0)), dtype=jnp.float64)
    rotated = evaluate(permutation @ deformation)
    assert bool(rotated.successful)
    np.testing.assert_allclose(
        rotated.first_piola, permutation @ response.first_piola, atol=2e-8
    )
    direction = jnp.asarray(
        ((0.1, 0.03, 0), (0.02, -0.04, 0.015), (0, -0.01, 0.05)), dtype=jnp.float64
    )
    predicted = (
        linearized.algorithmic_tangent.reshape(9, 9) @ direction.reshape(9)
    ).reshape(3, 3)
    differences: list[float] = []
    # At TE, z3' and z2' vanish. The unsorted equal-principal map is C1 but
    # generally not C2: central FD therefore has O(step), not O(step**2), error.
    # Refinement distinguishes this truncation error from an incorrect tangent.
    steps = (1e-5,) if compression else (1e-5, 1e-6)
    for step in steps:
        plus = evaluate(deformation + step * direction)
        minus = evaluate(deformation - step * direction)
        assert bool(plus.successful) and bool(minus.successful)
        difference = (plus.first_piola - minus.first_piola) / (2 * step)
        differences.append(float(jnp.max(jnp.abs(predicted - difference))))
        print(
            f"NorSand {'TC' if compression else 'TE'} FD step={step}: max tangent error={differences[-1]:.8g}"
        )
    if not compression:
        assert differences[-1] < 0.5 * differences[0], (
            "TE tangent must converge under FD refinement."
        )
    np.testing.assert_allclose(
        predicted,
        difference,
        rtol=3e-3,
        atol=0.02,
    )


def test_float32_prestress_and_plastic_admission() -> None:
    reference = _Reference()
    par = jax.tree.map(lambda leaf: leaf.astype(jnp.float32), reference.parameters())
    plan = phx.applications.solid_mechanics.NorSandMPMConstitutivePlan()
    identity = jnp.eye(3, dtype=jnp.float32)
    initialized = plan.initialize_state_from_stress(
        -100 * identity,
        par,
        initial_void_ratio=reference.initial_void(-0.1),
        overconsolidation_ratio=1.0,
    )
    assert bool(initialized.admissible)
    evaluate = jax.jit(
        lambda f, h, p: plan.evaluate(
            f, h, jnp.asarray(1800.0, dtype=jnp.float32), p, 0.0, 1.0
        )
    )
    rest = evaluate(identity, initialized.state, par)
    assert bool(rest.successful)
    deformation = jnp.diag(
        jnp.exp(jnp.asarray((-0.0005, 0.00025, 0.00025), dtype=jnp.float32))
    )
    plastic = evaluate(deformation, initialized.state, par)
    assert bool(plastic.successful) and int(plastic.branch_code) == 1, plastic.diagnostics
    assert plastic.first_piola.dtype == jnp.float32
    assert np.all(np.isfinite(plastic.first_piola))


def test_image_strength_guard_and_semilog_identity() -> None:
    reference = _Reference(slope=0.08, chi=5.0)
    par = reference.parameters()
    plan = phx.applications.solid_mechanics.NorSandMPMConstitutivePlan()
    stress = -jnp.diag(
        jnp.asarray((100 + 40 / 3, 100 - 20 / 3, 100 - 20 / 3), dtype=jnp.float64)
    )
    void = 0.6
    initialized = plan.initialize_state_from_stress(
        stress, par, initial_void_ratio=void, overconsolidation_ratio=1.0
    )
    assert bool(initialized.admissible)
    pi = float(initialized.state[22])
    psi = void - reference.intercept + reference.slope * np.log(100)
    image = void - reference.intercept + reference.slope * np.log(pi)
    np.testing.assert_allclose(
        image, psi + reference.slope * np.log(pi / 100), atol=1e-14
    )
    mi = reference.friction - reference.coupling * (1.25 * reference.chi) * abs(image)
    np.testing.assert_allclose(0.2 / mi - 1 + np.log(100 / pi), 0, atol=1e-10)


def test_isotropic_branch_and_invalid_previous_deformation() -> None:
    reference = _Reference()
    par = reference.parameters()
    plan = phx.applications.solid_mechanics.NorSandMPMConstitutivePlan()
    identity = jnp.eye(3, dtype=jnp.float64)
    compressed = jnp.exp(jnp.asarray(-0.0001, dtype=jnp.float64)) * identity
    nc = plan.initialize_state_from_stress(
        -100 * identity, par, initial_void_ratio=0.8, overconsolidation_ratio=1.0
    )
    oc = plan.initialize_state_from_stress(
        -100 * identity, par, initial_void_ratio=0.8, overconsolidation_ratio=2.0
    )
    assert bool(nc.admissible) and bool(oc.admissible)
    evaluate = jax.jit(lambda f, h: plan.evaluate(f, h, 1800.0, par, 0.0, 1.0))
    elastic = evaluate(compressed, oc.state)
    assert bool(elastic.successful) and int(elastic.branch_code) == 0
    assert not bool(elastic.diagnostics["lode_defined"])
    np.testing.assert_allclose(
        elastic.first_piola, jnp.trace(elastic.first_piola) / 3 * identity, atol=1e-10
    )
    # The outer-only baseline has no isotropic inner-cap plastic flow: reject
    # rather than manufacturing a preferred principal direction at q=0.
    rejected = evaluate(compressed, nc.state)
    assert not bool(rejected.successful) and int(rejected.branch_code) == -1
    np.testing.assert_array_equal(rejected.trial_state, nc.state)
    malformed = oc.state.at[12:21].set(0.0)
    rejected_history = evaluate(identity, malformed)
    assert not bool(rejected_history.successful)
    np.testing.assert_array_equal(rejected_history.trial_state, malformed)


def test_pressure_unit_invariance() -> None:
    reference = _Reference()
    par = reference.parameters()
    scale = 1000.0
    scaled_par = phx.applications.solid_mechanics.NorSandParameters(
        scale * par.reference_shear_modulus,
        scale * par.reference_pressure,
        par.shear_pressure_exponent,
        par.poisson_ratio,
        par.critical_state_intercept,
        par.critical_state_slope,
        par.critical_stress_ratio,
        par.volumetric_coupling,
        par.dilatancy_coefficient,
        par.hardening_intercept,
        par.hardening_state_coefficient,
        critical_state_reference_pressure=scale * par.critical_state_reference_pressure,
    )
    plan = phx.applications.solid_mechanics.NorSandMPMConstitutivePlan()
    identity = jnp.eye(3, dtype=jnp.float64)
    initial = plan.initialize_state_from_stress(
        -100 * identity, par, initial_void_ratio=0.8, overconsolidation_ratio=1.0
    )
    scaled_initial = plan.initialize_state_from_stress(
        -scale * 100 * identity,
        scaled_par,
        initial_void_ratio=0.8,
        overconsolidation_ratio=1.0,
    )
    assert bool(initial.admissible) and bool(scaled_initial.admissible)
    evaluate = jax.jit(lambda f, h, rho, p: plan.evaluate(f, h, rho, p, 0.0, 1.0))
    deformation = jnp.diag(
        jnp.exp(jnp.asarray((-0.0005, 0.00025, 0.00025), dtype=jnp.float64))
    )
    # kPa/tonne/m/s and Pa/kg/m/s: both stress and density numbers scale.
    base = evaluate(deformation, initial.state, jnp.asarray(1.8, dtype=jnp.float64), par)
    scaled = evaluate(
        deformation,
        scaled_initial.state,
        jnp.asarray(1800.0, dtype=jnp.float64),
        scaled_par,
    )
    assert bool(base.successful) and bool(scaled.successful)
    assert int(base.branch_code) == int(scaled.branch_code) == 1
    np.testing.assert_allclose(
        scaled.first_piola / scale, base.first_piola, rtol=1e-9, atol=1e-8
    )
    np.testing.assert_allclose(scaled.trial_state[:6], base.trial_state[:6], atol=1e-10)
    np.testing.assert_allclose(
        scaled.trial_state[6:12] / scale, base.trial_state[6:12], rtol=1e-9, atol=1e-8
    )
    np.testing.assert_allclose(
        scaled.trial_state[12:22], base.trial_state[12:22], atol=1e-10
    )
    np.testing.assert_allclose(
        scaled.trial_state[22] / scale, base.trial_state[22], atol=1e-8
    )
    np.testing.assert_allclose(
        scaled.maximum_wave_speed, base.maximum_wave_speed, rtol=1e-9
    )


def test_inactive_image_strength_kink_has_elastic_tangent() -> None:
    reference = _Reference()
    par = reference.parameters()
    plan = phx.applications.solid_mechanics.NorSandMPMConstitutivePlan()
    identity = jnp.eye(3, dtype=jnp.float64)
    # At q=0, pi=OCR*p/exp(1), independent of image friction. Choose e=ec(pi).
    void = reference.intercept - reference.slope * (np.log(2 * reference.pressure) - 1)
    initialized = plan.initialize_state_from_stress(
        -100 * identity, par, initial_void_ratio=void, overconsolidation_ratio=2.0
    )
    assert bool(initialized.admissible)
    result = jax.jit(
        lambda f: plan.evaluate_linearized(f, initialized.state, 1800.0, par, 0.0, 1.0)
    )(identity)
    assert bool(result.response.successful) and bool(result.tangent_successful)
    assert np.all(np.isfinite(result.algorithmic_tangent))
    assert int(result.response.branch_code) == 0
