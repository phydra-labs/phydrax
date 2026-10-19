# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Modern NorSand with objective, endpoint-hypoelastic Hencky increments.

Jefferies (1993), doi:10.1680/geot.1993.43.1.91; Jefferies & Shuttle
(2002), doi:10.1680/geot.2002.52.9.625, Eqs. 7a,14–16; Jefferies & Been,
Soil Liquefaction, second edition (2016), doi:10.1201/b19114. The modern
log surface, image friction and limited hardening are cross-checked against
Itasca 9.8 Eqs. 9–19, not the distinct N-dependent 1993 surface.

Compression-positive source variables: f=q/(p Mi)-1+log(p/pi),
Mi=(Mtc-N chi_i abs(psi_i)) g(theta), D=Mi-q/p,
log(pi_new/pi_old)=gamma H g (p/pi)^2 (exp(-chi_i psi_i/Mi_tc)-pi/p).
H=H0-Hy psi; chi_i uses the documented 1.25 chi_tc threshold branch.
Principal plastic flow is J&S's work-conjugate TC/TE interpolation, not
an arbitrary associated gradient of the Lode-dependent yield function.

The finite-strain lift is a Phydrax convention, not source hyperelasticity:
rotate committed stress and elastic metric with the incremental polar rotation;
Delta h=log(Be_trial)/2-log(R Be_old R.T)/2. Endpoint K,G act on
Delta h-Delta ep. A scalar pressure root surrounds a five-unknown principal
return (three stresses, work-conjugate plastic shear, log image pressure).
The elastic metric is exp(2(h_trial-Delta ep)); native small solves pull it
back. This corotated logarithmic-increment convention is objective and agrees
with the published rate law under refinement, without inventing free energy.

Spectral stress inputs and Newton stress variables are normalized by p_ref.
Normalized inputs are rounded before spectral symmetry admission so a fused
quotient cannot manufacture a nonzero self-transpose residual. The Newton
shear coordinate is gamma/(p_ref/G_ref). Thus the native
confluent-spectrum criterion is unit-independent and the five-variable
Jacobian remains well scaled in float32. Root residual tolerances are
max(1e-10,64 machine epsilon). At q=0 no preferred plastic direction exists;
the outer-only baseline explicitly rejects isotropic plastic loading.

History: packed Cp_inverse(6), packed Kirchhoff stress(6), previous F(9),
void ratio e, image pressure pi (width 23). Prestress initializes Cp_inverse=I:
its reference is already loaded; stress is independent hypoelastic memory,
not an asserted unique elastic-strain inverse. Zero-pressure initialization
is intentionally unavailable. Branches 0 elastic, 1 outer plastic, -1 failure.
Inner-cap hardening, PSR, cyclic terms and extra softening are excluded.

NorSandTXL parity is not asserted: released CC-BY NorSand4AI diagnostics have
lagged Dp and suggest Mi=Mtc-N chi_tc abs(current psi) and
pi_max=p exp(-chi_tc current psi/Mi), rather than image-state quantities.
Its OCR<1 initialization also differs from the stipulated OCR definition.
Independent rate integrations of those inferred formulas still disagree
with loose undrained data; no conclusive primary workbook equations were
available. This implementation therefore retains the frozen modern law,
not an empirical hybrid. H0-Hy psi is the selected sign; dense workbook
comparisons favor it over H0+Hy psi, but do not establish full workbook parity.
"""

from __future__ import annotations

from typing import final

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike, DTypeLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...equations import (
    AbstractImplicitMPMConstitutivePlan,
    MPMConstitutiveCapabilities,
    MPMConstitutiveResponse,
    MPMLinearizedConstitutiveResponse,
)
from ...linalg import (
    hermitian_exp,
    hermitian_inverse_sqrt,
    hermitian_log,
    hermitian_spectral_map,
    HermitianSpectralMapResult,
    SmallLinearSolvePlan,
    SmallLinearSolveResult,
    solve_small_linear,
)
from ...nonlinear import LocalRootPlan, VectorLocalRootPlan
from ._principal_hencky import (
    _positive_metric,
    pack_metric,
    PrestressedMaterialState,
    unpack_metric,
)


@final
class NorSandParameters(StrictModule, NonTrainableState):
    reference_shear_modulus: Array
    reference_pressure: Array
    shear_pressure_exponent: Array
    poisson_ratio: Array
    critical_state_intercept: Array
    critical_state_slope: Array
    critical_stress_ratio: Array
    volumetric_coupling: Array
    dilatancy_coefficient: Array
    hardening_intercept: Array
    hardening_state_coefficient: Array
    critical_state_reference_pressure: Array

    def __init__(
        self,
        reference_shear_modulus: ArrayLike,
        reference_pressure: ArrayLike,
        shear_pressure_exponent: ArrayLike,
        poisson_ratio: ArrayLike,
        critical_state_intercept: ArrayLike,
        critical_state_slope: ArrayLike,
        critical_stress_ratio: ArrayLike,
        volumetric_coupling: ArrayLike,
        dilatancy_coefficient: ArrayLike,
        hardening_intercept: ArrayLike,
        hardening_state_coefficient: ArrayLike,
        /,
        *,
        critical_state_reference_pressure: ArrayLike = 1.0,
    ) -> None:
        values = tuple(
            jnp.asarray(v)
            for v in (
                reference_shear_modulus,
                reference_pressure,
                shear_pressure_exponent,
                poisson_ratio,
                critical_state_intercept,
                critical_state_slope,
                critical_stress_ratio,
                volumetric_coupling,
                dilatancy_coefficient,
                hardening_intercept,
                hardening_state_coefficient,
                critical_state_reference_pressure,
            )
        )
        if any(v.shape != () or not bool(jnp.isfinite(v)) for v in values):
            raise ValueError("NorSand parameters must be finite scalars.")
        g, p, m, nu, _, lam, mtc, n, chi, h, _, pcsl = values
        if (
            g <= 0
            or p <= 0
            or pcsl <= 0
            or not 0 <= m <= 1
            or not -1 < nu < 0.5
            or lam <= 0
            or mtc <= 0
            or n < 0
            or chi < 0
            or h <= 0
        ):
            raise ValueError("NorSand parameters are inadmissible.")
        (
            self.reference_shear_modulus,
            self.reference_pressure,
            self.shear_pressure_exponent,
            self.poisson_ratio,
            self.critical_state_intercept,
            self.critical_state_slope,
            self.critical_stress_ratio,
            self.volumetric_coupling,
            self.dilatancy_coefficient,
            self.hardening_intercept,
            self.hardening_state_coefficient,
            self.critical_state_reference_pressure,
        ) = values


def _root_tolerance(dtype: DTypeLike) -> float:
    """Dimensionless residual tolerance, relative to native compute precision."""
    return float(max(1e-10, 64 * jnp.finfo(dtype).eps))


def _moduli(p: Array, par: NorSandParameters) -> tuple[Array, Array]:
    g = (
        par.reference_shear_modulus
        * (p / par.reference_pressure) ** par.shear_pressure_exponent
    )
    return g, 2 * g * (1 + par.poisson_ratio) / (3 * (1 - 2 * par.poisson_ratio))


def _ec(p: Array, par: NorSandParameters) -> Array:
    return par.critical_state_intercept - par.critical_state_slope * jnp.log(
        p / par.critical_state_reference_pressure
    )


def _geometry(sigma: Array) -> tuple[Array, Array, Array]:
    p = jnp.mean(sigma)
    # Pairwise differences preserve exact q=0 at equal principal stresses;
    # subtracting their rounded mean can manufacture a nonzero deviator.
    q2 = 0.5 * (
        (sigma[0] - sigma[1]) ** 2
        + (sigma[1] - sigma[2]) ** 2
        + (sigma[2] - sigma[0]) ** 2
    )
    q = jax.lax.cond(q2 > 0, jnp.sqrt, jnp.zeros_like, q2)
    theta = jax.lax.cond(
        q2 > 0,
        lambda _: jnp.arctan2(
            sigma[0] + sigma[2] - 2 * sigma[1], jnp.sqrt(3.0) * (sigma[0] - sigma[2])
        ),
        lambda _: jnp.zeros_like(p),
        None,
    )
    return p, q, theta


def _image(
    p: Array, theta: Array, e: Array, pi: Array, par: NorSandParameters
) -> tuple[Array, Array, Array, Array, Array]:
    ratio = (
        par.critical_state_slope * par.dilatancy_coefficient / par.critical_stress_ratio
    )
    chi = jax.lax.cond(
        ratio > 0.2,
        lambda _: 1.25 * par.dilatancy_coefficient,
        lambda _: par.dilatancy_coefficient / (1 - ratio),
        None,
    )
    psi_i = e - _ec(pi, par)
    mtc = par.critical_stress_ratio - par.volumetric_coupling * chi * jnp.abs(psi_i)
    c = 3 / (3 + par.critical_stress_ratio)
    g = (2 * c**4 / (c**4 + 1 + (c**4 - 1) * jnp.sin(3 * theta))) ** 0.25
    mi = mtc * g
    limit = p * jnp.exp(-chi * psi_i / mtc)
    h = par.hardening_intercept - par.hardening_state_coefficient * (e - _ec(p, par))
    return mi, mtc, g, limit, h


def _flow(
    sigma: Array, mi: Array, mtc: Array, theta: Array, par: NorSandParameters
) -> Array:
    p, q, _ = _geometry(sigma)
    d = mi - q / p
    a = (jnp.sin(theta) + jnp.sqrt(3.0) * jnp.cos(theta)) / 3
    b = -2 * jnp.sin(theta) / 3
    ctheta = (jnp.sin(theta) - jnp.sqrt(3.0) * jnp.cos(theta)) / 3
    dtc, dte = d * mtc / mi, d * (3 / (3 + par.critical_stress_ratio)) * mtc / mi
    ztc, zte = (2 * dtc - 3) / (6 + 2 * dtc), (2 * dte - 6) / (3 + 2 * dte)
    z3 = ztc - (ztc - zte) * jnp.cos((3 * theta + jnp.pi / 2) / 2)
    z2 = (a * d - 1 + z3 * (ctheta * d - 1)) / (1 - b * d)
    return jnp.stack((jnp.ones_like(p), z2, z3)) / (a + b * z2 + ctheta * z3)


def _principal_return(
    eigenvalues: Array, data: tuple[NorSandParameters, Array, Array, Array]
) -> tuple[Array, dict[str, Array]]:
    par, e, pi, endpoint_p = data
    pressure = jnp.mean(eigenvalues) * par.reference_pressure

    def rejected(_: None) -> tuple[Array, dict[str, Array]]:
        zero = jnp.zeros((), dtype=eigenvalues.dtype)
        result = {
            key: zero
            for key in (
                "plastic_multiplier",
                "yield_residual",
                "dissipation",
                "image_limit",
                "limit_stress_ratio",
                "lode_factor",
                "dilatancy",
            )
        }
        result.update(
            {
                "pressure": pressure,
                "image_pressure": pi,
                "successful": jnp.asarray(False),
                "lode_defined": jnp.asarray(False),
                "branch_code": jnp.asarray(-1, dtype=jnp.int32),
            }
        )
        return jnp.zeros((2, 3), dtype=eigenvalues.dtype), result

    return jax.lax.cond(
        (pressure > 0) & (e > 0) & (pi > 0) & (endpoint_p > 0),
        lambda _: _admitted_principal_return(eigenvalues * par.reference_pressure, data),
        rejected,
        None,
    )


def _admitted_principal_return(
    eigenvalues: Array, data: tuple[NorSandParameters, Array, Array, Array]
) -> tuple[Array, dict[str, Array]]:
    par, e, pi_old, endpoint_p = data
    order = jnp.argsort(-eigenvalues)
    undo = jnp.argsort(order)
    trial = eigenvalues[order]
    p, q, theta = _geometry(trial)
    mi, mtc, _, _, _ = _image(p, theta, e, pi_old, par)
    ftrial = q / (p * mi) - 1 + jnp.log(p / pi_old)
    g, k = _moduli(endpoint_p, par)
    # Pressure-scale stresses and elastic-strain-scale plastic shear keep the
    # five-variable Jacobian well scaled even when G_ref/p_ref is very large.
    scale = par.reference_pressure
    shear_scale = scale / par.reference_shear_modulus

    def plastic(_: None) -> tuple[Array, Array, Array, Array]:
        def unpack(x: Array) -> tuple[Array, Array, Array]:
            pressure = par.reference_pressure * jnp.exp(x[0])
            dev = scale * jnp.stack((x[1], x[2], -x[1] - x[2]))
            return (
                pressure + dev,
                shear_scale * x[3],
                par.reference_pressure * jnp.exp(x[4]),
            )

        def residual(x: Array) -> Array:
            sigma, gamma, pi = unpack(x)
            pressure, shear, angle = _geometry(sigma)
            image, image_tc, lode, limit, h = _image(pressure, angle, e, pi, par)
            flow = _flow(sigma, image, image_tc, angle, par)
            elastic = 2 * g * (flow - jnp.mean(flow)) + k * jnp.sum(flow)
            stress = (sigma - trial + gamma * elastic) / scale
            return jnp.concatenate(
                (
                    stress,
                    jnp.stack(
                        (
                            shear / (pressure * image) - 1 + jnp.log(pressure / pi),
                            jnp.log(pi / pi_old)
                            - gamma
                            * h
                            * lode
                            * (pressure / pi) ** 2
                            * (limit / pressure - pi / pressure),
                        )
                    ),
                )
            )

        guess = jnp.stack(
            (
                jnp.log(p / par.reference_pressure),
                (trial[0] - p) / scale,
                (trial[1] - p) / scale,
                jnp.zeros_like(p),
                jnp.log(pi_old / par.reference_pressure),
            )
        )
        root, evidence = VectorLocalRootPlan(
            5,
            maximum_steps=25,
            tolerance=_root_tolerance(eigenvalues.dtype),
            plan_id="norsand-outer-principal",
        ).solve_with_diagnostics(residual, guess)
        sigma, gamma, pi = unpack(root)
        return sigma, gamma, pi, evidence.converged

    sigma, gamma, pi, converged = jax.lax.cond(
        (ftrial > _root_tolerance(eigenvalues.dtype)) & (q > 0),
        plastic,
        lambda _: (trial, jnp.zeros_like(p), pi_old, jnp.asarray(True)),
        None,
    )
    pn, qn, angle = _geometry(sigma)
    image, image_tc, lode, limit, h = _image(pn, angle, e, pi, par)
    flow = jax.lax.cond(
        gamma != 0,
        lambda _: gamma * _flow(sigma, image, image_tc, angle, par),
        lambda _: jnp.zeros_like(sigma),
        None,
    )
    fy = qn / (pn * image) - 1 + jnp.log(pn / pi)
    valid = (
        converged
        & (p > 0)
        & (pn > 0)
        & (pi > 0)
        & (image_tc > 0)
        & (e > 0)
        & (h >= 0)
        & (gamma >= 0)
        & ((ftrial <= _root_tolerance(eigenvalues.dtype)) | (q > 0))
        & jnp.all(jnp.isfinite(flow))
        & (fy <= max(1e-8, _root_tolerance(eigenvalues.dtype)))
        & jnp.all(
            sigma[:-1]
            >= sigma[1:]
            - max(1e-8, _root_tolerance(eigenvalues.dtype)) * jnp.max(jnp.abs(sigma))
        )
    )
    return jnp.stack((-sigma[undo], flow[undo])), {
        "pressure": pn,
        "image_pressure": pi,
        "plastic_multiplier": gamma,
        "yield_residual": fy,
        "successful": valid,
        "branch_code": jnp.where(valid, jnp.where(gamma > 0, 1, 0), -1).astype(jnp.int32),
        "dissipation": gamma * pn * image,
        "image_limit": limit,
        "limit_stress_ratio": image * (1 + jnp.log(limit / pn)),
        "lode_factor": jnp.where(qn > 0, lode, 1.0),
        "lode_defined": qn > 0,
        "dilatancy": jnp.where(qn > 0, image - qn / pn, 0.0),
    }


def _initial_image_map(
    values: Array, data: tuple[NorSandParameters, Array, Array]
) -> tuple[Array, dict[str, Array]]:
    par, ev, ov = data
    sigma = jnp.sort(-values * par.reference_pressure)[::-1]
    p, q, theta = _geometry(sigma)

    def admitted(_: None) -> tuple[Array, Array, Array]:
        def residual(ell: Array) -> Array:
            pi = par.reference_pressure * jnp.exp(ell)
            mi = _image(p, theta, ev, pi, par)[0]
            return q / (p * mi) - 1 + jnp.log(p / pi)

        ell, diag = LocalRootPlan(
            plan_id="norsand-initial-image", tolerance=_root_tolerance(values.dtype)
        ).solve_with_diagnostics(residual, jnp.log(p / par.reference_pressure) - 1)
        pi = ov * par.reference_pressure * jnp.exp(ell)
        mi = _image(p, theta, ev, pi, par)[0]
        f = q / (p * mi) - 1 + jnp.log(p / pi)
        return (
            pi,
            f,
            diag.converged & (mi > 0) & (f <= max(1e-8, _root_tolerance(values.dtype))),
        )

    pi, f, valid = jax.lax.cond(
        (p > 0) & (ev > 0) & (ov > 0),
        admitted,
        lambda _: (jnp.ones_like(p), jnp.full_like(p, jnp.inf), jnp.asarray(False)),
        None,
    )
    return values[None, :], {"pi": pi, "yield": f, "valid": valid}


@final
class NorSandMPMConstitutivePlan(AbstractImplicitMPMConstitutivePlan, NonTrainableState):
    dimension: int = eqx.field(static=True)
    kinematics: str = eqx.field(static=True)
    state_shape: tuple[int, ...] = eqx.field(static=True)
    capabilities: MPMConstitutiveCapabilities
    plan_id: str = eqx.field(static=True)

    def __init__(self) -> None:
        self.dimension, self.kinematics, self.state_shape = 3, "three_dimensional", (23,)
        self.capabilities = MPMConstitutiveCapabilities(
            stateful=True,
            has_free_energy=False,
            has_algorithmic_tangent=True,
            has_dissipation=True,
            supports_implicit=True,
        )
        self.plan_id = canonical_fingerprint(
            {
                "kind": "norsand-modern",
                "elasticity": "corotated-endpoint-hypoelastic",
                "chi": "guarded",
                "hardening": "H0-Hy-psi",
            }
        )

    def initialize_state(
        self, batch_shape: tuple[int, ...], dtype: DTypeLike, /
    ) -> Array:
        del batch_shape, dtype
        raise ValueError(
            "NorSand requires initialize_state_from_stress with positive pressure, void ratio and OCR."
        )

    def initialize_state_from_stress(
        self,
        kirchhoff_stress: ArrayLike,
        parameters: NorSandParameters,
        /,
        *,
        initial_void_ratio: ArrayLike,
        overconsolidation_ratio: ArrayLike,
    ) -> PrestressedMaterialState:
        tau = jnp.asarray(kirchhoff_stress)
        if tau.shape[-2:] != (3, 3):
            raise ValueError("Kirchhoff stress must end in 3x3.")
        batch = tau.shape[:-2]
        e = jnp.broadcast_to(jnp.asarray(initial_void_ratio, dtype=tau.dtype), batch)
        ocr = jnp.broadcast_to(
            jnp.asarray(overconsolidation_ratio, dtype=tau.dtype), batch
        )

        def point(stress: Array, void: Array, ratio: Array) -> tuple[Array, Array, Array]:
            mapped = hermitian_spectral_map(
                jax.lax.optimization_barrier(stress / parameters.reference_pressure),
                _initial_image_map,
                (parameters, void, ratio),
                function_id="norsand-initial-stress",
            )
            identity = jnp.eye(3, dtype=stress.dtype)
            state = jnp.concatenate(
                (
                    pack_metric(identity),
                    pack_metric(stress),
                    identity.reshape(-1),
                    jnp.stack((void, mapped.invariants["pi"])),
                )
            )
            valid = (
                mapped.valid
                & mapped.invariants["valid"]
                & jnp.all(stress == stress.T)
                & jnp.all(jnp.isfinite(state))
            )
            return state, valid, mapped.invariants["yield"]

        state, valid, value = jax.vmap(point)(
            tau.reshape((-1, 3, 3)), e.reshape(-1), ocr.reshape(-1)
        )
        return PrestressedMaterialState(
            state.reshape(batch + (23,)), valid.reshape(batch), value.reshape(batch)
        )

    def _point(
        self,
        deformation: Array,
        history: Array,
        density: Array,
        par: NorSandParameters,
        step: Array,
    ) -> tuple[Array, Array, Array, Array, Array, Array, dict[str, Array]]:
        old_pressure = -jnp.sum(history[6:9]) / 3
        metric = unpack_metric(history[:6])
        determinant = solve_small_linear(
            SmallLinearSolvePlan(3), deformation, jnp.zeros_like(deformation)
        ).determinant
        old_f = history[12:21].reshape(3, 3)
        old_geometry = solve_small_linear(SmallLinearSolvePlan(3), old_f.T, deformation.T)
        positive = _positive_metric(metric)
        admitted = (
            (old_pressure > 0)
            & (history[21] > 0)
            & (history[22] > 0)
            & (density > 0)
            & (determinant > 0)
            & old_geometry.successful
            & (old_geometry.determinant > 0)
            & positive
            & jnp.all(jnp.isfinite(history))
            & jnp.all(jnp.isfinite(deformation))
        )

        def rejected(
            _: None,
        ) -> tuple[Array, Array, Array, Array, Array, Array, dict[str, Array]]:
            zero = jnp.zeros((), dtype=deformation.dtype)
            false = jnp.asarray(False)
            branch = jnp.asarray(-1, dtype=jnp.int32)
            diagnostics = {
                key: zero
                for key in (
                    "pressure",
                    "image_pressure",
                    "plastic_multiplier",
                    "yield_residual",
                    "dissipation",
                    "image_limit",
                    "limit_stress_ratio",
                    "lode_factor",
                    "dilatancy",
                    "pressure_root_residual",
                )
            }
            diagnostics.update(
                {
                    "successful": false,
                    "lode_defined": false,
                    "tangent_successful": false,
                    "branch_code": branch,
                    "suggested_step": 0.5 * step,
                }
            )
            return (
                jnp.zeros_like(deformation),
                history,
                zero,
                zero,
                branch,
                false,
                diagnostics,
            )

        return jax.lax.cond(
            admitted,
            lambda _: self._admitted_point(
                deformation, history, density, par, step, determinant, old_geometry
            ),
            rejected,
            None,
        )

    def _admitted_point(
        self,
        deformation: Array,
        history: Array,
        density: Array,
        par: NorSandParameters,
        step: Array,
        determinant: Array,
        relative: SmallLinearSolveResult,
    ) -> tuple[Array, Array, Array, Array, Array, Array, dict[str, Array]]:
        plan = SmallLinearSolvePlan(3)
        old_f = history[12:21].reshape(3, 3)
        increment = relative.value.T
        polar = hermitian_inverse_sqrt(increment.T @ increment)
        rotation = increment @ polar.value
        metric = unpack_metric(history[:6])
        trial_be = deformation @ metric @ deformation.T
        old_be = rotation @ old_f @ metric @ old_f.T @ rotation.T
        trial_log, old_log = hermitian_log(trial_be), hermitian_log(old_be)
        delta = 0.5 * (trial_log.value - old_log.value)
        old_tau = rotation @ unpack_metric(history[6:12]) @ rotation.T
        old_p = -jnp.trace(old_tau) / 3
        old_s = old_tau + old_p * jnp.eye(3, dtype=deformation.dtype)
        void = (1 + history[21]) * determinant / relative.determinant - 1

        def mapped_at(ell: Array) -> HermitianSpectralMapResult:
            pressure = par.reference_pressure * jnp.exp(ell)
            shear, bulk = _moduli(pressure, par)
            predictor = (
                -old_s
                - 2
                * shear
                * (delta - jnp.trace(delta) / 3 * jnp.eye(3, dtype=deformation.dtype))
                + (old_p - bulk * jnp.trace(delta)) * jnp.eye(3, dtype=deformation.dtype)
            )
            # Preserve one rounded tensor for both sides of symmetry admission.
            normalized = jax.lax.optimization_barrier(predictor / par.reference_pressure)
            return hermitian_spectral_map(
                normalized,
                _principal_return,
                (par, void, history[22], pressure),
                function_id="norsand-principal-return",
            )

        def pressure_residual(ell: Array) -> Array:
            mapped = mapped_at(ell)
            return (
                mapped.invariants["pressure"] - par.reference_pressure * jnp.exp(ell)
            ) / par.reference_pressure

        ell, root_diag = LocalRootPlan(
            plan_id="norsand-endpoint-pressure",
            tolerance=_root_tolerance(deformation.dtype),
        ).solve_with_diagnostics(
            pressure_residual, jnp.log(old_p / par.reference_pressure)
        )
        mapped = mapped_at(ell)
        tau, plastic = mapped.values[0], mapped.values[1]
        corrected = hermitian_exp(trial_log.value + 2 * plastic)
        left = solve_small_linear(plan, deformation, corrected.value)
        right = solve_small_linear(plan, deformation, left.value.T)
        stress = solve_small_linear(plan, deformation, tau.T)
        candidate = jnp.concatenate(
            (
                pack_metric(right.value.T),
                pack_metric(tau),
                deformation.reshape(-1),
                jnp.stack((void, mapped.invariants["image_pressure"])),
            )
        )
        p = mapped.invariants["pressure"]
        g, k = _moduli(p, par)
        speed = jnp.sqrt((k + 4 * g / 3) / density)
        valid = (
            root_diag.converged
            & relative.successful
            & polar.valid
            & trial_log.valid
            & old_log.valid
            & mapped.valid
            & mapped.invariants["successful"]
            & corrected.valid
            & left.successful
            & right.successful
            & stress.successful
            & (determinant > 0)
            & (relative.determinant > 0)
            & (density > 0)
            & jnp.all(jnp.isfinite(candidate))
            & jnp.isfinite(speed)
        )
        diagnostics = dict(mapped.invariants)
        strict_branch = (diagnostics["plastic_multiplier"] > 0) | (
            diagnostics["yield_residual"] < -_root_tolerance(deformation.dtype)
        )
        image_kink = (
            (diagnostics["plastic_multiplier"] > 0)
            & (par.volumetric_coupling > 0)
            & (
                jnp.abs(void - _ec(diagnostics["image_pressure"], par))
                <= _root_tolerance(deformation.dtype)
            )
        )
        diagnostics["tangent_successful"] = (
            valid
            & mapped.derivative_valid
            & polar.derivative_valid
            & trial_log.derivative_valid
            & old_log.derivative_valid
            & strict_branch
            & ~image_kink
        )
        diagnostics["pressure_root_residual"] = root_diag.residual
        diagnostics["suggested_step"] = jnp.where(valid, jnp.inf, 0.5 * step)
        return (
            jnp.where(valid, stress.value.T, 0),
            jnp.where(valid, candidate, history),
            jnp.where(valid, speed, 0),
            jnp.where(valid, mapped.invariants["dissipation"], 0),
            jnp.where(valid, mapped.invariants["branch_code"], -1).astype(jnp.int32),
            valid,
            diagnostics,
        )

    def evaluate(
        self,
        deformation_gradient: ArrayLike,
        committed_state: ArrayLike,
        reference_density: ArrayLike,
        parameters: NorSandParameters,
        time: ArrayLike,
        step_size: ArrayLike,
        /,
    ) -> MPMConstitutiveResponse:
        del time
        f = jnp.asarray(deformation_gradient)
        batch = f.shape[:-2]
        state = jnp.asarray(committed_state, dtype=f.dtype)
        rho = jnp.asarray(reference_density, dtype=f.dtype)
        if f.shape[-2:] != (3, 3) or state.shape != batch + (23,) or rho.shape != batch:
            raise ValueError("NorSand deformation/state/density shapes are incompatible.")
        result = jax.vmap(
            lambda ff, hh, rr: self._point(
                ff, hh, rr, parameters, jnp.asarray(step_size, dtype=f.dtype)
            )
        )(f.reshape((-1, 3, 3)), state.reshape((-1, 23)), rho.reshape(-1))
        stress, candidate, speed, dissipation, branch, valid, diagnostics = result
        return MPMConstitutiveResponse(
            stress.reshape(batch + (3, 3)),
            candidate.reshape(batch + (23,)),
            jnp.zeros(batch, dtype=f.dtype),
            speed.reshape(batch),
            dissipation_increment=dissipation.reshape(batch),
            branch_code=branch.reshape(batch),
            successful=valid.reshape(batch),
            admissible=valid.reshape(batch),
            suggested_step=diagnostics["suggested_step"].reshape(batch),
            diagnostics={
                key: val.reshape(batch + val.shape[1:])
                for key, val in diagnostics.items()
            },
        )

    def evaluate_linearized(
        self,
        deformation_gradient: ArrayLike,
        committed_state: ArrayLike,
        reference_density: ArrayLike,
        parameters: NorSandParameters,
        time: ArrayLike,
        step_size: ArrayLike,
        /,
    ) -> MPMLinearizedConstitutiveResponse:
        response = self.evaluate(
            deformation_gradient,
            committed_state,
            reference_density,
            parameters,
            time,
            step_size,
        )
        f = jnp.asarray(deformation_gradient)
        history = jnp.asarray(committed_state, dtype=f.dtype)
        rho = jnp.asarray(reference_density, dtype=f.dtype)
        batch = f.shape[:-2]
        tangent = jax.vmap(
            jax.jacfwd(
                lambda ff, hh, rr: self._point(
                    ff, hh, rr, parameters, jnp.asarray(step_size, dtype=f.dtype)
                )[0],
                argnums=0,
            )
        )(f.reshape((-1, 3, 3)), history.reshape((-1, 23)), rho.reshape(-1)).reshape(
            batch + (3, 3, 3, 3)
        )
        valid = (
            response.successful
            & response.diagnostics["tangent_successful"]
            & jnp.all(jnp.isfinite(tangent), axis=(-4, -3, -2, -1))
        )
        return MPMLinearizedConstitutiveResponse(response, tangent, valid)
