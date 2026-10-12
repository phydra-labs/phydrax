# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Borja–Tamagnini multiplicative hyperelastic Modified Cam–Clay.

Reference: Borja and Tamagnini (1998), CMAME 155, 73–95,
https://doi.org/10.1016/S0045-7825(97)00141-2, Eqs. (3.4–3.12),
(3.24–3.32). Stress is tension-positive and p=-tr(tau)/3>0.
With a=tr(epsilon_e), b=sqrt(2 dev(epsilon_e):dev(epsilon_e)/3),
R=kappa/(1+e0), D=(lambda-kappa)/(1+e0), B=p0 exp(-a/R),
mu=mu0+alpha B, the energy is psi=B R+3 mu b²/2. Its derivatives
are p=B(1+3 alpha b²/(2R)), q=3 mu b and tau=-p I+2mu dev(epsilon_e).
Thus mu0 is additive, not the initial total shear modulus mu0+alpha p0.
The fixed reference elastic volumetric strain is zero; p0 is the initial
mean pressure, independent of the preconsolidation pressure pc.

The unscaled ellipse f=q²/M²+p(p-pc) has multiplier gamma in inverse
stress units. Associated flow and exact natural-volume hardening give
pc=pc_n exp((a-a_trial)/D), and the three residuals
(a-a_trial-gamma(2p-pc), b-b_trial+2gamma q/M², f).
The native vector root solves these with stresses normalized by p0 and
g=gamma p0; this is an exact change of units, not a different flow rule.
Implicit-function derivatives compose with the principal spectral map;
the full first-Piola tangent is AD of the discrete multiplicative update.
Neither plastic tangents nor the spectral directions are frozen/symmetrized.

History: inverse plastic metric (11,22,33,23,13,12), pc, p0, initial e0.
The reference e0 is fixed; physical current void ratio is (1+e0)det(F)-1.
Initialization must specify a compressive effective Kirchhoff stress. The
shape-only initializer is deliberately unadmitted (pc=p0=0). Invalid
pressure, metric, energy Hessian, or local-root results do not commit history.
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
from ...linalg import hermitian_exp
from ...nonlinear import LocalRootPlan, VectorLocalRootPlan
from ...typing import checked
from ._principal_hencky import (
    evaluate_principal_point,
    initialize_metric_state,
    pack_metric,
    PrestressedMaterialState,
)


_RETURN_ROOT = VectorLocalRootPlan(
    3, maximum_steps=40, tolerance=1e-11, plan_id="borja-tamagnini-mcc-principal-return"
)
_INVERSE_ROOT = LocalRootPlan(
    maximum_steps=40, tolerance=1e-11, plan_id="borja-tamagnini-mcc-inverse-elasticity"
)


@final
class ModifiedCamClayParameters(StrictModule, NonTrainableState):
    """Raw e–ln(p) slopes and the two-invariant hyperelastic coefficients."""

    reference_shear_modulus: Array
    shear_pressure_coupling: Array
    critical_state_slope: Array
    compression_index: Array
    swelling_index: Array

    def __init__(
        self,
        reference_shear_modulus: ArrayLike,
        shear_pressure_coupling: ArrayLike,
        critical_state_slope: ArrayLike,
        compression_index: ArrayLike,
        swelling_index: ArrayLike,
        /,
    ) -> None:
        values = tuple(
            jnp.asarray(value)
            for value in (
                reference_shear_modulus,
                shear_pressure_coupling,
                critical_state_slope,
                compression_index,
                swelling_index,
            )
        )
        if any(value.shape != () or not bool(jnp.isfinite(value)) for value in values):
            raise ValueError("Modified Cam-Clay parameters must be finite scalars.")
        mu, alpha, slope, compression, swelling = values
        if (
            mu < 0
            or alpha < 0
            or (mu == 0 and alpha == 0)
            or slope <= 0
            or swelling <= 0
            or compression <= swelling
        ):
            raise ValueError("Modified Cam-Clay parameters are inadmissible.")
        (
            self.reference_shear_modulus,
            self.shear_pressure_coupling,
            self.critical_state_slope,
            self.compression_index,
            self.swelling_index,
        ) = values


def _elastic_invariants(
    a: Array, b: Array, parameters: ModifiedCamClayParameters, p0: Array, e0: Array
) -> tuple[Array, Array, Array, Array, Array, Array]:
    r = parameters.swelling_index / (1.0 + e0)
    base = p0 * jnp.exp(-a / r)
    mu = parameters.reference_shear_modulus + parameters.shear_pressure_coupling * base
    p = base * (1.0 + 1.5 * parameters.shear_pressure_coupling * b**2 / r)
    q = 3.0 * mu * b
    energy = base * r + 1.5 * mu * b**2
    # The remaining four deviatoric Hessian eigenvalues are 2mu. The
    # volume/shear 2x2 block is positive iff its determinant is positive.
    coupling = 3.0 * parameters.shear_pressure_coupling * base * b / r
    stable = (p > 0.0) & (mu > 0.0) & (3.0 * mu * p / r > coupling**2)
    # Trace plus the coupling norm bounds the elastic tensor operator norm.
    wave = (
        3.0 * p / r
        + 2.0 * mu
        + 2.0 * jnp.sqrt(6.0) * jnp.abs(parameters.shear_pressure_coupling * base * b / r)
    )
    return p, q, mu, energy, stable, wave


def _shear_norm(deviator: Array) -> Array:
    squared = 2.0 * jnp.sum(deviator**2) / 3.0
    positive = squared > 0.0
    # Batched cond lowers to select: an inactive sqrt(0) derivative would
    # transpose as 0*inf. A neutral inactive operand preserves the exact norm
    # (no strain floor) and the isotropic tensor derivative carried by ratio.
    return jnp.where(positive, jnp.sqrt(jnp.where(positive, squared, 1.0)), 0.0)


def _mcc_principal(
    eigenvalues: Array, data: tuple[ModifiedCamClayParameters, Array]
) -> tuple[Array, dict[str, Array]]:
    parameters, scalars = data
    pc_old, p0, e0 = scalars
    admitted = (pc_old > 0.0) & (p0 > 0.0) & (e0 >= 0.0)
    # Neutral values only evaluate rejected inputs finitely, never repair history.
    reference = jnp.where(admitted, p0, 1.0)
    void = jnp.where(admitted, e0, 0.0)
    strain = 0.5 * jnp.log(eigenvalues)
    a_trial = jnp.sum(strain)
    deviator = strain - a_trial / 3.0
    b_trial = _shear_norm(deviator)
    p_trial, q_trial, _, _, trial_stable, _ = _elastic_invariants(
        a_trial, b_trial, parameters, reference, void
    )
    f_trial = q_trial**2 / parameters.critical_state_slope**2 + p_trial * (
        p_trial - pc_old
    )
    difference = (parameters.compression_index - parameters.swelling_index) / (1.0 + void)
    tolerance = jnp.maximum(1e-10, 64.0 * jnp.finfo(eigenvalues.dtype).eps)
    root_plan = VectorLocalRootPlan(
        3,
        maximum_steps=_RETURN_ROOT.maximum_steps,
        tolerance=max(
            _RETURN_ROOT.tolerance, 64.0 * float(jnp.finfo(eigenvalues.dtype).eps)
        ),
        plan_id="borja-tamagnini-mcc-principal-return",
    )

    def plastic(_: None) -> tuple[Array, Array, Array, Array, Array, Array, Array, Array]:
        def residual(x: Array) -> Array:
            a, b, g = x
            p, q, _, _, _, _ = _elastic_invariants(a, b, parameters, reference, void)
            pc = pc_old * jnp.exp((a - a_trial) / difference)
            pn, qn, pcn = p / reference, q / reference, pc / reference
            return jnp.stack(
                (
                    a - a_trial - g * (2.0 * pn - pcn),
                    b - b_trial + 2.0 * g * qn / parameters.critical_state_slope**2,
                    qn**2 / parameters.critical_state_slope**2 + pn * (pn - pcn),
                )
            )

        x, evidence = root_plan.solve_with_diagnostics(
            residual, jnp.stack((a_trial, b_trial, jnp.zeros_like(a_trial)))
        )
        a, b, g = x
        pc = pc_old * jnp.exp((a - a_trial) / difference)
        return (
            a,
            b,
            g / reference,
            pc,
            evidence.converged,
            evidence.iterations,
            evidence.residual_norm,
            evidence.condition_estimate,
        )

    def elastic(_: None) -> tuple[Array, Array, Array, Array, Array, Array, Array, Array]:
        return (
            a_trial,
            b_trial,
            jnp.zeros_like(a_trial),
            pc_old,
            jnp.asarray(True),
            jnp.asarray(0, dtype=jnp.int32),
            jnp.zeros_like(a_trial),
            jnp.ones_like(a_trial),
        )

    active = admitted & trial_stable & (f_trial > tolerance * reference**2)
    a, b, gamma, pc, root_ok, iterations, root_norm, root_condition = jax.lax.cond(
        active, plastic, elastic, None
    )
    p, q, mu, energy, stable, wave = _elastic_invariants(
        a, b, parameters, reference, void
    )
    # r2 gives this ratio without dividing by zero at an isotropic plastic
    # return. It retains the correct deviatoric tangent in that limit.
    ratio = 1.0 / (1.0 + 6.0 * gamma * mu / parameters.critical_state_slope**2)
    corrected_strain = a / 3.0 + ratio * deviator
    tau = -p + 2.0 * mu * ratio * deviator
    residual = q**2 / parameters.critical_state_slope**2 + p * (p - pc)
    scale = jnp.maximum(reference, jnp.maximum(pc, jnp.maximum(p, q)))
    successful = (
        admitted
        & trial_stable
        & stable
        & root_ok
        & (b >= 0.0)
        & (gamma >= 0.0)
        & (pc > 0.0)
        & (residual <= tolerance * scale**2)
        & jnp.isfinite(energy)
    )
    # tau:delta epsilon_p from associated normality; exactly zero elastically.
    dissipation = gamma * (
        2.0 * q**2 / parameters.critical_state_slope**2 + p * (2.0 * p - pc)
    )
    return jnp.stack((tau, jnp.exp(2.0 * corrected_strain))), {
        "scalars": jnp.stack((pc, p0, e0)),
        "successful": successful,
        "branch_code": jnp.where(active, 1, 0).astype(jnp.int32),
        "strict_branch": jnp.abs(f_trial - tolerance * reference**2)
        > tolerance
        * jnp.maximum(reference, jnp.maximum(pc_old, jnp.maximum(p_trial, q_trial))) ** 2,
        "energy": energy,
        "dissipation": dissipation,
        "yield_residual": residual,
        "plastic_multiplier": gamma,
        "mean_pressure": p,
        "equivalent_stress": q,
        "preconsolidation_pressure": pc,
        "reference_pressure": p0,
        "initial_void_ratio": e0,
        "elastic_volume_strain": a,
        "elastic_shear_strain": b,
        "root_successful": root_ok,
        "root_iterations": iterations,
        "elastic_tangent_positive": stable,
        "root_residual_norm": jax.lax.stop_gradient(root_norm),
        "root_condition_estimate": jax.lax.stop_gradient(root_condition),
        "wave_modulus": wave,
    }


def _initialize_point(
    stress: Array, pc: Array, e0: Array, parameters: ModifiedCamClayParameters
) -> PrestressedMaterialState:
    identity = jnp.eye(3, dtype=stress.dtype)
    p0 = -jnp.trace(stress) / 3.0
    s = stress + p0 * identity
    q_squared = 1.5 * jnp.sum(s**2)
    valid_input = (p0 > 0.0) & (pc > 0.0) & (e0 >= 0.0)
    reference = jnp.where(valid_input, p0, 1.0)
    void = jnp.where(valid_input, e0, 0.0)
    r = parameters.swelling_index / (1.0 + void)
    inverse_plan = LocalRootPlan(
        maximum_steps=_INVERSE_ROOT.maximum_steps,
        tolerance=max(_INVERSE_ROOT.tolerance, 64.0 * float(jnp.finfo(stress.dtype).eps)),
        plan_id="borja-tamagnini-mcc-inverse-elasticity",
    )

    def coupled(_: None) -> tuple[Array, Array]:
        def inverse_residual(log_ratio: Array) -> Array:
            base = reference * jnp.exp(log_ratio)
            mu = (
                parameters.reference_shear_modulus
                + parameters.shear_pressure_coupling * base
            )
            return (
                base
                / reference
                * (
                    1.0
                    + parameters.shear_pressure_coupling * q_squared / (6.0 * r * mu**2)
                )
                - 1.0
            )

        ratio, evidence = inverse_plan.solve_with_diagnostics(
            inverse_residual, jnp.zeros_like(reference)
        )
        return reference * jnp.exp(ratio), evidence.converged & evidence.finite

    base, root_ok = jax.lax.cond(
        parameters.shear_pressure_coupling > 0.0,
        coupled,
        lambda _: (reference, jnp.asarray(True)),
        None,
    )
    mu = parameters.reference_shear_modulus + parameters.shear_pressure_coupling * base
    a = -r * jnp.log(base / reference)
    strain = s / (2.0 * mu) + a * identity / 3.0
    metric = hermitian_exp(2.0 * strain)
    state = jnp.concatenate((pack_metric(metric.value), jnp.stack((pc, p0, e0))))
    # Off-diagonal prestresses also contribute to the Hessian condition.
    b = _shear_norm(s / (2.0 * mu))
    _, _, _, _, stable, _ = _elastic_invariants(a, b, parameters, reference, void)
    value = q_squared / parameters.critical_state_slope**2 + p0 * (p0 - pc)
    tolerance = jnp.maximum(1e-10, 64.0 * jnp.finfo(stress.dtype).eps)
    scale = jnp.maximum(jnp.abs(p0), jnp.abs(pc))
    admissible = (
        valid_input
        & root_ok
        & metric.valid
        & stable
        & jnp.all(stress == stress.T)
        & jnp.all(jnp.isfinite(state))
        & (value <= tolerance * scale**2)
    )
    return PrestressedMaterialState(state, admissible, value)


@final
class ModifiedCamClayMPMConstitutivePlan(
    AbstractImplicitMPMConstitutivePlan, NonTrainableState
):
    dimension: int = eqx.field(static=True)
    kinematics: str = eqx.field(static=True)
    state_shape: tuple[int, ...] = eqx.field(static=True)
    capabilities: MPMConstitutiveCapabilities
    plan_id: str = eqx.field(static=True)

    def __init__(self) -> None:
        self.dimension, self.kinematics, self.state_shape = 3, "three_dimensional", (9,)
        self.capabilities = MPMConstitutiveCapabilities(
            stateful=True,
            has_free_energy=True,
            has_algorithmic_tangent=True,
            has_dissipation=True,
            supports_implicit=True,
        )
        self.plan_id = canonical_fingerprint(
            {
                "kind": "modified-cam-clay-mpm",
                "elasticity": "borja-tamagnini-hyperelastic",
                "history": "plastic-metric-pc-p0-e0",
                "return_root": _RETURN_ROOT.plan_id,
                "inverse_root": _INVERSE_ROOT.plan_id,
            }
        )

    def initialize_state(
        self, batch_shape: tuple[int, ...], dtype: DTypeLike, /
    ) -> Array:
        """Allocate unadmitted history; initialize reference data from stress."""
        return initialize_metric_state(batch_shape, dtype, scalars=(0.0, 0.0, 0.0))

    @checked
    def initialize_state_from_stress(
        self,
        kirchhoff_stress: ArrayLike,
        parameters: ModifiedCamClayParameters,
        /,
        *,
        preconsolidation_pressure: ArrayLike,
        initial_void_ratio: ArrayLike,
    ) -> PrestressedMaterialState:
        stress = jnp.asarray(kirchhoff_stress)
        if stress.shape[-2:] != (3, 3):
            raise ValueError("Kirchhoff stress must end in 3x3.")
        shape = stress.shape[:-2]
        pc = jnp.broadcast_to(
            jnp.asarray(preconsolidation_pressure, dtype=stress.dtype), shape
        )
        e0 = jnp.broadcast_to(jnp.asarray(initial_void_ratio, dtype=stress.dtype), shape)
        initialized = jax.vmap(
            lambda tau, cap, void: _initialize_point(tau, cap, void, parameters)
        )(stress.reshape((-1, 3, 3)), pc.reshape((-1,)), e0.reshape((-1,)))
        return PrestressedMaterialState(
            initialized.state.reshape(shape + (9,)),
            initialized.admissible.reshape(shape),
            initialized.yield_value.reshape(shape),
        )

    def _point(
        self,
        f: Array,
        history: Array,
        rho: Array,
        parameters: ModifiedCamClayParameters,
        dt: Array,
    ) -> tuple[Array, Array, Array, Array, Array, Array, Array, dict[str, Array]]:
        return evaluate_principal_point(
            f,
            history,
            rho,
            parameters,
            _mcc_principal,
            function_id="borja-tamagnini-mcc",
            step_size=dt,
            wave_modulus=parameters.reference_shear_modulus,
        )

    @checked
    def evaluate(
        self,
        deformation_gradient: ArrayLike,
        committed_state: ArrayLike,
        reference_density: ArrayLike,
        parameters: ModifiedCamClayParameters,
        time: ArrayLike,
        step_size: ArrayLike,
        /,
    ) -> MPMConstitutiveResponse:
        del time
        f = jnp.asarray(deformation_gradient)
        if f.shape[-2:] != (3, 3):
            raise ValueError("Modified Cam-Clay deformation must end in 3x3.")
        shape = f.shape[:-2]
        history = jnp.asarray(committed_state, dtype=f.dtype)
        if history.shape != shape + (9,):
            raise ValueError("Modified Cam-Clay history shape changed.")
        rho = jnp.broadcast_to(jnp.asarray(reference_density, dtype=f.dtype), shape)
        dt = jnp.asarray(step_size, dtype=f.dtype)
        outputs = jax.vmap(
            lambda value, state, density: self._point(
                value, state, density, parameters, dt
            )
        )(f.reshape((-1, 3, 3)), history.reshape((-1, 9)), rho.reshape((-1,)))
        stress, state, energy, speed, dissipation, branch, valid, diagnostics = outputs
        diagnostics["tangent_successful"] &= diagnostics["strict_branch"]
        return MPMConstitutiveResponse(
            stress.reshape(shape + (3, 3)),
            state.reshape(shape + (9,)),
            energy.reshape(shape),
            speed.reshape(shape),
            dissipation_increment=dissipation.reshape(shape),
            branch_code=branch.reshape(shape),
            suggested_step=jnp.where(valid, jnp.inf, 0.5 * dt).reshape(shape),
            successful=valid.reshape(shape),
            admissible=valid.reshape(shape),
            diagnostics={
                name: value.reshape(shape) for name, value in diagnostics.items()
            },
        )

    @checked
    def evaluate_linearized(
        self,
        deformation_gradient: ArrayLike,
        committed_state: ArrayLike,
        reference_density: ArrayLike,
        parameters: ModifiedCamClayParameters,
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
        shape = f.shape[:-2]
        history = jnp.asarray(committed_state, dtype=f.dtype)
        rho = jnp.broadcast_to(jnp.asarray(reference_density, dtype=f.dtype), shape)
        dt = jnp.asarray(step_size, dtype=f.dtype)

        def stress(value: Array, state: Array, density: Array) -> Array:
            return self._point(value, state, density, parameters, dt)[0]

        tangent = jax.vmap(jax.jacfwd(stress))(
            f.reshape((-1, 3, 3)), history.reshape((-1, 9)), rho.reshape((-1,))
        ).reshape(shape + (3, 3, 3, 3))
        successful = response.diagnostics["tangent_successful"] & jnp.all(
            jnp.isfinite(tangent), axis=(-4, -3, -2, -1)
        )
        return MPMLinearizedConstitutiveResponse(response, tangent, successful)
