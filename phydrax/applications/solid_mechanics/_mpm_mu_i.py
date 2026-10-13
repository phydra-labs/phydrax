# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Multiplicative Hencky elastic–viscoplastic granular material.

Dunatunga & Kamrin (2015), JFM 779:483–513, doi:10.1017/jfm.2015.383,
https://arxiv.org/abs/1411.5447, Eqs. (2.6), (2.10)–(2.13), (3.12)–(3.13).
Their shear norm is t = sqrt(s:s/2) = q/sqrt(3), not the von Mises q.
Their rate is r = sqrt(2 Dp:Dp). Thus t_tr-G dt r = p mu(I), with
Dp = r s/(2 t), and I = r d sqrt(rho_s/p_cauchy). For Kirchhoff stress,
p_cauchy=p/J while t and p in the return residual are Kirchhoff quantities.
The returned Hencky deviator is scaled by t/t_tr, preserving elastic volume
and hence det(B_e) on connected branches. Plastic work is t dt r >= 0.

This is the finite-strain Hencky realization requested by the material contract,
not the paper's Jaumann hypoelastic update or density equation of state.
At p_tr <= 0 the corrected B_e is identity and stress-free; all trial elastic
energy is released as dissipation. No hidden critical-density parameter is used.
The principal map is permutation equivariant; all diagnostics are invariant.
"""

from __future__ import annotations

from typing import final

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike, DTypeLike

from ..._fingerprint import canonical_fingerprint
from ..._trainable import NonTrainableState
from ...equations import (
    AbstractImplicitMPMConstitutivePlan,
    MPMConstitutiveCapabilities,
    MPMConstitutiveResponse,
    MPMLinearizedConstitutiveResponse,
)
from ...linalg import hermitian_exp, SmallLinearSolvePlan, solve_small_linear
from ...nonlinear import LocalRootPlan
from ...rheology import MuILaw, MuIParameters
from ...typing import checked
from ._principal_hencky import (
    elastic_principal_metric,
    evaluate_elastic_metric_point,
    pack_metric,
    PrestressedMaterialState,
    principal_trial,
    unpack_metric,
)


_JOP = MuILaw()
# Under simple shear, the globally scaled determinant decays as gamma^-3,
# while the solve condition grows only as gamma^2. Retain the native dtype
# singularity floor and condition limit without the default early cutoff.
_SOLVE = SmallLinearSolvePlan(3, singular_tolerance=1.0e-15)
_ROOT = LocalRootPlan(
    maximum_steps=4,
    tolerance=1.0e-6,
    minimum_derivative=1.0e-14,
    plan_id="mu-i-mpm-logistic-return",
)


def _mu_i_principal(
    eigenvalues: Array,
    data: tuple[tuple[MuIParameters, Array, Array, Array, Array], Array],
) -> tuple[Array, dict[str, Array]]:
    (parameters, shear, bulk, step, jacobian), scalars = data
    _, trial_tau, pressure, q, deviator = principal_trial(eigenvalues, shear, bulk)
    trial_t = q / jnp.sqrt(3.0)
    trial_energy = q**2 / (6.0 * shear) + pressure**2 / (2.0 * bulk)
    zero = jnp.zeros_like(pressure)
    connected = pressure > 0.0
    physical_pressure = jnp.where(connected, pressure / jacobian, 1.0)

    def dense(_: None) -> tuple[Array, Array, Array, Array, Array, Array]:
        excess = trial_t - pressure * parameters.mu_s

        def flow(_: None) -> tuple[Array, Array, Array, Array, Array, Array]:
            upper = excess / (shear * step)

            def fraction_residual(fraction: Array) -> Array:
                law = _JOP.evaluate(upper * fraction, physical_pressure, parameters)
                return (
                    trial_t - excess * fraction - pressure * law.friction_coefficient
                ) / excess

            def bisect(
                _: int | Array, bracket: tuple[Array, Array]
            ) -> tuple[Array, Array]:
                lower, higher = bracket
                midpoint = (lower + higher) / 2.0
                positive = fraction_residual(midpoint) > 0.0
                return jnp.where(positive, midpoint, lower), jnp.where(
                    positive, higher, midpoint
                )

            lower, higher = jax.lax.fori_loop(0, 40, bisect, (zero, jnp.ones_like(zero)))
            fraction = jax.lax.stop_gradient((lower + higher) / 2.0)
            guess = jnp.log(fraction) - jnp.log1p(-fraction)

            def transformed_residual(value: Array) -> Array:
                return fraction_residual(jax.nn.sigmoid(value))

            root, evidence = _ROOT.solve_with_diagnostics(transformed_residual, guess)
            rate = upper * jax.nn.sigmoid(root)
            law = _JOP.evaluate(rate, physical_pressure, parameters)
            returned_t = pressure * law.friction_coefficient
            # d(t)/d(r)/G is the Maxwell tangent relaxation time in seconds.
            slope = jax.grad(
                lambda r: (
                    _JOP.evaluate(r, physical_pressure, parameters).friction_coefficient
                )
            )(rate)
            relaxation = pressure * slope / shear
            valid = (
                evidence.converged
                & law.successful
                & (rate > 0.0)
                & (returned_t <= trial_t)
            )
            return (
                returned_t,
                rate,
                relaxation,
                valid,
                evidence.residual,
                jnp.asarray(1, dtype=jnp.int32),
            )

        return jax.lax.cond(
            (excess > 0.0) & (step > 0.0),
            flow,
            lambda _: (
                trial_t,
                zero,
                zero,
                jnp.asarray(True),
                excess,
                jnp.asarray(0, dtype=jnp.int32),
            ),
            None,
        )

    returned_t, rate, relaxation, converged, residual, branch = jax.lax.cond(
        connected,
        dense,
        lambda _: (
            zero,
            zero,
            zero,
            jnp.asarray(True),
            zero,
            jnp.asarray(2, dtype=jnp.int32),
        ),
        None,
    )
    denominator = jnp.where(trial_t > 0.0, trial_t, 1.0)
    tau = jnp.where(
        connected,
        jnp.where(
            branch == 0, trial_tau, deviator * (returned_t / denominator) - pressure
        ),
        jnp.zeros_like(trial_tau),
    )
    corrected = elastic_principal_metric(tau, shear, bulk)
    energy = jnp.where(
        connected, returned_t**2 / (2.0 * shear) + pressure**2 / (2.0 * bulk), zero
    )
    dissipation = jnp.where(connected, returned_t * step * rate, trial_energy)
    successful = (
        converged
        & (step >= 0.0)
        & jnp.isfinite(step)
        & (jacobian > 0.0)
        & (~connected | (trial_t <= pressure * parameters.mu_s) | (step > 0.0))
    )
    return jnp.stack((tau, corrected)), {
        "scalars": scalars,
        "successful": successful,
        "branch_code": branch,
        "energy": energy,
        "dissipation": dissipation,
        "yield_residual": residual,
        "plastic_multiplier": step * rate / jnp.sqrt(3.0),
        "plastic_shear_rate": rate,
        "relaxation_time": relaxation,
        "physical_pressure": jnp.where(connected, physical_pressure, zero),
        "inertial_number": jnp.where(
            connected,
            rate
            * parameters.grain_diameter_m
            * jnp.sqrt(
                parameters.grain_density_kg_m3
                / jnp.where(connected, physical_pressure, 1.0)
            ),
            zero,
        ),
        "strict_branch": (pressure != 0.0)
        & (~connected | (trial_t != pressure * parameters.mu_s)),
    }


@final
class MuIMPMConstitutivePlan(AbstractImplicitMPMConstitutivePlan, NonTrainableState):
    """Three-dimensional Jop μ(I) material; elastic moduli are dynamic leaves.

    Pass canonical ``MuIParameters`` to evaluate. Branch codes are 0 elastic,
    1 viscoplastic, 2 disconnected, -1 failure. History stores six packed B_e
    components followed by nine row-major previous-F entries, with no scalars.
    """

    shear_modulus: Array
    bulk_modulus: Array
    dimension: int = eqx.field(static=True)
    kinematics: str = eqx.field(static=True)
    state_shape: tuple[int, ...] = eqx.field(static=True)
    capabilities: MPMConstitutiveCapabilities
    plan_id: str = eqx.field(static=True)

    def __init__(self, shear_modulus: ArrayLike, bulk_modulus: ArrayLike, /) -> None:
        shear, bulk = jnp.asarray(shear_modulus), jnp.asarray(bulk_modulus)
        if shear.shape != () or bulk.shape != ():
            raise ValueError("μ(I) elastic moduli must be scalar.")
        dtype = jnp.result_type(shear, bulk)
        if not jnp.issubdtype(dtype, jnp.floating):
            dtype = jnp.dtype(jnp.float32)
        shear, bulk = shear.astype(dtype), bulk.astype(dtype)
        shear, bulk = eqx.error_if(
            (shear, bulk),
            ~jnp.isfinite(shear) | ~jnp.isfinite(bulk) | (shear <= 0.0) | (bulk <= 0.0),
            "μ(I) elastic moduli must be positive and finite.",
        )
        self.shear_modulus, self.bulk_modulus = shear, bulk
        self.dimension, self.kinematics, self.state_shape = 3, "three_dimensional", (15,)
        self.capabilities = MPMConstitutiveCapabilities(
            stateful=True,
            has_free_energy=True,
            has_algorithmic_tangent=True,
            has_dissipation=True,
            supports_implicit=True,
        )
        self.plan_id = canonical_fingerprint(
            {
                "kind": "mu-i-mpm",
                "elasticity": "multiplicative-hencky",
                "history": "spatial-elastic-metric-relative-deformation",
                "law_id": _JOP.law_id,
                "root_plan_id": _ROOT.plan_id,
                "relative_solve_plan_id": _SOLVE.plan_id,
            }
        )

    def initialize_state(
        self, batch_shape: tuple[int, ...], dtype: DTypeLike, /
    ) -> Array:
        identity = jnp.eye(3, dtype=dtype)
        initial = jnp.concatenate((pack_metric(identity), identity.reshape((9,))))
        return jnp.broadcast_to(initial, batch_shape + self.state_shape)

    @checked
    def initialize_state_from_stress(
        self, kirchhoff_stress: ArrayLike, parameters: MuIParameters, /
    ) -> PrestressedMaterialState:
        """Invert Hencky elasticity at F=I without projecting prestress.
        Stress is tension-positive: hydrostatic compression is -p I. At
        F_previous=I the supplied Kirchhoff stress also equals Cauchy stress.

        Admissibility is the static friction cone (or stress-free disconnection);
        a viscoplastic overstress is reported, not silently returned.
        """
        stress = jnp.asarray(kirchhoff_stress)
        if stress.shape[-2:] != (3, 3):
            raise ValueError("Kirchhoff stress must end in 3x3.")
        identity = jnp.broadcast_to(jnp.eye(3, dtype=stress.dtype), stress.shape)
        mean = jnp.trace(stress, axis1=-2, axis2=-1) / 3.0
        deviator = stress - mean[..., None, None] * identity
        strain = deviator / (2.0 * self.shear_modulus) + (
            mean[..., None, None] * identity / (3.0 * self.bulk_modulus)
        )
        metric = hermitian_exp(2.0 * strain)
        state = jnp.concatenate(
            (pack_metric(metric.value), identity.reshape(stress.shape[:-2] + (9,))),
            axis=-1,
        )
        shear = jnp.sqrt(jnp.sum(deviator**2, axis=(-2, -1)) / 2.0)
        pressure = -mean
        value = shear - pressure * parameters.mu_s
        scale = jnp.maximum(
            jnp.maximum(self.shear_modulus, self.bulk_modulus),
            jnp.max(jnp.abs(stress), axis=(-2, -1)),
        )
        symmetric = jnp.all(stress == jnp.swapaxes(stress, -1, -2), axis=(-2, -1))
        admissible = (
            metric.valid
            & symmetric
            & jnp.all(jnp.isfinite(state), axis=-1)
            & (pressure >= 0.0)
            & (value <= 1.0e-10 * scale)
        )
        return PrestressedMaterialState(state, admissible, value)

    def _point(
        self,
        deformation: Array,
        history: Array,
        density: Array,
        parameters: MuIParameters,
        step: Array,
    ) -> tuple[Array, Array, Array, Array, Array, Array, Array, dict[str, Array]]:
        kinematics = solve_small_linear(_SOLVE, deformation, jnp.zeros_like(deformation))
        previous = history[6:15].reshape((3, 3))
        relative_solve = solve_small_linear(_SOLVE, previous.T, deformation.T)
        relative = relative_solve.value.T
        trial = relative @ unpack_metric(history[:6]) @ relative.T
        context = (
            parameters,
            self.shear_modulus,
            self.bulk_modulus,
            step,
            kinematics.determinant,
        )
        result = evaluate_elastic_metric_point(
            deformation,
            trial,
            history,
            density,
            context,
            _mu_i_principal,
            function_id="mu-i-hencky",
            step_size=step,
            wave_modulus=self.bulk_modulus + 4.0 * self.shear_modulus / 3.0,
            solve_plan=_SOLVE,
            kinematics=kinematics,
            history_successful=relative_solve.successful
            & (relative_solve.determinant > 0.0),
        )
        diagnostics = result[-1]
        diagnostics["previous_deformation_condition"] = relative_solve.condition_estimate
        diagnostics["relative_deformation_residual"] = relative_solve.residual_norm
        diagnostics["tangent_successful"] = (
            diagnostics["tangent_successful"] & diagnostics["strict_branch"]
        )
        diagnostics["suggested_step"] = jnp.where(
            result[6],
            jnp.where(result[5] == 1, diagnostics["relaxation_time"], jnp.inf),
            0.5 * step,
        )
        return result

    @checked
    def evaluate(
        self,
        deformation_gradient: ArrayLike,
        committed_state: ArrayLike,
        reference_density: ArrayLike,
        parameters: MuIParameters,
        time: ArrayLike,
        step_size: ArrayLike,
        /,
    ) -> MPMConstitutiveResponse:
        del time
        deformation = jnp.asarray(deformation_gradient)
        if deformation.shape[-2:] != (3, 3):
            raise ValueError("μ(I) deformation must end in 3x3.")
        shape = deformation.shape[:-2]
        history = jnp.asarray(committed_state, dtype=deformation.dtype)
        density = jnp.broadcast_to(
            jnp.asarray(reference_density, dtype=deformation.dtype), shape
        )
        step = jnp.asarray(step_size, dtype=deformation.dtype)
        if history.shape != shape + self.state_shape or step.shape != ():
            raise ValueError("μ(I) history shape or scalar step size changed.")
        outputs = jax.vmap(
            lambda f, state, rho: self._point(f, state, rho, parameters, step)
        )(
            deformation.reshape((-1, 3, 3)),
            history.reshape((-1, 15)),
            density.reshape((-1,)),
        )
        stress, trial, energy, speed, dissipation, branch, successful, diagnostics = (
            outputs
        )
        return MPMConstitutiveResponse(
            stress.reshape(shape + (3, 3)),
            trial.reshape(shape + self.state_shape),
            energy.reshape(shape),
            speed.reshape(shape),
            dissipation_increment=dissipation.reshape(shape),
            branch_code=branch.reshape(shape),
            suggested_step=diagnostics["suggested_step"].reshape(shape),
            successful=successful.reshape(shape),
            admissible=successful.reshape(shape),
            diagnostics={
                key: value.reshape(shape + value.shape[1:])
                for key, value in diagnostics.items()
            },
        )

    def evaluate_linearized(
        self,
        deformation_gradient: ArrayLike,
        committed_state: ArrayLike,
        reference_density: ArrayLike,
        parameters: MuIParameters,
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
        deformation = jnp.asarray(deformation_gradient)
        shape = deformation.shape[:-2]
        history = jnp.asarray(committed_state, dtype=deformation.dtype)
        density = jnp.broadcast_to(
            jnp.asarray(reference_density, dtype=deformation.dtype), shape
        )
        step = jnp.asarray(step_size, dtype=deformation.dtype)

        def stress(f: Array, state: Array, rho: Array) -> Array:
            return self._point(f, state, rho, parameters, step)[0]

        tangent = jax.vmap(jax.jacfwd(stress))(
            deformation.reshape((-1, 3, 3)),
            history.reshape((-1, 15)),
            density.reshape((-1,)),
        ).reshape(shape + (3, 3, 3, 3))
        successful = response.diagnostics["tangent_successful"] & jnp.all(
            jnp.isfinite(tangent), axis=(-4, -3, -2, -1)
        )
        return MPMLinearizedConstitutiveResponse(response, tangent, successful)
