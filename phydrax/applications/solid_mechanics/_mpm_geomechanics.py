#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Multiplicative Hencky Drucker–Prager integration.

Stress is tension-positive: p=-tr(tau)/3, q=sqrt(3 s:s/2).
The outer Mohr–Coulomb fit gives M=6 sin(phi)/(3-sin(phi)) and
k=6 c cos(phi)/(3-sin(phi))+H alpha. Thus f=q-M p-k and
its nonassociated potential g=q-Mpsi p has gradient
3 s/(2q)+Mpsi I/3. Elastic contraction of this gradient gives
q=qtrial-3G gamma and p=ptrial+K Mpsi gamma. Substitution in f=0
therefore gives gamma=ftrial/(3G+K M Mpsi+H).

If that return crosses q=0, all deviatoric elastic strain is removed.
Writing v=tr(delta epsilon_p), alpha+=v/Mpsi and p=ptrial+K v
in the apex equation -M p-k=0 gives
v=-(k+M ptrial)/(H/Mpsi+M K). Zero dilation cannot reach the apex
and is explicitly refused (unless a tensile cutoff admits another branch).

For the optional mean tension cutoff ft=-p-sigma_t, associated flow is
I/3. Its multiplier t gives p+=K t. Tension alone has t=fttrial/K.
At the cone/tension corner p=-sigma_t and
shear gamma=(qtrial-k+M sigma_t)/(3G+H),
t=fttrial/K-Mpsi gamma. Each branch is admitted only if its multipliers
are nonnegative and both yield inequalities hold to a relative tolerance.
Plastic metric reconstruction follows F=Fe Fp, not additive reference strain.

Corrected tau_i=(q/qtrial) s_trial_i-p determines elastic strain
epsilon_i=(tau_i-mean(tau))/(2G)+mean(tau)/(3K) and B_e,i=exp(2 epsilon_i).
The recoverable Hencky energy is tau:epsilon/2. Plastic stress work is
tau:delta epsilon_p=q gamma-p v (v=Mpsi gamma+t); at the apex q=0.
The admissibility tolerance is max(1e-10,64 machine epsilon) times a stress
scale, so converting all stress/modulus units does not change branch admission.
"""

from __future__ import annotations

from math import pi
from typing import final

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
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
from ...typing import checked
from ._principal_hencky import (
    elastic_principal_metric,
    evaluate_principal_point,
    initialize_metric_state,
    initialize_state_from_stress,
    PrestressedMaterialState,
    principal_trial,
)


@final
class DruckerPragerParameters(StrictModule, NonTrainableState):
    shear_modulus: Array
    bulk_modulus: Array
    cohesion: Array
    friction_angle: Array
    dilation_angle: Array
    hardening_modulus: Array
    tensile_strength: Array | None

    def __init__(
        self,
        shear_modulus: ArrayLike,
        bulk_modulus: ArrayLike,
        cohesion: ArrayLike,
        friction_angle: ArrayLike,
        dilation_angle: ArrayLike,
        hardening_modulus: ArrayLike = 0.0,
        /,
        *,
        tensile_strength: ArrayLike | None = None,
    ) -> None:
        values = tuple(
            jnp.asarray(v)
            for v in (
                shear_modulus,
                bulk_modulus,
                cohesion,
                friction_angle,
                dilation_angle,
                hardening_modulus,
            )
        )
        if any(v.shape != () or not bool(jnp.isfinite(v)) for v in values):
            raise ValueError("Drucker-Prager parameters must be finite scalars.")
        g, k, c, phi, psi, h = values
        if (
            g <= 0
            or k <= 0
            or c < 0
            or not 0 <= phi < pi / 2
            or not -pi / 2 < psi < pi / 2
            or h < 0
        ):
            raise ValueError("Drucker-Prager parameters are inadmissible.")
        tension = None if tensile_strength is None else jnp.asarray(tensile_strength)
        if tension is not None:
            m, strength, _ = _coefficients_values(c, phi, psi)
            if (
                tension.shape != ()
                or not bool(jnp.isfinite(tension))
                or tension < 0
                or (m > 0 and m * tension > strength)
            ):
                raise ValueError("Tensile strength must lie between zero and k/M.")
        self.shear_modulus, self.bulk_modulus, self.cohesion = g, k, c
        self.friction_angle, self.dilation_angle, self.hardening_modulus = phi, psi, h
        self.tensile_strength = tension


def _coefficients_values(c: Array, phi: Array, psi: Array) -> tuple[Array, Array, Array]:
    return (
        6 * jnp.sin(phi) / (3 - jnp.sin(phi)),
        6 * c * jnp.cos(phi) / (3 - jnp.sin(phi)),
        6 * jnp.sin(psi) / (3 - jnp.sin(psi)),
    )


def _dp_yield(
    stress: Array, parameters: DruckerPragerParameters, scalars: Array
) -> Array:
    p = -jnp.trace(stress) / 3
    s = stress + p * jnp.eye(3, dtype=stress.dtype)
    q = jnp.sqrt(1.5 * jnp.sum(s * s))
    m, c, _ = _coefficients_values(
        parameters.cohesion, parameters.friction_angle, parameters.dilation_angle
    )
    f = q - m * p - c - parameters.hardening_modulus * scalars[0]
    return (
        f
        if parameters.tensile_strength is None
        else jnp.maximum(f, -p - parameters.tensile_strength)
    )


def _dp_principal(
    eigenvalues: Array, data: tuple[DruckerPragerParameters, Array]
) -> tuple[Array, dict[str, Array]]:
    parameters, scalars = data
    g, bulk, h = (
        parameters.shear_modulus,
        parameters.bulk_modulus,
        parameters.hardening_modulus,
    )
    _, _, p, q, s = principal_trial(eigenvalues, g, bulk)
    m, cohesion, dilation = _coefficients_values(
        parameters.cohesion, parameters.friction_angle, parameters.dilation_angle
    )
    strength = cohesion + h * scalars[0]
    f = q - m * p - strength
    scale = jnp.maximum(jnp.maximum(jnp.abs(p), q), jnp.maximum(strength, g))
    tol = jnp.maximum(1e-10, 64 * jnp.finfo(eigenvalues.dtype).eps) * scale
    tension = parameters.tensile_strength
    ft = jnp.asarray(0.0, dtype=p.dtype) if tension is None else -p - tension

    def elastic(_: None) -> tuple[Array, Array, Array, Array, Array]:
        return p, q, jnp.zeros_like(p), jnp.zeros_like(p), jnp.asarray(0, dtype=jnp.int32)

    def plastic(_: None) -> tuple[Array, Array, Array, Array, Array]:
        gamma = f / (3 * g + bulk * m * dilation + h)
        cone_p, cone_q = p + bulk * dilation * gamma, q - 3 * g * gamma
        cone_ok = (gamma >= 0) & (cone_q >= 0)
        if tension is not None:
            cone_ok &= -cone_p - tension <= tol
            t = ft / bulk
            tension_ok = (t >= 0) & (q + m * tension - strength <= tol)
            corner_gamma = (q - strength + m * tension) / (3 * g + h)
            corner_t = ft / bulk - dilation * corner_gamma
            corner_q = q - 3 * g * corner_gamma
            corner_ok = (corner_gamma >= 0) & (corner_t >= 0) & (corner_q >= 0)
            return jax.lax.cond(
                cone_ok,
                lambda _: (
                    cone_p,
                    cone_q,
                    gamma,
                    dilation * gamma,
                    jnp.asarray(1, dtype=jnp.int32),
                ),
                lambda _: jax.lax.cond(
                    tension_ok,
                    lambda _: (
                        -tension,
                        q,
                        jnp.zeros_like(q),
                        t,
                        jnp.asarray(4, dtype=jnp.int32),
                    ),
                    lambda _: (
                        -tension,
                        corner_q,
                        corner_gamma,
                        dilation * corner_gamma + corner_t,
                        jnp.where(corner_ok, 5, -1).astype(jnp.int32),
                    ),
                    None,
                ),
                None,
            )

        def apex(_: None) -> tuple[Array, Array, Array, Array, Array]:
            safe_dilation = jnp.where(dilation > 0, dilation, 1.0)
            denominator = h / safe_dilation + m * bulk
            v = -(strength + m * p) / jnp.where(denominator > 0, denominator, 1.0)
            valid = (dilation > 0) & (denominator > 0) & (v >= 0)
            return (
                p + bulk * v,
                jnp.zeros_like(q),
                v / safe_dilation,
                v,
                jnp.where(valid, 2, -1).astype(jnp.int32),
            )

        return jax.lax.cond(
            cone_ok,
            lambda _: (
                cone_p,
                cone_q,
                gamma,
                dilation * gamma,
                jnp.asarray(1, dtype=jnp.int32),
            ),
            apex,
            None,
        )

    pn, qn, gamma, volume, branch = jax.lax.cond(
        (f <= 0) & (ft <= 0), elastic, plastic, None
    )
    ratio = jnp.where((branch == 0) | (branch == 4), 1.0, qn / jnp.where(q > 0, q, 1.0))
    tau = ratio * s - pn
    alpha = scalars[0] + gamma
    residual = qn - m * pn - cohesion - h * alpha
    successful = (branch >= 0) & (residual <= tol) & (scalars[0] >= 0.0)
    if tension is not None:
        successful &= -pn - tension <= tol
    metric = elastic_principal_metric(tau, g, bulk)
    elastic_strain = 0.5 * jnp.log(metric)
    energy = 0.5 * jnp.sum(tau * elastic_strain)
    # tau : delta epsilon_p; apex deviatoric work vanishes.
    dissipation = qn * gamma - pn * volume
    cone_gamma = f / (3 * g + bulk * m * dilation + h)
    cone_q = q - 3 * g * cone_gamma
    strict_branch = jnp.where(
        branch == 0,
        f < -tol,
        jnp.where(
            branch == 1,
            (f > tol) & (qn > tol),
            (branch == 2) & (cone_q < -tol) & (bulk * volume > tol),
        ),
    )
    if tension is not None:
        strict_branch = jnp.where(
            branch == 4,
            (ft > tol) & (q + m * tension - strength < -tol),
            jnp.where(
                branch == 5,
                (3 * g * gamma > tol)
                & (bulk * (volume - dilation * gamma) > tol)
                & (qn > tol),
                strict_branch & (-pn - tension < -tol),
            ),
        )
    return jnp.stack((tau, metric)), {
        "scalars": alpha[None],
        "branch_code": branch,
        "successful": successful,
        "strict_branch": strict_branch,
        "energy": energy,
        "dissipation": dissipation,
        "yield_residual": residual,
        "plastic_multiplier": gamma,
        "apex_or_corner": (branch == 2) | (branch == 5),
    }


@final
class DruckerPragerMPMConstitutivePlan(
    AbstractImplicitMPMConstitutivePlan, NonTrainableState
):
    dimension: int = eqx.field(static=True)
    kinematics: str = eqx.field(static=True)
    state_shape: tuple[int, ...] = eqx.field(static=True)
    capabilities: MPMConstitutiveCapabilities
    plan_id: str = eqx.field(static=True)

    def __init__(self) -> None:
        self.dimension, self.kinematics, self.state_shape = 3, "three_dimensional", (7,)
        self.capabilities = MPMConstitutiveCapabilities(
            stateful=True,
            has_free_energy=True,
            has_algorithmic_tangent=True,
            has_dissipation=True,
            supports_implicit=True,
        )
        self.plan_id = canonical_fingerprint(
            {
                "kind": "drucker-prager-mpm",
                "history": "plastic-metric",
                "flow": "non-associated",
            }
        )

    def initialize_state(
        self, batch_shape: tuple[int, ...], dtype: DTypeLike, /
    ) -> Array:
        return initialize_metric_state(batch_shape, dtype, scalars=(0.0,))

    def initialize_state_from_stress(
        self,
        kirchhoff_stress: ArrayLike,
        parameters: DruckerPragerParameters,
        /,
        *,
        equivalent_plastic_strain: ArrayLike = 0.0,
    ) -> PrestressedMaterialState:
        stress = jnp.asarray(kirchhoff_stress)
        if stress.shape[-2:] != (3, 3):
            raise ValueError("Kirchhoff stress must end in 3x3.")
        alpha = jnp.broadcast_to(
            jnp.asarray(equivalent_plastic_strain, dtype=stress.dtype), stress.shape[:-2]
        )
        result = initialize_state_from_stress(
            stress, parameters, alpha[..., None], _dp_yield
        )
        return PrestressedMaterialState(
            result.state, result.admissible & (alpha >= 0.0), result.yield_value
        )

    @checked
    def evaluate(
        self,
        deformation_gradient: ArrayLike,
        committed_state: ArrayLike,
        reference_density: ArrayLike,
        parameters: DruckerPragerParameters,
        time: ArrayLike,
        step_size: ArrayLike,
        /,
    ) -> MPMConstitutiveResponse:
        del time
        deformation = jnp.asarray(deformation_gradient)
        state = jnp.asarray(committed_state, dtype=deformation.dtype)
        if state.shape != deformation.shape[:-2] + (7,):
            raise ValueError("Drucker-Prager history shape changed.")
        density = jnp.broadcast_to(
            jnp.asarray(reference_density, dtype=deformation.dtype),
            deformation.shape[:-2],
        )
        shape = deformation.shape[:-2]
        outputs = jax.vmap(
            lambda f, history, rho: evaluate_principal_point(
                f,
                history,
                rho,
                parameters,
                _dp_principal,
                function_id="drucker-prager",
                step_size=jnp.asarray(step_size, dtype=deformation.dtype),
                wave_modulus=parameters.bulk_modulus + 4 * parameters.shear_modulus / 3,
            )
        )(deformation.reshape((-1, 3, 3)), state.reshape((-1, 7)), density.reshape((-1,)))
        stress, history, energy, speed, dissipation, branch, valid, diagnostics = outputs
        diagnostics["tangent_successful"] &= diagnostics["strict_branch"]
        return MPMConstitutiveResponse(
            stress.reshape(shape + (3, 3)),
            history.reshape(shape + (7,)),
            energy.reshape(shape),
            speed.reshape(shape),
            dissipation_increment=dissipation.reshape(shape),
            branch_code=branch.reshape(shape),
            suggested_step=jnp.where(
                valid, jnp.inf, 0.5 * jnp.asarray(step_size)
            ).reshape(shape),
            successful=valid.reshape(shape),
            admissible=valid.reshape(shape),
            diagnostics={
                name: value.reshape(shape) for name, value in diagnostics.items()
            },
        )

    def evaluate_linearized(
        self,
        deformation_gradient: ArrayLike,
        committed_state: ArrayLike,
        reference_density: ArrayLike,
        parameters: DruckerPragerParameters,
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
        state = jnp.asarray(committed_state, dtype=deformation.dtype)
        density = jnp.broadcast_to(
            jnp.asarray(reference_density, dtype=deformation.dtype), shape
        )

        def point_stress(f: Array, history: Array, rho: Array) -> Array:
            return evaluate_principal_point(
                f,
                history,
                rho,
                parameters,
                _dp_principal,
                function_id="drucker-prager",
                step_size=jnp.asarray(step_size, dtype=deformation.dtype),
                wave_modulus=parameters.bulk_modulus + 4 * parameters.shear_modulus / 3,
            )[0]

        tangent = jax.vmap(jax.jacfwd(point_stress))(
            deformation.reshape((-1, 3, 3)),
            state.reshape((-1, 7)),
            density.reshape((-1,)),
        ).reshape(shape + (3, 3, 3, 3))
        return MPMLinearizedConstitutiveResponse(
            response,
            tangent,
            response.diagnostics["tangent_successful"]
            & jnp.all(jnp.isfinite(tangent), axis=(-4, -3, -2, -1)),
        )


@final
class NonlocalSofteningPlan(StrictModule, NonTrainableState):
    viscosity: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(self, *, viscosity: float = 0.0) -> None:
        value = float(viscosity)
        if not np.isfinite(value) or value < 0:
            raise ValueError(
                "Nonlocal softening viscosity must be finite and nonnegative."
            )
        self.viscosity = value
        self.plan_id = canonical_fingerprint(
            {"kind": "nonlocal-softening-plan", "viscosity": value}
        )

    def regularize(
        self,
        local_history: ArrayLike,
        nonlocal_history: ArrayLike,
        step_size: ArrayLike,
        /,
    ) -> Array:
        local = jnp.asarray(local_history)
        nonlocal_ = jnp.asarray(nonlocal_history, dtype=local.dtype)
        dt = jnp.asarray(step_size, dtype=local.dtype)
        return (
            nonlocal_
            if self.viscosity == 0
            else local + dt / (self.viscosity + dt) * (nonlocal_ - local)
        )
