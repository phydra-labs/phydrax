# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Multiplicative Hencky Mohr–Coulomb with ordered closed-form returns.

Algorithmic structure: de Souza Neto, Perić & Owen, Computational Methods for
Plasticity (Wiley, 2008), §8.2. The following formulas follow directly from
consistency and isotropic elasticity; no generic complementarity root is used.

For tension-positive ordered trial stresses t1 >= t2 >= t3, define the six
planes nij.t - k <= 0, nij=(1+sin(phi))ei-(1-sin(phi))ej, i != j,
k=2(c+H alpha)cos(phi). Nonassociated flow uses bij with phi replaced by psi.
C=2G I+(K-2G/3)11^T and delta_epsilon_p=sum(gamma_ij bij).
The hardening convention is alpha_new=alpha+sum(gamma_ij); H is the slope of
cohesion, not of k. Thus consistency on active planes is
(n_i C b_j + 2 H cos(phi)) gamma_j = n_i.t_trial-k.

The main plane is (1,3). Its denominator is
D=4G(1+sin(phi)sin(psi))+4(K-2G/3)sin(phi)sin(psi)+2H cos(phi).
If the corrected ordering is violated, the left edge adds (2,3), and the
right edge adds (1,2). Their diagonal entries are D; off-diagonal O is
2G(1-sin(phi))(1-sin(psi))+4(K-2G/3)sin(phi)sin(psi)+2H cos(phi)
on the left and the same expression with both minus signs changed to plus
on the right. The closed-form sum/difference solution is
s=(f1+f2)/(D+O), d=(f1-f2)/(D-O), gamma1=(s+d)/2, gamma2=(s-d)/2.
Ordering, nonnegative multipliers and all six inequalities admit a branch.

At the apex t_i=t, gamma=sum(gamma_ij), tr(delta_epsilon_p)=2sin(psi)gamma.
Therefore t=mean(t_trial)-2K sin(psi)gamma and
 gamma=(sin(phi)mean(t_trial)-(c+H alpha)cos(phi)) /
       (2K sin(phi)sin(psi)+H cos(phi)).
The flow cone is the convex hull of the six permutations of
(1+sin(psi),0,-1+sin(psi)), scaled by gamma. Its exact membership conditions
are max(delta_epsilon_p)<=(1+sin(psi))gamma and
min(delta_epsilon_p)>=-(1-sin(psi))gamma, plus the trace identity.
A singular apex equation is admissible only when its numerator vanishes;
then the minimum gamma satisfying these bounds fixes the hardening convention.
Otherwise the return fails explicitly, without floors or stress projection.

Sort/unsort guarantees principal-output permutation equivariance. Diagnostics
are ordered-principal, hence permutation-invariant. History is Cp_inverse
(11,22,33,23,13,12), then alpha: seven entries, identity metric at rest.
Corrected elastic Hencky strains reconstruct Be=exp(2 epsilon_e); the shared
kernel pulls it back multiplicatively. Energy is recoverable elastic energy;
dissipation_increment is plastic stress work tau:delta_epsilon_p (hardening
is phenomenological, with no additional stored-energy claim). Branch codes:
0 elastic, 1 face, 2 edge (edge_side -1 left, +1 right), 3 apex, -1 failure.
Tangent validity excludes branch boundaries; strict repeated-spectrum branches
are differentiated through the native isotropic spectral substrate.
"""

from __future__ import annotations

from math import pi
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
from ...linalg import hermitian_spectral_map
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
class MohrCoulombParameters(StrictModule, NonTrainableState):
    """Constant Hencky moduli, angles in radians, and linear cohesion hardening."""

    shear_modulus: Array
    bulk_modulus: Array
    cohesion: Array
    friction_angle: Array
    dilation_angle: Array
    hardening_modulus: Array

    def __init__(
        self,
        shear_modulus: ArrayLike,
        bulk_modulus: ArrayLike,
        cohesion: ArrayLike,
        friction_angle: ArrayLike,
        dilation_angle: ArrayLike,
        hardening_modulus: ArrayLike = 0.0,
        /,
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
            raise ValueError("Mohr-Coulomb parameters must be finite scalars.")
        g, k, c, phi, psi, h = values
        if (
            g <= 0
            or k <= 0
            or c < 0
            or not 0 <= phi < pi / 2
            or not -pi / 2 < psi < pi / 2
            or h < 0
        ):
            raise ValueError("Mohr-Coulomb parameters are inadmissible.")
        (
            self.shear_modulus,
            self.bulk_modulus,
            self.cohesion,
            self.friction_angle,
            self.dilation_angle,
            self.hardening_modulus,
        ) = values


def _plane_normals(angle: Array) -> Array:
    # Sorted-principal plane order: 13, 23, 12, 31, 32, 21.
    identity = jnp.eye(3, dtype=angle.dtype)
    high = identity[jnp.asarray((0, 1, 0, 2, 2, 1), dtype=jnp.int32)]
    low = identity[jnp.asarray((2, 2, 1, 0, 1, 0), dtype=jnp.int32)]
    return (1 + jnp.sin(angle)) * high - (1 - jnp.sin(angle)) * low


def _mc_stress_yield(
    eigenvalues: Array, data: tuple[MohrCoulombParameters, Array]
) -> tuple[Array, dict[str, Array]]:
    parameters, scalars = data
    strength = (
        2
        * (parameters.cohesion + parameters.hardening_modulus * scalars[0])
        * jnp.cos(parameters.friction_angle)
    )
    values = (
        _plane_normals(parameters.friction_angle) @ jnp.sort(eigenvalues)[::-1] - strength
    )
    return eigenvalues[None, :], {"yield": jnp.max(values)}


def _mc_yield(stress: Array, parameters: MohrCoulombParameters, scalars: Array) -> Array:
    return hermitian_spectral_map(
        stress,
        _mc_stress_yield,
        (parameters, scalars),
        function_id="mohr-coulomb-prestress",
    ).invariants["yield"]


def _ordered_return(
    trial: Array, alpha: Array, parameters: MohrCoulombParameters, tolerance: Array
) -> tuple[Array, Array, Array, Array, Array]:
    """Return sorted stress, total multiplier, branch, edge side, strictness."""
    g, bulk, h = (
        parameters.shear_modulus,
        parameters.bulk_modulus,
        parameters.hardening_modulus,
    )
    phi, psi = parameters.friction_angle, parameters.dilation_angle
    n, b = _plane_normals(phi), _plane_normals(psi)
    elasticity = 2 * g * jnp.eye(3, dtype=trial.dtype) + (bulk - 2 * g / 3) * jnp.ones(
        (3, 3), dtype=trial.dtype
    )
    strength = 2 * (parameters.cohesion + h * alpha) * jnp.cos(phi)
    f = n @ trial - strength
    d = n[0] @ elasticity @ b[0] + 2 * h * jnp.cos(phi)
    zero = jnp.zeros_like(alpha)

    def elastic(_: None) -> tuple[Array, Array, Array, Array, Array]:
        return (
            trial,
            zero,
            jnp.asarray(0, jnp.int32),
            jnp.asarray(0, jnp.int32),
            f[0] < -tolerance,
        )

    def plastic(_: None) -> tuple[Array, Array, Array, Array, Array]:
        gamma = f[0] / jnp.where(d != 0, d, 1)
        face = trial - elasticity @ (gamma * b[0])
        gaps = face[:-1] - face[1:]
        face_ok = (d > 0) & (gamma >= 0) & jnp.all(gaps >= 0)

        def edge_or_apex(_: None) -> tuple[Array, Array, Array, Array, Array]:
            def edge(index: int) -> tuple[Array, Array, Array, Array]:
                o = n[0] @ elasticity @ b[index] + 2 * h * jnp.cos(phi)
                summed = (f[0] + f[index]) / jnp.where(d + o != 0, d + o, 1)
                difference = (f[0] - f[index]) / jnp.where(d - o != 0, d - o, 1)
                multipliers = jnp.stack(
                    ((summed + difference) / 2, (summed - difference) / 2)
                )
                returned = trial - elasticity @ (
                    multipliers[0] * b[0] + multipliers[1] * b[index]
                )
                residual = n @ returned - strength - 2 * h * jnp.cos(phi) * summed
                ok = (
                    (d + o > 0)
                    & (d - o > 0)
                    & jnp.all(multipliers >= 0)
                    & jnp.all(returned[:-1] - returned[1:] >= -tolerance)
                    & jnp.all(residual <= tolerance)
                )
                gap = (
                    returned[1] - returned[2] if index == 1 else returned[0] - returned[1]
                )
                strict = (gap > tolerance) & jnp.all(multipliers > tolerance / g)
                return returned, summed, ok, strict

            left, left_gamma, left_ok, left_strict = edge(1)
            right, right_gamma, right_ok, right_strict = edge(2)

            def apex(_: None) -> tuple[Array, Array, Array, Array, Array]:
                mean = jnp.mean(trial)
                numerator = jnp.sin(phi) * mean - (
                    parameters.cohesion + h * alpha
                ) * jnp.cos(phi)
                denominator = 2 * bulk * jnp.sin(phi) * jnp.sin(psi) + h * jnp.cos(phi)
                increment = (trial - mean) / (2 * g)
                minimum_gamma = jnp.maximum(
                    increment[0] / (1 + jnp.sin(psi)), -increment[2] / (1 - jnp.sin(psi))
                )
                total = jnp.where(
                    denominator != 0,
                    numerator / jnp.where(denominator != 0, denominator, 1),
                    minimum_gamma,
                )
                hydro = mean - 2 * bulk * jnp.sin(psi) * total
                increment = (trial - mean) / (2 * g) + (mean - hydro) / (3 * bulk)
                upper = (1 + jnp.sin(psi)) * total - increment[0]
                lower = increment[2] + (1 - jnp.sin(psi)) * total
                ok = (
                    (total >= 0)
                    & (upper >= -tolerance / g)
                    & (lower >= -tolerance / g)
                    & (
                        (denominator > 0)
                        | ((denominator == 0) & (jnp.abs(numerator) <= tolerance))
                    )
                )
                strict = (
                    (upper > tolerance / g) & (lower > tolerance / g) & (denominator > 0)
                )
                return (
                    jnp.full_like(trial, hydro),
                    total,
                    jnp.where(ok, 3, -1).astype(jnp.int32),
                    jnp.asarray(0, jnp.int32),
                    strict,
                )

            return jax.lax.cond(
                left_ok,
                lambda _: (
                    left,
                    left_gamma,
                    jnp.asarray(2, jnp.int32),
                    jnp.asarray(-1, jnp.int32),
                    left_strict,
                ),
                lambda _: jax.lax.cond(
                    right_ok,
                    lambda _: (
                        right,
                        right_gamma,
                        jnp.asarray(2, jnp.int32),
                        jnp.asarray(1, jnp.int32),
                        right_strict,
                    ),
                    apex,
                    None,
                ),
                None,
            )

        return jax.lax.cond(
            face_ok,
            lambda _: (
                face,
                gamma,
                jnp.asarray(1, jnp.int32),
                jnp.asarray(0, jnp.int32),
                jnp.all(gaps > tolerance) & (gamma > tolerance / g),
            ),
            edge_or_apex,
            None,
        )

    return jax.lax.cond(f[0] <= 0, elastic, plastic, None)


def _mc_principal(
    eigenvalues: Array, data: tuple[MohrCoulombParameters, Array]
) -> tuple[Array, dict[str, Array]]:
    parameters, scalars = data
    g, bulk = parameters.shear_modulus, parameters.bulk_modulus
    _, trial, _, _, _ = principal_trial(eigenvalues, g, bulk)
    order = jnp.argsort(trial)[::-1]
    ordered = trial[order]
    scale = jnp.maximum(
        jnp.maximum(g, bulk),
        jnp.maximum(
            jnp.max(jnp.abs(trial)),
            parameters.cohesion + parameters.hardening_modulus * scalars[0],
        ),
    )
    tolerance = jnp.maximum(1e-10, 64 * jnp.finfo(eigenvalues.dtype).eps) * scale
    returned, gamma, branch, side, strict = _ordered_return(
        ordered, scalars[0], parameters, tolerance
    )
    alpha = scalars[0] + gamma
    residuals = _plane_normals(parameters.friction_angle) @ returned - 2 * (
        parameters.cohesion + parameters.hardening_modulus * alpha
    ) * jnp.cos(parameters.friction_angle)
    tau = returned[jnp.argsort(order)]
    metric = elastic_principal_metric(tau, g, bulk)
    mean = jnp.mean(returned)
    increment = (ordered - returned - jnp.mean(ordered - returned)) / (2 * g) + jnp.mean(
        ordered - returned
    ) / (3 * bulk)
    return jnp.stack((tau, metric)), {
        "scalars": alpha[None],
        "successful": (branch >= 0) & (scalars[0] >= 0) & jnp.all(residuals <= tolerance),
        "branch_code": branch,
        "edge_side": side,
        "strict_branch": strict,
        "energy": jnp.sum((returned - mean) ** 2) / (4 * g) + mean**2 / (2 * bulk),
        "dissipation": returned @ increment,
        "yield_residual": jnp.max(residuals),
        "plastic_multiplier": gamma,
        "ordered_trial_stress": ordered,
        "ordered_returned_stress": returned,
        "ordered_plastic_increment": increment,
        "plane_yield_values": residuals,
    }


@final
class MohrCoulombMPMConstitutivePlan(
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
                "kind": "mohr-coulomb-mpm",
                "history": "plastic-metric",
                "return": "ordered-face-edge-apex",
                "hardening": "cohesion-sum-multipliers",
            }
        )

    def initialize_state(
        self, batch_shape: tuple[int, ...], dtype: DTypeLike, /
    ) -> Array:
        return initialize_metric_state(batch_shape, dtype)

    def initialize_state_from_stress(
        self,
        kirchhoff_stress: ArrayLike,
        parameters: MohrCoulombParameters,
        /,
        *,
        equivalent_plastic_strain: ArrayLike = 0.0,
    ) -> PrestressedMaterialState:
        stress = jnp.asarray(kirchhoff_stress)
        alpha = jnp.broadcast_to(
            jnp.asarray(equivalent_plastic_strain, dtype=stress.dtype), stress.shape[:-2]
        )
        result = initialize_state_from_stress(
            stress, parameters, alpha[..., None], _mc_yield
        )
        return PrestressedMaterialState(
            result.state, result.admissible & (alpha >= 0), result.yield_value
        )

    def _point(
        self,
        deformation: Array,
        history: Array,
        density: Array,
        parameters: MohrCoulombParameters,
        step_size: Array,
    ) -> tuple[Array, Array, Array, Array, Array, Array, Array, dict[str, Array]]:
        return evaluate_principal_point(
            deformation,
            history,
            density,
            parameters,
            _mc_principal,
            function_id="mohr-coulomb-hencky",
            step_size=step_size,
            wave_modulus=parameters.bulk_modulus + 4 * parameters.shear_modulus / 3,
        )

    @checked
    def evaluate(
        self,
        deformation_gradient: ArrayLike,
        committed_state: ArrayLike,
        reference_density: ArrayLike,
        parameters: MohrCoulombParameters,
        time: ArrayLike,
        step_size: ArrayLike,
        /,
    ) -> MPMConstitutiveResponse:
        del time
        deformation = jnp.asarray(deformation_gradient)
        if deformation.shape[-2:] != (3, 3):
            raise ValueError("Mohr-Coulomb deformation must end in 3x3.")
        shape = deformation.shape[:-2]
        state = jnp.asarray(committed_state, dtype=deformation.dtype)
        if state.shape != shape + (7,):
            raise ValueError("Mohr-Coulomb state must have seven entries per point.")
        density = jnp.broadcast_to(
            jnp.asarray(reference_density, dtype=deformation.dtype), shape
        )
        outputs = jax.vmap(
            lambda f, history, rho: self._point(
                f,
                history,
                rho,
                parameters,
                jnp.asarray(step_size, dtype=deformation.dtype),
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
            suggested_step=diagnostics["suggested_step"].reshape(shape),
            successful=valid.reshape(shape),
            admissible=valid.reshape(shape),
            diagnostics={
                name: value.reshape(shape + value.shape[1:])
                for name, value in diagnostics.items()
            },
        )

    def evaluate_linearized(
        self,
        deformation_gradient: ArrayLike,
        committed_state: ArrayLike,
        reference_density: ArrayLike,
        parameters: MohrCoulombParameters,
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
            return self._point(
                f, history, rho, parameters, jnp.asarray(step_size, dtype=f.dtype)
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
