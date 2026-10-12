# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Shared multiplicative isotropic Hencky kinematics, not a return solver.

History is the symmetric plastic metric C_p^-1 packed (11,22,33,23,13,12),
followed by material scalars. B_e,tr = F C_p^-1 F^T. A permutation-equivariant
principal map returns Kirchhoff stress and corrected B_e; two native solves
pull B_e back to the reference frame. Solving F P^T = tau gives P=tau F^-T.
Invalid kinematics never commit the candidate metric. First-order derivatives
are owned by the spectral substrate; no eigenvector derivatives are taken.
The additive supplied-elastic-metric entry instead stores spatial B_e and
previous F; its caller owns the relative-deformation solve.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import final, Protocol

import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike, DTypeLike

from ..._strict import StrictModule
from ...linalg import (
    hermitian_exp,
    hermitian_spectral_map,
    SmallLinearSolvePlan,
    SmallLinearSolveResult,
    solve_small_linear,
)


class HenckyParameters(Protocol):
    shear_modulus: Array
    bulk_modulus: Array


@final
class PrestressedMaterialState(StrictModule):
    state: Array
    admissible: Array
    yield_value: Array


def pack_metric(matrix: Array) -> Array:
    return jnp.stack(
        (
            matrix[..., 0, 0],
            matrix[..., 1, 1],
            matrix[..., 2, 2],
            matrix[..., 1, 2],
            matrix[..., 0, 2],
            matrix[..., 0, 1],
        ),
        axis=-1,
    )


def unpack_metric(packed: Array) -> Array:
    a, b, c, d, e, f = (packed[..., i] for i in range(6))
    return jnp.stack(
        (
            jnp.stack((a, f, e), axis=-1),
            jnp.stack((f, b, d), axis=-1),
            jnp.stack((e, d, c), axis=-1),
        ),
        axis=-2,
    )


def initialize_metric_state(
    batch_shape: tuple[int, ...], dtype: DTypeLike, scalars: tuple[float, ...] = (0.0,)
) -> Array:
    initial = jnp.asarray((1.0, 1.0, 1.0, 0.0, 0.0, 0.0, *scalars), dtype=dtype)
    return jnp.broadcast_to(initial, batch_shape + initial.shape)


def principal_trial(
    eigenvalues: Array, shear: Array, bulk: Array
) -> tuple[Array, Array, Array, Array, Array]:
    strain = 0.5 * jnp.log(eigenvalues)
    volume = jnp.sum(strain)
    s = 2.0 * shear * (strain - volume / 3.0)
    p = -bulk * volume
    squared_q = 1.5 * jnp.sum(s * s)
    positive = squared_q > 0.0
    # vmap turns cond into select; keep the inactive sqrt derivative finite.
    q = jnp.where(positive, jnp.sqrt(jnp.where(positive, squared_q, 1.0)), 0.0)
    return strain, s - p, p, q, s


def elastic_principal_metric(tau: Array, shear: Array, bulk: Array) -> Array:
    mean = jnp.mean(tau)
    strain = (tau - mean) / (2.0 * shear) + mean / (3.0 * bulk)
    return jnp.exp(2.0 * strain)


def _positive_metric(metric: Array) -> Array:
    # Sylvester's criterion avoids a second eigendecomposition of the history.
    a, b, c = metric[0, 0], metric[1, 1], metric[2, 2]
    d, e, f = metric[1, 2], metric[0, 2], metric[0, 1]
    determinant = a * b * c + 2.0 * d * e * f - a * d * d - b * e * e - c * f * f
    return (
        jnp.all(jnp.isfinite(metric))
        & (a > 0.0)
        & (a * b - f * f > 0.0)
        & (determinant > 0.0)
    )


def evaluate_principal_point[P](
    deformation: Array,
    history: Array,
    density: Array,
    parameters: P,
    eigenvalue_map: Callable[[Array, tuple[P, Array]], tuple[Array, dict[str, Array]]],
    *,
    function_id: str,
    step_size: Array,
    wave_modulus: Array,
) -> tuple[Array, Array, Array, Array, Array, Array, Array, dict[str, Array]]:
    """Return P, history, energy, speed, dissipation, branch, valid, diagnostics.

    The static principal callback receives (eigenvalues, (parameters,
    history[6:])); it must be permutation-equivariant and return shape (2,3)
    principal outputs ordered (tau, corrected B_e). Its invariant dictionary
    contains scalars (a vector), successful, branch_code, energy, dissipation,
    yield_residual, and plastic_multiplier. Additional invariant arrays are
    preserved as diagnostics. wave_modulus is the caller's longitudinal
    elastic stiffness, optionally overridden by the map's scalar wave_modulus
    invariant for nonlinear elasticity; the kernel imposes no material law.
    """
    metric = unpack_metric(history[:6])
    plan = SmallLinearSolvePlan(3)
    identity = jnp.eye(3, dtype=deformation.dtype)
    kinematic_solve = solve_small_linear(plan, deformation, jnp.zeros_like(deformation))
    admitted = (
        kinematic_solve.successful
        & (kinematic_solve.determinant > 0.0)
        & _positive_metric(metric)
        & jnp.all(jnp.isfinite(history))
        & (density > 0.0)
        & jnp.isfinite(density)
    )
    trial = deformation @ metric @ deformation.T
    # A rejected input is evaluated on a neutral tensor only to keep failed
    # responses finite; admission remains false and history is never repaired.
    mapped = hermitian_spectral_map(
        jnp.where(admitted, trial, identity),
        eigenvalue_map,
        (parameters, history[6:]),
        function_id=function_id,
    )
    tau, corrected = mapped.values[0], mapped.values[1]
    stress_solve = solve_small_linear(plan, deformation, tau.T)
    left_solve = solve_small_linear(plan, deformation, corrected)
    right_solve = solve_small_linear(plan, deformation, left_solve.value.T)
    next_metric = right_solve.value.T
    invariants = mapped.invariants
    state = jnp.concatenate((pack_metric(next_metric), invariants["scalars"]))
    # For singular values a>=b>=c, ab <= ||F||_F²/2 and
    # ||F^-1||_2=ab/det(F). This conservative wave bound forms no inverse.
    determinant = jnp.where(admitted, kinematic_solve.determinant, 1.0)
    norm_bound = jnp.sum(deformation**2) / (2.0 * determinant)
    wave_modulus = invariants.get("wave_modulus", wave_modulus)
    speed = jnp.sqrt(wave_modulus / jnp.where(density > 0.0, density, 1.0)) * jnp.maximum(
        norm_bound, 1.0
    )
    valid = (
        admitted
        & mapped.valid
        & jnp.all(mapped.spectrum.eigenvalues > 0.0)
        & invariants["successful"]
        & stress_solve.successful
        & left_solve.successful
        & right_solve.successful
        & _positive_metric(next_metric)
        & jnp.all(jnp.isfinite(state))
        & (wave_modulus > 0.0)
        & jnp.isfinite(wave_modulus)
        & jnp.isfinite(speed)
    )
    diagnostics = {key: value for key, value in invariants.items() if key != "scalars"}
    diagnostics["tangent_successful"] = valid & mapped.derivative_valid
    diagnostics["suggested_step"] = jnp.where(
        valid, jnp.asarray(jnp.inf, dtype=deformation.dtype), 0.5 * step_size
    )
    diagnostics["deformation_determinant"] = kinematic_solve.determinant
    return (
        jnp.where(valid, stress_solve.value.T, 0.0),
        jnp.where(valid, state, history),
        jnp.where(valid, invariants["energy"], 0.0),
        jnp.where(valid, speed, 0.0),
        jnp.where(valid, invariants["dissipation"], 0.0),
        jnp.where(valid, invariants["branch_code"], -1).astype(jnp.int32),
        valid,
        diagnostics,
    )


def evaluate_elastic_metric_point[P](
    deformation: Array,
    trial_metric: Array,
    history: Array,
    density: Array,
    parameters: P,
    eigenvalue_map: Callable[[Array, tuple[P, Array]], tuple[Array, dict[str, Array]]],
    *,
    function_id: str,
    step_size: Array,
    wave_modulus: Array,
    solve_plan: SmallLinearSolvePlan,
    kinematics: SmallLinearSolveResult,
    history_successful: Array,
) -> tuple[Array, Array, Array, Array, Array, Array, Array, dict[str, Array]]:
    """Evaluate a supplied elastic trial metric without reference pullback.

    History is packed spatial B_e (six), row-major previous F (nine), then
    material scalars. The caller forms the relative-deformation trial and owns
    its solve evidence. The principal callback has the same contract as
    evaluate_principal_point; no existing plastic-metric material is changed.
    """
    admitted = (
        history_successful
        & kinematics.successful
        & (kinematics.determinant > 0.0)
        & _positive_metric(unpack_metric(history[:6]))
        & jnp.all(jnp.isfinite(history))
        & (density > 0.0)
        & jnp.isfinite(density)
    )
    mapped = hermitian_spectral_map(
        jnp.where(admitted, trial_metric, jnp.eye(3, dtype=deformation.dtype)),
        eigenvalue_map,
        (parameters, history[15:]),
        function_id=function_id,
    )
    tau, corrected = mapped.values[0], mapped.values[1]
    stress_solve = solve_small_linear(solve_plan, deformation, tau.T)
    invariants = mapped.invariants
    state = jnp.concatenate(
        (pack_metric(corrected), deformation.reshape((9,)), invariants["scalars"])
    )
    determinant = jnp.where(admitted, kinematics.determinant, 1.0)
    norm_bound = jnp.sum(deformation**2) / (2.0 * determinant)
    wave_modulus = invariants.get("wave_modulus", wave_modulus)
    speed = jnp.sqrt(wave_modulus / jnp.where(density > 0.0, density, 1.0)) * jnp.maximum(
        norm_bound, 1.0
    )
    valid = (
        admitted
        & mapped.valid
        & jnp.all(mapped.spectrum.eigenvalues > 0.0)
        & invariants["successful"]
        & stress_solve.successful
        & _positive_metric(corrected)
        & jnp.all(jnp.isfinite(state))
        & (wave_modulus > 0.0)
        & jnp.isfinite(wave_modulus)
        & jnp.isfinite(speed)
    )
    diagnostics = {key: value for key, value in invariants.items() if key != "scalars"}
    diagnostics["tangent_successful"] = valid & mapped.derivative_valid
    diagnostics["suggested_step"] = jnp.where(
        valid, jnp.asarray(jnp.inf, dtype=deformation.dtype), 0.5 * step_size
    )
    diagnostics["deformation_determinant"] = kinematics.determinant
    diagnostics["deformation_condition"] = kinematics.condition_estimate
    diagnostics["trial_elastic_metric_condition"] = mapped.spectrum.condition_number
    return (
        jnp.where(valid, stress_solve.value.T, 0.0),
        jnp.where(valid, state, history),
        jnp.where(valid, invariants["energy"], 0.0),
        jnp.where(valid, speed, 0.0),
        jnp.where(valid, invariants["dissipation"], 0.0),
        jnp.where(valid, invariants["branch_code"], -1).astype(jnp.int32),
        valid,
        diagnostics,
    )


def initialize_state_from_stress[P: HenckyParameters](
    stress: ArrayLike,
    parameters: P,
    scalars: Array,
    yield_function: Callable[[Array, P, Array], Array],
    *,
    tolerance: float = 1.0e-10,
) -> PrestressedMaterialState:
    """Invert tau=2G dev(epsilon)+K tr(epsilon) I at F=I.

    The initializer reports yield admissibility without projecting the stress.
    Its tolerance is relative to stress and elastic-modulus scales.
    """
    stress = jnp.asarray(stress)
    if stress.shape[-2:] != (3, 3):
        raise ValueError("Kirchhoff stress must end in 3x3.")
    identity = jnp.broadcast_to(jnp.eye(3, dtype=stress.dtype), stress.shape)
    mean = jnp.trace(stress, axis1=-2, axis2=-1) / 3.0
    strain = (stress - mean[..., None, None] * identity) / (
        2.0 * parameters.shear_modulus
    ) + mean[..., None, None] * identity / (3.0 * parameters.bulk_modulus)
    metric = hermitian_exp(2.0 * strain)
    scalar_batch = jnp.broadcast_to(scalars, stress.shape[:-2] + scalars.shape[-1:])
    state = jnp.concatenate((pack_metric(metric.value), scalar_batch), axis=-1)
    value = jax.vmap(lambda tau, state: yield_function(tau, parameters, state))(
        stress.reshape((-1, 3, 3)), scalar_batch.reshape((-1, scalar_batch.shape[-1]))
    ).reshape(stress.shape[:-2])
    scale = jnp.maximum(
        jnp.maximum(parameters.shear_modulus, parameters.bulk_modulus),
        jnp.max(jnp.abs(stress), axis=(-2, -1)),
    )
    symmetric = jnp.all(stress == jnp.swapaxes(stress, -1, -2), axis=(-2, -1))
    admissible = (
        metric.valid
        & symmetric
        & jnp.all(jnp.isfinite(state), axis=-1)
        & (value <= tolerance * scale)
    )
    return PrestressedMaterialState(state, admissible, value)
