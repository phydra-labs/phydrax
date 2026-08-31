#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ._dem_contact import (
    _effective_mass,
    _pair_parameter,
    _safe_vector_norm,
    AbstractDEMNormalContactPlan,
    AbstractDEMTangentialContactPlan,
    DEMNormalResponse,
    DEMTangentialResponse,
)


class SmoothPenaltyNormalPlan(AbstractDEMNormalContactPlan):
    stiffness: Array
    gap_smoothing: float = eqx.field(static=True)
    force_smoothing: float = eqx.field(static=True)
    cutoff_multiple: float = eqx.field(static=True)
    maximum_range: float = eqx.field(static=True)
    normal_law_id: str = eqx.field(static=True)

    def __init__(
        self,
        stiffness: ArrayLike,
        /,
        *,
        gap_smoothing: float,
        force_smoothing: float,
        cutoff_multiple: float = 12.0,
        normal_law_id: str | None = None,
    ):
        values = np.asarray(stiffness)
        gap = float(gap_smoothing)
        force = float(force_smoothing)
        cutoff = float(cutoff_multiple)
        if values.ndim not in (0, 2) or (
            values.ndim == 2
            and (
                values.shape[0] != values.shape[1] or not np.array_equal(values, values.T)
            )
        ):
            raise ValueError(
                "Smooth normal stiffness must be scalar or symmetric pair table."
            )
        if (
            np.any(~np.isfinite(values))
            or np.any(values <= 0.0)
            or not np.isfinite(gap)
            or gap <= 0.0
            or not np.isfinite(force)
            or force <= 0.0
            or not np.isfinite(cutoff)
            or cutoff <= 1.0
        ):
            raise ValueError("Smooth contact stiffness and smoothing scales are invalid.")
        generated = canonical_fingerprint(
            {
                "kind": "smooth-penalty-normal",
                "stiffness": array_tree_fingerprint(values),
                "gap_smoothing": gap,
                "force_smoothing": force,
                "cutoff_multiple": cutoff,
            }
        )
        self.stiffness = jnp.asarray(values)
        self.gap_smoothing = gap
        self.force_smoothing = force
        self.cutoff_multiple = cutoff
        self.maximum_range = gap * cutoff
        self.normal_law_id = generated if normal_law_id is None else str(normal_law_id)
        if not self.normal_law_id:
            raise ValueError("normal_law_id must be nonempty.")

    def evaluate(
        self,
        batch,
        previous_history,
        left_inverse_mass,
        right_inverse_mass,
        left_radius,
        right_radius,
        left_material,
        right_material,
        materials,
        step_size,
        /,
    ):
        del left_radius, right_radius, previous_history
        dtype = batch.gap.dtype
        stiffness = _pair_parameter(
            self.stiffness, left_material, right_material, materials.material_count
        ).astype(dtype)
        epsilon = jnp.asarray(self.gap_smoothing, dtype=dtype)
        delta = -batch.gap
        smooth_overlap = epsilon * jax.nn.softplus(delta / epsilon)
        derivative = jax.nn.sigmoid(delta / epsilon)
        elastic = stiffness * smooth_overlap * derivative
        tangent_stiffness = stiffness * (
            derivative**2 + smooth_overlap * derivative * (1.0 - derivative) / epsilon
        )
        effective_mass = _effective_mass(left_inverse_mass, right_inverse_mass)
        restitution = materials.pair_restitution(left_material, right_material).astype(
            dtype
        )
        beta = jnp.log(restitution) / jnp.sqrt(jnp.pi**2 + jnp.log(restitution) ** 2)
        damping = (
            -2.0 * beta * jnp.sqrt(effective_mass * jnp.maximum(tangent_stiffness, 0.0))
        )
        trial = elastic - damping * batch.normal_velocity
        epsilon_force = jnp.asarray(self.force_smoothing, dtype=dtype)
        force = epsilon_force * jax.nn.softplus(trial / epsilon_force)
        active = (
            batch.valid
            & (batch.gap < self.maximum_range)
            & (left_inverse_mass + right_inverse_mass > 0.0)
        )
        force = jnp.where(active, force, 0.0)
        energy = jnp.where(active, 0.5 * stiffness * smooth_overlap**2, 0.0)
        viscous = jnp.where(
            active,
            damping * batch.normal_velocity**2 * jnp.asarray(step_size),
            0.0,
        )
        finite = (
            jnp.all(jnp.isfinite(force))
            & jnp.all(jnp.isfinite(energy))
            & jnp.all(jnp.isfinite(viscous))
        )
        return DEMNormalResponse(
            force,
            force,
            tangent_stiffness,
            damping,
            jnp.full_like(force, jnp.inf),
            energy,
            jnp.maximum(viscous, 0.0),
            jnp.zeros_like(force),
            jnp.zeros_like(force),
            jnp.zeros_like(force),
            jnp.where(active, batch.overlap, 0.0),
            active,
            finite,
        )


class SmoothCoulombTangentialPlan(AbstractDEMTangentialContactPlan):
    stiffness: Array
    direction_smoothing: float = eqx.field(static=True)
    projection_order: int = eqx.field(static=True)
    tangential_law_id: str = eqx.field(static=True)

    def __init__(
        self,
        stiffness: ArrayLike,
        /,
        *,
        direction_smoothing: float,
        projection_order: int = 4,
        tangential_law_id: str | None = None,
    ):
        values = np.asarray(stiffness)
        epsilon = float(direction_smoothing)
        order = int(projection_order)
        if values.ndim not in (0, 2) or (
            values.ndim == 2
            and (
                values.shape[0] != values.shape[1] or not np.array_equal(values, values.T)
            )
        ):
            raise ValueError("Smooth tangential stiffness schema is invalid.")
        if (
            np.any(~np.isfinite(values))
            or np.any(values <= 0.0)
            or not np.isfinite(epsilon)
            or epsilon <= 0.0
            or order < 2
        ):
            raise ValueError("Smooth tangential parameters are invalid.")
        generated = canonical_fingerprint(
            {
                "kind": "smooth-coulomb-tangential",
                "stiffness": array_tree_fingerprint(values),
                "direction_smoothing": epsilon,
                "projection_order": order,
            }
        )
        self.stiffness = jnp.asarray(values)
        self.direction_smoothing = epsilon
        self.projection_order = order
        self.tangential_law_id = (
            generated if tangential_law_id is None else str(tangential_law_id)
        )
        if not self.tangential_law_id:
            raise ValueError("tangential_law_id must be nonempty.")

    def evaluate(
        self,
        batch,
        normal,
        transported_displacement,
        left_inverse_mass,
        right_inverse_mass,
        left_radius,
        right_radius,
        left_material,
        right_material,
        materials,
        step_size,
        /,
    ):
        del left_radius, right_radius
        stiffness = _pair_parameter(
            self.stiffness, left_material, right_material, materials.material_count
        ).astype(batch.overlap.dtype)
        effective_mass = _effective_mass(left_inverse_mass, right_inverse_mass)
        restitution = materials.pair_restitution(left_material, right_material).astype(
            batch.overlap.dtype
        )
        beta = jnp.log(restitution) / jnp.sqrt(jnp.pi**2 + jnp.log(restitution) ** 2)
        damping = -2.0 * beta * jnp.sqrt(effective_mass * stiffness)
        displacement = transported_displacement + step_size * batch.tangential_velocity
        trial = (
            -stiffness[:, None] * displacement
            - damping[:, None] * batch.tangential_velocity
        )
        epsilon = jnp.asarray(self.direction_smoothing, dtype=trial.dtype)
        norm = jnp.sqrt(jnp.sum(trial**2, axis=-1) + epsilon**2)
        limit = (
            materials.pair_friction(left_material, right_material) * normal.friction_load
        )
        order = self.projection_order
        denominator = (norm**order + limit**order + epsilon**order) ** (1.0 / order)
        scale = limit / denominator
        force = scale[:, None] * trial
        active = normal.active
        force = jnp.where(active[:, None], force, 0.0)
        corrected = (
            -(force + damping[:, None] * batch.tangential_velocity) / stiffness[:, None]
        )
        corrected = jnp.where(active[:, None], corrected, 0.0)
        energy = jnp.where(active, 0.5 * stiffness * jnp.sum(corrected**2, axis=-1), 0.0)
        loss = jnp.where(
            active,
            damping * jnp.sum(batch.tangential_velocity**2, axis=-1) * step_size,
            0.0,
        )
        force_norm = _safe_vector_norm(force)
        defect = jnp.maximum(force_norm - limit, 0.0)
        finite = (
            jnp.all(jnp.isfinite(force))
            & jnp.all(jnp.isfinite(corrected))
            & jnp.all(jnp.isfinite(loss))
        )
        return DEMTangentialResponse(
            force,
            corrected,
            energy,
            jnp.maximum(loss, 0.0),
            active & (force_norm < 0.99 * limit),
            active & (force_norm >= 0.99 * limit),
            defect,
            jnp.full_like(limit, jnp.inf),
            finite,
        )


class DEMSurrogateBiasCertificate(StrictModule):
    force_relative_error: Array
    energy_relative_error: Array
    observable_relative_error: Array
    qualified: Array
    certificate_id: str = eqx.field(static=True)


def surrogate_bias_certificate(
    sharp_force: Array,
    smooth_force: Array,
    sharp_energy: Array,
    smooth_energy: Array,
    sharp_observable: Array,
    smooth_observable: Array,
    /,
    *,
    tolerance: float,
) -> DEMSurrogateBiasCertificate:
    tolerance_ = float(tolerance)
    if not np.isfinite(tolerance_) or tolerance_ <= 0.0:
        raise ValueError("Surrogate bias tolerance must be finite and positive.")

    def relative(left, right):
        scale = jnp.maximum(jnp.linalg.norm(left), 1.0e-30)
        return jnp.linalg.norm(right - left) / scale

    force_error = relative(sharp_force, smooth_force)
    energy_error = relative(sharp_energy, smooth_energy)
    observable_error = relative(sharp_observable, smooth_observable)
    qualified = (
        (force_error <= tolerance_)
        & (energy_error <= tolerance_)
        & (observable_error <= tolerance_)
    )
    return DEMSurrogateBiasCertificate(
        force_error,
        energy_error,
        observable_error,
        qualified,
        canonical_fingerprint(
            {
                "kind": "dem-surrogate-bias-certificate",
                "tolerance": tolerance_,
            }
        ),
    )


__all__ = [
    "DEMSurrogateBiasCertificate",
    "SmoothCoulombTangentialPlan",
    "SmoothPenaltyNormalPlan",
    "surrogate_bias_certificate",
]
