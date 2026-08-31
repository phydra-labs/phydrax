#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ._dem_contact import AbstractDEMRotationalContactPlan, DEMRotationalResponse
from ._dem_contact_state import DEMRotationalHistory


class ElasticRollingTorsionalResistancePlan(AbstractDEMRotationalContactPlan):
    """History spring-dashpot rolling and torsion with bounded return mapping."""

    rolling_stiffness: Array
    torsional_stiffness: Array
    rolling_damping: Array
    torsional_damping: Array
    torsional_friction: Array
    rotational_law_id: str = eqx.field(static=True)

    def __init__(
        self,
        rolling_stiffness: ArrayLike,
        torsional_stiffness: ArrayLike,
        /,
        *,
        rolling_damping: ArrayLike = 0.0,
        torsional_damping: ArrayLike = 0.0,
        torsional_friction: ArrayLike = 0.0,
        rotational_law_id: str | None = None,
    ):
        values = tuple(
            np.asarray(value)
            for value in (
                rolling_stiffness,
                torsional_stiffness,
                rolling_damping,
                torsional_damping,
                torsional_friction,
            )
        )
        names = (
            "rolling_stiffness",
            "torsional_stiffness",
            "rolling_damping",
            "torsional_damping",
            "torsional_friction",
        )
        first = values[0]
        if any(
            value.ndim != first.ndim or value.shape != first.shape for value in values
        ):
            raise ValueError("Rotational parameter schemas must match.")
        for index, (name, value) in enumerate(zip(names, values, strict=True)):
            if value.ndim not in (0, 2):
                raise ValueError(f"{name} must be scalar or a square pair table.")
            if value.ndim == 2 and (
                value.shape[0] != value.shape[1] or not np.array_equal(value, value.T)
            ):
                raise ValueError(f"{name} pair table must be square and symmetric.")
            if np.any(~np.isfinite(value)) or np.any(value < 0.0):
                raise ValueError(f"{name} must be finite and nonnegative.")
            if index < 2 and np.any(value <= 0.0):
                raise ValueError(f"{name} must be positive.")
        generated = canonical_fingerprint(
            {
                "kind": "elastic-rolling-torsional-resistance",
                "values": array_tree_fingerprint(
                    {name: value for name, value in zip(names, values, strict=True)}
                ),
            }
        )
        identifier = generated if rotational_law_id is None else str(rotational_law_id)
        if not identifier:
            raise ValueError("rotational_law_id must be nonempty.")
        (
            rolling_stiffness_,
            torsional_stiffness_,
            rolling_damping_,
            torsional_damping_,
            torsional_friction_,
        ) = values
        self.rolling_stiffness = jnp.asarray(rolling_stiffness_)
        self.torsional_stiffness = jnp.asarray(torsional_stiffness_)
        self.rolling_damping = jnp.asarray(rolling_damping_)
        self.torsional_damping = jnp.asarray(torsional_damping_)
        self.torsional_friction = jnp.asarray(torsional_friction_)
        self.rotational_law_id = identifier

    def evaluate(
        self,
        batch,
        normal,
        history,
        context,
        materials,
        ambient_dimension,
        /,
    ):
        if not isinstance(history, DEMRotationalHistory):
            raise TypeError("history must be a DEMRotationalHistory.")
        rolling_stiffness = _pair_value(
            self.rolling_stiffness,
            context.left_material,
            context.right_material,
            materials.material_count,
        ).astype(batch.gap.dtype)
        torsional_stiffness = _pair_value(
            self.torsional_stiffness,
            context.left_material,
            context.right_material,
            materials.material_count,
        ).astype(batch.gap.dtype)
        rolling_damping = _pair_value(
            self.rolling_damping,
            context.left_material,
            context.right_material,
            materials.material_count,
        ).astype(batch.gap.dtype)
        torsional_damping = _pair_value(
            self.torsional_damping,
            context.left_material,
            context.right_material,
            materials.material_count,
        ).astype(batch.gap.dtype)
        torsional_friction = _pair_value(
            self.torsional_friction,
            context.left_material,
            context.right_material,
            materials.material_count,
        ).astype(batch.gap.dtype)
        rolling_friction = materials.pair_rolling_friction(
            context.left_material, context.right_material
        ).astype(batch.gap.dtype)
        relative = batch.left_angular_velocity - batch.right_angular_velocity
        continued = context.continued & history.previous_normal.any(axis=-1)
        frame_margin = jnp.full_like(batch.gap, jnp.inf)
        frame_valid = jnp.asarray(True)
        if ambient_dimension == 2:
            rolling_rate = relative
            torsional_rate = jnp.zeros_like(relative)
            rolling_previous = history.rolling_displacement
            torsional_previous = jnp.zeros_like(history.torsional_displacement)
        elif ambient_dimension == 3:
            normal_rate = jnp.sum(relative * batch.normal, axis=-1, keepdims=True)
            torsional_rate = normal_rate * batch.normal
            rolling_rate = relative - torsional_rate
            rolling_previous, frame_valid, frame_margin = _transport_tangent(
                history.rolling_displacement,
                history.previous_normal,
                batch.normal,
                continued,
            )
            torsional_scalar = jnp.sum(
                history.torsional_displacement * history.previous_normal,
                axis=-1,
                keepdims=True,
            )
            torsional_previous = jnp.where(
                continued[:, None], torsional_scalar * batch.normal, 0.0
            )
        else:
            raise ValueError("Rotational resistance requires dimension 2 or 3.")
        dt = jnp.asarray(context.step_size, dtype=batch.gap.dtype)
        rolling_trial_displacement = rolling_previous + dt * rolling_rate
        torsional_trial_displacement = torsional_previous + dt * torsional_rate
        rolling_trial_torque = (
            -rolling_stiffness[:, None] * rolling_trial_displacement
            - rolling_damping[:, None] * rolling_rate
        )
        torsional_trial_torque = (
            -torsional_stiffness[:, None] * torsional_trial_displacement
            - torsional_damping[:, None] * torsional_rate
        )
        radius = batch.effective_radius
        rolling_limit = rolling_friction * radius * normal.friction_load
        torsional_limit = torsional_friction * radius * normal.friction_load
        rolling_torque, rolling_yielded, rolling_margin = _bounded_vector(
            rolling_trial_torque, rolling_limit, normal.active
        )
        torsional_torque, torsional_yielded, torsional_margin = _bounded_vector(
            torsional_trial_torque,
            torsional_limit,
            normal.active & (ambient_dimension == 3),
        )
        rolling_displacement = jnp.where(
            rolling_yielded[:, None],
            -(rolling_torque + rolling_damping[:, None] * rolling_rate)
            / rolling_stiffness[:, None],
            rolling_trial_displacement,
        )
        torsional_displacement = jnp.where(
            torsional_yielded[:, None],
            -(torsional_torque + torsional_damping[:, None] * torsional_rate)
            / torsional_stiffness[:, None],
            torsional_trial_displacement,
        )
        active = normal.active & (
            (_norm(rolling_rate) > 0.0)
            | (_norm(torsional_rate) > 0.0)
            | (_norm(rolling_displacement) > 0.0)
            | (_norm(torsional_displacement) > 0.0)
        )
        rolling_torque = jnp.where(normal.active[:, None], rolling_torque, 0.0)
        torsional_torque = jnp.where(normal.active[:, None], torsional_torque, 0.0)
        total_torque = rolling_torque + torsional_torque
        old_rolling_energy = (
            0.5 * rolling_stiffness * _norm(history.rolling_displacement) ** 2
        )
        old_torsional_energy = (
            0.5 * torsional_stiffness * _norm(history.torsional_displacement) ** 2
        )
        rolling_energy = 0.5 * rolling_stiffness * _norm(rolling_displacement) ** 2
        torsional_energy = 0.5 * torsional_stiffness * _norm(torsional_displacement) ** 2
        rolling_work = -jnp.sum(rolling_torque * rolling_rate, axis=-1) * dt
        torsional_work = -jnp.sum(torsional_torque * torsional_rate, axis=-1) * dt
        rolling_loss = jnp.maximum(
            rolling_work - (rolling_energy - old_rolling_energy), 0.0
        )
        torsional_loss = jnp.maximum(
            torsional_work - (torsional_energy - old_torsional_energy), 0.0
        )
        rolling_loss = jnp.where(normal.active, rolling_loss, 0.0)
        torsional_loss = jnp.where(normal.active, torsional_loss, 0.0)
        rolling_displacement = jnp.where(
            normal.active[:, None], rolling_displacement, 0.0
        )
        torsional_displacement = jnp.where(
            normal.active[:, None], torsional_displacement, 0.0
        )
        next_history = DEMRotationalHistory(
            rolling_displacement,
            torsional_displacement,
            jnp.where(normal.active[:, None], batch.normal, 0.0),
            rolling_yielded,
            torsional_yielded,
        )
        finite = (
            frame_valid
            & jnp.all(jnp.isfinite(total_torque))
            & jnp.all(jnp.isfinite(rolling_energy))
            & jnp.all(jnp.isfinite(torsional_energy))
            & jnp.all(jnp.isfinite(rolling_loss))
            & jnp.all(jnp.isfinite(torsional_loss))
        )
        del frame_margin
        return DEMRotationalResponse(
            total_torque,
            -total_torque,
            rolling_torque,
            -rolling_torque,
            torsional_torque,
            -torsional_torque,
            jnp.where(normal.active, rolling_energy + torsional_energy, 0.0),
            rolling_loss + torsional_loss,
            rolling_loss,
            torsional_loss,
            rolling_yielded,
            torsional_yielded,
            rolling_margin,
            torsional_margin,
            next_history,
            active,
            finite,
        )


def _transport_tangent(values, old_normal, new_normal, continued):
    cross = jnp.cross(old_normal, new_normal)
    dot = jnp.sum(old_normal * new_normal, axis=-1)
    denominator = 1.0 + dot
    safe = jnp.where(jnp.abs(denominator) > 1.0e-10, denominator, 1.0)
    rotated = (
        values
        + jnp.cross(cross, values)
        + jnp.cross(cross, jnp.cross(cross, values)) / safe[:, None]
    )
    rotated = rotated - jnp.sum(rotated * new_normal, axis=-1, keepdims=True) * new_normal
    result = jnp.where(continued[:, None], rotated, 0.0)
    margin = jnp.min(jnp.where(continued, denominator, jnp.inf))
    valid = jnp.all(~continued | (denominator > 1.0e-10))
    return result, valid, margin


def _bounded_vector(trial, limit, active):
    magnitude = _norm(trial)
    safe = jnp.where(magnitude > 0.0, magnitude, 1.0)
    yielded = active & (magnitude > limit)
    bounded = jnp.where(yielded[:, None], trial * (limit / safe)[:, None], trial)
    bounded = jnp.where(active[:, None], bounded, 0.0)
    margin = jnp.where(active, jnp.abs(limit - magnitude), jnp.inf)
    return bounded, yielded, margin


def _norm(value):
    squared = jnp.sum(value * value, axis=-1)
    return jnp.where(squared > 0.0, jnp.sqrt(squared), 0.0)


def _pair_value(parameter, left, right, material_count):
    if parameter.ndim == 0:
        return jnp.broadcast_to(parameter, left.shape)
    if parameter.shape != (material_count, material_count):
        raise ValueError("Rotational pair table does not match material count.")
    return parameter[left, right]


__all__ = ["ElasticRollingTorsionalResistancePlan"]
