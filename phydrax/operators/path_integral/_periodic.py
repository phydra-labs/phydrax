#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Finite periodic Euclidean path measures and reference-based partition estimates."""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._strict import StrictModule


class PeriodicPathPlan(StrictModule):
    """Static ring-polymer plan with an explicit treatment of the centroid mode."""

    bead_count: int = eqx.field(static=True)
    inverse_temperature: float = eqx.field(static=True)
    mass: float = eqx.field(static=True)
    hbar: float = eqx.field(static=True)
    zero_mode: Literal["confining-potential", "periodic-cell", "fixed-centroid"] = (
        eqx.field(static=True)
    )
    periodic_cell_lengths: tuple[float, ...] | None = eqx.field(static=True)
    fixed_centroid: tuple[float, ...] | None = eqx.field(static=True)

    def __init__(
        self,
        bead_count: int,
        inverse_temperature: float,
        /,
        *,
        mass: float = 1.0,
        hbar: float = 1.0,
        zero_mode: Literal["confining-potential", "periodic-cell", "fixed-centroid"],
        periodic_cell_lengths: tuple[float, ...] | None = None,
        fixed_centroid: tuple[float, ...] | None = None,
    ):
        beads = int(bead_count)
        beta, mass_, hbar_ = float(inverse_temperature), float(mass), float(hbar)
        if beads < 2 or any(not np.isfinite(v) or v <= 0.0 for v in (beta, mass_, hbar_)):
            raise ValueError(
                "bead_count >= 2 and finite positive beta/mass/hbar are required."
            )
        if zero_mode not in ("confining-potential", "periodic-cell", "fixed-centroid"):
            raise ValueError("A normalizable zero_mode treatment is required.")
        cell = (
            None
            if periodic_cell_lengths is None
            else tuple(float(v) for v in periodic_cell_lengths)
        )
        centroid = (
            None if fixed_centroid is None else tuple(float(v) for v in fixed_centroid)
        )
        if zero_mode == "periodic-cell" and (
            cell is None or not cell or any(v <= 0.0 or not np.isfinite(v) for v in cell)
        ):
            raise ValueError(
                "periodic-cell zero mode requires finite positive cell lengths."
            )
        if zero_mode == "fixed-centroid" and (
            centroid is None or not centroid or any(not np.isfinite(v) for v in centroid)
        ):
            raise ValueError("fixed-centroid zero mode requires a finite centroid.")
        if zero_mode != "periodic-cell" and cell is not None:
            raise ValueError(
                "periodic_cell_lengths are only valid for periodic-cell zero mode."
            )
        if zero_mode != "fixed-centroid" and centroid is not None:
            raise ValueError("fixed_centroid is only valid for fixed-centroid zero mode.")
        self.bead_count = beads
        self.inverse_temperature = beta
        self.mass = mass_
        self.hbar = hbar_
        self.zero_mode = zero_mode
        self.periodic_cell_lengths = cell
        self.fixed_centroid = centroid

    @property
    def imaginary_time_step(self) -> float:
        return self.inverse_temperature / self.bead_count


def periodic_path_action(
    beads: ArrayLike,
    potential: Callable[[Array], Array],
    /,
    *,
    plan: PeriodicPathPlan,
) -> Array:
    """Evaluate the ring action with algebraic closure from last bead to first."""
    if not isinstance(plan, PeriodicPathPlan):
        raise TypeError("plan must be PeriodicPathPlan.")
    values = jnp.asarray(beads)
    if values.ndim < 2 or values.shape[-2:] != (plan.bead_count, values.shape[-1]):
        raise ValueError("beads must have trailing shape (bead_count, state_dimension).")
    if int(values.shape[-1]) < 1:
        raise ValueError("Periodic paths require a nonempty state dimension.")
    following = jnp.roll(values, -1, axis=-2)
    difference = following - values
    if plan.periodic_cell_lengths is not None:
        cell = jnp.asarray(plan.periodic_cell_lengths, dtype=values.dtype)
        if cell.shape != (int(values.shape[-1]),):
            raise ValueError(
                "periodic cell dimension must match the path state dimension."
            )
        difference = difference - cell * jnp.round(difference / cell)
    step = plan.imaginary_time_step
    kinetic = (
        plan.mass
        * jnp.sum(difference * difference, axis=(-2, -1))
        / (2.0 * plan.hbar**2 * step)
    )
    flat = values.reshape((-1, int(values.shape[-1])))
    potential_values = jnp.asarray(jax.vmap(potential)(flat))
    if potential_values.shape != (flat.shape[0],) or jnp.iscomplexobj(potential_values):
        raise ValueError("potential must return one real scalar per bead.")
    potential_values = potential_values.reshape(values.shape[:-1])
    potential_action = step * jnp.sum(potential_values, axis=-1)
    action = kinetic + potential_action
    if plan.fixed_centroid is not None:
        centroid = jnp.mean(values, axis=-2)
        required = jnp.asarray(plan.fixed_centroid, dtype=values.dtype)
        tolerance = 32.0 * jnp.finfo(values.dtype).eps
        action = jnp.where(
            jnp.max(jnp.abs(centroid - required), axis=-1) <= tolerance, action, jnp.inf
        )
    return action


class PathPartitionEstimate(StrictModule):
    log_partition: Array
    thermodynamic_integral: Array
    standard_error: Array
    quadrature_refinement_difference: Array
    valid: Array
    reference_id: str = eqx.field(static=True)
    claim: str = eqx.field(static=True)


def estimate_path_partition_function(
    lambda_derivative_samples: ArrayLike,
    lambda_schedule: ArrayLike,
    /,
    *,
    reference_log_partition: ArrayLike,
    reference_id: str,
) -> PathPartitionEstimate:
    """Estimate log Z only relative to a caller-declared known reference."""
    samples = jnp.asarray(lambda_derivative_samples)
    schedule = jnp.asarray(lambda_schedule, dtype=float)
    if samples.ndim < 2 or schedule.ndim != 1 or samples.shape[0] != schedule.shape[0]:
        raise ValueError(
            "samples need shape (lambda, draws...) matching lambda_schedule."
        )
    if int(schedule.shape[0]) < 2:
        raise ValueError("lambda_schedule requires at least two nodes.")
    if not isinstance(reference_id, str) or not reference_id:
        raise ValueError("reference_id must be nonempty.")
    flat = samples.reshape((samples.shape[0], -1))
    means = jnp.mean(flat, axis=-1)
    widths = jnp.diff(schedule)
    trapezoid = jnp.sum(0.5 * widths * (means[:-1] + means[1:]))
    left = jnp.sum(widths * means[:-1])
    draw_integrals = jnp.sum(0.5 * widths[:, None] * (flat[:-1] + flat[1:]), axis=0)
    count = int(draw_integrals.shape[0])
    standard_error = (
        jnp.std(draw_integrals, ddof=1) / jnp.sqrt(float(count))
        if count > 1
        else jnp.asarray(jnp.nan)
    )
    reference = jnp.asarray(reference_log_partition)
    valid = (
        jnp.all(jnp.isfinite(samples))
        & jnp.all(jnp.isfinite(schedule))
        & jnp.all(jnp.diff(schedule) > 0.0)
        & jnp.isfinite(reference)
    )
    return PathPartitionEstimate(
        log_partition=reference - trapezoid,
        thermodynamic_integral=trapezoid,
        standard_error=standard_error,
        quadrature_refinement_difference=trapezoid - left,
        valid=valid,
        reference_id=reference_id,
        claim="finite-schedule-reference-relative-log-partition",
    )


__all__ = [
    "PathPartitionEstimate",
    "PeriodicPathPlan",
    "estimate_path_partition_function",
    "periodic_path_action",
]
