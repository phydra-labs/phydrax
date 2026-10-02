#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import final

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...sparse import canonical_row_route_ids, EdgeRelation, RelationExecutionPlan
from ...typing import checked
from ._binning import PICCellBinningPlan
from ._transfer import PreparedPICParticleCochainTransfer
from ._types import PICCurrentDepositResult


@final
class ChargeConservingCurrentPlan(StrictModule, NonTrainableState):
    """Physical tail-to-head spline-Whitney current with ``rho_dot - delta(J) = 0``.

    The shape order is the transfer's ``shape_order``. All orders use the
    owning cubical kernel's exact polynomial moments on knot-split segments.

    Nonperiodic axes clip every path at the closed domain box: a path whose
    head leaves the box ends at its exit point, where its charge is deposited,
    and is reported in ``PICCurrentDepositResult.boundary_exit``. Continuity
    therefore holds exactly for the deposited path. Order one keeps every
    in-box stencil inside the grid; orders two and three refuse stencils that
    reach beyond a nonperiodic boundary (the endpoint splat rejects them).
    """

    transfer: PreparedPICParticleCochainTransfer
    binning: PICCellBinningPlan
    periodic: tuple[bool, bool, bool] = eqx.field(static=True)
    maximum_segments_per_particle: int = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        transfer: PreparedPICParticleCochainTransfer,
        /,
        *,
        maximum_segments_per_particle: int = 4,
        tolerance: float = 1.0e-10,
    ) -> None:
        if transfer.bridge.dimension != 3:
            raise ValueError("Charge-conserving current currently requires a 3-D bridge.")
        axes = transfer.bridge.grid.structured_axes
        segments = int(maximum_segments_per_particle)
        tolerance_ = float(tolerance)
        if segments < 1:
            raise ValueError("maximum_segments_per_particle must be positive.")
        if not np.isfinite(tolerance_) or tolerance_ <= 0.0:
            raise ValueError("tolerance must be positive and finite.")
        periodic = (
            bool(axes[0].periodic),
            bool(axes[1].periodic),
            bool(axes[2].periodic),
        )
        self.transfer = transfer
        self.binning = PICCellBinningPlan(
            tuple(float(axis.bounds[0]) for axis in axes),
            tuple(float(axis.bounds[1]) for axis in axes),
            tuple(axis.interval_centers.size for axis in axes),
            periodic,
        )
        self.periodic = periodic
        self.maximum_segments_per_particle = segments
        self.tolerance = tolerance_
        # The transfer identity already fixes the grid and its periodicity.
        self.plan_id = canonical_fingerprint(
            {
                "kind": "charge-conserving-whitney-current",
                "transfer": transfer.prepared_id,
                "segments": segments,
                "tolerance": tolerance_,
            }
        )

    def _flux(
        self, start: Array, end: Array, charges: Array, active: Array, dt: Array, /
    ) -> tuple[Array, Array, Array, Array]:
        query = self.transfer.kernel.integrate_segments(
            start, end, maximum_segments=self.maximum_segments_per_particle
        )
        route_valid = query.valid & active[:, None]
        counts = jnp.sum(
            jnp.any(
                route_valid.reshape(
                    (start.shape[0], self.maximum_segments_per_particle, -1)
                ),
                axis=-1,
            ),
            axis=1,
            dtype=jnp.int32,
        )
        bins = self.binning.bin(
            start,
            active,
            identity=(
                *(start[:, axis] for axis in range(3)),
                *(end[:, axis] for axis in range(3)),
                charges,
            ),
        )
        relation = EdgeRelation(
            jnp.arange(query.indices.size, dtype=jnp.int32),
            query.indices.reshape((-1,)),
            source_size=query.indices.size,
            target_size=query.dof_count,
            valid=route_valid.reshape((-1,)),
        )
        execution = RelationExecutionPlan().prepare(
            relation,
            stable_route_ids=canonical_row_route_ids(bins.order, query.indices.shape[1]),
        )
        values = (query.coefficients * charges[:, None] / dt).reshape((-1,))
        flux_content, evidence = execution.reduce(
            values, accumulation="fast", output="dense"
        )
        return (
            flux_content,
            counts,
            jnp.any(query.overflow & active),
            evidence.successful & jnp.all(query.successful | ~active),
        )

    def _clip_to_domain(
        self, start: Array, end: Array, active: Array, /
    ) -> tuple[Array, Array]:
        """Return the path head clipped at the closed nonperiodic box and exits.

        The exit parameter is the first crossing of any nonperiodic face along
        the straight tail-to-head path; the crossed coordinate is snapped onto
        its face so the deposited head lies exactly in the closed box.
        """
        axes = self.transfer.bridge.grid.structured_axes
        lower = jnp.asarray([axis.bounds[0] for axis in axes], dtype=start.dtype)
        upper = jnp.asarray([axis.bounds[1] for axis in axes], dtype=start.dtype)
        bounded = jnp.asarray(tuple(not value for value in self.periodic))
        delta = end - start
        above = bounded & (end > upper)
        below = bounded & (end < lower)
        safe = jnp.where(above | below, delta, 1.0)
        fraction = jnp.where(
            above, (upper - start) / safe, jnp.where(below, (lower - start) / safe, 1.0)
        )
        parameter = jnp.clip(jnp.min(fraction, axis=-1), 0.0, 1.0)
        leaves = active & jnp.any(above | below, axis=-1)
        clipped = start + parameter[:, None] * delta
        crossed = fraction <= parameter[:, None]
        clipped = jnp.where(crossed & above, upper, clipped)
        clipped = jnp.where(crossed & below, lower, clipped)
        clipped = jnp.where(bounded, jnp.clip(clipped, lower, upper), clipped)
        return jnp.where(leaves[:, None], clipped, end), leaves

    def deposit(
        self,
        start_position: ArrayLike,
        end_position: ArrayLike,
        step_size: ArrayLike,
        /,
        *,
        macrocharge: ArrayLike | None = None,
        active_mask: ArrayLike | None = None,
    ) -> PICCurrentDepositResult:
        """Deposit tail-to-head current and endpoint charge.

        ``macrocharge`` and ``active_mask`` are the runtime population charge and
        activity; omitted, the prepared species charges and activity are used.
        Heads beyond a nonperiodic face are clipped at their exit point (see the
        class docstring); ``deposited_end`` and ``boundary_exit`` report them.
        """
        start = jnp.asarray(start_position)
        end = jnp.asarray(end_position, dtype=start.dtype)
        expected = (self.transfer.species.capacity, 3)
        if start.shape != expected or end.shape != expected:
            raise ValueError(f"Current-deposition positions must have shape {expected}.")
        charges = (
            self.transfer.species.charges
            if macrocharge is None
            else jnp.asarray(macrocharge)
        ).astype(start.dtype)
        active = self.transfer.species.particles.active_mask
        if active_mask is not None:
            runtime_active = jnp.asarray(active_mask, dtype=jnp.bool_)
            if runtime_active.shape != active.shape:
                raise ValueError("active_mask must have particle-capacity shape.")
            active = active & runtime_active
        if charges.shape != active.shape:
            raise ValueError("macrocharge must have particle-capacity shape.")
        dt = jnp.asarray(step_size, dtype=start.dtype).reshape(())
        dt = eqx.error_if(
            dt, ~jnp.isfinite(dt) | (dt <= 0.0), "step_size must be positive and finite."
        )
        deposited_end, boundary_exit = self._clip_to_domain(start, end, active)
        start_routes = self.transfer.build(start, active_mask=active_mask)
        end_routes = self.transfer.build(deposited_end, active_mask=active_mask)
        start_charge = self.transfer.deposit_macrocharge(start_routes, charges)
        end_charge = self.transfer.deposit_macrocharge(end_routes, charges)
        flux_content, counts, overflow, reduced = self._flux(
            start, deposited_end, charges, active, dt
        )
        bridge = self.transfer.bridge
        current = bridge.cochain.inverse_hodge_star(1, flux_content)
        continuity = (
            end_charge.cochain - start_charge.cochain
        ) / dt - bridge.codifferential(1, current)
        maximum = jnp.max(jnp.abs(continuity), initial=0.0)
        # The residual of a short step is a difference of charges of size |ρ|, so
        # its roundoff floor is ε|ρ|/Δt even when the charge change is tiny; the
        # unsigned deposit keeps that floor when opposite charges coincide.
        magnitude = self.transfer.deposit_macrocharge(end_routes, jnp.abs(charges))
        scale = (
            jnp.max(
                jnp.abs(end_charge.cochain - start_charge.cochain)
                + 2.0 * magnitude.cochain,
                initial=0.0,
            )
            / dt
        )
        finite = (
            jnp.all(jnp.isfinite(current))
            & jnp.all(jnp.isfinite(continuity))
            & jnp.all(jnp.isfinite(start))
            & jnp.all(jnp.isfinite(end))
        )
        successful = (
            start_charge.successful
            & end_charge.successful
            & ~overflow
            & reduced
            & finite
            & (maximum <= self.tolerance * scale)
        )
        return PICCurrentDepositResult(
            start_charge,
            end_charge,
            current,
            continuity,
            maximum,
            scale,
            jnp.sum(counts, dtype=jnp.int32),
            overflow,
            deposited_end,
            boundary_exit,
            finite,
            successful,
            self.plan_id,
        )


@final
class PICMaxwellCurrentArguments(StrictModule):
    particle_current: Array
    external_arguments: object


__all__ = [
    "ChargeConservingCurrentPlan",
    "PICMaxwellCurrentArguments",
]
