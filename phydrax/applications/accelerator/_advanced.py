#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...linalg import DenseLinearOperator, DenseLU, LinearSolvePolicy, LinearSystem, solve
from ...typing import checked
from ._beam import _late_sign, AcceleratorBunch, AcceleratorConvention


def _symplectic_form(late_sign: float, /) -> np.ndarray:
    """Symplectic form of ``(x, px/p₀, y, py/p₀, ζ, δ)``.

    The canonical partner of ``δ`` is the positive-early coordinate
    ``z = −σ ζ``, so the longitudinal block carries the sign ``−σ``.
    """
    form = np.zeros((6, 6), dtype=np.float64)
    for index, sign in ((0, 1.0), (2, 1.0), (4, -late_sign)):
        form[index, index + 1] = sign
        form[index + 1, index] = -sign
    return form


class SymplecticMapPlan(StrictModule, NonTrainableState):
    matrix: Array
    offset: Array
    symplectic_residual: Array
    convention: AcceleratorConvention
    element_id: str = eqx.field(static=True)
    maximum_symplectic_residual: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        matrix: ArrayLike,
        offset: ArrayLike,
        convention: AcceleratorConvention,
        /,
        *,
        element_id: str,
        maximum_symplectic_residual: float = 1.0e-10,
    ) -> None:
        matrix_ = np.asarray(matrix, dtype=np.float64)
        offset_ = np.asarray(offset, dtype=np.float64)
        maximum = float(maximum_symplectic_residual)
        element = str(element_id).strip()
        if (
            matrix_.shape != (6, 6)
            or offset_.shape != (6,)
            or np.any(~np.isfinite(matrix_))
            or np.any(~np.isfinite(offset_))
        ):
            raise ValueError(
                "Symplectic map matrix/offset must be finite six-dimensional values."
            )
        if (
            not isinstance(convention, AcceleratorConvention)
            or not element
            or not math.isfinite(maximum)
            or maximum < 0.0
        ):
            raise ValueError("Map convention, identity, and residual policy are invalid.")
        form = _symplectic_form(_late_sign(convention))
        residual = float(np.linalg.norm(matrix_.T @ form @ matrix_ - form, ord=np.inf))
        if residual > maximum:
            raise ValueError("Transfer map exceeds the declared symplectic residual.")
        self.matrix = jnp.asarray(matrix_)
        self.offset = jnp.asarray(offset_)
        self.symplectic_residual = jnp.asarray(residual)
        self.convention = convention
        self.element_id = element
        self.maximum_symplectic_residual = maximum
        self.plan_id = canonical_fingerprint(
            {
                "kind": "accelerator-symplectic-map",
                "arrays": array_tree_fingerprint((matrix_, offset_)),
                "convention": convention.convention_id,
                "element": element,
                "maximum_residual": maximum,
            }
        )


class RingTrackingPlan(StrictModule, NonTrainableState):
    one_turn: SymplecticMapPlan
    turn_count: int = eqx.field(static=True)
    horizontal_aperture: float = eqx.field(static=True)
    vertical_aperture: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        one_turn: SymplecticMapPlan,
        turn_count: int,
        /,
        *,
        horizontal_aperture: float,
        vertical_aperture: float,
    ) -> None:
        turns = int(turn_count)
        horizontal = float(horizontal_aperture)
        vertical = float(vertical_aperture)
        if (
            turns < 1
            or not math.isfinite(horizontal)
            or horizontal <= 0.0
            or not math.isfinite(vertical)
            or vertical <= 0.0
        ):
            raise ValueError("Ring turn and aperture policy is invalid.")
        self.one_turn = one_turn
        self.turn_count = turns
        self.horizontal_aperture = horizontal
        self.vertical_aperture = vertical
        self.plan_id = canonical_fingerprint(
            {
                "kind": "accelerator-ring-tracking-plan",
                "one_turn": one_turn.plan_id,
                "turns": turns,
                "apertures": [horizontal, vertical],
            }
        )


class RingTrackingResult(StrictModule, NonTrainableState):
    bunch: AcceleratorBunch
    coordinate_history: Array
    active_history: Array
    loss_turn: Array
    finite: Array
    plan_id: str = eqx.field(static=True)


def track_ring(plan: RingTrackingPlan, bunch: AcceleratorBunch, /) -> RingTrackingResult:
    if not isinstance(plan, RingTrackingPlan) or not isinstance(bunch, AcceleratorBunch):
        raise TypeError("plan and bunch must use accelerator types.")
    if plan.one_turn.convention.convention_id != bunch.convention.convention_id:
        raise ValueError("Ring map and bunch coordinate conventions differ.")

    def turn(
        carry: tuple[Array, Array, Array], turn_index: Array
    ) -> tuple[tuple[Array, Array, Array], tuple[Array, Array]]:
        coordinates, active, loss_turn = carry
        candidate = coordinates @ plan.one_turn.matrix.T + plan.one_turn.offset
        inside = (jnp.abs(candidate[:, 0]) <= plan.horizontal_aperture) & (
            jnp.abs(candidate[:, 2]) <= plan.vertical_aperture
        )
        finite = jnp.all(jnp.isfinite(candidate), axis=-1)
        next_active = active & inside & finite & (candidate[:, 5] > -1.0)
        newly_lost = active & ~next_active & (loss_turn < 0)
        next_loss = jnp.where(newly_lost, turn_index, loss_turn)
        next_coordinates = jnp.where(active[:, None], candidate, coordinates)
        return (next_coordinates, next_active, next_loss), (next_coordinates, next_active)

    initial = (
        bunch.coordinates,
        bunch.active & bunch.valid,
        jnp.full((bunch.capacity,), -1, dtype=jnp.int32),
    )
    (coordinates, active, loss_turn), history = jax.lax.scan(
        turn, initial, jnp.arange(plan.turn_count, dtype=jnp.int32)
    )
    coordinate_history, active_history = history
    result_bunch = AcceleratorBunch(
        coordinates,
        bunch.weights,
        bunch.particle_ids,
        active=active,
        reference_rest_energy=float(bunch.reference_rest_energy),
        reference_momentum=float(bunch.reference_momentum),
        reference_charge=float(bunch.reference_charge),
        convention=bunch.convention,
        bunch_id=f"{bunch.bunch_id}:{plan.plan_id}",
    )
    return RingTrackingResult(
        result_bunch,
        coordinate_history,
        active_history,
        loss_turn,
        jnp.all(jnp.isfinite(coordinate_history)),
        plan.plan_id,
    )


class LinearRingOptics(StrictModule, NonTrainableState):
    closed_orbit: Array
    horizontal_tune: Array
    vertical_tune: Array
    stable: Array
    symplectic_residual: Array
    plan_id: str = eqx.field(static=True)


def linear_ring_optics(plan: SymplecticMapPlan, /) -> LinearRingOptics:
    if not isinstance(plan, SymplecticMapPlan):
        raise TypeError("plan must be SymplecticMapPlan.")
    system = jnp.eye(6, dtype=plan.matrix.dtype) - plan.matrix
    orbit = solve(
        LinearSystem(DenseLinearOperator(system)),
        plan.offset,
        policy=LinearSolvePolicy(DenseLU()),
    )
    trace_x = 0.5 * (plan.matrix[0, 0] + plan.matrix[1, 1])
    trace_y = 0.5 * (plan.matrix[2, 2] + plan.matrix[3, 3])
    stable = (
        jnp.all(orbit.status == 0) & (jnp.abs(trace_x) <= 1.0) & (jnp.abs(trace_y) <= 1.0)
    )
    tune_x = jnp.arccos(jnp.clip(trace_x, -1.0, 1.0)) / (2.0 * jnp.pi)
    tune_y = jnp.arccos(jnp.clip(trace_y, -1.0, 1.0)) / (2.0 * jnp.pi)
    return LinearRingOptics(
        orbit.value, tune_x, tune_y, stable, plan.symplectic_residual, plan.plan_id
    )


__all__ = [
    "LinearRingOptics",
    "RingTrackingPlan",
    "RingTrackingResult",
    "SymplecticMapPlan",
    "linear_ring_optics",
    "track_ring",
]
