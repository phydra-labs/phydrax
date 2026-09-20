#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import enum

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState


class NonequilibriumStatus(enum.IntEnum):
    SUCCESS = 0
    NONFINITE = 1
    CONSTRAINT_VIOLATION = 2
    CAPACITY_EXCEEDED = 3
    UNSTABLE_STEP = 4


class ClosedTimePathPlan(StrictModule, NonTrainableState):
    """Physical time grid and hard contour allocation budget."""

    time_nodes: Array
    maximum_contour_points: int = eqx.field(static=True)
    maximum_two_point_elements: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        time_nodes: ArrayLike,
        /,
        *,
        maximum_contour_points: int = 4096,
        maximum_two_point_elements: int = 16_777_216,
    ):
        times = np.asarray(time_nodes, dtype=np.float64)
        contour_capacity = int(maximum_contour_points)
        two_point_capacity = int(maximum_two_point_elements)
        if (
            times.ndim != 1
            or times.size < 2
            or np.any(~np.isfinite(times))
            or np.any(np.diff(times) <= 0.0)
            or 2 * times.size > contour_capacity
            or two_point_capacity <= 0
        ):
            raise ValueError(
                "Closed-time-path coordinates or resource budget are invalid."
            )
        self.time_nodes = jnp.asarray(times)
        self.maximum_contour_points = contour_capacity
        self.maximum_two_point_elements = two_point_capacity
        self.plan_id = canonical_fingerprint(
            {
                "kind": "closed-time-path-plan",
                "time_nodes": array_tree_fingerprint(times),
                "maximum_contour_points": contour_capacity,
                "maximum_two_point_elements": two_point_capacity,
                "branches": ("forward", "backward"),
            }
        )

    def prepare(self, /) -> "ClosedTimePathGrid":
        return ClosedTimePathGrid(self)


class ClosedTimePathGrid(StrictModule, NonTrainableState):
    """Prepared Schwinger--Keldysh contour with signed trapezoidal measure."""

    plan: ClosedTimePathPlan
    contour_times: Array
    physical_indices: Array
    branch_indices: Array
    branch_orientation: Array
    contour_weights: Array
    causal_mask: Array
    grid_id: str = eqx.field(static=True)

    def __init__(self, plan: ClosedTimePathPlan, /):
        if not isinstance(plan, ClosedTimePathPlan):
            raise TypeError("plan must be ClosedTimePathPlan.")
        times = np.asarray(plan.time_nodes)
        differences = np.diff(times)
        physical_weights = np.empty_like(times)
        physical_weights[0] = 0.5 * differences[0]
        physical_weights[-1] = 0.5 * differences[-1]
        if times.size > 2:
            physical_weights[1:-1] = 0.5 * (differences[:-1] + differences[1:])
        contour_times = np.concatenate((times, times[::-1]))
        physical_indices = np.concatenate(
            (np.arange(times.size), np.arange(times.size - 1, -1, -1))
        ).astype(np.int32)
        branch_indices = np.concatenate(
            (np.zeros(times.size, dtype=np.int32), np.ones(times.size, dtype=np.int32))
        )
        orientation = np.concatenate((np.ones(times.size), -np.ones(times.size)))
        weights = np.concatenate((physical_weights, -physical_weights[::-1]))
        causal = np.arange(times.size)[:, None] >= np.arange(times.size)[None, :]
        self.plan = plan
        self.contour_times = jnp.asarray(contour_times)
        self.physical_indices = jnp.asarray(physical_indices)
        self.branch_indices = jnp.asarray(branch_indices)
        self.branch_orientation = jnp.asarray(orientation)
        self.contour_weights = jnp.asarray(weights)
        self.causal_mask = jnp.asarray(causal)
        self.grid_id = canonical_fingerprint(
            {
                "kind": "prepared-closed-time-path-grid",
                "plan": plan.plan_id,
                "contour_times": array_tree_fingerprint(contour_times),
                "physical_indices": array_tree_fingerprint(physical_indices),
                "branch_indices": array_tree_fingerprint(branch_indices),
                "contour_weights": array_tree_fingerprint(weights),
            }
        )


class KeldyshTwoPointFunctions(StrictModule):
    statistical: Array
    spectral: Array
    retarded: Array
    advanced: Array
    spectral_first_time_derivative: Array
    finite: Array
    status: Array
    grid_id: str = eqx.field(static=True)
    source_id: str = eqx.field(static=True)


class KeldyshIdentityEvidence(StrictModule):
    statistical_symmetry_residual: Array
    spectral_antisymmetry_residual: Array
    equal_time_spectral_residual: Array
    canonical_commutator_residual: Array
    finite: Array
    satisfied: Array
    source_id: str = eqx.field(static=True)


class FreeKeldyshPlan(StrictModule, NonTrainableState):
    """Free real-scalar mode propagators on one prepared physical time grid."""

    __hash__ = object.__hash__

    grid: ClosedTimePathGrid
    frequencies: Array
    time_differences: Array
    maximum_modes: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        grid: ClosedTimePathGrid,
        frequencies: ArrayLike,
        /,
        *,
        maximum_modes: int = 4096,
    ):
        if not isinstance(grid, ClosedTimePathGrid):
            raise TypeError("grid must be ClosedTimePathGrid.")
        frequency = np.asarray(frequencies, dtype=np.float64)
        mode_capacity = int(maximum_modes)
        required = grid.plan.time_nodes.size**2 * frequency.size
        if (
            frequency.ndim != 1
            or frequency.size == 0
            or frequency.size > mode_capacity
            or np.any(~np.isfinite(frequency))
            or np.any(frequency <= 0.0)
            or required > grid.plan.maximum_two_point_elements
        ):
            raise ValueError("Free Keldysh modes exceed the fixed two-point budget.")
        delta = (
            np.asarray(grid.plan.time_nodes)[:, None]
            - np.asarray(grid.plan.time_nodes)[None, :]
        )
        self.grid = grid
        self.frequencies = jnp.asarray(frequency)
        self.time_differences = jnp.asarray(delta)
        self.maximum_modes = mode_capacity
        self.plan_id = canonical_fingerprint(
            {
                "kind": "free-keldysh-propagator-plan",
                "grid": grid.grid_id,
                "frequencies": array_tree_fingerprint(frequency),
                "maximum_modes": mode_capacity,
            }
        )

    def evaluate(self, occupations: ArrayLike, /) -> KeldyshTwoPointFunctions:
        occupation = jnp.asarray(occupations, dtype=self.frequencies.dtype)
        if occupation.shape != self.frequencies.shape:
            raise ValueError("Occupations must provide one value per free mode.")
        phase = self.time_differences[..., None] * self.frequencies
        spectral = jnp.sin(phase) / self.frequencies
        statistical = (occupation + 0.5) * jnp.cos(phase) / self.frequencies
        derivative = jnp.cos(phase)
        retarded = jnp.where(self.grid.causal_mask[..., None], spectral, 0.0)
        advanced = -jnp.swapaxes(retarded, 0, 1)
        finite = (
            jnp.all(jnp.isfinite(occupation))
            & jnp.all(occupation >= 0.0)
            & jnp.all(jnp.isfinite(statistical))
            & jnp.all(jnp.isfinite(spectral))
        )
        status = jnp.where(
            finite,
            int(NonequilibriumStatus.SUCCESS),
            int(NonequilibriumStatus.NONFINITE),
        ).astype(jnp.int32)
        return KeldyshTwoPointFunctions(
            statistical,
            spectral,
            retarded,
            advanced,
            derivative,
            finite,
            status,
            self.grid.grid_id,
            self.plan_id,
        )

    def identity_evidence(
        self,
        functions: KeldyshTwoPointFunctions,
        /,
        *,
        tolerance: float = 1.0e-10,
    ) -> KeldyshIdentityEvidence:
        if not isinstance(functions, KeldyshTwoPointFunctions):
            raise TypeError("functions must be KeldyshTwoPointFunctions.")
        if functions.source_id != self.plan_id:
            raise ValueError("Two-point functions belong to another free plan.")
        tolerance_ = float(tolerance)
        if not np.isfinite(tolerance_) or tolerance_ <= 0.0:
            raise ValueError("Keldysh identity tolerance must be finite and positive.")
        statistical_symmetry = jnp.max(
            jnp.abs(functions.statistical - jnp.swapaxes(functions.statistical, 0, 1))
        )
        spectral_antisymmetry = jnp.max(
            jnp.abs(functions.spectral + jnp.swapaxes(functions.spectral, 0, 1))
        )
        diagonal_spectral = jnp.max(
            jnp.abs(jnp.diagonal(functions.spectral, axis1=0, axis2=1))
        )
        diagonal_derivative = jnp.diagonal(
            functions.spectral_first_time_derivative, axis1=0, axis2=1
        )
        canonical = jnp.max(jnp.abs(diagonal_derivative - 1.0))
        finite = functions.finite & jnp.all(
            jnp.isfinite(
                jnp.stack(
                    (
                        statistical_symmetry,
                        spectral_antisymmetry,
                        diagonal_spectral,
                        canonical,
                    )
                )
            )
        )
        satisfied = finite & (
            jnp.maximum(
                jnp.maximum(statistical_symmetry, spectral_antisymmetry),
                jnp.maximum(diagonal_spectral, canonical),
            )
            <= tolerance_
        )
        return KeldyshIdentityEvidence(
            statistical_symmetry,
            spectral_antisymmetry,
            diagonal_spectral,
            canonical,
            finite,
            satisfied,
            self.plan_id,
        )


__all__ = [
    "ClosedTimePathGrid",
    "ClosedTimePathPlan",
    "FreeKeldyshPlan",
    "KeldyshIdentityEvidence",
    "KeldyshTwoPointFunctions",
    "NonequilibriumStatus",
]
