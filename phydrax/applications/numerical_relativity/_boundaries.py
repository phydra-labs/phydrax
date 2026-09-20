#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import abc
from collections.abc import Callable
from math import isfinite

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._derivatives import FourthOrderDerivatives
from ._grid import FixedGridGeometry
from ._state import Z4C_CHANNEL_COUNT, Z4cState


class Z4cBoundaryEvidence(StrictModule):
    applied_points: Array
    maximum_correction: Array
    finite: Array
    analytic: Array
    characteristic: Array
    successful: Array


class Z4cBoundaryResult(StrictModule):
    state: Z4cState
    evidence: Z4cBoundaryEvidence


def _evidence(
    state: Z4cState,
    correction: Array,
    applied_points: ArrayLike,
    /,
    *,
    analytic: bool,
    characteristic: bool,
    successful: ArrayLike,
) -> Z4cBoundaryEvidence:
    finite = jnp.all(jnp.isfinite(state.values)) & jnp.isfinite(correction)
    return Z4cBoundaryEvidence(
        jnp.asarray(applied_points, dtype=jnp.int32).reshape(()),
        jnp.asarray(correction).reshape(()),
        finite,
        jnp.asarray(analytic),
        jnp.asarray(characteristic),
        jnp.asarray(successful, dtype=jnp.bool_).reshape(()) & finite,
    )


class AbstractZ4cBoundary(StrictModule, NonTrainableState):
    boundary_id: eqx.AbstractVar[str]

    @abc.abstractmethod
    def apply_state(
        self, time: Array, state: Z4cState, grid: FixedGridGeometry, /
    ) -> Z4cBoundaryResult:
        raise NotImplementedError

    @abc.abstractmethod
    def apply_rates(
        self,
        time: Array,
        state: Z4cState,
        rates: Z4cState,
        grid: FixedGridGeometry,
        derivatives: FourthOrderDerivatives,
        /,
    ) -> Z4cBoundaryResult:
        raise NotImplementedError


class PeriodicBoundary(AbstractZ4cBoundary):
    boundary_id: str = "z4c-boundary:periodic"

    def apply_state(
        self, time: Array, state: Z4cState, grid: FixedGridGeometry, /
    ) -> Z4cBoundaryResult:
        del time
        successful = jnp.asarray(grid.periodic)
        return Z4cBoundaryResult(
            state,
            _evidence(
                state,
                jnp.zeros((), dtype=state.values.dtype),
                0,
                analytic=False,
                characteristic=False,
                successful=successful,
            ),
        )

    def apply_rates(
        self,
        time: Array,
        state: Z4cState,
        rates: Z4cState,
        grid: FixedGridGeometry,
        derivatives: FourthOrderDerivatives,
        /,
    ) -> Z4cBoundaryResult:
        del time, state, derivatives
        successful = jnp.asarray(grid.periodic)
        return Z4cBoundaryResult(
            rates,
            _evidence(
                rates,
                jnp.zeros((), dtype=rates.values.dtype),
                0,
                analytic=False,
                characteristic=False,
                successful=successful,
            ),
        )


class AnalyticBoundary(AbstractZ4cBoundary):
    """Strong outer-shell values supplied by an identified analytic solution."""

    solution: Callable[[Array, Array], Array | Z4cState] = eqx.field(static=True)
    width: int = eqx.field(static=True)
    boundary_id: str = eqx.field(static=True)

    def __init__(
        self,
        solution: Callable[[Array, Array], Array | Z4cState],
        solution_id: str,
        /,
        *,
        width: int = 3,
    ):
        if not callable(solution):
            raise TypeError("solution must be callable.")
        identifier = str(solution_id)
        width_ = int(width)
        if not identifier:
            raise ValueError("solution_id must be non-empty.")
        if width_ < 1:
            raise ValueError("width must be positive.")
        self.solution = solution
        self.width = width_
        self.boundary_id = canonical_fingerprint(
            {
                "kind": "z4c-analytic-boundary",
                "solution": identifier,
                "width": width_,
            }
        )

    def _exact(
        self, time: Array, state: Z4cState, grid: FixedGridGeometry, /
    ) -> Z4cState:
        value = self.solution(jnp.asarray(time), grid.coordinates)
        exact = (
            value
            if isinstance(value, Z4cState)
            else Z4cState(value, grid_id=grid.grid_id)
        )
        if exact.grid_id != grid.grid_id or exact.values.shape != state.values.shape:
            raise ValueError("Analytic boundary solution does not match the fixed grid.")
        return exact

    def apply_state(
        self, time: Array, state: Z4cState, grid: FixedGridGeometry, /
    ) -> Z4cBoundaryResult:
        exact = self._exact(time, state, grid)
        mask = grid.boundary_mask(self.width)
        values = jnp.where(mask[None, ...], exact.values, state.values)
        result = state.with_values(values)
        correction = jnp.max(jnp.abs(values - state.values))
        return Z4cBoundaryResult(
            result,
            _evidence(
                result,
                correction,
                jnp.sum(mask, dtype=jnp.int32),
                analytic=True,
                characteristic=False,
                successful=~jnp.asarray(grid.periodic),
            ),
        )

    def apply_rates(
        self,
        time: Array,
        state: Z4cState,
        rates: Z4cState,
        grid: FixedGridGeometry,
        derivatives: FourthOrderDerivatives,
        /,
    ) -> Z4cBoundaryResult:
        del time, state, derivatives
        mask = grid.boundary_mask(self.width)
        return Z4cBoundaryResult(
            rates,
            _evidence(
                rates,
                jnp.zeros((), dtype=rates.values.dtype),
                jnp.sum(mask, dtype=jnp.int32),
                analytic=True,
                characteristic=False,
                successful=~jnp.asarray(grid.periodic),
            ),
        )


class CharacteristicRadiativeBoundary(AbstractZ4cBoundary):
    """Outgoing characteristic/Sommerfeld path on a Cartesian outer shell."""

    asymptotic_values: Array
    center: tuple[float, float, float] = eqx.field(static=True)
    speed: float = eqx.field(static=True)
    width: int = eqx.field(static=True)
    boundary_id: str = eqx.field(static=True)

    def __init__(
        self,
        asymptotic_values: ArrayLike,
        /,
        *,
        center: tuple[float, float, float] = (0.0, 0.0, 0.0),
        speed: float = 1.0,
        width: int = 1,
    ):
        values = np.asarray(asymptotic_values)
        center_ = tuple(float(value) for value in center)
        speed_ = float(speed)
        width_ = int(width)
        if values.shape != (Z4C_CHANNEL_COUNT,) or not np.all(np.isfinite(values)):
            raise ValueError("asymptotic_values must be a finite shape-(25,) vector.")
        if len(center_) != 3 or any(not isfinite(value) for value in center_):
            raise ValueError("center must contain three finite coordinates.")
        if not isfinite(speed_) or speed_ <= 0.0:
            raise ValueError("speed must be finite and positive.")
        if width_ < 1:
            raise ValueError("width must be positive.")
        self.asymptotic_values = jnp.asarray(values)
        self.center = center_
        self.speed = speed_
        self.width = width_
        self.boundary_id = canonical_fingerprint(
            {
                "kind": "z4c-characteristic-radiative-boundary",
                "asymptotic": array_tree_fingerprint(values),
                "center": list(center_),
                "speed": speed_,
                "width": width_,
            }
        )

    def apply_state(
        self, time: Array, state: Z4cState, grid: FixedGridGeometry, /
    ) -> Z4cBoundaryResult:
        del time
        return Z4cBoundaryResult(
            state,
            _evidence(
                state,
                jnp.zeros((), dtype=state.values.dtype),
                0,
                analytic=False,
                characteristic=True,
                successful=~jnp.asarray(grid.periodic),
            ),
        )

    def apply_rates(
        self,
        time: Array,
        state: Z4cState,
        rates: Z4cState,
        grid: FixedGridGeometry,
        derivatives: FourthOrderDerivatives,
        /,
    ) -> Z4cBoundaryResult:
        del time
        radius, radial_normal = grid.radial_geometry(self.center)
        difference = state.values - self.asymptotic_values[:, None, None, None]
        radial_derivative = sum(
            radial_normal[axis][None, ...] * derivatives.first(difference, axis)
            for axis in range(3)
        )
        safe_radius = jnp.where(radius > 0.0, radius, 1.0)
        outgoing = -self.speed * (radial_derivative + difference / safe_radius[None, ...])
        mask = grid.boundary_mask(self.width)
        values = jnp.where(mask[None, ...], outgoing, rates.values)
        result = rates.with_values(values)
        correction = jnp.max(jnp.abs(values - rates.values))
        successful = (
            ~jnp.asarray(grid.periodic)
            & jnp.all(jnp.where(mask, radius > 0.0, True))
            & jnp.all(jnp.where(mask[None, ...], jnp.isfinite(outgoing), True))
        )
        return Z4cBoundaryResult(
            result,
            _evidence(
                result,
                correction,
                jnp.sum(mask, dtype=jnp.int32),
                analytic=False,
                characteristic=True,
                successful=successful,
            ),
        )


__all__ = [
    "AbstractZ4cBoundary",
    "AnalyticBoundary",
    "CharacteristicRadiativeBoundary",
    "PeriodicBoundary",
    "Z4cBoundaryEvidence",
    "Z4cBoundaryResult",
]
