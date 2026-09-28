#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Separable binomial PIC filter with compensation.

`PICFilterPlan` applies ``passes`` three-point binomial passes
``(1/4, 1/2, 1/4)`` along each filtered axis, optionally followed by one
compensation pass ``(−n/4, 1 + n/2, −n/4)`` that cancels the leading
``(kΔ)²`` attenuation, so the per-axis response is

``T(kΔ) = cos²ⁿ(kΔ/2) · (1 + n sin²(kΔ/2))`` (``cos²ⁿ(kΔ/2)`` uncompensated).

The same operator acts on deposited charge, deposited current, and the field
view used for the gather, component by component through the solver's
`PICTensorLayout`. On periodic axes every pass is a circulant stencil that
commutes with the discrete divergence, so filtered current advances the Gauss
charge exactly to the filtered deposited charge, and the symmetric stencil
makes the gather filter the transpose of the deposit filter (no filter-induced
self-force or particle/field power mismatch). Near nonperiodic walls each pass
extends the component by mirror reflection with its declared parity; the
mirror commutes with the divergence while the wall-normal current vanishes and
otherwise leaves a defect measured by `continuity_report`.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array

from .._fingerprint import canonical_fingerprint
from .._trainable import NonTrainableState
from ..discretization import AxisEntityKind
from ._pic_field_solver import (
    AbstractPICFieldFilter,
    AbstractPreparedPICFieldSolver,
    PICFilterContinuityReport,
    PICTensorKind,
    PICTensorLayout,
    PICTensorMap,
)


def _three_point(
    value: Array,
    axis: int,
    weight: float,
    periodic: bool,
    location: AxisEntityKind,
    parity: int,
    /,
) -> Array:
    """``(1 − 2w)·x_i + w·(x_{i−1} + x_{i+1})`` with periodic or mirror extension."""
    if periodic:
        lower = jnp.roll(value, 1, axis=axis)
        upper = jnp.roll(value, -1, axis=axis)
    else:
        count = value.shape[axis]
        # Walls sit at points: point arrays reflect about their end entries,
        # interval arrays about the wall between the end entry and its image.
        inner = 1 if location == "point" else 0
        lower_image = parity * jnp.take(value, jnp.asarray([inner]), axis=axis)
        upper_image = parity * jnp.take(
            value, jnp.asarray([count - 1 - inner]), axis=axis
        )
        lower = jnp.concatenate(
            (lower_image, jnp.take(value, jnp.arange(count - 1), axis=axis)), axis=axis
        )
        upper = jnp.concatenate(
            (jnp.take(value, jnp.arange(1, count), axis=axis), upper_image), axis=axis
        )
    return (1.0 - 2.0 * weight) * value + weight * (lower + upper)


def _layout(solver: AbstractPreparedPICFieldSolver, /) -> PICTensorLayout:
    if not isinstance(solver, PICTensorLayout):
        raise TypeError(
            "PICFilterPlan requires a structured field solver implementing "
            "PICTensorLayout."
        )
    return solver


class PICFilterPlan(AbstractPICFieldFilter, NonTrainableState):
    """Binomial filter with optional compensation on selected grid axes."""

    passes: int = eqx.field(static=True)
    compensation: bool = eqx.field(static=True)
    axes: tuple[int, ...] | None = eqx.field(static=True)
    filter_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        passes: int = 1,
        compensation: bool = True,
        axes: Sequence[int] | None = None,
    ) -> None:
        count = int(passes)
        if count < 1:
            raise ValueError("passes must be positive.")
        selected = None if axes is None else tuple(int(value) for value in axes)
        if selected is not None and (
            not selected or len(set(selected)) != len(selected) or min(selected) < 0
        ):
            raise ValueError("axes must be distinct nonnegative axis indices.")
        self.passes = count
        self.compensation = bool(compensation)
        self.axes = None if selected is None else tuple(sorted(selected))
        self.filter_id = canonical_fingerprint(
            {
                "kind": "pic-binomial-filter",
                "passes": count,
                "compensation": self.compensation,
                "axes": None if self.axes is None else list(self.axes),
            }
        )

    def _filtered_axes(self, dimension: int, /) -> tuple[int, ...]:
        if self.axes is None:
            return tuple(range(dimension))
        if max(self.axes) >= dimension:
            raise ValueError("PIC filter axes exceed the field solver dimension.")
        return self.axes

    def _function(self, periodic: tuple[bool, ...], /) -> PICTensorMap:
        axes = self._filtered_axes(len(periodic))

        def apply(
            value: Array, location: tuple[AxisEntityKind, ...], parity: tuple[int, ...]
        ) -> Array:
            for axis in axes:
                arguments = (periodic[axis], location[axis], parity[axis])
                for _ in range(self.passes):
                    value = _three_point(value, axis, 0.25, *arguments)
                if self.compensation:
                    value = _three_point(value, axis, -0.25 * self.passes, *arguments)
            return value

        return apply

    def _apply(
        self, solver: AbstractPreparedPICFieldSolver, kind: PICTensorKind, value: Any, /
    ) -> Any:
        layout = _layout(solver)
        return layout.map_tensors(kind, value, self._function(layout.tensor_periodic))

    def filter_charge(
        self, solver: AbstractPreparedPICFieldSolver, charge: Array, /
    ) -> Array:
        return self._apply(solver, "charge", charge)

    def filter_current(
        self, solver: AbstractPreparedPICFieldSolver, current: Any, /
    ) -> Any:
        return self._apply(solver, "current", current)

    def filter_field(self, solver: AbstractPreparedPICFieldSolver, field: Any, /) -> Any:
        return self._apply(solver, "field", field)

    def continuity_report(
        self, solver: AbstractPreparedPICFieldSolver, /
    ) -> PICFilterContinuityReport:
        layout = _layout(solver)
        periodic = layout.tensor_periodic
        self._filtered_axes(len(periodic))
        generator = np.random.default_rng(20260927)

        def random(
            value: Array, location: tuple[AxisEntityKind, ...], parity: tuple[int, ...]
        ) -> Array:
            del location, parity
            return jnp.asarray(generator.standard_normal(value.shape), dtype=value.dtype)

        def interior(
            value: Array, location: tuple[AxisEntityKind, ...], parity: tuple[int, ...]
        ) -> Array:
            # Zero the wall-normal (odd, point-located) entries on every wall.
            for axis, (wrapped, kind, sign) in enumerate(
                zip(periodic, location, parity, strict=True)
            ):
                if not wrapped and kind == "point" and sign < 0:
                    wall = jnp.zeros(value.shape[axis], dtype=jnp.bool_)
                    wall = wall.at[0].set(True).at[-1].set(True)
                    shape = [1] * value.ndim
                    shape[axis] = value.shape[axis]
                    value = jnp.where(wall.reshape(shape), 0.0, value)
            return value

        step = 0.5 * jnp.asarray(solver.stable_step)
        zero_charge = layout.tensor_template("charge")
        start = jnp.zeros((), dtype=zero_charge.dtype)

        def charge_change(current: Any) -> Array:
            advanced = solver.advance(
                start, solver.field_with_charge(zero_charge), current, step
            )
            return solver.field_charge(advanced.field)

        def commutation(current: Any) -> float:
            change = charge_change(current)
            defect = self.filter_charge(solver, change) - charge_change(
                self.filter_current(solver, current)
            )
            return float(
                jnp.max(jnp.abs(defect))
                / jnp.maximum(jnp.max(jnp.abs(change)), jnp.finfo(change.dtype).tiny)
            )

        full = layout.map_tensors("current", layout.tensor_template("current"), random)
        interior_probe = layout.map_tensors("current", full, interior)
        charge = layout.map_tensors("charge", zero_charge, random)
        filtered = self.filter_charge(solver, charge - jnp.mean(charge))
        field, initialized = solver.initialize_field(filtered)
        relaxed = solver.advance(start, field, layout.tensor_template("current"), step)
        gauss = float(
            relaxed.electric_constraint
            / jnp.maximum(jnp.max(jnp.abs(filtered)), jnp.finfo(filtered.dtype).tiny)
        )
        if not bool(initialized):
            raise ValueError("Filtered Gauss initialization probe failed.")
        return PICFilterContinuityReport(
            self.filter_id,
            solver.solver_id,
            periodic,
            commutation(interior_probe),
            commutation(full),
            gauss,
        )


__all__ = ["PICFilterPlan"]
