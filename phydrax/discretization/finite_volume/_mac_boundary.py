#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Literal, TypeAlias

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._incompressible import FaceVelocity, PreparedMACOperators


MACBoundaryKind: TypeAlias = Literal[
    "no-slip",
    "free-slip",
    "symmetry",
    "velocity-inflow",
    "normal-flux-inflow",
    "pressure-outlet",
    "traction-open",
]
MACBoundarySideName: TypeAlias = Literal["lower", "upper"]
MACPressureClosureKind: TypeAlias = Literal["neumann", "dirichlet"]
MACBoundaryProviderFunction: TypeAlias = Callable[
    [Array, tuple[Array, ...], Any], tuple[ArrayLike, ArrayLike]
]

_ESSENTIAL_KINDS = (
    "no-slip",
    "free-slip",
    "symmetry",
    "velocity-inflow",
    "normal-flux-inflow",
)
_VECTOR_KINDS = ("no-slip", "velocity-inflow", "traction-open")
_OPEN_KINDS = ("pressure-outlet", "traction-open")


def _axis_boundary(value: Array, axis: int, index: int, /) -> Array:
    location = [slice(None)] * value.ndim
    location[axis] = index
    return value[tuple(location)]


def _set_axis_boundary(
    value: Array, axis: int, index: int, target: ArrayLike, /
) -> Array:
    location = [slice(None)] * value.ndim
    location[axis] = index
    return value.at[tuple(location)].set(target)


def _broadcast_scalar(
    value: ArrayLike, shape: tuple[int, ...], dtype: jnp.dtype, /
) -> Array:
    array = jnp.asarray(value, dtype=dtype)
    if array.shape == ():
        return jnp.broadcast_to(array, shape)
    if array.shape != shape:
        raise ValueError(
            f"Scalar MAC boundary data must have shape {shape} or be scalar."
        )
    return array


def _broadcast_vector(
    value: ArrayLike,
    dimension: int,
    shape: tuple[int, ...],
    dtype: jnp.dtype,
    /,
) -> Array:
    array = jnp.asarray(value, dtype=dtype)
    target = (dimension,) + shape
    if array.shape == ():
        return jnp.broadcast_to(array, target)
    if array.shape == (dimension,):
        return jnp.broadcast_to(array.reshape((dimension,) + (1,) * len(shape)), target)
    if array.shape != target:
        raise ValueError(
            f"Vector MAC boundary data must have shape {(dimension,)}, {target}, "
            "or be scalar."
        )
    return array


class MACBoundaryProvider(StrictModule, NonTrainableState):
    """Constant or stage-evaluated boundary value and material time derivative."""

    value: Array
    rate: Array
    function: MACBoundaryProviderFunction | None
    time_dependent: bool = eqx.field(static=True)
    provider_id: str = eqx.field(static=True)

    def __init__(
        self,
        value: ArrayLike = 0.0,
        /,
        *,
        rate: ArrayLike | None = None,
        function: MACBoundaryProviderFunction | None = None,
        provider_id: str | None = None,
    ):
        value_ = jnp.asarray(value)
        if not jnp.issubdtype(value_.dtype, jnp.inexact):
            value_ = value_.astype(float)
        rate_ = (
            jnp.zeros_like(value_)
            if rate is None
            else jnp.asarray(rate, dtype=value_.dtype)
        )
        if function is not None and not callable(function):
            raise TypeError("MAC boundary provider function must be callable or None.")
        if function is not None and (provider_id is None or not str(provider_id)):
            raise ValueError("Time-dependent MAC boundary providers require provider_id.")
        identifier = canonical_fingerprint(
            {
                "kind": "mac-boundary-provider",
                "time_dependent": function is not None,
                "external_id": None if provider_id is None else str(provider_id),
                "template": array_tree_fingerprint((value_, rate_)),
            }
        )
        self.value = value_
        self.rate = rate_
        self.function = function
        self.time_dependent = function is not None
        self.provider_id = identifier

    def evaluate(
        self,
        time: ArrayLike,
        coordinates: tuple[Array, ...],
        args: Any = None,
        /,
    ) -> tuple[Array, Array]:
        if self.function is None:
            return self.value, self.rate
        value, rate = self.function(jnp.asarray(time), coordinates, args)
        return jnp.asarray(value), jnp.asarray(rate)


class MACBoundarySide(StrictModule, NonTrainableState):
    """One named Cartesian boundary side and its physical closure."""

    axis: str = eqx.field(static=True)
    side: MACBoundarySideName = eqx.field(static=True)
    kind: MACBoundaryKind = eqx.field(static=True)
    provider: MACBoundaryProvider
    backflow_coefficient: float = eqx.field(static=True)
    side_id: str = eqx.field(static=True)

    def __init__(
        self,
        axis: str,
        side: MACBoundarySideName,
        kind: MACBoundaryKind,
        /,
        *,
        provider: MACBoundaryProvider | None = None,
        backflow_coefficient: float = 0.0,
    ):
        axis_ = str(axis)
        if not axis_:
            raise ValueError("MAC boundary axis must be non-empty.")
        if side not in ("lower", "upper"):
            raise ValueError("MAC boundary side must be 'lower' or 'upper'.")
        allowed = _ESSENTIAL_KINDS + _OPEN_KINDS
        if kind not in allowed:
            raise ValueError("Unknown MAC boundary kind.")
        provider_ = MACBoundaryProvider() if provider is None else provider
        if not isinstance(provider_, MACBoundaryProvider):
            raise TypeError("provider must be MACBoundaryProvider or None.")
        coefficient = float(backflow_coefficient)
        if not np.isfinite(coefficient) or coefficient < 0.0:
            raise ValueError(
                "MAC open-boundary backflow coefficient must be finite and nonnegative."
            )
        if kind != "traction-open" and coefficient != 0.0:
            raise ValueError(
                "Only traction-open boundaries accept backflow stabilization."
            )
        self.axis = axis_
        self.side = side
        self.kind = kind
        self.provider = provider_
        self.backflow_coefficient = coefficient
        self.side_id = canonical_fingerprint(
            {
                "kind": "mac-boundary-side",
                "axis": axis_,
                "side": side,
                "boundary_kind": kind,
                "provider": provider_.provider_id,
                "backflow_coefficient": coefficient,
            }
        )


class MACBoundaryStageData(StrictModule, NonTrainableState):
    """Dynamic provider leaves and fail-closed evidence for one MAC stage."""

    values: tuple[Array, ...]
    rates: tuple[Array, ...]
    prescribed_mass_flux: Array
    compatibility_defect: Array
    finite: Array
    compatible: Array
    successful: Array
    boundary_id: str = eqx.field(static=True)
    stage_id: str = eqx.field(static=True)


class MACBoundaryPlan(StrictModule, NonTrainableState):
    """Complete static physical-boundary declaration for one MAC grid."""

    operators: PreparedMACOperators
    sides: tuple[MACBoundarySide, ...]
    closure_kind: MACPressureClosureKind = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        operators: PreparedMACOperators,
        sides: Sequence[MACBoundarySide] | None = None,
        /,
    ):
        if not isinstance(operators, PreparedMACOperators):
            raise TypeError("operators must be PreparedMACOperators.")
        grid = operators.discretization.grid
        supplied = () if sides is None else tuple(sides)
        if not all(isinstance(value, MACBoundarySide) for value in supplied):
            raise TypeError("MAC boundary sides must be MACBoundarySide values.")
        indexed = {(value.axis, value.side): value for value in supplied}
        if len(indexed) != len(supplied):
            raise ValueError(
                "Each MAC axis side must have exactly one boundary declaration."
            )
        unknown = set(indexed).difference(
            (name, side) for name in grid.axis_names for side in ("lower", "upper")
        )
        if unknown:
            raise ValueError(f"Unknown MAC boundary sides {sorted(unknown)!r}.")
        complete: list[MACBoundarySide] = []
        for name, axis in zip(grid.axis_names, grid.structured_axes, strict=True):
            lower = indexed.get((name, "lower"))
            upper = indexed.get((name, "upper"))
            if axis.periodic:
                if lower is not None or upper is not None:
                    raise ValueError(
                        "Periodic MAC axes do not accept physical boundaries."
                    )
                continue
            if sides is not None and (lower is None or upper is None):
                raise ValueError(
                    "Every nonperiodic MAC axis requires lower and upper sides."
                )
            complete.extend(
                (
                    MACBoundarySide(name, "lower", "no-slip") if lower is None else lower,
                    MACBoundarySide(name, "upper", "no-slip") if upper is None else upper,
                )
            )
        closure: MACPressureClosureKind = (
            "dirichlet"
            if any(value.kind in _OPEN_KINDS for value in complete)
            else "neumann"
        )
        self.operators = operators
        self.sides = tuple(complete)
        self.closure_kind = closure
        self.plan_id = canonical_fingerprint(
            {
                "kind": "mac-boundary-plan",
                "operators": operators.prepared_id,
                "sides": [value.side_id for value in complete],
                "pressure_closure": closure,
            }
        )

    def prepare(self, /) -> PreparedMACBoundaryPlan:
        return PreparedMACBoundaryPlan(self)


class PreparedMACBoundaryPlan(StrictModule, NonTrainableState):
    """Prepared side geometry with pure stage evaluation and trace actions."""

    operators: PreparedMACOperators
    sides: tuple[MACBoundarySide, ...]
    side_axes: tuple[int, ...] = eqx.field(static=True)
    side_indices: tuple[int, ...] = eqx.field(static=True)
    side_shapes: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    coordinates: tuple[tuple[Array, ...], ...]
    closure_kind: MACPressureClosureKind = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)

    def __init__(self, plan: MACBoundaryPlan, /):
        if not isinstance(plan, MACBoundaryPlan):
            raise TypeError("plan must be MACBoundaryPlan.")
        grid = plan.operators.discretization.grid
        axes: list[int] = []
        indices: list[int] = []
        shapes: list[tuple[int, ...]] = []
        coordinates: list[tuple[Array, ...]] = []
        for boundary in plan.sides:
            axis = grid.axis_names.index(boundary.axis)
            side_index = 0 if boundary.side == "lower" else -1
            shape = tuple(
                int(count)
                for index, count in enumerate(plan.operators.discretization.cell_shape)
                if index != axis
            )
            side_coordinates: list[Array] = []
            for coordinate_axis, structured_axis in enumerate(grid.structured_axes):
                if coordinate_axis == axis:
                    coordinate = structured_axis.bounds[0 if side_index == 0 else 1]
                    side_coordinates.append(
                        jnp.full(
                            shape,
                            coordinate,
                            dtype=structured_axis.interval_centers.dtype,
                        )
                    )
                else:
                    local_axis = (
                        coordinate_axis if coordinate_axis < axis else coordinate_axis - 1
                    )
                    reshape = [1] * len(shape)
                    reshape[local_axis] = int(structured_axis.interval_centers.size)
                    side_coordinates.append(
                        jnp.broadcast_to(
                            structured_axis.interval_centers.reshape(tuple(reshape)),
                            shape,
                        )
                    )
            axes.append(axis)
            indices.append(side_index)
            shapes.append(shape)
            coordinates.append(tuple(side_coordinates))
        self.operators = plan.operators
        self.sides = plan.sides
        self.side_axes = tuple(axes)
        self.side_indices = tuple(indices)
        self.side_shapes = tuple(shapes)
        self.coordinates = tuple(coordinates)
        self.closure_kind = plan.closure_kind
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-mac-boundary-plan",
                "plan": plan.plan_id,
                "side_shapes": [list(value) for value in shapes],
            }
        )

    @property
    def dimension(self) -> int:
        return len(self.operators.discretization.cell_shape)

    def _side_position(self, axis: int, side: MACBoundarySideName, /) -> int:
        side_index = 0 if side == "lower" else -1
        return next(
            index
            for index, (candidate_axis, candidate_side) in enumerate(
                zip(self.side_axes, self.side_indices, strict=True)
            )
            if candidate_axis == axis and candidate_side == side_index
        )

    def side_kind(self, axis: int, side: MACBoundarySideName, /) -> MACBoundaryKind:
        return self.sides[self._side_position(axis, side)].kind

    def evaluate(self, time: ArrayLike, args: Any = None, /) -> MACBoundaryStageData:
        dtype = self.operators.pressure_space.dtype
        values: list[Array] = []
        rates: list[Array] = []
        finite_terms: list[Array] = []
        flux_terms: list[Array] = []
        flux_scale_terms: list[Array] = []
        for boundary, axis, side_index, shape, coordinates in zip(
            self.sides,
            self.side_axes,
            self.side_indices,
            self.side_shapes,
            self.coordinates,
            strict=True,
        ):
            raw_value, raw_rate = boundary.provider.evaluate(time, coordinates, args)
            if boundary.kind in _VECTOR_KINDS:
                value = _broadcast_vector(raw_value, self.dimension, shape, dtype)
                rate = _broadcast_vector(raw_rate, self.dimension, shape, dtype)
            else:
                value = _broadcast_scalar(raw_value, shape, dtype)
                rate = _broadcast_scalar(raw_rate, shape, dtype)
            values.append(value)
            rates.append(rate)
            finite_terms.extend(
                (jnp.all(jnp.isfinite(value)), jnp.all(jnp.isfinite(rate)))
            )
            if boundary.kind in _ESSENTIAL_KINDS:
                normal = value[axis] if boundary.kind in _VECTOR_KINDS else value
                measure = _axis_boundary(
                    self.operators.discretization.face_measures[axis], axis, side_index
                )
                signed_flux = (-1.0 if side_index == 0 else 1.0) * jnp.sum(
                    measure * normal
                )
                flux_terms.append(signed_flux)
                flux_scale_terms.append(jnp.sum(measure * jnp.abs(normal)))
        zero = jnp.asarray(0.0, dtype=dtype)
        prescribed_flux = sum(flux_terms, start=zero)
        flux_scale = sum(flux_scale_terms, start=zero)
        finite = (
            jnp.all(jnp.stack(tuple(finite_terms))) if finite_terms else jnp.asarray(True)
        )
        epsilon = jnp.finfo(dtype).eps
        tolerance = 4096.0 * epsilon * jnp.maximum(1.0, flux_scale)
        compatibility_defect = jnp.abs(prescribed_flux)
        compatible = (
            jnp.asarray(True)
            if self.closure_kind == "dirichlet"
            else compatibility_defect <= tolerance
        )
        successful = finite & compatible
        return MACBoundaryStageData(
            values=tuple(values),
            rates=tuple(rates),
            prescribed_mass_flux=prescribed_flux,
            compatibility_defect=compatibility_defect,
            finite=finite,
            compatible=compatible,
            successful=successful,
            boundary_id=self.prepared_id,
            stage_id=canonical_fingerprint(
                {"kind": "mac-boundary-stage", "boundaries": self.prepared_id}
            ),
        )

    def homogeneous_stage(self, /) -> MACBoundaryStageData:
        dtype = self.operators.pressure_space.dtype
        values = tuple(
            jnp.zeros(
                ((self.dimension,) + shape if boundary.kind in _VECTOR_KINDS else shape),
                dtype=dtype,
            )
            for boundary, shape in zip(self.sides, self.side_shapes, strict=True)
        )
        zero = jnp.asarray(0.0, dtype=dtype)
        true = jnp.asarray(True)
        return MACBoundaryStageData(
            values=values,
            rates=values,
            prescribed_mass_flux=zero,
            compatibility_defect=zero,
            finite=true,
            compatible=true,
            successful=true,
            boundary_id=self.prepared_id,
            stage_id=canonical_fingerprint(
                {"kind": "homogeneous-mac-boundary-stage", "boundaries": self.prepared_id}
            ),
        )

    def validate_stage(self, stage: MACBoundaryStageData, /) -> MACBoundaryStageData:
        if not isinstance(stage, MACBoundaryStageData):
            raise TypeError("stage must be MACBoundaryStageData.")
        if stage.boundary_id != self.prepared_id:
            raise ValueError("MAC boundary stage belongs to a different prepared plan.")
        return stage

    def _normal_target(
        self, position: int, stage: MACBoundaryStageData, /, *, rate: bool
    ) -> Array:
        boundary = self.sides[position]
        values = stage.rates if rate else stage.values
        value = values[position]
        return (
            value[self.side_axes[position]] if boundary.kind in _VECTOR_KINDS else value
        )

    def enforce(
        self, velocity: FaceVelocity, stage: MACBoundaryStageData, /
    ) -> FaceVelocity:
        stage_ = self.validate_stage(stage)
        values = list(self.operators.validate_velocity(velocity))
        for position, (boundary, axis, side_index) in enumerate(
            zip(self.sides, self.side_axes, self.side_indices, strict=True)
        ):
            if boundary.kind in _ESSENTIAL_KINDS:
                target = self._normal_target(position, stage_, rate=False)
                safe_target = jnp.where(stage_.successful, target, 0.0)
                values[axis] = _set_axis_boundary(
                    values[axis], axis, side_index, safe_target
                )
        return tuple(values)

    def enforce_rate(
        self, rate: FaceVelocity, stage: MACBoundaryStageData, /
    ) -> FaceVelocity:
        stage_ = self.validate_stage(stage)
        values = list(self.operators.validate_velocity(rate))
        for position, (boundary, axis, side_index) in enumerate(
            zip(self.sides, self.side_axes, self.side_indices, strict=True)
        ):
            if boundary.kind in _ESSENTIAL_KINDS:
                target = self._normal_target(position, stage_, rate=True)
                safe_target = jnp.where(stage_.successful, target, 0.0)
                values[axis] = _set_axis_boundary(
                    values[axis], axis, side_index, safe_target
                )
        return tuple(values)

    def homogeneous_rate(self, rate: FaceVelocity, /) -> FaceVelocity:
        values = list(self.operators.validate_velocity(rate))
        for boundary, axis, side_index in zip(
            self.sides, self.side_axes, self.side_indices, strict=True
        ):
            if boundary.kind in _ESSENTIAL_KINDS:
                values[axis] = _set_axis_boundary(values[axis], axis, side_index, 0.0)
        return tuple(values)

    def defect(self, velocity: FaceVelocity, stage: MACBoundaryStageData, /) -> Array:
        stage_ = self.validate_stage(stage)
        values = self.operators.validate_velocity(velocity)
        defects: list[Array] = []
        for position, (boundary, axis, side_index) in enumerate(
            zip(self.sides, self.side_axes, self.side_indices, strict=True)
        ):
            if boundary.kind in _ESSENTIAL_KINDS:
                defects.append(
                    jnp.max(
                        jnp.abs(
                            _axis_boundary(values[axis], axis, side_index)
                            - self._normal_target(position, stage_, rate=False)
                        )
                    )
                )
        return (
            jnp.max(jnp.stack(tuple(defects)))
            if defects
            else jnp.asarray(0.0, dtype=self.operators.pressure_space.dtype)
        )

    def tangential_dirichlet(self, axis: int, side: MACBoundarySideName, /) -> bool:
        return self.side_kind(axis, side) in ("no-slip", "velocity-inflow")

    def tangential_value(
        self,
        axis: int,
        side: MACBoundarySideName,
        component: int,
        stage: MACBoundaryStageData,
        /,
        *,
        homogeneous: bool,
    ) -> Array:
        position = self._side_position(axis, side)
        self.validate_stage(stage)
        if not self.tangential_dirichlet(axis, side):
            return jnp.asarray(0.0, dtype=self.operators.pressure_space.dtype)
        if homogeneous:
            return jnp.zeros(
                self.side_shapes[position], dtype=self.operators.pressure_space.dtype
            )
        return stage.values[position][component]

    def pressure_gradient(
        self,
        pressure: ArrayLike,
        stage: MACBoundaryStageData | None,
        /,
        *,
        homogeneous: bool,
    ) -> FaceVelocity:
        value = self.operators.validate_pressure(pressure)
        if not homogeneous:
            if stage is None:
                raise ValueError(
                    "Inhomogeneous MAC pressure gradients require stage data."
                )
            stage_ = self.validate_stage(stage)
        else:
            stage_ = stage
        gradient = list(self.operators.gradient(value))
        grid = self.operators.discretization.grid
        for position, (boundary, axis, side_index) in enumerate(
            zip(self.sides, self.side_axes, self.side_indices, strict=True)
        ):
            if boundary.kind not in _OPEN_KINDS:
                continue
            moved = jnp.moveaxis(value, axis, 0)
            structured_axis = grid.structured_axes[axis]
            if homogeneous:
                datum = jnp.zeros(self.side_shapes[position], dtype=value.dtype)
            elif boundary.kind == "pressure-outlet":
                datum = stage_.values[position]
            else:
                traction = stage_.values[position]
                outward_sign = -1.0 if side_index == 0 else 1.0
                datum = -outward_sign * traction[axis]
            if not homogeneous:
                datum = jnp.where(stage_.successful, datum, 0.0)
            if side_index == 0:
                derivative = (moved[0] - datum) / (
                    structured_axis.interval_centers[0] - structured_axis.bounds[0]
                )
            else:
                derivative = (datum - moved[-1]) / (
                    structured_axis.bounds[1] - structured_axis.interval_centers[-1]
                )
            gradient[axis] = _set_axis_boundary(
                gradient[axis], axis, side_index, derivative
            )
        return tuple(gradient)

    def integrated_mass_flux(self, velocity: FaceVelocity, /) -> Array:
        divergence = self.operators.divergence(velocity)
        volumes = self.operators.discretization.cell_volumes.astype(divergence.dtype)
        return jnp.sum(volumes * divergence)


__all__ = [
    "MACBoundaryKind",
    "MACBoundaryPlan",
    "MACBoundaryProvider",
    "MACBoundaryProviderFunction",
    "MACBoundarySide",
    "MACBoundarySideName",
    "MACBoundaryStageData",
    "MACPressureClosureKind",
    "PreparedMACBoundaryPlan",
]
