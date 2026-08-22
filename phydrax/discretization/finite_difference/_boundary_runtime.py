#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any, Literal, TypeAlias

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState


GhostConditionKind: TypeAlias = Literal[
    "periodic",
    "dirichlet",
    "neumann",
    "robin",
]
CornerPolicy: TypeAlias = Literal["error", "axis_separable", "tensor_product"]


class BoundaryStageContext(StrictModule):
    """One time-stage state, runtime arguments, and structured coordinates."""

    time: Array
    state: Any
    args: Any
    axis_names: tuple[str, ...] = eqx.field(static=True)
    coordinates: tuple[Array, ...]
    stage_id: str = eqx.field(static=True)

    def __init__(
        self,
        time: ArrayLike,
        state: Any,
        args: Any,
        axis_names: tuple[str, ...],
        coordinates: tuple[ArrayLike, ...],
        /,
        *,
        stage_id: str,
    ):
        names = tuple(str(value) for value in axis_names)
        coordinate_values = tuple(jnp.asarray(value) for value in coordinates)
        identifier = str(stage_id)
        if (
            not names
            or len(names) != len(coordinate_values)
            or len(set(names)) != len(names)
            or not identifier
        ):
            raise ValueError("Boundary stage axes, coordinates, and ID must align.")
        if any(value.ndim != 1 or value.size == 0 for value in coordinate_values):
            raise ValueError("Boundary stage coordinates must be non-empty vectors.")
        self.time = jnp.asarray(time)
        self.state = state
        self.args = args
        self.axis_names = names
        self.coordinates = coordinate_values
        self.stage_id = identifier

    def coordinate(self, axis: str, /) -> Array:
        if axis not in self.axis_names:
            raise KeyError(f"Unknown boundary stage axis {axis!r}.")
        return self.coordinates[self.axis_names.index(axis)]


def _boundary_shape(shape: tuple[int, ...], axis: int, /) -> tuple[int, ...]:
    return shape[:axis] + shape[axis + 1 :]


def _broadcast_boundary(
    value: ArrayLike,
    shape: tuple[int, ...],
    axis: int,
    /,
) -> Array:
    boundary = jnp.asarray(value)
    expected = _boundary_shape(shape, axis)
    if boundary.shape == ():
        boundary = jnp.broadcast_to(boundary, expected)
    if boundary.shape != expected:
        raise ValueError(f"Boundary data must have shape {expected} or be scalar.")
    return jnp.expand_dims(boundary, axis=axis)


class CellGhostBoundary(StrictModule, NonTrainableState):
    """Executable arbitrary-depth ghost realization for uniform cell fields."""

    axis: int = eqx.field(static=True)
    lower_kind: GhostConditionKind = eqx.field(static=True)
    upper_kind: GhostConditionKind = eqx.field(static=True)
    spacing: float = eqx.field(static=True)
    lower_width: int = eqx.field(static=True)
    upper_width: int = eqx.field(static=True)
    lower_alpha: float = eqx.field(static=True)
    lower_beta: float = eqx.field(static=True)
    upper_alpha: float = eqx.field(static=True)
    upper_beta: float = eqx.field(static=True)
    runtime_id: str = eqx.field(static=True)

    def __init__(
        self,
        axis: int,
        lower_kind: GhostConditionKind,
        upper_kind: GhostConditionKind,
        spacing: float,
        /,
        *,
        lower_width: int = 1,
        upper_width: int = 1,
        lower_alpha: float = 1.0,
        lower_beta: float = 1.0,
        upper_alpha: float = 1.0,
        upper_beta: float = 1.0,
    ):
        axis_ = int(axis)
        allowed = ("periodic", "dirichlet", "neumann", "robin")
        if axis_ < 0 or lower_kind not in allowed or upper_kind not in allowed:
            raise ValueError("Invalid ghost boundary axis or condition kind.")
        if (lower_kind == "periodic") != (upper_kind == "periodic"):
            raise ValueError("Periodicity must be declared on both boundary sides.")
        spacing_ = float(spacing)
        lower_width_ = int(lower_width)
        upper_width_ = int(upper_width)
        coefficients = tuple(
            float(value) for value in (lower_alpha, lower_beta, upper_alpha, upper_beta)
        )
        if (
            not np.isfinite(spacing_)
            or spacing_ <= 0.0
            or lower_width_ < 0
            or upper_width_ < 0
            or lower_width_ + upper_width_ == 0
            or any(not np.isfinite(value) for value in coefficients)
        ):
            raise ValueError("Ghost widths, spacing, and Robin coefficients are invalid.")
        for side, width, kind, alpha, beta in (
            ("lower", lower_width_, lower_kind, coefficients[0], coefficients[1]),
            ("upper", upper_width_, upper_kind, coefficients[2], coefficients[3]),
        ):
            orientation = -1.0 if side == "lower" else 1.0
            distances = (2.0 * np.arange(width) + 1.0) * spacing_
            denominators = 0.5 * alpha + orientation * beta / distances
            if kind == "robin" and np.any(np.abs(denominators) <= np.finfo(float).eps):
                raise ValueError("Robin ghost relation is singular.")
        if lower_kind == "dirichlet" and coefficients[0] == 0.0:
            raise ValueError("Lower Dirichlet coefficient must be nonzero.")
        if upper_kind == "dirichlet" and coefficients[2] == 0.0:
            raise ValueError("Upper Dirichlet coefficient must be nonzero.")
        if lower_kind == "neumann" and coefficients[1] == 0.0:
            raise ValueError("Lower Neumann coefficient must be nonzero.")
        if upper_kind == "neumann" and coefficients[3] == 0.0:
            raise ValueError("Upper Neumann coefficient must be nonzero.")
        self.axis = axis_
        self.lower_kind = lower_kind
        self.upper_kind = upper_kind
        self.spacing = spacing_
        self.lower_width = lower_width_
        self.upper_width = upper_width_
        self.lower_alpha = coefficients[0]
        self.lower_beta = coefficients[1]
        self.upper_alpha = coefficients[2]
        self.upper_beta = coefficients[3]
        self.runtime_id = canonical_fingerprint(
            {
                "kind": "cell-ghost-boundary",
                "axis": axis_,
                "lower_kind": lower_kind,
                "upper_kind": upper_kind,
                "spacing": spacing_,
                "widths": [lower_width_, upper_width_],
                "coefficients": list(coefficients),
            }
        )

    def _ghost_layers(
        self,
        interior_near: Array,
        value: Array,
        width: int,
        kind: GhostConditionKind,
        alpha: float,
        beta: float,
        side: Literal["lower", "upper"],
        /,
    ) -> Array:
        if width == 0:
            return interior_near
        distance_shape = [1] * interior_near.ndim
        distance_shape[self.axis] = width
        distances = (
            (2.0 * jnp.arange(width, dtype=interior_near.dtype) + 1.0) * self.spacing
        ).reshape(distance_shape)
        if kind == "dirichlet":
            return 2.0 * value / alpha - interior_near
        orientation = -1.0 if side == "lower" else 1.0
        if kind == "neumann":
            return interior_near + orientation * distances * value / beta
        if kind == "robin":
            numerator = (
                value - (0.5 * alpha - orientation * beta / distances) * interior_near
            )
            denominator = 0.5 * alpha + orientation * beta / distances
            return numerator / denominator
        raise ValueError("Periodic ghosts are handled by direct wrapping.")

    def fill(
        self,
        values: ArrayLike,
        lower_value: ArrayLike = 0.0,
        upper_value: ArrayLike = 0.0,
        /,
    ) -> Array:
        array = jnp.asarray(values)
        if self.axis >= array.ndim or array.shape[self.axis] < max(
            self.lower_width, self.upper_width
        ):
            raise ValueError("Ghost boundary axis is shorter than its halo depth.")
        if self.lower_kind == "periodic":
            lower_indices = jnp.arange(
                array.shape[self.axis] - self.lower_width,
                array.shape[self.axis],
            )
            upper_indices = jnp.arange(self.upper_width)
            lower_ghost = jnp.take(array, lower_indices, axis=self.axis)
            upper_ghost = jnp.take(array, upper_indices, axis=self.axis)
        else:
            lower_data = _broadcast_boundary(lower_value, array.shape, self.axis)
            upper_data = _broadcast_boundary(upper_value, array.shape, self.axis)
            lower_interior = jnp.take(
                array,
                jnp.arange(self.lower_width),
                axis=self.axis,
            )
            upper_interior = jnp.take(
                array,
                array.shape[self.axis] - 1 - jnp.arange(self.upper_width),
                axis=self.axis,
            )
            lower_near = self._ghost_layers(
                lower_interior,
                lower_data,
                self.lower_width,
                self.lower_kind,
                self.lower_alpha,
                self.lower_beta,
                "lower",
            )
            lower_ghost = jnp.flip(lower_near, axis=self.axis)
            upper_ghost = self._ghost_layers(
                upper_interior,
                upper_data,
                self.upper_width,
                self.upper_kind,
                self.upper_alpha,
                self.upper_beta,
                "upper",
            )
        return jnp.concatenate((lower_ghost, array, upper_ghost), axis=self.axis)


class NodalBoundaryRuntime(StrictModule, NonTrainableState):
    """Strong nodal state and coordinate-derivative boundary realization."""

    axis: int = eqx.field(static=True)
    lower_kind: GhostConditionKind = eqx.field(static=True)
    upper_kind: GhostConditionKind = eqx.field(static=True)
    lower_alpha: float = eqx.field(static=True)
    lower_beta: float = eqx.field(static=True)
    upper_alpha: float = eqx.field(static=True)
    upper_beta: float = eqx.field(static=True)
    runtime_id: str = eqx.field(static=True)

    def __init__(
        self,
        axis: int,
        lower_kind: GhostConditionKind,
        upper_kind: GhostConditionKind,
        /,
        *,
        lower_alpha: float = 1.0,
        lower_beta: float = 1.0,
        upper_alpha: float = 1.0,
        upper_beta: float = 1.0,
    ):
        axis_ = int(axis)
        kinds = (lower_kind, upper_kind)
        coefficients = tuple(
            float(value) for value in (lower_alpha, lower_beta, upper_alpha, upper_beta)
        )
        if (
            axis_ < 0
            or any(kind not in ("dirichlet", "neumann", "robin") for kind in kinds)
            or any(not np.isfinite(value) for value in coefficients)
        ):
            raise ValueError("Nodal boundary axis, kinds, or coefficients are invalid.")
        if lower_kind == "dirichlet" and coefficients[0] == 0.0:
            raise ValueError("Lower Dirichlet coefficient must be nonzero.")
        if upper_kind == "dirichlet" and coefficients[2] == 0.0:
            raise ValueError("Upper Dirichlet coefficient must be nonzero.")
        if lower_kind in ("neumann", "robin") and coefficients[1] == 0.0:
            raise ValueError("Lower derivative coefficient must be nonzero.")
        if upper_kind in ("neumann", "robin") and coefficients[3] == 0.0:
            raise ValueError("Upper derivative coefficient must be nonzero.")
        self.axis = axis_
        self.lower_kind = lower_kind
        self.upper_kind = upper_kind
        self.lower_alpha = coefficients[0]
        self.lower_beta = coefficients[1]
        self.upper_alpha = coefficients[2]
        self.upper_beta = coefficients[3]
        self.runtime_id = canonical_fingerprint(
            {
                "kind": "nodal-boundary-runtime",
                "axis": axis_,
                "conditions": list(kinds),
                "coefficients": list(coefficients),
            }
        )

    def _set_side(
        self,
        values: Array,
        side: Literal["lower", "upper"],
        replacement: ArrayLike,
        /,
    ) -> Array:
        index = [slice(None)] * values.ndim
        index[self.axis] = (
            slice(0, 1)
            if side == "lower"
            else slice(
                values.shape[self.axis] - 1,
                values.shape[self.axis],
            )
        )
        boundary = _broadcast_boundary(replacement, values.shape, self.axis)
        return values.at[tuple(index)].set(boundary)

    def apply_state(
        self,
        values: ArrayLike,
        lower_value: ArrayLike,
        upper_value: ArrayLike,
        /,
    ) -> Array:
        array = jnp.asarray(values)
        if self.axis >= array.ndim or array.shape[self.axis] < 2:
            raise ValueError("Nodal boundary axis must have at least two entries.")
        result = array
        if self.lower_kind == "dirichlet":
            result = self._set_side(
                result,
                "lower",
                jnp.asarray(lower_value) / self.lower_alpha,
            )
        if self.upper_kind == "dirichlet":
            result = self._set_side(
                result,
                "upper",
                jnp.asarray(upper_value) / self.upper_alpha,
            )
        return result

    def apply_coordinate_derivative(
        self,
        derivative: ArrayLike,
        state: ArrayLike,
        lower_value: ArrayLike,
        upper_value: ArrayLike,
        /,
    ) -> Array:
        derivative_ = jnp.asarray(derivative)
        state_ = jnp.asarray(state)
        if derivative_.shape != state_.shape:
            raise ValueError("Nodal derivative and state shapes must match.")
        result = derivative_
        lower_state = jnp.take(state_, 0, axis=self.axis)
        upper_state = jnp.take(
            state_,
            state_.shape[self.axis] - 1,
            axis=self.axis,
        )
        if self.lower_kind != "dirichlet":
            lower_replacement = (
                jnp.asarray(lower_value) / self.lower_beta
                if self.lower_kind == "neumann"
                else (jnp.asarray(lower_value) - self.lower_alpha * lower_state)
                / self.lower_beta
            )
            result = self._set_side(result, "lower", lower_replacement)
        if self.upper_kind != "dirichlet":
            upper_replacement = (
                jnp.asarray(upper_value) / self.upper_beta
                if self.upper_kind == "neumann"
                else (jnp.asarray(upper_value) - self.upper_alpha * upper_state)
                / self.upper_beta
            )
            result = self._set_side(result, "upper", upper_replacement)
        return result

    def apply_time_derivative(
        self,
        derivative: ArrayLike,
        lower_rate: ArrayLike,
        upper_rate: ArrayLike,
        /,
    ) -> Array:
        result = jnp.asarray(derivative)
        if self.lower_kind == "dirichlet":
            result = self._set_side(
                result,
                "lower",
                jnp.asarray(lower_rate) / self.lower_alpha,
            )
        if self.upper_kind == "dirichlet":
            result = self._set_side(
                result,
                "upper",
                jnp.asarray(upper_rate) / self.upper_alpha,
            )
        return result


class BoundaryWorkspace(StrictModule):
    """Stage-cached original, axis-separable, and optional corner-filled values."""

    original_values: Array
    axis_names: tuple[str, ...] = eqx.field(static=True)
    axis_values: tuple[Array, ...]
    lower_values: tuple[Array, ...]
    upper_values: tuple[Array, ...]
    tensor_values: Array | None
    runtime_ids: tuple[str, ...] = eqx.field(static=True)
    stage_id: str = eqx.field(static=True)
    workspace_id: str = eqx.field(static=True)

    def __init__(
        self,
        original_values: ArrayLike,
        axis_names: tuple[str, ...],
        axis_values: tuple[ArrayLike, ...],
        lower_values: tuple[ArrayLike, ...],
        upper_values: tuple[ArrayLike, ...],
        tensor_values: ArrayLike | None,
        runtime_ids: tuple[str, ...],
        stage_id: str,
        /,
    ):
        original = jnp.asarray(original_values)
        names = tuple(str(value) for value in axis_names)
        values = tuple(jnp.asarray(value) for value in axis_values)
        lower = tuple(jnp.asarray(value) for value in lower_values)
        upper = tuple(jnp.asarray(value) for value in upper_values)
        identifiers = tuple(str(value) for value in runtime_ids)
        stage = str(stage_id)
        if (
            len(names) != len(values)
            or len(names) != len(lower)
            or len(names) != len(upper)
            or len(names) != len(identifiers)
            or len(set(names)) != len(names)
            or any(not value for value in names + identifiers)
            or not stage
        ):
            raise ValueError(
                "Boundary workspace axes, values, targets, runtimes, and stage must align."
            )
        self.original_values = original
        self.axis_names = names
        self.axis_values = values
        self.lower_values = lower
        self.upper_values = upper
        self.tensor_values = None if tensor_values is None else jnp.asarray(tensor_values)
        self.runtime_ids = identifiers
        self.stage_id = stage
        self.workspace_id = canonical_fingerprint(
            {
                "kind": "boundary-workspace",
                "runtime_ids": list(identifiers),
                "stage_id": stage,
                "original_shape": list(original.shape),
                "axis_shapes": [list(value.shape) for value in values],
                "tensor_shape": (
                    None if self.tensor_values is None else list(self.tensor_values.shape)
                ),
            }
        )

    def for_axis(self, axis: str, /) -> Array:
        if axis not in self.axis_names:
            raise KeyError(f"Boundary workspace has no axis {axis!r}.")
        return self.axis_values[self.axis_names.index(axis)]

    def target_values(self, axis: str, /) -> tuple[Array, Array]:
        if axis not in self.axis_names:
            raise KeyError(f"Boundary workspace has no axis {axis!r}.")
        index = self.axis_names.index(axis)
        return self.lower_values[index], self.upper_values[index]


class ConformingInterfaceRuntime(StrictModule, NonTrainableState):
    """Symmetric trace correction for conforming field and outward-flux jumps."""

    field_name: str = eqx.field(static=True)
    axis: str = eqx.field(static=True)
    runtime_id: str = eqx.field(static=True)

    def __init__(self, field_name: str, axis: str, /):
        field = str(field_name)
        axis_ = str(axis)
        if not field or not axis_:
            raise ValueError("Interface runtime field and axis must be non-empty.")
        self.field_name = field
        self.axis = axis_
        self.runtime_id = canonical_fingerprint(
            {
                "kind": "conforming-interface-runtime",
                "field": field,
                "axis": axis_,
            }
        )

    def couple(
        self,
        left_value: ArrayLike,
        right_value: ArrayLike,
        left_outward_flux: ArrayLike,
        right_outward_flux: ArrayLike,
        /,
        *,
        field_jump: ArrayLike = 0.0,
        flux_jump: ArrayLike = 0.0,
    ) -> tuple[Array, Array, Array, Array]:
        left = jnp.asarray(left_value)
        right = jnp.asarray(right_value)
        left_flux = jnp.asarray(left_outward_flux)
        right_flux = jnp.asarray(right_outward_flux)
        if (
            left.shape != right.shape
            or left_flux.shape != right_flux.shape
            or left.shape != left_flux.shape
        ):
            raise ValueError("Conforming interface traces must have identical shapes.")
        field_jump_ = jnp.broadcast_to(jnp.asarray(field_jump), left.shape)
        flux_jump_ = jnp.broadcast_to(jnp.asarray(flux_jump), left.shape)
        field_average = 0.5 * (left + right)
        corrected_left = field_average - 0.5 * field_jump_
        corrected_right = field_average + 0.5 * field_jump_
        flux_mismatch = left_flux + right_flux - flux_jump_
        corrected_left_flux = left_flux - 0.5 * flux_mismatch
        corrected_right_flux = right_flux - 0.5 * flux_mismatch
        return (
            corrected_left,
            corrected_right,
            corrected_left_flux,
            corrected_right_flux,
        )


__all__ = [
    "BoundaryStageContext",
    "BoundaryWorkspace",
    "CellGhostBoundary",
    "ConformingInterfaceRuntime",
    "CornerPolicy",
    "GhostConditionKind",
    "NodalBoundaryRuntime",
]
