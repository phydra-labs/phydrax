#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable, Sequence
from math import prod
from typing import Any, cast, Literal, TypeAlias

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._strict import StrictModule
from ..metrix import AbstractStateGeometry, EuclideanStateGeometry
from ._layout import InputLayout


DAERole: TypeAlias = Literal["differential", "algebraic"]
AutonomousDifferentialAlgebraicResidual: TypeAlias = Callable[
    [Array, Array, Array, Any], ArrayLike
]
InputDifferentialAlgebraicResidual: TypeAlias = Callable[
    [Array, Array, Array, Array, Any], ArrayLike
]
DifferentialAlgebraicResidual: TypeAlias = (
    AutonomousDifferentialAlgebraicResidual | InputDifferentialAlgebraicResidual
)


def _identifier(value: str, owner: str, /) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{owner} must be a non-empty string.")
    return value


def _shape(value: Sequence[int], owner: str, /) -> tuple[int, ...]:
    shape = tuple(int(size) for size in value)
    if not shape or any(size <= 0 for size in shape):
        raise ValueError(f"{owner} dimensions must be positive.")
    return shape


def _roles(value: Sequence[DAERole], owner: str, /) -> tuple[DAERole, ...]:
    roles = tuple(value)
    if not roles:
        raise ValueError(f"{owner} must not be empty.")
    if any(role not in ("differential", "algebraic") for role in roles):
        raise ValueError(f"{owner} entries must be 'differential' or 'algebraic'.")
    return roles


def _inexact(value: ArrayLike, /) -> Array:
    array = jnp.asarray(value)
    return array if jnp.issubdtype(array.dtype, jnp.inexact) else array.astype(float)


def _positive_scale(
    value: ArrayLike | None, shape: tuple[int, ...], owner: str, /
) -> Array:
    scale = jnp.ones(shape) if value is None else _inexact(value)
    scale = jnp.broadcast_to(scale, shape)
    if jnp.issubdtype(scale.dtype, jnp.complexfloating):
        raise TypeError(f"{owner} must be real-valued.")
    return eqx.error_if(
        scale,
        jnp.any(~jnp.isfinite(scale)) | jnp.any(scale <= 0.0),
        f"{owner} must be finite and positive.",
    )


class DAEStructure(StrictModule):
    """Differential/algebraic roles broadcast along one state component axis."""

    variable_roles: tuple[DAERole, ...] = eqx.field(static=True)
    equation_roles: tuple[DAERole, ...] = eqx.field(static=True)
    component_axis: int | None = eqx.field(static=True)

    def __init__(
        self,
        variable_roles: Sequence[DAERole],
        /,
        *,
        equation_roles: Sequence[DAERole] | None = None,
        component_axis: int | None = -1,
    ):
        variables = _roles(variable_roles, "variable_roles")
        equations = (
            variables
            if equation_roles is None
            else _roles(equation_roles, "equation_roles")
        )
        if len(equations) != len(variables):
            raise ValueError("Variable and equation role counts must match.")
        if variables.count("differential") != equations.count("differential"):
            raise ValueError(
                "Differential variable and equation component counts must match."
            )
        self.variable_roles = variables
        self.equation_roles = equations
        self.component_axis = None if component_axis is None else int(component_axis)

    def resolved_axis(self, state_shape: Sequence[int], /) -> int | None:
        shape = tuple(int(size) for size in state_shape)
        if self.component_axis is None:
            if len(self.variable_roles) != 1:
                raise ValueError(
                    "DAE component_axis=None requires one role applied to the full state."
                )
            return None
        rank = len(shape)
        axis = (
            self.component_axis + rank if self.component_axis < 0 else self.component_axis
        )
        if axis < 0 or axis >= rank:
            raise ValueError(
                f"DAE component_axis {self.component_axis} is invalid for shape {shape}."
            )
        if shape[axis] != len(self.variable_roles):
            raise ValueError(
                "DAE role count must match the selected component axis: "
                f"got {len(self.variable_roles)} roles for axis size {shape[axis]}."
            )
        return axis

    def _mask(
        self,
        state_shape: Sequence[int],
        roles: tuple[DAERole, ...],
        selected: DAERole,
        /,
    ) -> Array:
        shape = tuple(int(size) for size in state_shape)
        axis = self.resolved_axis(shape)
        if axis is None:
            return jnp.full(shape, roles[0] == selected, dtype=bool)
        component_mask = jnp.asarray(
            tuple(role == selected for role in roles),
            dtype=bool,
        )
        broadcast_shape = [1] * len(shape)
        broadcast_shape[axis] = len(roles)
        return jnp.broadcast_to(component_mask.reshape(tuple(broadcast_shape)), shape)

    def differential_variable_mask(self, state_shape: Sequence[int], /) -> Array:
        return self._mask(state_shape, self.variable_roles, "differential")

    def algebraic_variable_mask(self, state_shape: Sequence[int], /) -> Array:
        return self._mask(state_shape, self.variable_roles, "algebraic")

    def differential_equation_mask(self, state_shape: Sequence[int], /) -> Array:
        return self._mask(state_shape, self.equation_roles, "differential")

    def algebraic_equation_mask(self, state_shape: Sequence[int], /) -> Array:
        return self._mask(state_shape, self.equation_roles, "algebraic")


class DifferentialAlgebraicSystem(StrictModule):
    """State-shaped implicit residual independent of initialization and integration."""

    residual: DifferentialAlgebraicResidual
    input_layout: InputLayout | None
    structure: DAEStructure
    state_scale: Array
    state_rate_scale: Array
    residual_scale: Array
    state_geometry: AbstractStateGeometry
    state_shape: tuple[int, ...] = eqx.field(static=True)
    system_id: str = eqx.field(static=True)

    def __init__(
        self,
        residual: DifferentialAlgebraicResidual,
        /,
        *,
        state_shape: Sequence[int],
        structure: DAEStructure,
        input_layout: InputLayout | None = None,
        state_scale: ArrayLike | None = None,
        state_rate_scale: ArrayLike | None = None,
        residual_scale: ArrayLike | None = None,
        state_geometry: AbstractStateGeometry | None = None,
        system_id: str,
    ):
        if not callable(residual):
            raise TypeError("DifferentialAlgebraicSystem residual must be callable.")
        if not isinstance(structure, DAEStructure):
            raise TypeError("structure must be a DAEStructure.")
        if input_layout is not None and not isinstance(input_layout, InputLayout):
            raise TypeError("input_layout must be an InputLayout or None.")
        shape = _shape(state_shape, "DifferentialAlgebraicSystem state_shape")
        structure.resolved_axis(shape)
        geometry = EuclideanStateGeometry() if state_geometry is None else state_geometry
        if not isinstance(geometry, AbstractStateGeometry):
            raise TypeError("state_geometry must be an AbstractStateGeometry or None.")
        if not geometry.trivial:
            raise ValueError(
                "DifferentialAlgebraicSystem currently requires Euclidean state "
                "geometry; manifold BDF needs independent local tangent coordinates."
            )
        resolved_state_scale = _positive_scale(state_scale, shape, "state_scale")
        resolved_rate_scale = (
            resolved_state_scale
            if state_rate_scale is None
            else _positive_scale(state_rate_scale, shape, "state_rate_scale")
        )
        self.residual = residual
        self.structure = structure
        self.input_layout = input_layout
        self.state_scale = resolved_state_scale
        self.state_rate_scale = resolved_rate_scale
        self.residual_scale = _positive_scale(
            residual_scale,
            shape,
            "residual_scale",
        )
        self.state_geometry = geometry
        self.state_shape = shape
        self.system_id = _identifier(
            system_id,
            "DifferentialAlgebraicSystem system_id",
        )

    @property
    def state_size(self) -> int:
        return prod(self.state_shape)

    def evaluate(
        self,
        time: ArrayLike,
        state: ArrayLike,
        state_rate: ArrayLike,
        args: Any = None,
        /,
        *,
        inputs: ArrayLike | None = None,
    ) -> Array:
        time_array = jnp.asarray(time)
        if time_array.shape != () or jnp.issubdtype(
            time_array.dtype,
            jnp.complexfloating,
        ):
            raise ValueError("DAE residual time must be one real scalar.")
        state_array = _inexact(state)
        rate_array = _inexact(state_rate)
        if state_array.shape != self.state_shape:
            raise ValueError(
                f"state must have shape {self.state_shape}; got {state_array.shape}."
            )
        if rate_array.shape != self.state_shape:
            raise ValueError(
                f"state_rate must have shape {self.state_shape}; got {rate_array.shape}."
            )
        if state_array.dtype != rate_array.dtype:
            raise TypeError("state and state_rate must have the same dtype.")
        if self.input_layout is None:
            if inputs is not None:
                raise ValueError(
                    "An autonomous DifferentialAlgebraicSystem does not accept inputs."
                )
            residual = cast(AutonomousDifferentialAlgebraicResidual, self.residual)
            value = _inexact(residual(time_array, state_array, rate_array, args))
        else:
            if inputs is None:
                raise ValueError(
                    "This DifferentialAlgebraicSystem requires explicit inputs."
                )
            input_array = _inexact(inputs)
            if input_array.shape != self.input_layout.shape:
                raise ValueError(
                    f"inputs must have shape {self.input_layout.shape}; "
                    f"got {input_array.shape}."
                )
            residual = cast(InputDifferentialAlgebraicResidual, self.residual)
            value = _inexact(
                residual(time_array, state_array, rate_array, input_array, args)
            )
        if value.shape != self.state_shape:
            raise ValueError(
                "DifferentialAlgebraicSystem residual returned shape "
                f"{value.shape}; expected {self.state_shape}."
            )
        return value

    def scaled_residual(
        self,
        time: ArrayLike,
        state: ArrayLike,
        state_rate: ArrayLike,
        args: Any = None,
        /,
        *,
        inputs: ArrayLike | None = None,
    ) -> Array:
        residual = self.evaluate(time, state, state_rate, args, inputs=inputs)
        return residual / self.residual_scale.astype(residual.dtype)

    def __call__(
        self,
        time: ArrayLike,
        state: ArrayLike,
        state_rate: ArrayLike,
        args: Any = None,
        /,
        *,
        inputs: ArrayLike | None = None,
    ) -> Array:
        return self.evaluate(time, state, state_rate, args, inputs=inputs)

    @classmethod
    def from_mass_matrix(
        cls,
        mass_matrix: Any,
        vector_field: Callable[..., ArrayLike],
        /,
        *,
        state_shape: Sequence[int],
        structure: DAEStructure,
        input_layout: InputLayout | None = None,
        state_scale: ArrayLike | None = None,
        state_rate_scale: ArrayLike | None = None,
        residual_scale: ArrayLike | None = None,
        state_geometry: AbstractStateGeometry | None = None,
        system_id: str,
    ) -> "DifferentialAlgebraicSystem":
        """Construct ``M(t, state, args) @ state_rate - f(t, state, args) = 0``."""
        from ..linalg import AbstractLinearOperator

        if not callable(vector_field):
            raise TypeError("vector_field must be callable.")
        if input_layout is not None and not isinstance(input_layout, InputLayout):
            raise TypeError("input_layout must be an InputLayout or None.")
        shape = _shape(state_shape, "Mass-matrix DAE state_shape")
        size = prod(shape)
        constant_operator = isinstance(mass_matrix, AbstractLinearOperator)
        dynamic_mass = callable(mass_matrix) and not constant_operator
        constant_array = None
        if not dynamic_mass and not constant_operator:
            constant_array = _inexact(mass_matrix)
            if constant_array.shape != (size, size):
                raise ValueError(
                    f"mass_matrix must have shape {(size, size)}; "
                    f"got {constant_array.shape}."
                )

        def apply_mass(time, state, state_rate, args):
            resolved_mass = (
                mass_matrix(time, state, args) if dynamic_mass else mass_matrix
            )
            if isinstance(resolved_mass, AbstractLinearOperator):
                return resolved_mass.mv(state_rate)
            matrix = (
                constant_array if constant_array is not None else _inexact(resolved_mass)
            )
            if matrix.shape != (size, size):
                raise ValueError(
                    f"Dynamic mass matrix must have shape {(size, size)}; "
                    f"got {matrix.shape}."
                )
            return (matrix @ state_rate.reshape((size,))).reshape(shape)

        def validate_drift(value):
            drift = _inexact(value)
            if drift.shape != shape:
                raise ValueError(
                    f"vector_field must return shape {shape}; got {drift.shape}."
                )
            return drift

        if input_layout is None:

            def residual(time, state, state_rate, args):
                return apply_mass(time, state, state_rate, args) - validate_drift(
                    vector_field(time, state, args)
                )

        else:

            def residual(time, state, state_rate, inputs, args):
                return apply_mass(time, state, state_rate, args) - validate_drift(
                    vector_field(time, state, inputs, args)
                )

        return cls(
            residual,
            state_shape=shape,
            structure=structure,
            input_layout=input_layout,
            state_scale=state_scale,
            state_rate_scale=state_rate_scale,
            residual_scale=residual_scale,
            state_geometry=state_geometry,
            system_id=system_id,
        )


__all__ = [
    "AutonomousDifferentialAlgebraicResidual",
    "DAERole",
    "DAEStructure",
    "DifferentialAlgebraicResidual",
    "DifferentialAlgebraicSystem",
    "InputDifferentialAlgebraicResidual",
]
