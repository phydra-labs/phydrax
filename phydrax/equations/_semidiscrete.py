#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from typing import Any, Literal

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._strict import StrictModule
from ._ir import PDEExpression, PDEField, PDEProblemIR
from ._validate import infer_expression_type, validate_pde_ir


SemidiscreteCompilationMethod = Literal["auto", "direct", "semilinear"]
ResolvedSemidiscreteMethod = Literal[
    "direct",
    "semilinear-matrix-free",
    "semilinear-spectral",
]


def _stable_id(*parts: str) -> str:
    digest = hashlib.sha256()
    for part in parts:
        encoded = str(part).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def _value_fingerprint(value: Any | None, /) -> str:
    if value is None:
        return "none"
    array = np.asarray(value)
    if array.dtype.kind not in "biuf":
        raise TypeError("Compiled PDE parameter values must be real numeric arrays.")
    if np.any(~np.isfinite(array)):
        raise ValueError("Compiled PDE parameter values must be finite.")
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("utf-8"))
    digest.update(repr(tuple(int(size) for size in array.shape)).encode("utf-8"))
    digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


class SemidiscreteFieldLayout(StrictModule):
    """Static packing of PDE fields after a leading spatial discretization."""

    field_names: tuple[str, ...] = eqx.field(static=True)
    component_counts: tuple[int, ...] = eqx.field(static=True)
    scalar_fields: tuple[bool, ...] = eqx.field(static=True)
    component_offsets: tuple[int, ...] = eqx.field(static=True)
    spatial_shape: tuple[int, ...] = eqx.field(static=True)
    state_shape: tuple[int, ...] = eqx.field(static=True)
    squeezed: bool = eqx.field(static=True)
    layout_id: str = eqx.field(static=True)

    def __init__(
        self,
        fields: Sequence[PDEField],
        spatial_shape: Sequence[int],
        /,
    ):
        field_values = tuple(fields)
        if not field_values or any(not isinstance(field, PDEField) for field in field_values):
            raise TypeError("fields must be a non-empty sequence of PDEField objects.")
        names = tuple(field.name for field in field_values)
        if len(set(names)) != len(names):
            raise ValueError("Semidiscrete field names must be unique.")
        components = tuple(int(field.components) for field in field_values)
        scalar_fields = tuple(
            field.representation in ("scalar", "pseudoscalar")
            for field in field_values
        )
        shape = tuple(int(size) for size in spatial_shape)
        if not shape or any(size <= 0 for size in shape):
            raise ValueError("spatial_shape must contain positive dimensions.")
        offsets: list[int] = []
        offset = 0
        for count in components:
            offsets.append(offset)
            offset += count
        squeezed = (
            len(field_values) == 1
            and components == (1,)
            and scalar_fields == (True,)
        )
        state_shape = shape if squeezed else shape + (offset,)
        self.field_names = names
        self.component_counts = components
        self.scalar_fields = scalar_fields
        self.component_offsets = tuple(offsets)
        self.spatial_shape = shape
        self.state_shape = state_shape
        self.squeezed = squeezed
        self.layout_id = _stable_id(
            "semidiscrete-field-layout-v2",
            repr(tuple(zip(names, components, scalar_fields, strict=True))),
            repr(shape),
        )

    @property
    def total_components(self) -> int:
        return sum(self.component_counts)

    def _field_index(self, name: str, /) -> int:
        identifier = str(name)
        if identifier not in self.field_names:
            raise KeyError(f"Unknown semidiscrete field {identifier!r}.")
        return self.field_names.index(identifier)

    def field_shape(self, name: str, /) -> tuple[int, ...]:
        index = self._field_index(name)
        count = self.component_counts[index]
        if count == 1 and self.scalar_fields[index]:
            return self.spatial_shape
        return self.spatial_shape + (count,)

    def field(self, state: ArrayLike, name: str, /) -> Array:
        value = jnp.asarray(state)
        if tuple(value.shape) != self.state_shape:
            raise ValueError(
                f"Packed state must have shape {self.state_shape}; got {value.shape}."
            )
        index = self._field_index(name)
        if self.squeezed:
            return value
        offset = self.component_offsets[index]
        count = self.component_counts[index]
        if count == 1 and self.scalar_fields[index]:
            return value[..., offset]
        return value[..., offset : offset + count]

    def unpack(self, state: ArrayLike, /) -> dict[str, Array]:
        return {name: self.field(state, name) for name in self.field_names}

    def pack(self, fields: Mapping[str, ArrayLike], /) -> Array:
        if set(fields) != set(self.field_names):
            raise ValueError(
                "Packed field keys must exactly match "
                f"{self.field_names}; got {tuple(fields)}."
            )
        values: list[Array] = []
        for index, (name, count) in enumerate(
            zip(
                self.field_names,
                self.component_counts,
                strict=True,
            )
        ):
            value = jnp.asarray(fields[name])
            expected = self.field_shape(name)
            if tuple(value.shape) != expected:
                raise ValueError(
                    f"Field {name!r} must have shape {expected}; got {value.shape}."
                )
            values.append(
                value[..., None]
                if count == 1 and self.scalar_fields[index]
                else value
            )
        if self.squeezed:
            return values[0][..., 0]
        return jnp.concatenate(tuple(values), axis=-1)


class BoundaryLift(StrictModule):
    """Explicit lift whose subtraction leaves homogeneous boundary data."""

    value: Any
    time_derivative: Any | None
    field_name: str = eqx.field(static=True)
    lift_id: str = eqx.field(static=True)

    def __init__(
        self,
        field_name: str,
        value: ArrayLike | Any,
        /,
        *,
        lift_id: str,
        time_derivative: ArrayLike | Any | None = None,
    ):
        name = str(field_name)
        identifier = str(lift_id)
        if not name:
            raise ValueError("BoundaryLift field_name must be non-empty.")
        if not identifier:
            raise ValueError("BoundaryLift lift_id must be non-empty.")
        if callable(value) and time_derivative is None:
            raise ValueError(
                "Callable BoundaryLift values require an explicit time_derivative."
            )
        self.value = value
        self.time_derivative = time_derivative
        self.field_name = name
        self.lift_id = identifier

    def evaluate(self, time: Array, args: Any, /) -> Array:
        value = self.value(time, args) if callable(self.value) else self.value
        return jnp.asarray(value)

    def derivative(self, time: Array, args: Any, /) -> Array:
        if self.time_derivative is None:
            return jnp.zeros_like(self.evaluate(time, args))
        value = (
            self.time_derivative(time, args)
            if callable(self.time_derivative)
            else self.time_derivative
        )
        return jnp.asarray(value)


class _SemidiscreteEvaluator(StrictModule):
    layout: SemidiscreteFieldLayout
    discretization: Any
    boundary_lifts: tuple[BoundaryLift, ...]
    parameter_defaults: tuple[Any | None, ...]
    rhs_expressions: tuple[PDEExpression, ...] = eqx.field(static=True)
    parameter_names: tuple[str, ...] = eqx.field(static=True)
    parameter_components: tuple[int, ...] = eqx.field(static=True)
    parameter_functional: tuple[bool, ...] = eqx.field(static=True)
    spatial_coordinate_axes: tuple[tuple[str, tuple[int, ...]], ...] = eqx.field(
        static=True
    )
    time_coordinate: str = eqx.field(static=True)
    region_axes: tuple[tuple[str, tuple[int, ...]], ...] = eqx.field(static=True)

    def __init__(
        self,
        problem: PDEProblemIR,
        rhs_expressions: Sequence[PDEExpression],
        layout: SemidiscreteFieldLayout,
        discretization: Any,
        boundary_lifts: Sequence[BoundaryLift],
        parameter_defaults: Sequence[Any | None],
        spatial_coordinate_axes: Sequence[tuple[str, tuple[int, ...]]],
        time_coordinate: str,
        region_axes: Sequence[tuple[str, tuple[int, ...]]],
        /,
    ):
        self.layout = layout
        self.discretization = discretization
        self.boundary_lifts = tuple(boundary_lifts)
        self.parameter_defaults = tuple(parameter_defaults)
        self.rhs_expressions = tuple(rhs_expressions)
        self.parameter_names = tuple(parameter.name for parameter in problem.parameters)
        self.parameter_components = tuple(
            parameter.components for parameter in problem.parameters
        )
        self.parameter_functional = tuple(
            parameter.functional for parameter in problem.parameters
        )
        self.spatial_coordinate_axes = tuple(spatial_coordinate_axes)
        self.time_coordinate = str(time_coordinate)
        self.region_axes = tuple(region_axes)

    def _parameter(self, name: str, args: Any, /) -> Array:
        if args is not None and not isinstance(args, Mapping):
            raise TypeError("Semidiscrete PDE args must be a parameter mapping or None.")
        index = self.parameter_names.index(name)
        if args is not None and name in args:
            value = args[name]
        else:
            value = self.parameter_defaults[index]
        if value is None:
            raise KeyError(f"No value supplied for PDE parameter {name!r}.")
        array = jnp.asarray(value)
        if array.dtype.kind not in "biuf":
            raise TypeError(f"PDE parameter {name!r} must be real-valued.")
        components = self.parameter_components[index]
        functional = self.parameter_functional[index]
        if components == 1 and array.shape == (1,):
            return array[0]
        if components == 1:
            allowed = ((), self.layout.spatial_shape) if functional else ((),)
            if array.shape not in allowed:
                raise ValueError(
                    f"PDE parameter {name!r} must be scalar"
                    + (" or spatially field-valued" if functional else "")
                    + f"; got {array.shape}."
                )
        else:
            allowed = (
                ((components,), self.layout.spatial_shape + (components,))
                if functional
                else ((components,),)
            )
            if array.shape not in allowed:
                raise ValueError(
                    f"PDE parameter {name!r} must end in {components} components; "
                    f"got {array.shape}."
                )
        return array

    def _axes(self, coordinate: str, /) -> tuple[int, ...]:
        for name, axes in self.spatial_coordinate_axes:
            if name == coordinate:
                return axes
        raise ValueError(
            f"Coordinate {coordinate!r} is not a compiled spatial coordinate."
        )

    def _coordinate(self, name: str, time: Array, /) -> Array:
        if name == self.time_coordinate:
            return jnp.asarray(time)
        axes = self._axes(name)
        points = self.discretization.points
        if points is None:
            raise ValueError(
                "Coordinate expressions require a spatial discretization with points."
            )
        components = tuple(
            self.discretization.unflatten(points[:, axis]) for axis in axes
        )
        if len(components) == 1:
            return components[0]
        return jnp.stack(components, axis=-1)

    def _region(self, name: str, /) -> tuple[int, ...]:
        for region, axes in self.region_axes:
            if region == name:
                return axes
        raise ValueError(f"Region {name!r} is not a compiled spatial region.")

    def _lift(self, name: str, time: Array, args: Any, /) -> Array | None:
        for lift in self.boundary_lifts:
            if lift.field_name == name:
                value = lift.evaluate(time, args)
                expected = self.layout.field_shape(name)
                if tuple(value.shape) != expected:
                    raise ValueError(
                        f"Boundary lift {lift.lift_id!r} must have shape "
                        f"{expected}; got {value.shape}."
                    )
                return value
        return None

    def _physical_fields(
        self,
        time: Array,
        state: Array,
        args: Any,
        /,
    ) -> dict[str, Array]:
        fields = self.layout.unpack(state)
        for name in self.layout.field_names:
            lift = self._lift(name, time, args)
            if lift is not None:
                fields[name] = fields[name] + lift
        return fields

    def _expression_lift(
        self,
        expression: PDEExpression,
        time: Array,
        args: Any,
        /,
    ) -> Array | None:
        if expression.op == "field":
            assert expression.symbol is not None
            return self._lift(expression.symbol, time, args)
        if expression.op == "component":
            assert expression.axis is not None
            source = self._expression_lift(expression.args[0], time, args)
            return None if source is None else source[..., expression.axis]
        if expression.op == "negate":
            source = self._expression_lift(expression.args[0], time, args)
            return None if source is None else -source
        if expression.op == "add":
            lifts = tuple(
                self._expression_lift(argument, time, args)
                for argument in expression.args
            )
            present = tuple(lift for lift in lifts if lift is not None)
            if not present:
                return None
            result = present[0]
            for lift in present[1:]:
                result = result + lift
            return result
        if expression.op == "multiply":
            lifts = tuple(
                self._expression_lift(argument, time, args)
                for argument in expression.args
            )
            indices = tuple(
                index for index, lift in enumerate(lifts) if lift is not None
            )
            if len(indices) != 1:
                return None
            selected = indices[0]
            if any(
                _field_degree(argument) > 0
                for index, argument in enumerate(expression.args)
                if index != selected
            ):
                return None
            result = lifts[selected]
            result_components = self._components(expression.args[selected])
            for index, argument in enumerate(expression.args):
                if index == selected:
                    continue
                factor = self._evaluate(argument, time, args, {})
                factor_components = self._components(argument)
                left = self._align_semantic_scalar(
                    result,
                    result_components,
                    factor_components,
                    factor,
                )
                right = self._align_semantic_scalar(
                    factor,
                    factor_components,
                    result_components,
                    result,
                )
                result = left * right
                result_components = max(result_components, factor_components)
            return result
        if expression.op == "divide":
            source = self._expression_lift(expression.args[0], time, args)
            if source is None or _field_degree(expression.args[1]) > 0:
                return None
            denominator = self._evaluate(expression.args[1], time, args, {})
            numerator_components = self._components(expression.args[0])
            denominator_components = self._components(expression.args[1])
            numerator = self._align_semantic_scalar(
                source,
                numerator_components,
                denominator_components,
                denominator,
            )
            aligned_denominator = self._align_semantic_scalar(
                denominator,
                denominator_components,
                numerator_components,
                source,
            )
            return numerator / aligned_denominator
        if expression.op not in (
            "derivative",
            "gradient",
            "divergence",
            "curl",
            "laplacian",
        ):
            return None
        source = self._expression_lift(expression.args[0], time, args)
        if source is None:
            return None
        assert expression.coordinate is not None
        axes = self._axes(expression.coordinate)
        if expression.op == "derivative":
            axis = axes[0] if expression.axis is None else axes[expression.axis]
            return self._lift_partial(source, axis=axis, order=expression.order)
        if expression.op == "gradient":
            return jnp.stack(
                tuple(
                    self._lift_partial(source, axis=axis, order=1)
                    for axis in axes
                ),
                axis=-1,
            )
        if expression.op == "divergence":
            result = jnp.zeros_like(source[..., 0])
            for component, axis in enumerate(axes):
                result = result + self._lift_partial(
                    source[..., component],
                    axis=axis,
                    order=1,
                )
            return result
        if expression.op == "curl":
            first, second, third = axes
            return jnp.stack(
                (
                    self._lift_partial(source[..., 2], axis=second, order=1)
                    - self._lift_partial(source[..., 1], axis=third, order=1),
                    self._lift_partial(source[..., 0], axis=third, order=1)
                    - self._lift_partial(source[..., 2], axis=first, order=1),
                    self._lift_partial(source[..., 1], axis=first, order=1)
                    - self._lift_partial(source[..., 0], axis=second, order=1),
                ),
                axis=-1,
            )
        result = jnp.zeros_like(source)
        for axis in axes:
            result = result + self._lift_partial(source, axis=axis, order=2)
        return result

    def _spatially_neutral(self, expression: PDEExpression, /) -> bool:
        if expression.op == "constant":
            return True
        if expression.op == "parameter":
            assert expression.symbol is not None
            index = self.parameter_names.index(expression.symbol)
            return not self.parameter_functional[index]
        if expression.op == "coordinate":
            return expression.symbol == self.time_coordinate
        if expression.op == "field" or not expression.args:
            return False
        return all(
            self._spatially_neutral(argument) for argument in expression.args
        )

    def _lift_partial(
        self,
        value: Array,
        /,
        *,
        axis: int,
        order: int,
    ) -> Array:
        from ..operators.differential._array_ops import _fd_nth_derivative
        from ..solver._spatial import TensorGridDiscretization

        if not isinstance(self.discretization, TensorGridDiscretization):
            return self.discretization.partial_derivative(
                value,
                axis=axis,
                order=order,
            )
        basis = self.discretization.basis[axis]
        if basis in ("uniform", "fourier"):
            return self.discretization.partial_derivative(
                value,
                axis=axis,
                order=order,
            )
        spacing = self.discretization.axes[axis].nodes[1] - self.discretization.axes[
            axis
        ].nodes[0]
        return _fd_nth_derivative(
            value,
            dx=spacing,
            axis=axis,
            order=order,
            periodic=False,
        )

    def _components(self, expression: PDEExpression, /) -> int:
        if expression.op == "field":
            assert expression.symbol is not None
            index = self.layout.field_names.index(expression.symbol)
            return self.layout.component_counts[index]
        if expression.op == "parameter":
            assert expression.symbol is not None
            return self.parameter_components[self.parameter_names.index(expression.symbol)]
        if expression.op == "coordinate":
            assert expression.symbol is not None
            if expression.symbol == self.time_coordinate:
                return 1
            return len(self._axes(expression.symbol))
        if expression.op in ("component", "dot", "divergence"):
            return 1
        if expression.op == "gradient":
            assert expression.coordinate is not None
            return len(self._axes(expression.coordinate))
        if expression.op == "curl":
            return 3
        if not expression.args:
            return 1
        if expression.op == "multiply":
            return max(self._components(argument) for argument in expression.args)
        return self._components(expression.args[0])

    def _unknown_parities(self, components: int, /) -> tuple[tuple[int, ...], ...]:
        rank = len(self.discretization.state_shape)
        return tuple((2,) * rank for _ in range(components))

    def _toggle_parity(
        self,
        parity: tuple[int, ...],
        axis: int,
        /,
        *,
        gradient: bool,
        order: int = 1,
    ) -> tuple[int, ...]:
        from ..solver._spatial import TensorGridDiscretization

        if not isinstance(self.discretization, TensorGridDiscretization):
            return parity
        basis = self.discretization.basis[axis]
        output = list(parity)
        if basis in ("sine", "cosine"):
            if output[axis] in (0, 1) and order % 2:
                output[axis] = 1 - output[axis]
        elif basis == "uniform" and gradient:
            output[axis] = 1
        elif basis == "uniform" and order % 2:
            output[axis] = 2
        return tuple(output)

    def _parities(
        self,
        expression: PDEExpression,
        /,
    ) -> tuple[tuple[int, ...], ...] | None:
        rank = len(self.discretization.state_shape)
        components = self._components(expression)
        if expression.op == "field":
            return tuple((0,) * rank for _ in range(components))
        if expression.op == "coordinate":
            return None
        if expression.op == "constant":
            return None
        if expression.op == "parameter":
            return None
        argument_parities = tuple(
            self._parities(argument) for argument in expression.args
        )
        if expression.op == "component":
            assert expression.axis is not None
            source = argument_parities[0]
            return None if source is None else (source[expression.axis],)
        if expression.op == "derivative":
            source = argument_parities[0]
            if source is None:
                return None
            assert expression.coordinate is not None
            axes = self._axes(expression.coordinate)
            axis = axes[0] if expression.axis is None else axes[expression.axis]
            return tuple(
                self._toggle_parity(
                    parity,
                    axis,
                    gradient=False,
                    order=expression.order,
                )
                for parity in source
            )
        if expression.op == "gradient":
            source = argument_parities[0]
            if source is None:
                return None
            assert expression.coordinate is not None
            return tuple(
                self._toggle_parity(
                    source[0],
                    axis,
                    gradient=True,
                )
                for axis in self._axes(expression.coordinate)
            )
        if expression.op == "laplacian":
            return argument_parities[0]
        if expression.op == "divergence":
            source = argument_parities[0]
            if source is None:
                return None
            assert expression.coordinate is not None
            differentiated = tuple(
                self._toggle_parity(parity, axis, gradient=False)
                for parity, axis in zip(
                    source,
                    self._axes(expression.coordinate),
                    strict=True,
                )
            )
            if all(parity == differentiated[0] for parity in differentiated[1:]):
                return (differentiated[0],)
            return self._unknown_parities(1)
        if expression.op == "curl":
            return self._unknown_parities(3)
        if expression.op == "integral":
            source = argument_parities[0]
            if source is None:
                return None
            return self._unknown_parities(components)
        if expression.op == "negate":
            return argument_parities[0]
        spatial = tuple(parity for parity in argument_parities if parity is not None)
        if not spatial:
            return None
        if expression.op == "add":
            if len(spatial) != len(argument_parities):
                return self._unknown_parities(components)
            if all(parity == spatial[0] for parity in spatial[1:]):
                return spatial[0]
            return self._unknown_parities(components)
        if expression.op == "multiply":
            if len(spatial) == 1 and all(
                parity is not None or self._spatially_neutral(argument)
                for argument, parity in zip(
                    expression.args,
                    argument_parities,
                    strict=True,
                )
            ):
                return spatial[0]
            return self._unknown_parities(components)
        if expression.op == "divide":
            if (
                argument_parities[0] is not None
                and argument_parities[1] is None
                and self._spatially_neutral(expression.args[1])
            ):
                return argument_parities[0]
            return self._unknown_parities(components)
        if expression.op == "dot":
            return self._unknown_parities(components)
        if expression.op in ("power", "sin", "cos", "exp", "log", "sqrt"):
            return self._unknown_parities(components)
        return spatial[0]

    def _partial_with_parity(
        self,
        value: Array,
        /,
        *,
        axis: int,
        order: int,
        parity: int,
    ) -> Array:
        from ..solver._spatial import (
            _dual_basis_first_derivative,
            TensorGridDiscretization,
        )

        if (
            isinstance(self.discretization, TensorGridDiscretization)
            and self.discretization.basis[axis] == "uniform"
            and parity == 1
        ):
            spacing = (
                self.discretization.axes[axis].nodes[1]
                - self.discretization.axes[axis].nodes[0]
            )
            result = (value - jnp.roll(value, 1, axis=axis)) / spacing
            if order > 1:
                result = self.discretization.partial_derivative(
                    result,
                    axis=axis,
                    order=order - 1,
                )
            return result
        if (
            isinstance(self.discretization, TensorGridDiscretization)
            and self.discretization.basis[axis] in ("sine", "cosine")
            and parity == 2
        ):
            raise ValueError(
                "Cannot infer sine/cosine extension parity for this differentiated "
                "composite expression; rewrite it into boundary-compatible terms."
            )
        if (
            not isinstance(self.discretization, TensorGridDiscretization)
            or self.discretization.basis[axis] not in ("sine", "cosine")
            or parity != 1
        ):
            return self.discretization.partial_derivative(
                value,
                axis=axis,
                order=order,
            )
        result = value
        current = parity
        for _ in range(int(order)):
            if current == 1:
                result = _dual_basis_first_derivative(
                    result,
                    self.discretization.axes[axis].nodes,
                    axis=axis,
                    basis=self.discretization.basis[axis],
                )
            else:
                result = self.discretization.partial_derivative(
                    result,
                    axis=axis,
                )
            current = 1 - current
        return result

    def _differentiate_components(
        self,
        value: Array,
        expression: PDEExpression,
        /,
        *,
        axis: int,
        order: int,
    ) -> Array:
        components = self._components(expression)
        parities = self._parities(expression)
        if components == 1:
            parity = 2 if parities is None else parities[0][axis]
            return self._partial_with_parity(
                value,
                axis=axis,
                order=order,
                parity=parity,
            )
        results = tuple(
            self._partial_with_parity(
                value[..., component],
                axis=axis,
                order=order,
                parity=2 if parities is None else parities[component][axis],
            )
            for component in range(components)
        )
        return jnp.stack(results, axis=-1)

    def _align_semantic_scalar(
        self,
        value: Any,
        components: int,
        other_components: int,
        other_value: Any,
        /,
    ) -> Any:
        array = jnp.asarray(value)
        other = jnp.asarray(other_value)
        spatial_rank = len(self.discretization.state_shape)
        if (
            components == 1
            and array.ndim == spatial_rank
            and (
                other_components > 1
                or other.ndim == spatial_rank + 1
            )
        ):
            return array[..., None]
        return value

    def _evaluate(
        self,
        node: PDEExpression,
        time: Array,
        args: Any,
        fields: Mapping[str, Array],
        /,
    ) -> Any:
        if (
            node.op == "divergence"
            and node.args[0].op == "gradient"
            and node.coordinate == node.args[0].coordinate
        ):
            return self._evaluate(
                PDEExpression(
                    "laplacian",
                    (node.args[0].args[0],),
                    coordinate=node.coordinate,
                ),
                time,
                args,
                fields,
            )
        if node.op == "constant":
            assert node.value is not None
            return node.value
        if node.op == "field":
            assert node.symbol is not None
            return fields[node.symbol]
        if node.op == "parameter":
            assert node.symbol is not None
            return self._parameter(node.symbol, args)
        if node.op == "coordinate":
            assert node.symbol is not None
            return self._coordinate(node.symbol, time)

        values = tuple(self._evaluate(arg, time, args, fields) for arg in node.args)
        if node.op == "add":
            result = values[0]
            for value in values[1:]:
                result = result + value
            return result
        if node.op == "multiply":
            result = values[0]
            result_components = self._components(node.args[0])
            for argument, value in zip(node.args[1:], values[1:], strict=True):
                value_components = self._components(argument)
                left = self._align_semantic_scalar(
                    result,
                    result_components,
                    value_components,
                    value,
                )
                right = self._align_semantic_scalar(
                    value,
                    value_components,
                    result_components,
                    result,
                )
                result = left * right
                result_components = max(result_components, value_components)
            return result
        if node.op == "divide":
            numerator_components = self._components(node.args[0])
            denominator_components = self._components(node.args[1])
            numerator = self._align_semantic_scalar(
                values[0],
                numerator_components,
                denominator_components,
                values[1],
            )
            denominator = self._align_semantic_scalar(
                values[1],
                denominator_components,
                numerator_components,
                values[0],
            )
            return numerator / denominator
        if node.op == "negate":
            return -values[0]
        if node.op == "power":
            return values[0] ** values[1]
        if node.op == "sin":
            return jnp.sin(values[0])
        if node.op == "cos":
            return jnp.cos(values[0])
        if node.op == "exp":
            return jnp.exp(values[0])
        if node.op == "log":
            return jnp.log(values[0])
        if node.op == "sqrt":
            return jnp.sqrt(values[0])
        if node.op == "component":
            assert node.axis is not None
            return values[0][..., node.axis]
        if node.op == "dot":
            return jnp.sum(values[0] * values[1], axis=-1)
        if node.op in (
            "derivative",
            "gradient",
            "divergence",
            "curl",
            "laplacian",
        ):
            assert node.coordinate is not None
            if node.coordinate == self.time_coordinate:
                raise ValueError(
                    "Temporal derivatives may only appear as evolution derivatives."
                )
            axes = self._axes(node.coordinate)
            lift = self._expression_lift(node.args[0], time, args)
            operand = values[0] if lift is None else values[0] - lift
            if node.args[0].op == "coordinate" and node.op != "derivative":
                assert node.args[0].symbol is not None
                source_axes = self._axes(node.args[0].symbol)
                source = jnp.asarray(values[0])
                if node.op == "gradient":
                    return jnp.stack(
                        tuple(
                            jnp.ones_like(source)
                            if axis in source_axes
                            else jnp.zeros_like(source)
                            for axis in axes
                        ),
                        axis=-1,
                    )
                if node.op == "divergence":
                    result = jnp.zeros_like(source[..., 0])
                    for component, axis in enumerate(axes):
                        if source_axes[component] == axis:
                            result = result + 1.0
                    return result
                if node.op == "curl":
                    return jnp.zeros_like(source)
                return jnp.zeros_like(source)
            if node.op == "laplacian" and self.discretization.points is None:
                return self.discretization.laplacian(operand)
            if node.op == "derivative":
                if node.axis is None:
                    if len(axes) != 1:
                        raise ValueError(
                            "Derivatives of grouped coordinates require an axis."
                        )
                    axis = axes[0]
                else:
                    axis = axes[node.axis]
                if node.args[0].op == "coordinate":
                    assert node.args[0].symbol is not None
                    source_axes = self._axes(node.args[0].symbol)
                    source = jnp.asarray(values[0])
                    result = jnp.zeros_like(source)
                    if node.order == 1 and axis in source_axes:
                        if len(source_axes) == 1:
                            result = jnp.ones_like(source)
                        else:
                            component = source_axes.index(axis)
                            result = result.at[..., component].set(1.0)
                    return result
                result = self._differentiate_components(
                    operand,
                    node.args[0],
                    axis=axis,
                    order=node.order,
                )
                if lift is not None:
                    result = result + self._lift_partial(
                        lift,
                        axis=axis,
                        order=node.order,
                    )
                return result
            if node.op == "gradient":
                from ..solver._spatial import TensorGridDiscretization

                source_parities = self._parities(node.args[0])
                components = []
                for axis in axes:
                    parity = (
                        2
                        if source_parities is None
                        else source_parities[0][axis]
                    )
                    if (
                        isinstance(self.discretization, TensorGridDiscretization)
                        and self.discretization.basis[axis] == "uniform"
                        and parity != 1
                    ):
                        component = self.discretization.gradient(
                            operand,
                            axes=(axis,),
                        )[..., 0]
                    else:
                        component = self._differentiate_components(
                            operand,
                            node.args[0],
                            axis=axis,
                            order=1,
                        )
                    components.append(component)
                result = jnp.stack(tuple(components), axis=-1)
                if lift is not None:
                    result = result + jnp.stack(
                        tuple(
                            self._lift_partial(lift, axis=axis, order=1)
                            for axis in axes
                        ),
                        axis=-1,
                    )
                return result
            if node.op == "divergence":
                source_parities = self._parities(node.args[0])
                result = jnp.zeros_like(operand[..., 0])
                for component, axis in enumerate(axes):
                    parity = (
                        2
                        if source_parities is None
                        else source_parities[component][axis]
                    )
                    if parity == 1:
                        derivative = self.discretization.divergence(
                            operand[..., component, None],
                            axes=(axis,),
                            dual=True,
                        )
                    else:
                        derivative = self._partial_with_parity(
                            operand[..., component],
                            axis=axis,
                            order=1,
                            parity=parity,
                        )
                    result = result + derivative
                if lift is not None:
                    correction = jnp.zeros_like(lift[..., 0])
                    for component, axis in enumerate(axes):
                        correction = correction + self._lift_partial(
                            lift[..., component],
                            axis=axis,
                            order=1,
                        )
                    result = result + correction
                return result
            if node.op == "curl":
                source_parities = self._parities(node.args[0])

                def curl_partial(component: int, axis: int) -> Array:
                    parity = (
                        2
                        if source_parities is None
                        else source_parities[component][axis]
                    )
                    return self._partial_with_parity(
                        operand[..., component],
                        axis=axis,
                        order=1,
                        parity=parity,
                    )

                first, second, third = axes
                result = jnp.stack(
                    (
                        curl_partial(2, second) - curl_partial(1, third),
                        curl_partial(0, third) - curl_partial(2, first),
                        curl_partial(1, first) - curl_partial(0, second),
                    ),
                    axis=-1,
                )
                if lift is not None:
                    correction = jnp.stack(
                        (
                            self._lift_partial(lift[..., 2], axis=second, order=1)
                            - self._lift_partial(lift[..., 1], axis=third, order=1),
                            self._lift_partial(lift[..., 0], axis=third, order=1)
                            - self._lift_partial(lift[..., 2], axis=first, order=1),
                            self._lift_partial(lift[..., 1], axis=first, order=1)
                            - self._lift_partial(lift[..., 0], axis=second, order=1),
                        ),
                        axis=-1,
                    )
                    result = result + correction
                return result
            result = jnp.zeros_like(operand)
            for axis in axes:
                result = result + self._differentiate_components(
                    operand,
                    node.args[0],
                    axis=axis,
                    order=2,
                )
            if lift is not None:
                for axis in axes:
                    result = result + self._lift_partial(lift, axis=axis, order=2)
            return result
        if node.op == "integral":
            assert node.region is not None
            integrated_axes = self._region(node.region)
            result = self.discretization.integral(
                values[0],
                axes=integrated_axes,
            )
            spatial_rank = len(self.discretization.state_shape)
            trailing = result.shape[spatial_rank - len(integrated_axes) :]
            if len(integrated_axes) < spatial_rank:
                remaining_shape = iter(
                    result.shape[: spatial_rank - len(integrated_axes)]
                )
                spatial_shape = tuple(
                    1 if axis in integrated_axes else next(remaining_shape)
                    for axis in range(spatial_rank)
                )
                result = result.reshape(spatial_shape + trailing)
            return jnp.broadcast_to(
                result,
                self.discretization.state_shape + trailing,
            )
        raise ValueError(f"Unsupported semidiscrete PDE operation {node.op!r}.")

    def _coerce_field_value(self, name: str, value: Any, /) -> Array:
        result = jnp.asarray(value)
        expected = self.layout.field_shape(name)
        count = self.layout.component_counts[self.layout.field_names.index(name)]
        if result.shape == () or (count > 1 and result.shape == (count,)):
            return jnp.broadcast_to(result, expected)
        if (
            count == 1
            and expected == self.layout.spatial_shape
            and result.shape == expected + (1,)
        ):
            return result[..., 0]
        compatible = (
            result.ndim == len(expected)
            and all(
                actual in (1, target)
                for actual, target in zip(result.shape, expected, strict=True)
            )
        )
        if compatible:
            return jnp.broadcast_to(result, expected)
        raise ValueError(
            f"Evolution RHS for field {name!r} must have shape {expected}; "
            f"got {result.shape}."
        )

    def physical_state(self, time: Array, state: ArrayLike, args: Any, /) -> Array:
        value = jnp.asarray(state)
        if tuple(value.shape) != self.layout.state_shape:
            raise ValueError(
                f"Semidiscrete state must have shape {self.layout.state_shape}; "
                f"got {value.shape}."
            )
        return self.layout.pack(self._physical_fields(time, value, args))

    def __call__(self, time: Array, state: Array, args: Any) -> Array:
        value = jnp.asarray(state)
        if tuple(value.shape) != self.layout.state_shape:
            raise ValueError(
                f"Semidiscrete state must have shape {self.layout.state_shape}; "
                f"got {value.shape}."
            )
        fields = self._physical_fields(time, value, args)
        derivatives: dict[str, Array] = {}
        for name, expression in zip(
            self.layout.field_names,
            self.rhs_expressions,
            strict=True,
        ):
            rhs = self._coerce_field_value(
                name,
                self._evaluate(expression, time, args, fields),
            )
            for lift in self.boundary_lifts:
                if lift.field_name == name:
                    derivative = lift.derivative(time, args)
                    if tuple(derivative.shape) != self.layout.field_shape(name):
                        raise ValueError(
                            f"Boundary lift derivative {lift.lift_id!r} has shape "
                            f"{derivative.shape}; expected {self.layout.field_shape(name)}."
                        )
                    rhs = rhs - derivative
            derivatives[name] = rhs
        return self.layout.pack(derivatives)


class _FieldwiseLaplacianOperator(StrictModule):
    layout: SemidiscreteFieldLayout
    discretization: Any
    coefficients: tuple[Array, ...]

    def __call__(self, state: Array) -> Array:
        fields = self.layout.unpack(state)
        return self.layout.pack(
            {
                name: coefficient * self.discretization.laplacian(fields[name])
                for name, coefficient in zip(
                    self.layout.field_names,
                    self.coefficients,
                    strict=True,
                )
            }
        )


class _LinearizedOperator(StrictModule):
    evaluator: _SemidiscreteEvaluator

    def __call__(self, state: Array) -> Array:
        value = jnp.asarray(state)
        zero = jnp.zeros_like(value)
        time = jnp.asarray(0.0, dtype=value.dtype)
        return self.evaluator(time, value, None) - self.evaluator(time, zero, None)


class _SemilinearRemainder(StrictModule):
    evaluator: _SemidiscreteEvaluator
    linear_operator: Any

    def __call__(self, time: Array, state: Array, args: Any) -> Array:
        return self.evaluator(time, state, args) - self.linear_operator(state)


class CompiledSpatialDynamics(StrictModule):
    """State-shaped method-of-lines dynamics with compilation provenance."""

    drift: Any
    layout: SemidiscreteFieldLayout
    spatial_discretization: Any
    semilinear_drift: Any | None
    boundary_lifts: tuple[BoundaryLift, ...]
    _evaluator: _SemidiscreteEvaluator
    compilation_id: str = eqx.field(static=True)
    source_hash: str = eqx.field(static=True)
    resolved_method: ResolvedSemidiscreteMethod = eqx.field(static=True)

    def __init__(
        self,
        drift: Any,
        layout: SemidiscreteFieldLayout,
        spatial_discretization: Any,
        evaluator: _SemidiscreteEvaluator,
        /,
        *,
        semilinear_drift: Any | None,
        boundary_lifts: Sequence[BoundaryLift],
        compilation_id: str,
        source_hash: str,
        resolved_method: ResolvedSemidiscreteMethod,
    ):
        self.drift = drift
        self.layout = layout
        self.spatial_discretization = spatial_discretization
        self.semilinear_drift = semilinear_drift
        self.boundary_lifts = tuple(boundary_lifts)
        self._evaluator = evaluator
        self.compilation_id = str(compilation_id)
        self.source_hash = str(source_hash)
        self.resolved_method = resolved_method

    @property
    def state_shape(self) -> tuple[int, ...]:
        return self.layout.state_shape

    def physical_state(self, time: Array, state: ArrayLike, args: Any = None) -> Array:
        return self._evaluator.physical_state(time, state, args)

    def __call__(self, time: Array, state: Array, args: Any) -> Array:
        return self.drift(time, state, args)


def _signed_additive_terms(
    expression: PDEExpression,
    sign: float = 1.0,
    /,
) -> tuple[tuple[float, PDEExpression], ...]:
    if expression.op == "add":
        return tuple(
            term
            for argument in expression.args
            for term in _signed_additive_terms(argument, sign)
        )
    if expression.op == "negate":
        return _signed_additive_terms(expression.args[0], -sign)
    return ((sign, expression),)


def _temporal_field(
    expression: PDEExpression,
    time_coordinate: str,
    /,
) -> str | None:
    if (
        expression.op != "derivative"
        or expression.coordinate != time_coordinate
        or expression.order != 1
        or expression.axis not in (None, 0)
        or len(expression.args) != 1
        or expression.args[0].op != "field"
    ):
        return None
    return expression.args[0].symbol


def _temporal_occurrences(
    expression: PDEExpression,
    time_coordinate: str,
    /,
) -> tuple[str, ...]:
    field_name = _temporal_field(expression, time_coordinate)
    if field_name is not None:
        return (field_name,)
    return tuple(
        occurrence
        for argument in expression.args
        for occurrence in _temporal_occurrences(argument, time_coordinate)
    )


def _contains_temporal_derivative(
    expression: PDEExpression,
    time_coordinate: str,
    /,
) -> bool:
    return (
        expression.op == "derivative"
        and expression.coordinate == time_coordinate
    ) or any(
        _contains_temporal_derivative(argument, time_coordinate)
        for argument in expression.args
    )


def _requires_coordinate_frame(
    expression: PDEExpression,
    spatial_coordinates: set[str],
    /,
) -> bool:
    if expression.op == "coordinate" and expression.symbol in spatial_coordinates:
        return True
    if expression.op in ("gradient", "divergence", "curl"):
        return True
    if (
        expression.op == "derivative"
        and expression.coordinate in spatial_coordinates
    ):
        return True
    return any(
        _requires_coordinate_frame(argument, spatial_coordinates)
        for argument in expression.args
    )


def _integral_regions(expression: PDEExpression, /) -> tuple[str, ...]:
    current = (
        (expression.region,)
        if expression.op == "integral" and expression.region is not None
        else ()
    )
    return current + tuple(
        region
        for argument in expression.args
        for region in _integral_regions(argument)
    )


def _expression_nodes(expression: PDEExpression, /) -> tuple[PDEExpression, ...]:
    return (expression,) + tuple(
        node
        for argument in expression.args
        for node in _expression_nodes(argument)
    )


def _evolution_rhs(
    problem: PDEProblemIR,
    time_coordinate: str,
    /,
) -> tuple[PDEExpression, ...]:
    equations: dict[str, PDEExpression] = {}
    field_names = {field.name for field in problem.fields}
    for equation in problem.equations:
        left_field = _temporal_field(equation.lhs, time_coordinate)
        right_field = _temporal_field(equation.rhs, time_coordinate)
        if left_field is not None and not _temporal_occurrences(
            equation.rhs, time_coordinate
        ):
            if left_field in equations:
                raise ValueError(
                    f"Field {left_field!r} has more than one temporal evolution equation."
                )
            equations[left_field] = equation.rhs
            continue
        if right_field is not None and not _temporal_occurrences(
            equation.lhs, time_coordinate
        ):
            if right_field in equations:
                raise ValueError(
                    f"Field {right_field!r} has more than one temporal evolution equation."
                )
            equations[right_field] = equation.lhs
            continue
        temporal: list[tuple[float, str]] = []
        remainder: list[tuple[float, PDEExpression]] = []
        for sign, term in _signed_additive_terms(equation.residual):
            field_name = _temporal_field(term, time_coordinate)
            if field_name is None:
                remainder.append((sign, term))
            else:
                temporal.append((sign, field_name))
        if len(temporal) != 1:
            raise ValueError(
                f"PDE equation {equation.name!r} must contain exactly one direct "
                "first temporal derivative."
            )
        coefficient, field_name = temporal[0]
        if field_name not in field_names:
            raise ValueError(
                f"Evolution equation {equation.name!r} references unknown field "
                f"{field_name!r}."
            )
        if field_name in equations:
            raise ValueError(
                f"Field {field_name!r} has more than one temporal evolution equation."
            )
        residual = PDEExpression.constant(0.0)
        for sign, term in remainder:
            residual = residual + (term if sign > 0.0 else -term)
        if coefficient == 1.0:
            equations[field_name] = -residual
        elif coefficient == -1.0:
            equations[field_name] = residual
        else:
            equations[field_name] = -residual / coefficient
    missing = field_names - set(equations)
    if missing:
        raise ValueError(
            "Every PDE field requires exactly one temporal evolution equation; "
            f"missing {sorted(missing)}."
        )
    rhs = tuple(equations[field.name] for field in problem.fields)
    if any(
        _contains_temporal_derivative(expression, time_coordinate)
        for expression in rhs
    ):
        raise ValueError("Evolution right-hand sides cannot contain temporal derivatives.")
    return rhs


def _coordinate_axis_map(
    problem: PDEProblemIR,
    discretization: Any,
    /,
) -> tuple[tuple[str, tuple[int, ...]], ...]:
    from ..solver._spatial import TensorGridDiscretization
    spatial = tuple(
        coordinate for coordinate in problem.coordinates if coordinate.kind == "space"
    )
    if not spatial:
        raise ValueError("Semidiscrete PDE compilation requires spatial coordinates.")
    rank = sum(coordinate.size for coordinate in spatial)
    if isinstance(discretization, TensorGridDiscretization):
        if rank != len(discretization.state_shape):
            raise ValueError(
                "PDE spatial coordinate size must match the tensor-grid rank; "
                f"got {rank} and {len(discretization.state_shape)}."
            )
        output: list[tuple[str, tuple[int, ...]]] = []
        offset = 0
        for coordinate in spatial:
            axes = tuple(range(offset, offset + coordinate.size))
            for axis in axes:
                periodic = discretization.boundary_conditions[axis] == "periodic"
                if coordinate.periodic != periodic:
                    raise ValueError(
                        f"PDE coordinate {coordinate.name!r} periodic={coordinate.periodic} "
                        f"is incompatible with basis {discretization.basis[axis]!r}."
                    )
                if coordinate.bounds is not None:
                    nodes = np.asarray(discretization.axes[axis].nodes)
                    spacing = float(nodes[1] - nodes[0])
                    basis = discretization.basis[axis]
                    if basis == "sine":
                        actual_bounds = (
                            float(nodes[0] - 0.5 * spacing),
                            float(nodes[-1] + 0.5 * spacing),
                        )
                    elif basis in ("fourier", "uniform"):
                        actual_bounds = (
                            float(nodes[0]),
                            float(nodes[-1] + spacing),
                        )
                    else:
                        actual_bounds = (float(nodes[0]), float(nodes[-1]))
                    if not np.allclose(
                        actual_bounds,
                        coordinate.bounds,
                        rtol=1e-10,
                        atol=1e-12,
                    ):
                        raise ValueError(
                            f"PDE coordinate {coordinate.name!r} bounds "
                            f"{coordinate.bounds} do not match discretization axis "
                            f"bounds {actual_bounds}."
                        )
            output.append((coordinate.name, axes))
            offset += coordinate.size
        return tuple(output)
    if len(spatial) != 1 or spatial[0].size != 1:
        raise ValueError(
            "Manifold spectral compilation requires one scalar spatial coordinate."
        )
    return ((spatial[0].name, (0,)),)


def _region_axis_map(
    problem: PDEProblemIR,
    coordinate_axes: Sequence[tuple[str, tuple[int, ...]]],
    /,
) -> tuple[tuple[str, tuple[int, ...]], ...]:
    axes_by_name = dict(coordinate_axes)
    output: list[tuple[str, tuple[int, ...]]] = []
    for region in problem.regions:
        axes = tuple(
            axis
            for coordinate in region.coordinates
            if coordinate in axes_by_name
            for axis in axes_by_name[coordinate]
        )
        if region.kind == "interior" and not axes:
            raise ValueError(
                f"Interior region {region.name!r} has no compiled spatial coordinates."
            )
        output.append((region.name, axes))
    return tuple(output)


def _field_reference(expression: PDEExpression, /) -> str | None:
    node = expression
    if node.op == "component":
        node = node.args[0]
    if node.op == "field":
        return node.symbol
    return None


def _boundary_form(
    expression: PDEExpression,
    /,
) -> tuple[str, str, str | None, int | None, int] | None:
    field_name = _field_reference(expression)
    if field_name is not None:
        return ("dirichlet", field_name, None, None, 0)
    if expression.op != "derivative" or len(expression.args) != 1:
        return None
    field_name = _field_reference(expression.args[0])
    if field_name is None:
        return None
    return (
        "neumann",
        field_name,
        expression.coordinate,
        expression.axis,
        expression.order,
    )


def _homogeneous_target(expression: PDEExpression, /) -> bool:
    if expression.op == "constant":
        return expression.value == 0.0
    if expression.op == "negate":
        return _homogeneous_target(expression.args[0])
    if expression.op == "add":
        return all(_homogeneous_target(argument) for argument in expression.args)
    return False


def _validate_boundary_conditions(
    problem: PDEProblemIR,
    discretization: Any,
    coordinate_axes: Sequence[tuple[str, tuple[int, ...]]],
    boundary_lifts: Sequence[BoundaryLift],
    /,
) -> None:
    from ..solver._spatial import TensorGridDiscretization
    lifts = {lift.field_name: lift for lift in boundary_lifts}
    if any(condition.kind == "interface" for condition in problem.conditions):
        raise ValueError(
            "Semidiscrete single-grid compilation does not support interface conditions."
        )
    if not isinstance(discretization, TensorGridDiscretization):
        if any(condition.kind == "boundary" for condition in problem.conditions):
            raise ValueError(
                "Manifold spectral discretizations do not expose a boundary basis contract."
            )
        return
    regions = {region.name: region for region in problem.regions}
    axes_by_coordinate = dict(coordinate_axes)
    for condition in problem.conditions:
        if condition.kind != "boundary":
            continue
        region = regions[condition.region]
        if region.component is not None:
            raise ValueError(
                f"Boundary region {region.name!r} selects component "
                f"{region.component!r}, but tensor bases enforce both boundary sides."
            )
        if condition.coordinate is None or condition.coordinate not in axes_by_coordinate:
            raise ValueError(
                f"Boundary condition {condition.name!r} requires a compiled spatial "
                "coordinate."
            )
        region_coordinates = tuple(
            coordinate
            for coordinate in regions[condition.region].coordinates
            if coordinate in axes_by_coordinate
        )
        if region_coordinates != (condition.coordinate,):
            raise ValueError(
                f"Boundary condition {condition.name!r} normal coordinate "
                f"{condition.coordinate!r} does not match region "
                f"{condition.region!r} spatial coordinates {region_coordinates}."
            )
        form = _boundary_form(condition.expression)
        if form is None:
            raise ValueError(
                f"Boundary condition {condition.name!r} must directly constrain a "
                "field or its normal derivative."
            )
        kind, field_name, derivative_coordinate, derivative_axis, order = form
        for axis in axes_by_coordinate[condition.coordinate]:
            basis = discretization.basis[axis]
            if basis in ("uniform", "fourier"):
                raise ValueError(
                    f"Boundary condition {condition.name!r} is incompatible with "
                    f"periodic basis {basis!r}."
                )
            expected = "dirichlet" if basis == "sine" else "neumann"
            if kind != expected:
                raise ValueError(
                    f"Boundary condition {condition.name!r} is {kind}, but basis "
                    f"{basis!r} requires homogeneous {expected} residual data."
                )
        if kind == "neumann" and (
            derivative_coordinate != condition.coordinate
            or order != 1
            or (
                derivative_axis is None
                and len(axes_by_coordinate[condition.coordinate]) != 1
            )
        ):
            raise ValueError(
                f"Neumann condition {condition.name!r} must use one first derivative "
                "normal to its boundary coordinate."
            )
        if not _homogeneous_target(condition.target) and field_name not in lifts:
            raise ValueError(
                f"Nonhomogeneous boundary condition {condition.name!r} requires an "
                f"explicit BoundaryLift for field {field_name!r}."
            )
        if (
            kind == "dirichlet"
            and condition.target.op == "constant"
            and field_name in lifts
            and not callable(lifts[field_name].value)
        ):
            target = float(condition.target.value)
            lift_value = np.asarray(lifts[field_name].value)
            if (
                lift_value.size > 0
                and np.all(lift_value == lift_value.reshape((-1,))[0])
                and not np.allclose(lift_value, target)
            ):
                raise ValueError(
                    f"BoundaryLift {lifts[field_name].lift_id!r} is constant but "
                    f"does not match boundary target {target}."
                )


def _field_degree(expression: PDEExpression, /) -> int:
    if expression.op == "field":
        return 1
    degrees = tuple(_field_degree(argument) for argument in expression.args)
    if not degrees:
        return 0
    if expression.op in (
        "add",
        "negate",
        "component",
        "derivative",
        "gradient",
        "divergence",
        "curl",
        "laplacian",
        "integral",
    ):
        return max(degrees)
    if expression.op == "multiply":
        return min(sum(degrees), 2)
    if expression.op == "divide":
        return degrees[0] if degrees[1] == 0 else 2
    if expression.op == "dot":
        return min(sum(degrees), 2)
    if expression.op == "power":
        exponent = expression.args[1]
        if degrees[0] == 0:
            return 0
        if exponent.op == "constant" and exponent.value == 1.0:
            return degrees[0]
        return 2
    if expression.op in ("sin", "cos", "exp", "log", "sqrt"):
        return 0 if degrees[0] == 0 else 2
    return max(degrees)


def _depends_on_time(expression: PDEExpression, time_coordinate: str, /) -> bool:
    if expression.op == "coordinate" and expression.symbol == time_coordinate:
        return True
    return any(
        _depends_on_time(argument, time_coordinate) for argument in expression.args
    )


def _static_scalar(
    expression: PDEExpression,
    defaults: Mapping[str, Any],
    /,
) -> float | None:
    if expression.op == "constant":
        return expression.value
    if expression.op == "parameter":
        if expression.symbol not in defaults:
            return None
        value = np.asarray(defaults[expression.symbol])
        return float(value) if value.shape == () and np.isfinite(value) else None
    values = tuple(_static_scalar(argument, defaults) for argument in expression.args)
    if any(value is None for value in values):
        return None
    scalars = tuple(float(value) for value in values if value is not None)
    if expression.op == "add":
        return sum(scalars)
    if expression.op == "multiply":
        return float(np.prod(scalars))
    if expression.op == "divide" and scalars[1] != 0.0:
        return scalars[0] / scalars[1]
    if expression.op == "negate":
        return -scalars[0]
    if expression.op == "power":
        result = scalars[0] ** scalars[1]
        return float(result) if np.isfinite(result) else None
    return None


def _multiplicative_factors(expression: PDEExpression, /) -> tuple[PDEExpression, ...]:
    if expression.op == "multiply":
        return tuple(
            factor
            for argument in expression.args
            for factor in _multiplicative_factors(argument)
        )
    return (expression,)


def _diffusion_coefficients(
    rhs_expressions: Sequence[PDEExpression],
    layout: SemidiscreteFieldLayout,
    all_spatial_coordinates: set[str],
    defaults: Mapping[str, Any],
    /,
) -> tuple[float, ...] | None:
    coefficients: list[float] = []
    found = False
    for field_name, expression in zip(
        layout.field_names,
        rhs_expressions,
        strict=True,
    ):
        coefficient = 0.0
        for sign, term in _signed_additive_terms(expression):
            factors = _multiplicative_factors(term)
            laplacians = tuple(
                factor
                for factor in factors
                if factor.op == "laplacian"
                and len(factor.args) == 1
                and factor.args[0].op == "field"
                and factor.args[0].symbol == field_name
                and factor.coordinate in all_spatial_coordinates
            )
            if len(laplacians) != 1:
                continue
            scalar = 1.0
            valid = True
            for factor in factors:
                if factor is laplacians[0]:
                    continue
                value = _static_scalar(factor, defaults)
                if value is None:
                    valid = False
                    break
                scalar *= value
            if valid:
                coefficient += sign * scalar
                found = True
        coefficients.append(coefficient)
    return tuple(coefficients) if found else None


def _spectral_representation(
    discretization: Any,
    layout: SemidiscreteFieldLayout,
    coefficients: tuple[float, ...] | None,
    /,
) -> Any | None:
    from ..solver._matrix_functions import SpectralMatrixRepresentation
    from ..solver._spatial import SpectralSpatialDiscretization
    if (
        not isinstance(discretization, SpectralSpatialDiscretization)
        or discretization.plan.num_modes != discretization.plan.num_points
        or coefficients is None
        or not layout.squeezed
    ):
        return None
    coefficient = coefficients[0]
    return SpectralMatrixRepresentation(
        -coefficient * discretization.plan.eigenvalues,
        discretization.plan.analysis,
        discretization.plan.synthesis,
        state_shape=layout.state_shape,
        representation_id=_stable_id(
            "semidiscrete-spectral-v1",
            discretization.discretization_id,
            repr(coefficient),
        ),
    )


def compile_semidiscrete_pde(
    problem: PDEProblemIR,
    discretization: Any,
    /,
    *,
    parameter_values: Mapping[str, Any] | None = None,
    boundary_lifts: Sequence[BoundaryLift] = (),
    method: SemidiscreteCompilationMethod = "auto",
) -> CompiledSpatialDynamics:
    """Compile validated PDE IR into state-shaped method-of-lines dynamics."""
    from ..solver._semilinear_drift import SemilinearDrift
    from ..solver._spatial import (
        AbstractSpatialDiscretization,
        SpectralSpatialDiscretization,
    )
    if not isinstance(problem, PDEProblemIR):
        raise TypeError("problem must be a PDEProblemIR.")
    if not isinstance(discretization, AbstractSpatialDiscretization):
        raise TypeError("discretization must be an AbstractSpatialDiscretization.")
    if method not in ("auto", "direct", "semilinear"):
        raise ValueError("method must be 'auto', 'direct', or 'semilinear'.")
    validate_pde_ir(problem)

    time_coordinates = tuple(
        coordinate for coordinate in problem.coordinates if coordinate.kind == "time"
    )
    if len(time_coordinates) != 1 or time_coordinates[0].size != 1:
        raise ValueError(
            "Semidiscrete PDE compilation requires exactly one scalar time coordinate."
        )
    time_coordinate = time_coordinates[0].name
    spatial_coordinates = tuple(
        coordinate.name
        for coordinate in problem.coordinates
        if coordinate.kind == "space"
    )
    spatial_coordinate_set = set(spatial_coordinates)
    for field in problem.fields:
        if time_coordinate not in field.coordinates:
            raise ValueError(
                f"Evolution field {field.name!r} must depend on time coordinate "
                f"{time_coordinate!r}."
            )
        field_spatial_coordinates = tuple(
            coordinate
            for coordinate in field.coordinates
            if coordinate in spatial_coordinate_set
        )
        if field_spatial_coordinates != spatial_coordinates:
            raise ValueError(
                "All evolution fields must use the same complete spatial coordinate "
                f"layout {spatial_coordinates}; field {field.name!r} uses "
                f"{field_spatial_coordinates}."
            )

    lifts = tuple(boundary_lifts)
    if any(not isinstance(lift, BoundaryLift) for lift in lifts):
        raise TypeError("boundary_lifts must contain BoundaryLift objects.")
    lift_fields = tuple(lift.field_name for lift in lifts)
    if len(set(lift_fields)) != len(lift_fields):
        raise ValueError("At most one BoundaryLift may be supplied per field.")
    unknown_lifts = set(lift_fields) - {field.name for field in problem.fields}
    if unknown_lifts:
        raise ValueError(f"Boundary lifts reference unknown fields {sorted(unknown_lifts)}.")

    coordinate_axes = _coordinate_axis_map(problem, discretization)
    _validate_boundary_conditions(
        problem,
        discretization,
        coordinate_axes,
        lifts,
    )
    rhs_expressions = _evolution_rhs(problem, time_coordinate)
    regions_by_name = {region.name: region for region in problem.regions}
    unsupported_integrals = tuple(
        region_name
        for expression in rhs_expressions
        for region_name in _integral_regions(expression)
        if (
            regions_by_name[region_name].kind != "interior"
            or regions_by_name[region_name].component is not None
            or len(set(regions_by_name[region_name].coordinates))
            != len(regions_by_name[region_name].coordinates)
            or any(
                coordinate not in spatial_coordinate_set
                for coordinate in regions_by_name[region_name].coordinates
            )
        )
    )
    if unsupported_integrals:
        raise ValueError(
            "Semidiscrete volume quadrature only supports unpartitioned interior "
            f"spatial regions with unique coordinates; got {unsupported_integrals}."
        )
    if isinstance(discretization, SpectralSpatialDiscretization):
        if lifts:
            raise ValueError(
                "SpectralSpatialDiscretization cannot apply coordinate-space "
                "BoundaryLift objects without a coordinate frame."
            )
        framed_expressions = rhs_expressions + tuple(
            expression
            for condition in problem.conditions
            for expression in (condition.expression, condition.target)
        )
        if any(
            _requires_coordinate_frame(expression, spatial_coordinate_set)
            for expression in framed_expressions
        ):
            raise ValueError(
                "SpectralSpatialDiscretization has no coordinate frame for "
                "coordinate, derivative, gradient, divergence, or curl nodes."
            )
    layout = SemidiscreteFieldLayout(problem.fields, discretization.state_shape)

    supplied = {} if parameter_values is None else dict(parameter_values)
    parameter_names = {parameter.name for parameter in problem.parameters}
    unknown_parameters = set(supplied) - parameter_names
    if unknown_parameters:
        raise ValueError(
            f"parameter_values contains unknown PDE parameters {sorted(unknown_parameters)}."
        )
    defaults: list[Any | None] = []
    defaults_by_name: dict[str, Any] = {}
    for parameter in problem.parameters:
        value = supplied.get(parameter.name, parameter.value)
        if value is None:
            normalized = None
        else:
            normalized = jnp.asarray(value)
            if normalized.dtype.kind not in "biuf":
                raise TypeError(
                    f"Default for PDE parameter {parameter.name!r} must be real-valued."
                )
            if parameter.components == 1 and normalized.shape == (1,):
                normalized = normalized[0]
            if parameter.components == 1:
                allowed = (
                    ((), layout.spatial_shape)
                    if parameter.functional
                    else ((),)
                )
            else:
                allowed = (
                    ((parameter.components,), layout.spatial_shape + (parameter.components,))
                    if parameter.functional
                    else ((parameter.components,),)
                )
            if normalized.shape not in allowed:
                raise ValueError(
                    f"Default for PDE parameter {parameter.name!r} has shape "
                    f"{normalized.shape}; expected one of {allowed}."
                )
            _value_fingerprint(normalized)
        defaults.append(normalized)
        if normalized is not None:
            defaults_by_name[parameter.name] = normalized
    lift_bindings = tuple(
        sorted((lift.field_name, lift.lift_id) for lift in lifts)
    )
    parameter_bindings = tuple(
        (parameter.name, _value_fingerprint(value))
        for parameter, value in zip(problem.parameters, defaults, strict=True)
    )
    binding_id = _stable_id(
        "semidiscrete-bindings-v1",
        problem.canonical_hash,
        discretization.discretization_id,
        layout.layout_id,
        repr(lift_bindings),
        repr(parameter_bindings),
    )

    evaluator = _SemidiscreteEvaluator(
        problem,
        rhs_expressions,
        layout,
        discretization,
        lifts,
        defaults,
        coordinate_axes,
        time_coordinate,
        _region_axis_map(problem, coordinate_axes),
    )
    from ..solver._spatial import TensorGridDiscretization

    for expression in rhs_expressions:
        for node in _expression_nodes(expression):
            if node.op not in (
                "derivative",
                "gradient",
                "divergence",
                "curl",
                "laplacian",
            ):
                continue
            if (
                node.op in ("divergence", "curl")
                and infer_expression_type(node.args[0], problem).is_scalar
            ):
                raise ValueError(
                    f"{node.op} requires a vector-like operand, not a scalar "
                    "with the same component count."
                )
            assert node.coordinate is not None
            axes = evaluator._axes(node.coordinate)
            if node.op == "derivative" and node.axis is None and len(axes) != 1:
                raise ValueError(
                    "Derivatives of grouped coordinates require an explicit axis."
                )
            if not isinstance(discretization, TensorGridDiscretization) or not any(
                discretization.basis[axis] in ("sine", "cosine")
                for axis in axes
            ):
                continue
            if node.args[0].op == "coordinate":
                continue
            if (
                node.op == "divergence"
                and node.args[0].op == "gradient"
                and node.coordinate == node.args[0].coordinate
            ):
                continue
            parities = evaluator._parities(node.args[0])
            if parities is None or any(
                parity[axis] == 2
                for parity in parities
                for axis in axes
            ):
                raise ValueError(
                    "Cannot infer sine/cosine extension parity for a "
                    "differentiated composite expression; rewrite it into "
                    "boundary-compatible terms."
                )

    full_spatial_axes = tuple(range(len(discretization.state_shape)))
    full_spatial_coordinates = {
        name for name, axes in coordinate_axes if axes == full_spatial_axes
    }
    coefficients = _diffusion_coefficients(
        rhs_expressions,
        layout,
        full_spatial_coordinates,
        defaults_by_name,
    )
    linear_operator: Any | None = None
    mass_self_adjoint = False
    if coefficients is not None:
        linear_operator = _FieldwiseLaplacianOperator(
            layout,
            discretization,
            tuple(jnp.asarray(value) for value in coefficients),
        )
        mass_self_adjoint = True
    elif (
        all(_field_degree(expression) <= 1 for expression in rhs_expressions)
        and not any(
            _depends_on_time(expression, time_coordinate)
            for expression in rhs_expressions
        )
        and all(value is not None for value in defaults)
    ):
        linear_operator = _LinearizedOperator(evaluator)

    if method == "direct":
        linear_operator = None
    if method == "semilinear" and linear_operator is None:
        raise ValueError(
            "The requested semilinear method could not conservatively isolate a "
            "time-independent linear operator."
        )

    semilinear: Any | None = None
    if linear_operator is None:
        drift: Any = evaluator
        resolved: ResolvedSemidiscreteMethod = "direct"
    else:
        spectral = _spectral_representation(discretization, layout, coefficients)
        resolved = (
            "semilinear-spectral"
            if spectral is not None
            else "semilinear-matrix-free"
        )
        operator_id = _stable_id(
            "semidiscrete-linear-operator-v1",
            binding_id,
            type(linear_operator).__name__,
            repr(coefficients),
        )
        semilinear = SemilinearDrift(
            linear_operator,
            _SemilinearRemainder(evaluator, linear_operator),
            state_shape=layout.state_shape,
            operator_id=operator_id,
            mass_self_adjoint=mass_self_adjoint,
            mass_weights=discretization.quadrature_weights,
            spectral_representation=spectral,
        )
        drift = semilinear

    compilation_id = _stable_id(
        "semidiscrete-pde-compiler-v1",
        binding_id,
        resolved,
    )
    return CompiledSpatialDynamics(
        drift,
        layout,
        discretization,
        evaluator,
        semilinear_drift=semilinear,
        boundary_lifts=lifts,
        compilation_id=compilation_id,
        source_hash=problem.canonical_hash,
        resolved_method=resolved,
    )


__all__ = [
    "BoundaryLift",
    "CompiledSpatialDynamics",
    "ResolvedSemidiscreteMethod",
    "SemidiscreteCompilationMethod",
    "SemidiscreteFieldLayout",
    "compile_semidiscrete_pde",
]
