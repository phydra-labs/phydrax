#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, TYPE_CHECKING

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from ..discretization import (
    DiscreteFieldSpace,
    DiscretizationBundle,
    DiscretizationKey,
    DiscretizationRecord,
    DiscretizationRole,
    PseudospectralMethodPlan,
    TensorDofLayout,
    TensorSpectralDiscretization,
)
from ..linalg import (
    ArraySpace,
    DiagonalLinearOperator,
    FunctionLinearOperator,
    OperatorProperties,
)
from ._ir import PDEExpression, PDEField, PDEProblemIR
from ._validate import validate_pde_ir


if TYPE_CHECKING:
    from ..solver._semilinear_drift import SemilinearDrift


class SpectralStateLayout(StrictModule):
    """Static modal packing for PDE fields sharing one tensor spectral space."""

    field_names: tuple[str, ...] = eqx.field(static=True)
    field_spaces: tuple[DiscreteFieldSpace, ...]
    component_counts: tuple[int, ...] = eqx.field(static=True)
    scalar_fields: tuple[bool, ...] = eqx.field(static=True)
    component_offsets: tuple[int, ...] = eqx.field(static=True)
    modal_shape: tuple[int, ...] = eqx.field(static=True)
    physical_shape: tuple[int, ...] = eqx.field(static=True)
    state_shape: tuple[int, ...] = eqx.field(static=True)
    physical_state_shape: tuple[int, ...] = eqx.field(static=True)
    squeezed: bool = eqx.field(static=True)
    layout_id: str = eqx.field(static=True)

    def __init__(
        self,
        fields: Sequence[PDEField],
        discretization: TensorSpectralDiscretization,
        /,
    ):
        field_values = tuple(fields)
        if not field_values or not all(
            isinstance(field, PDEField) for field in field_values
        ):
            raise TypeError("fields must contain one or more PDEField values.")
        names = tuple(field.name for field in field_values)
        if len(set(names)) != len(names):
            raise ValueError("Spectral PDE field names must be unique.")
        components = tuple(int(field.components) for field in field_values)
        scalar = tuple(
            field.representation in ("scalar", "pseudoscalar") for field in field_values
        )
        spaces = []
        for field, count, is_scalar in zip(
            field_values,
            components,
            scalar,
            strict=True,
        ):
            component_shape = () if count == 1 and is_scalar else (count,)
            layout = TensorDofLayout(
                discretization.plan.axis_names,
                discretization.modal_shape,
                component_shape=component_shape,
            )
            spaces.append(
                DiscreteFieldSpace(
                    field.name,
                    discretization.support.support_id,
                    layout,
                    ArraySpace(
                        layout.value_shape,
                        dtype=jnp.dtype(discretization.plan.precision.coefficient_dtype),
                    ),
                    representation="modal_coefficient",
                    conformity="unrestricted",
                    projection_id=discretization.modal_space.projection_id,
                    reconstruction_id=discretization.modal_space.reconstruction_id,
                )
            )
        offsets = []
        offset = 0
        for count in components:
            offsets.append(offset)
            offset += count
        squeezed = len(field_values) == 1 and components == (1,) and scalar == (True,)
        state_shape = (
            discretization.modal_shape
            if squeezed
            else discretization.modal_shape + (offset,)
        )
        physical_state_shape = (
            discretization.physical_shape
            if squeezed
            else discretization.physical_shape + (offset,)
        )
        self.field_names = names
        self.field_spaces = tuple(spaces)
        self.component_counts = components
        self.scalar_fields = scalar
        self.component_offsets = tuple(offsets)
        self.modal_shape = discretization.modal_shape
        self.physical_shape = discretization.physical_shape
        self.state_shape = state_shape
        self.physical_state_shape = physical_state_shape
        self.squeezed = squeezed
        self.layout_id = canonical_fingerprint(
            {
                "kind": "spectral-state-layout",
                "fields": [space.field_space_id for space in spaces],
                "modal_shape": list(discretization.modal_shape),
                "physical_shape": list(discretization.physical_shape),
            }
        )

    def _index(self, name: str, /) -> int:
        identifier = str(name)
        if identifier not in self.field_names:
            raise KeyError(f"Unknown spectral field {identifier!r}.")
        return self.field_names.index(identifier)

    def field_shape(self, name: str, /, *, physical: bool = False) -> tuple[int, ...]:
        index = self._index(name)
        base = self.physical_shape if physical else self.modal_shape
        count = self.component_counts[index]
        return base if count == 1 and self.scalar_fields[index] else base + (count,)

    def field(self, state: ArrayLike, name: str, /, *, physical: bool = False) -> Array:
        value = jnp.asarray(state)
        expected = self.physical_state_shape if physical else self.state_shape
        if value.shape != expected:
            raise ValueError(
                f"Packed spectral state must have shape {expected}; got {value.shape}."
            )
        index = self._index(name)
        if self.squeezed:
            return value
        offset = self.component_offsets[index]
        count = self.component_counts[index]
        if count == 1 and self.scalar_fields[index]:
            return value[..., offset]
        return value[..., offset : offset + count]

    def unpack(self, state: ArrayLike, /, *, physical: bool = False) -> dict[str, Array]:
        return {
            name: self.field(state, name, physical=physical) for name in self.field_names
        }

    def pack(
        self,
        fields: Mapping[str, ArrayLike],
        /,
        *,
        physical: bool = False,
    ) -> Array:
        if set(fields) != set(self.field_names):
            raise ValueError("Packed spectral field keys must exactly match the layout.")
        values = []
        for index, name in enumerate(self.field_names):
            value = jnp.asarray(fields[name])
            expected = self.field_shape(name, physical=physical)
            if value.shape != expected:
                raise ValueError(
                    f"Field {name!r} must have shape {expected}; got {value.shape}."
                )
            count = self.component_counts[index]
            values.append(
                value[..., None] if count == 1 and self.scalar_fields[index] else value
            )
        if self.squeezed:
            return values[0][..., 0]
        return jnp.concatenate(tuple(values), axis=-1)


class _SpectralEvaluator(StrictModule):
    layout: SpectralStateLayout
    discretization: TensorSpectralDiscretization
    method: Any
    parameter_defaults: tuple[Any | None, ...]
    rhs_expressions: tuple[PDEExpression, ...] = eqx.field(static=True)
    output_names: tuple[str, ...] = eqx.field(static=True)
    output_components: tuple[int, ...] = eqx.field(static=True)
    parameter_names: tuple[str, ...] = eqx.field(static=True)
    parameter_components: tuple[int, ...] = eqx.field(static=True)
    parameter_functional: tuple[bool, ...] = eqx.field(static=True)
    coordinate_axes: tuple[tuple[str, tuple[int, ...]], ...] = eqx.field(static=True)
    time_coordinate: str | None = eqx.field(static=True)
    region_axes: tuple[tuple[str, tuple[int, ...]], ...] = eqx.field(static=True)

    def __init__(
        self,
        problem: PDEProblemIR,
        rhs_expressions: Sequence[PDEExpression],
        output_names: Sequence[str],
        output_components: Sequence[int],
        layout: SpectralStateLayout,
        discretization: TensorSpectralDiscretization,
        method: Any,
        parameter_defaults: Sequence[Any | None],
        coordinate_axes: Sequence[tuple[str, tuple[int, ...]]],
        time_coordinate: str | None,
        region_axes: Sequence[tuple[str, tuple[int, ...]]],
        /,
    ):
        self.layout = layout
        self.discretization = discretization
        self.method = method
        self.parameter_defaults = tuple(parameter_defaults)
        expressions = tuple(rhs_expressions)
        names = tuple(str(name) for name in output_names)
        components = tuple(int(count) for count in output_components)
        if (
            not expressions
            or len(expressions) != len(names)
            or len(expressions) != len(components)
            or any(not name for name in names)
            or any(count <= 0 for count in components)
        ):
            raise ValueError(
                "Spectral evaluator outputs require aligned expressions, names, "
                "and positive component counts."
            )
        self.rhs_expressions = expressions
        self.output_names = names
        self.output_components = components
        self.parameter_names = tuple(parameter.name for parameter in problem.parameters)
        self.parameter_components = tuple(
            parameter.components for parameter in problem.parameters
        )
        self.parameter_functional = tuple(
            parameter.functional for parameter in problem.parameters
        )
        self.coordinate_axes = tuple(coordinate_axes)
        self.time_coordinate = None if time_coordinate is None else str(time_coordinate)
        self.region_axes = tuple(region_axes)

    @property
    def evaluation(self) -> TensorSpectralDiscretization:
        return self.method.dealiasing.evaluation

    def _axes(self, coordinate: str, /) -> tuple[int, ...]:
        for name, axes in self.coordinate_axes:
            if name == coordinate:
                return axes
        raise ValueError(
            f"Coordinate {coordinate!r} is not a compiled spatial coordinate."
        )

    def _region(self, name: str, /) -> tuple[int, ...]:
        for region, axes in self.region_axes:
            if region == name:
                return axes
        raise ValueError(f"Region {name!r} is not a compiled spectral region.")

    def _parameter(self, name: str, args: Any, /) -> Array:
        if args is not None and not isinstance(args, Mapping):
            raise TypeError("Spectral PDE args must be a parameter mapping or None.")
        index = self.parameter_names.index(name)
        value = (
            args[name]
            if args is not None and name in args
            else self.parameter_defaults[index]
        )
        if value is None:
            raise KeyError(f"No value supplied for PDE parameter {name!r}.")
        array = jnp.asarray(value)
        if array.dtype.kind not in "biufc":
            raise TypeError(f"PDE parameter {name!r} must be numeric.")
        components = self.parameter_components[index]
        functional = self.parameter_functional[index]
        if not functional:
            expected = () if components == 1 else (components,)
            if array.shape not in (expected, (1,)):
                raise ValueError(f"PDE parameter {name!r} must have shape {expected}.")
            return array[0] if components == 1 and array.shape == (1,) else array
        base = self.discretization.physical_shape + (
            () if components == 1 else (components,)
        )
        target = self.evaluation.physical_shape + (
            () if components == 1 else (components,)
        )
        if array.shape == target:
            return array
        if array.shape != base:
            raise ValueError(
                f"Functional PDE parameter {name!r} must have shape {base} or {target}."
            )
        return self.method.dealiasing.reconstruct(self.discretization.project(array))

    def _coordinate(self, name: str, time: Array, /) -> Array:
        if self.time_coordinate is not None and name == self.time_coordinate:
            return jnp.asarray(time)
        axes = self._axes(name)
        components = []
        for axis in axes:
            nodes = self.evaluation.axes[axis].nodes
            shape = [1] * len(self.evaluation.physical_shape)
            shape[axis] = nodes.size
            components.append(
                jnp.broadcast_to(
                    nodes.reshape(tuple(shape)), self.evaluation.physical_shape
                )
            )
        return (
            components[0]
            if len(components) == 1
            else jnp.stack(tuple(components), axis=-1)
        )

    def _components(self, expression: PDEExpression, /) -> int:
        if expression.op == "field":
            assert expression.symbol is not None
            return self.layout.component_counts[
                self.layout.field_names.index(expression.symbol)
            ]
        if expression.op == "parameter":
            assert expression.symbol is not None
            return self.parameter_components[
                self.parameter_names.index(expression.symbol)
            ]
        if expression.op == "coordinate":
            assert expression.symbol is not None
            return (
                1
                if expression.symbol == self.time_coordinate
                else len(self._axes(expression.symbol))
            )
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

    def _align_scalar(
        self,
        value: Any,
        components: int,
        other_components: int,
        other_value: Any,
        /,
    ) -> Any:
        array = jnp.asarray(value)
        other = jnp.asarray(other_value)
        spatial_rank = len(self.evaluation.physical_shape)
        if (
            components == 1
            and array.ndim == spatial_rank
            and (other_components > 1 or other.ndim == spatial_rank + 1)
        ):
            return array[..., None]
        return value

    def _direct_field_derivative(
        self,
        field_name: str,
        fields: Mapping[str, Array],
        /,
        *,
        axis: int,
        order: int,
    ) -> Array:
        embedded = self.method.dealiasing.embed(fields[field_name])
        return self.evaluation.derivative_values(embedded, axis=axis, order=order)

    def _derivative(
        self,
        expression: PDEExpression,
        value: Array,
        fields: Mapping[str, Array],
        /,
        *,
        axis: int,
        order: int,
    ) -> Array:
        if expression.op == "field":
            assert expression.symbol is not None
            return self._direct_field_derivative(
                expression.symbol,
                fields,
                axis=axis,
                order=order,
            )
        coefficients = self.evaluation.project(value)
        return self.evaluation.derivative_values(coefficients, axis=axis, order=order)

    def _integrate_values(self, values: Array, axes: tuple[int, ...], /) -> Array:
        result = values
        for axis in sorted(axes, reverse=True):
            result = jnp.tensordot(
                self.evaluation.axes[axis].quadrature_weights,
                result,
                axes=((0,), (axis,)),
            )
        spatial_rank = len(self.evaluation.physical_shape)
        trailing = result.shape[spatial_rank - len(axes) :]
        if len(axes) < spatial_rank:
            remaining = iter(result.shape[: spatial_rank - len(axes)])
            spatial_shape = tuple(
                1 if axis in axes else next(remaining) for axis in range(spatial_rank)
            )
            result = result.reshape(spatial_shape + trailing)
        return jnp.broadcast_to(result, self.evaluation.physical_shape + trailing)

    def _evaluate(
        self,
        node: PDEExpression,
        time: Array,
        args: Any,
        fields: Mapping[str, Array],
        cache: dict[PDEExpression, Any],
        /,
    ) -> Any:
        if node in cache:
            return cache[node]
        if node.op == "constant":
            assert node.value is not None
            result: Any = node.value
        elif node.op == "field":
            assert node.symbol is not None
            result = self.method.dealiasing.reconstruct(fields[node.symbol])
        elif node.op == "parameter":
            assert node.symbol is not None
            result = self._parameter(node.symbol, args)
        elif node.op == "coordinate":
            assert node.symbol is not None
            result = self._coordinate(node.symbol, time)
        else:
            values = tuple(
                self._evaluate(argument, time, args, fields, cache)
                for argument in node.args
            )
            if node.op == "add":
                result = values[0]
                for value in values[1:]:
                    result = result + value
            elif node.op == "multiply":
                result = values[0]
                result_components = self._components(node.args[0])
                for argument, value in zip(node.args[1:], values[1:], strict=True):
                    value_components = self._components(argument)
                    left = self._align_scalar(
                        result,
                        result_components,
                        value_components,
                        value,
                    )
                    right = self._align_scalar(
                        value,
                        value_components,
                        result_components,
                        result,
                    )
                    result = left * right
                    result_components = max(result_components, value_components)
            elif node.op == "divide":
                left_components = self._components(node.args[0])
                right_components = self._components(node.args[1])
                result = self._align_scalar(
                    values[0], left_components, right_components, values[1]
                ) / self._align_scalar(
                    values[1], right_components, left_components, values[0]
                )
            elif node.op == "negate":
                result = -values[0]
            elif node.op == "power":
                result = values[0] ** values[1]
            elif node.op == "sin":
                result = jnp.sin(values[0])
            elif node.op == "cos":
                result = jnp.cos(values[0])
            elif node.op == "exp":
                result = jnp.exp(values[0])
            elif node.op == "log":
                result = jnp.log(values[0])
            elif node.op == "sqrt":
                result = jnp.sqrt(values[0])
            elif node.op == "component":
                assert node.axis is not None
                result = values[0][..., node.axis]
            elif node.op == "dot":
                result = jnp.sum(values[0] * values[1], axis=-1)
            elif node.op in ("derivative", "gradient", "divergence", "curl", "laplacian"):
                assert node.coordinate is not None
                axes = self._axes(node.coordinate)
                expression = node.args[0]
                if node.op == "derivative":
                    if node.axis is None:
                        if len(axes) != 1:
                            raise ValueError(
                                "Grouped-coordinate derivatives require an axis."
                            )
                        axis = axes[0]
                    else:
                        axis = axes[node.axis]
                    result = self._derivative(
                        expression,
                        values[0],
                        fields,
                        axis=axis,
                        order=node.order,
                    )
                elif node.op == "gradient":
                    result = jnp.stack(
                        tuple(
                            self._derivative(
                                expression,
                                values[0],
                                fields,
                                axis=axis,
                                order=1,
                            )
                            for axis in axes
                        ),
                        axis=-1,
                    )
                elif node.op == "divergence":
                    if values[0].shape[-1] != len(axes):
                        raise ValueError(
                            "Divergence components must match coordinate axes."
                        )
                    result = jnp.zeros_like(values[0][..., 0])
                    for component, axis in enumerate(axes):
                        result = result + self._derivative(
                            expression.component(component),
                            values[0][..., component],
                            fields,
                            axis=axis,
                            order=1,
                        )
                elif node.op == "curl":
                    if len(axes) != 3 or values[0].shape[-1] != 3:
                        raise ValueError("Curl requires three axes and components.")
                    first, second, third = axes
                    derivative = lambda component, axis: self._derivative(
                        expression.component(component),
                        values[0][..., component],
                        fields,
                        axis=axis,
                        order=1,
                    )
                    result = jnp.stack(
                        (
                            derivative(2, second) - derivative(1, third),
                            derivative(0, third) - derivative(2, first),
                            derivative(1, first) - derivative(0, second),
                        ),
                        axis=-1,
                    )
                else:
                    result = jnp.zeros_like(values[0])
                    for axis in axes:
                        result = result + self._derivative(
                            expression,
                            values[0],
                            fields,
                            axis=axis,
                            order=2,
                        )
            elif node.op == "integral":
                assert node.region is not None
                result = self._integrate_values(values[0], self._region(node.region))
            else:
                raise ValueError(f"Unsupported spectral PDE operation {node.op!r}.")
        cache[node] = result
        return result

    def _coerce_physical(
        self,
        name: str,
        components: int,
        value: Any,
        /,
    ) -> Array:
        result = jnp.asarray(value)
        expected = self.evaluation.physical_shape + (
            () if components == 1 else (components,)
        )
        if result.shape == () or (components > 1 and result.shape == (components,)):
            return jnp.broadcast_to(result, expected)
        if components == 1 and result.shape == expected + (1,):
            return result[..., 0]
        compatible = result.ndim == len(expected) and all(
            actual in (1, target)
            for actual, target in zip(result.shape, expected, strict=True)
        )
        if compatible:
            return jnp.broadcast_to(result, expected)
        raise ValueError(
            f"Spectral output {name!r} must have shape {expected}; got {result.shape}."
        )

    def physical_outputs(self, time: Array, state: Array, args: Any) -> tuple[Array, ...]:
        value = jnp.asarray(state)
        if value.shape != self.layout.state_shape:
            raise ValueError(
                f"Spectral state must have shape {self.layout.state_shape}; got {value.shape}."
            )
        fields = self.layout.unpack(value)
        cache: dict[PDEExpression, Any] = {}
        return tuple(
            self._coerce_physical(
                name,
                components,
                self._evaluate(expression, time, args, fields, cache),
            )
            for name, components, expression in zip(
                self.output_names,
                self.output_components,
                self.rhs_expressions,
                strict=True,
            )
        )

    def __call__(self, time: Array, state: Array, args: Any) -> Array:
        outputs = self.physical_outputs(time, state, args)
        if len(outputs) != len(self.layout.field_names):
            raise ValueError(
                "Spectral dynamics require one evaluator output per state field."
            )
        derivatives = {
            name: self.method.dealiasing.project(physical)
            for name, physical in zip(
                self.layout.field_names,
                outputs,
                strict=True,
            )
        }
        return self.layout.pack(derivatives)


class _LinearizedSpectralOperator(StrictModule):
    evaluator: _SpectralEvaluator

    def __call__(self, state: Array) -> Array:
        value = jnp.asarray(state)
        zero = jnp.zeros_like(value)
        time = jnp.asarray(0.0, dtype=value.real.dtype)
        return jax.jvp(
            lambda candidate: self.evaluator(time, candidate, None),
            (zero,),
            (value,),
        )[1]


class _SpectralRemainder(StrictModule):
    evaluator: _SpectralEvaluator
    linear_operator: Any

    def __call__(self, time: Array, state: Array, args: Any) -> Array:
        return self.evaluator(time, state, args) - self.linear_operator(state)


class CompiledSpectralDynamics(StrictModule):
    """Coefficient-resident spectral dynamics and physical projection interface."""

    drift: Any
    layout: SpectralStateLayout
    discretization: TensorSpectralDiscretization
    spatial_method: Any
    semilinear_drift: SemilinearDrift | None
    evaluator: _SpectralEvaluator
    discretization_bundle: DiscretizationBundle
    compilation_id: str = eqx.field(static=True)
    source_hash: str = eqx.field(static=True)
    resolved_method: str = eqx.field(static=True)

    def __init__(
        self,
        drift: Any,
        layout: SpectralStateLayout,
        discretization: TensorSpectralDiscretization,
        spatial_method: Any,
        evaluator: _SpectralEvaluator,
        /,
        *,
        semilinear_drift: SemilinearDrift | None,
        compilation_id: str,
        source_hash: str,
        resolved_method: str,
    ):
        residual_key = DiscretizationKey(
            "spectral_form",
            DiscretizationRole.RESIDUAL,
            domain_labels=discretization.key.domain_labels,
        )
        bundle = DiscretizationBundle(
            (
                DiscretizationRecord(
                    discretization.key,
                    type(discretization).__name__,
                    discretization.prepared_id,
                    numeric_version=discretization.numeric_version,
                    precision_evidence_id=discretization.precision_evidence_id,
                    resource_evidence_id=discretization.resource_evidence_id,
                ),
                DiscretizationRecord(
                    residual_key,
                    "compiled-spectral-form",
                    compilation_id,
                    dependency_key_ids=(discretization.key.key_id,),
                ),
            )
        )
        self.drift = drift
        self.layout = layout
        self.discretization = discretization
        self.spatial_method = spatial_method
        self.semilinear_drift = semilinear_drift
        self.evaluator = evaluator
        self.discretization_bundle = bundle
        self.compilation_id = str(compilation_id)
        self.source_hash = str(source_hash)
        self.resolved_method = str(resolved_method)

    @property
    def state_shape(self) -> tuple[int, ...]:
        return self.layout.state_shape

    def project_state(self, values: ArrayLike | Mapping[str, ArrayLike], /) -> Array:
        physical = (
            values
            if isinstance(values, Mapping)
            else self.layout.unpack(values, physical=True)
        )
        coefficients = {
            name: self.discretization.project(physical[name])
            for name in self.layout.field_names
        }
        return self.layout.pack(coefficients)

    def reconstruct_state(self, state: ArrayLike, /) -> Array:
        coefficients = self.layout.unpack(state)
        physical = {
            name: self.discretization.reconstruct(coefficients[name])
            for name in self.layout.field_names
        }
        return self.layout.pack(physical, physical=True)

    def physical_state(
        self, time: ArrayLike, state: ArrayLike, args: Any = None, /
    ) -> Array:
        del time, args
        return self.reconstruct_state(state)

    def __call__(self, time: Array, state: Array, args: Any) -> Array:
        return self.drift(jnp.asarray(time), state, args)


def _field_degree(expression: PDEExpression, /) -> int | None:
    if expression.op == "field":
        return 1
    if expression.op in ("constant", "coordinate", "parameter"):
        return 0
    degrees = tuple(_field_degree(argument) for argument in expression.args)
    if any(value is None for value in degrees):
        return None
    finite = tuple(int(value) for value in degrees if value is not None)
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
        return max(finite, default=0)
    if expression.op in ("multiply", "dot"):
        return sum(finite)
    if expression.op == "divide":
        return finite[0] if finite[1] == 0 else None
    if expression.op == "power":
        exponent = expression.args[1]
        if (
            exponent.op != "constant"
            or exponent.value is None
            or not exponent.value.is_integer()
        ):
            return None
        power = int(exponent.value)
        return finite[0] * power if power >= 0 else None
    if expression.op in ("sin", "cos", "exp", "log", "sqrt"):
        return 0 if finite[0] == 0 else None
    return None


def _linear_symbol(
    expression: PDEExpression,
    field_name: str,
    discretization: TensorSpectralDiscretization,
    coordinate_axes: Mapping[str, tuple[int, ...]],
    parameter_values: Mapping[str, Any],
    /,
) -> tuple[Array, Array] | None:
    dtype = jnp.dtype(discretization.plan.precision.coefficient_dtype)
    zero = jnp.zeros(discretization.modal_shape, dtype=dtype)
    if expression.op == "constant":
        assert expression.value is not None
        return jnp.asarray(expression.value, dtype=dtype), zero
    if expression.op == "parameter":
        assert expression.symbol is not None
        if expression.symbol not in parameter_values:
            return None
        value = jnp.asarray(parameter_values[expression.symbol], dtype=dtype)
        return (value, zero) if value.shape == () else None
    if expression.op == "coordinate":
        return None
    if expression.op == "field":
        assert expression.symbol is not None
        return (
            (
                jnp.asarray(0.0, dtype=dtype),
                jnp.ones(discretization.modal_shape, dtype=dtype),
            )
            if expression.symbol == field_name
            else None
        )
    children = tuple(
        _linear_symbol(
            argument,
            field_name,
            discretization,
            coordinate_axes,
            parameter_values,
        )
        for argument in expression.args
    )
    if expression.op == "add":
        if any(child is None for child in children):
            return None
        constant = jnp.asarray(0.0, dtype=dtype)
        symbol = zero
        for child in children:
            assert child is not None
            constant = constant + child[0]
            symbol = symbol + child[1]
        return constant, symbol
    if expression.op == "negate":
        child = children[0]
        return None if child is None else (-child[0], -child[1])
    if expression.op == "multiply":
        if any(child is None for child in children):
            return None
        constant = jnp.asarray(1.0, dtype=dtype)
        symbol = zero
        for child in children:
            assert child is not None
            symbol = symbol * child[0] + constant * child[1]
            constant = constant * child[0]
        return constant, symbol
    if expression.op == "divide":
        numerator, denominator = children
        if numerator is None or denominator is None:
            return None
        if bool(jnp.any(denominator[1] != 0)) or bool(denominator[0] == 0):
            return None
        return (
            numerator[0] / denominator[0],
            numerator[1] / denominator[0],
        )
    if expression.op == "power":
        base = children[0]
        exponent = expression.args[1]
        if (
            base is None
            or exponent.op != "constant"
            or exponent.value is None
            or not exponent.value.is_integer()
        ):
            return None
        power = int(exponent.value)
        if power < 0 or (power == 0 and bool(base[0] == 0)):
            return None
        if power == 0:
            return jnp.asarray(1.0, dtype=dtype), zero
        return base[0] ** power, power * base[0] ** (power - 1) * base[1]
    if expression.op in ("sin", "cos", "exp"):
        child = children[0]
        if child is None:
            return None
        if expression.op == "sin":
            return jnp.sin(child[0]), jnp.cos(child[0]) * child[1]
        if expression.op == "cos":
            return jnp.cos(child[0]), -jnp.sin(child[0]) * child[1]
        return jnp.exp(child[0]), jnp.exp(child[0]) * child[1]
    if expression.op in (
        "log",
        "sqrt",
        "component",
        "dot",
        "gradient",
        "divergence",
        "curl",
        "integral",
    ):
        return None
    if expression.op == "derivative":
        child = children[0]
        if child is None or expression.coordinate not in coordinate_axes:
            return None
        axes = coordinate_axes[expression.coordinate]
        if expression.axis is None:
            if len(axes) != 1:
                return None
            axis = axes[0]
        else:
            axis = axes[expression.axis]
        prepared = discretization.axes[axis]
        if (
            prepared.derivative_matrix is not None
            or prepared.family not in ("fourier", "sine", "cosine")
            or (prepared.family != "fourier" and expression.order % 2)
        ):
            return None
        multiplier = prepared.derivative_multiplier(expression.order)
        shape = [1] * len(discretization.modal_shape)
        shape[axis] = multiplier.size
        return (
            jnp.asarray(0.0, dtype=dtype),
            child[1] * multiplier.reshape(tuple(shape)),
        )
    if expression.op == "laplacian":
        child = children[0]
        if child is None or expression.coordinate not in coordinate_axes:
            return None
        symbol = zero
        for axis in coordinate_axes[expression.coordinate]:
            prepared = discretization.axes[axis]
            if prepared.derivative_matrix is not None or prepared.family not in (
                "fourier",
                "sine",
                "cosine",
            ):
                return None
            multiplier = discretization.axes[axis].derivative_multiplier(2)
            shape = [1] * len(discretization.modal_shape)
            shape[axis] = multiplier.size
            symbol = symbol + child[1] * multiplier.reshape(tuple(shape))
        return jnp.asarray(0.0, dtype=dtype), symbol
    return None


def _coordinate_axes(
    problem: PDEProblemIR,
    discretization: TensorSpectralDiscretization,
    /,
) -> tuple[str, tuple[tuple[str, tuple[int, ...]], ...]]:
    temporal = tuple(
        coordinate for coordinate in problem.coordinates if coordinate.kind == "time"
    )
    spatial = tuple(
        coordinate for coordinate in problem.coordinates if coordinate.kind == "space"
    )
    if len(temporal) != 1 or not spatial:
        raise ValueError(
            "Spectral PDE compilation requires one time and spatial coordinates."
        )
    if sum(coordinate.size for coordinate in spatial) != len(discretization.axes):
        raise ValueError("PDE spatial coordinate size must match spectral tensor rank.")
    output = []
    offset = 0
    for coordinate in spatial:
        axes = tuple(range(offset, offset + coordinate.size))
        for axis in axes:
            prepared = discretization.axes[axis]
            if coordinate.periodic != prepared.periodic:
                raise ValueError(
                    f"PDE coordinate {coordinate.name!r} periodicity does not match "
                    f"spectral basis {prepared.family!r}."
                )
            if coordinate.bounds is not None:
                actual_bounds = prepared.bounds
                if actual_bounds is None or not jnp.allclose(
                    jnp.asarray(coordinate.bounds),
                    actual_bounds,
                ):
                    raise ValueError(
                        f"PDE coordinate {coordinate.name!r} bounds do not match "
                        "the spectral axis domain."
                    )
        output.append((coordinate.name, axes))
        offset += coordinate.size
    return temporal[0].name, tuple(output)


def _region_axes(
    problem: PDEProblemIR,
    coordinate_axes: tuple[tuple[str, tuple[int, ...]], ...],
    /,
) -> tuple[tuple[str, tuple[int, ...]], ...]:
    lookup = dict(coordinate_axes)
    output = []
    for region in problem.regions:
        axes = tuple(
            axis
            for coordinate in region.coordinates
            if coordinate in lookup
            for axis in lookup[coordinate]
        )
        if region.kind != "interior" or region.component is not None or not axes:
            raise ValueError(
                "Spectral integration initially supports unpartitioned interior regions."
            )
        output.append((region.name, axes))
    return tuple(output)


def compile_spectral_pde(
    problem: PDEProblemIR,
    discretization: TensorSpectralDiscretization,
    method: PseudospectralMethodPlan,
    /,
    *,
    parameter_values: Mapping[str, Any] | None = None,
    boundary_lifts: Sequence[Any] = (),
    splitting: str = "auto",
) -> CompiledSpectralDynamics:
    """Lower PDE IR to coefficient-resident global pseudospectral dynamics."""
    if not isinstance(problem, PDEProblemIR):
        raise TypeError("problem must be a PDEProblemIR.")
    if not isinstance(discretization, TensorSpectralDiscretization):
        raise TypeError("discretization must be a TensorSpectralDiscretization.")
    if not isinstance(method, PseudospectralMethodPlan):
        raise TypeError("method must be a PseudospectralMethodPlan.")
    if boundary_lifts:
        raise ValueError(
            "Tensor spectral boundary lifts require the bounded constrained-basis "
            "compiler and are not accepted by the initial pseudospectral path."
        )
    if splitting not in ("auto", "direct", "semilinear"):
        raise ValueError("splitting must be 'auto', 'direct', or 'semilinear'.")
    validate_pde_ir(problem)
    from ._semidiscrete import _evolution_rhs

    time_coordinate, coordinate_axes = _coordinate_axes(problem, discretization)
    rhs = _evolution_rhs(problem, time_coordinate)
    degrees = tuple(_field_degree(expression) for expression in rhs)
    nonlinear = any(value is None or value > 1 for value in degrees)
    required_degree = None if any(value is None for value in degrees) else max(degrees)
    prepared_method = method.prepare(
        discretization,
        required_polynomial_degree=required_degree,
        nonlinear=nonlinear,
    )
    supplied = {} if parameter_values is None else dict(parameter_values)
    unknown = set(supplied) - {parameter.name for parameter in problem.parameters}
    if unknown:
        raise KeyError(
            f"Unknown spectral PDE parameter values: {tuple(sorted(unknown))}."
        )
    defaults = tuple(
        supplied.get(parameter.name, parameter.value) for parameter in problem.parameters
    )
    parameter_fingerprints = {
        parameter.name: (
            None if value is None else array_tree_fingerprint(jnp.asarray(value))
        )
        for parameter, value in zip(problem.parameters, defaults, strict=True)
    }
    unresolved_parameters = tuple(
        name for name, value in parameter_fingerprints.items() if value is None
    )
    if splitting == "semilinear" and unresolved_parameters:
        raise ValueError(
            "Semilinear spectral compilation requires concrete values for parameters "
            f"{unresolved_parameters}."
        )
    layout = SpectralStateLayout(problem.fields, discretization)
    evaluator = _SpectralEvaluator(
        problem,
        rhs,
        layout.field_names,
        layout.component_counts,
        layout,
        discretization,
        prepared_method,
        defaults,
        coordinate_axes,
        time_coordinate,
        _region_axes(problem, coordinate_axes),
    )
    semilinear = None
    use_direct = splitting == "direct" or (
        splitting == "auto" and bool(unresolved_parameters)
    )
    if use_direct:
        drift: Any = evaluator
        resolved = "spectral-direct"
    else:
        from ..solver._semilinear_drift import SemilinearDrift

        state_space = ArraySpace(
            layout.state_shape,
            dtype=jnp.dtype(discretization.plan.precision.coefficient_dtype),
        )
        defaults_by_name = {
            parameter.name: value
            for parameter, value in zip(problem.parameters, defaults, strict=True)
            if value is not None
        }
        diagonal_data = (
            _linear_symbol(
                rhs[0],
                layout.field_names[0],
                discretization,
                dict(coordinate_axes),
                defaults_by_name,
            )
            if layout.squeezed
            else None
        )
        operator_id = canonical_fingerprint(
            {
                "kind": (
                    "spectral-diagonal-linear-operator"
                    if diagonal_data is not None
                    else "spectral-linearized-operator"
                ),
                "problem": problem.canonical_hash,
                "discretization": discretization.prepared_id,
                "method": prepared_method.prepared_id,
                "parameters": parameter_fingerprints,
            }
        )
        if diagonal_data is None:
            operator = FunctionLinearOperator(
                _LinearizedSpectralOperator(evaluator),
                source=state_space,
                target=state_space,
                properties=OperatorProperties(),
                operator_id=operator_id,
            )
            resolved = "spectral-semilinear-matrix-free"
        else:
            operator = DiagonalLinearOperator(
                diagonal_data[1].reshape((-1,)),
                space=state_space,
                operator_id=operator_id,
            )
            resolved = "spectral-semilinear-diagonal"
        semilinear = SemilinearDrift(
            operator,
            _SpectralRemainder(evaluator, operator),
            state_shape=layout.state_shape,
            operator_id=operator_id,
        )
        drift = semilinear
    compilation_id = canonical_fingerprint(
        {
            "kind": "spectral-pde-compiler-v1",
            "problem": problem.canonical_hash,
            "discretization": discretization.prepared_id,
            "method": prepared_method.prepared_id,
            "layout": layout.layout_id,
            "splitting": resolved,
            "parameters": parameter_fingerprints,
        }
    )
    return CompiledSpectralDynamics(
        drift,
        layout,
        discretization,
        prepared_method,
        evaluator,
        semilinear_drift=semilinear,
        compilation_id=compilation_id,
        source_hash=problem.canonical_hash,
        resolved_method=resolved,
    )


__all__ = [
    "CompiledSpectralDynamics",
    "SpectralStateLayout",
    "compile_spectral_pde",
]
