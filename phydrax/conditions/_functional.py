#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import coordax as cx
import equinox as eqx
import jax.numpy as jnp
import opt_einsum as oe
from jaxtyping import Array, ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from ..domain import DomainFunction, PointBatch
from ..integration._linear import PreparedLinearReduction
from ..operators.differential._domain_ops import partial_n
from ._ir import (
    AbstractConditionOperator,
    Condition,
    ConditionCodomain,
    ConditionQuantifier,
    OperatorCapabilities,
    ProductFieldSpec,
)
from ._relations import Equality


def _point_axis(batch: PointBatch, /) -> tuple[str | None, int]:
    axes = batch.structure.axis_names
    if axes is None:
        raise ValueError("Point-functional layout must be canonicalized.")
    if not axes:
        return None, 1
    if len(axes) != 1:
        raise ValueError("Finite point functionals require one coupled sampling axis.")
    axis = axes[0]
    counts = tuple(
        int(field.data.shape[field.dims.index(axis)])
        for field in batch.points.values()
        if isinstance(field, cx.Field) and axis in field.dims
    )
    if not counts or len(set(counts)) != 1:
        raise ValueError("Point-functional coordinates disagree on sampling-axis size.")
    return axis, counts[0]


def _derivative_requests(
    requests: Sequence[tuple[str, int | None, int]], /
) -> tuple[tuple[str, int | None, int], ...]:
    values = tuple(
        (str(variable), axis, int(order)) for variable, axis, order in requests
    )
    if any(not variable or order <= 0 for variable, _, order in values):
        raise ValueError("Point derivative requests need names and positive orders.")
    if any(axis is not None and int(axis) < 0 for _, axis, _ in values):
        raise ValueError("Point derivative axes must be nonnegative or None.")
    return values


class EventLinearMap(StrictModule):
    """Fixed linear map over one trailing event axis."""

    matrix: Array
    map_id: str = eqx.field(static=True)

    def __init__(self, matrix: ArrayLike, /, *, map_id: str | None = None):
        value = jnp.asarray(matrix)
        if value.ndim != 2 or not jnp.issubdtype(value.dtype, jnp.inexact):
            raise TypeError("EventLinearMap.matrix must be a rank-two inexact array.")
        if int(value.shape[0]) <= 0 or int(value.shape[1]) <= 0:
            raise ValueError("EventLinearMap dimensions must be positive.")
        self.matrix = eqx.error_if(
            value, jnp.any(~jnp.isfinite(value)), "EventLinearMap must be finite."
        )
        self.map_id = (
            canonical_fingerprint(
                {"kind": "event-linear-map", "matrix": array_tree_fingerprint(value)}
            )
            if map_id is None
            else str(map_id)
        )
        if not self.map_id:
            raise ValueError("EventLinearMap.map_id must be nonempty.")

    def __call__(self, value: ArrayLike, /) -> Array:
        array = jnp.asarray(value)
        if array.ndim == 0 or int(array.shape[-1]) != int(self.matrix.shape[1]):
            raise ValueError("EventLinearMap input has the wrong trailing event size.")
        return oe.contract("oi,...i->...o", self.matrix, array)


class PointJetAction(AbstractConditionOperator):
    """Certified finite linear action on point values or mixed coordinate jets."""

    field: str = eqx.field(static=True)
    batch: PointBatch
    coefficients: Array
    derivatives: tuple[tuple[str, int | None, int], ...] = eqx.field(static=True)
    event_map: EventLinearMap | None
    capabilities: OperatorCapabilities = eqx.field(static=True)
    action_id: str = eqx.field(static=True)

    def __init__(
        self,
        field: str,
        batch: PointBatch,
        coefficients: ArrayLike,
        /,
        *,
        derivatives: Sequence[tuple[str, int | None, int]] = (),
        event_map: EventLinearMap | None = None,
    ):
        name = str(field)
        if not name or not isinstance(batch, PointBatch):
            raise TypeError("PointJetAction requires a field name and PointBatch.")
        _, count = _point_axis(batch)
        rows = jnp.asarray(coefficients)
        if rows.ndim != 2 or int(rows.shape[1]) != count:
            raise ValueError("Point-jet coefficients must have shape (rows, points).")
        if not jnp.issubdtype(rows.dtype, jnp.inexact):
            rows = rows.astype(float)
        if event_map is not None and not isinstance(event_map, EventLinearMap):
            raise TypeError("event_map must be EventLinearMap or None.")
        requests = _derivative_requests(derivatives)
        self.field = name
        self.batch = batch
        self.coefficients = eqx.error_if(
            rows, jnp.any(~jnp.isfinite(rows)), "Point-jet coefficients must be finite."
        )
        self.derivatives = requests
        self.event_map = event_map
        self.capabilities = OperatorCapabilities(is_linear=True)
        self.action_id = canonical_fingerprint(
            {
                "kind": "point-jet-action",
                "field": name,
                "batch": array_tree_fingerprint(batch.points),
                "layout": repr(batch.structure),
                "coefficients": array_tree_fingerprint(rows),
                "derivatives": requests,
                "event_map": None if event_map is None else event_map.map_id,
            }
        )

    def _apply(self, values: Mapping[str, Any], /, *, key=None, **kwargs: Any) -> Array:
        if self.field not in values:
            raise KeyError(f"Missing point-jet field {self.field!r}.")
        function = values[self.field]
        if not isinstance(function, DomainFunction):
            raise TypeError("PointJetAction acts on DomainFunction values.")
        for variable, axis, order in self.derivatives:
            function = partial_n(function, var=variable, order=order, axis=axis)
        evaluated = function(self.batch, key=key, **kwargs)
        if not isinstance(evaluated, cx.Field):
            raise TypeError("Point-jet evaluation must return coordax.Field.")
        sample_axis, count = _point_axis(self.batch)
        data = jnp.asarray(evaluated.data)
        if sample_axis is None:
            data = data[None, ...]
        else:
            if sample_axis not in evaluated.dims:
                raise ValueError("Point-jet output lost its sampling axis.")
            data = jnp.moveaxis(data, evaluated.dims.index(sample_axis), 0)
        if int(data.shape[0]) != count:
            raise ValueError("Point-jet output row count changed.")
        if self.event_map is not None:
            data = self.event_map(data)
        return oe.contract("rn,n...->r...", self.coefficients, data)

    def apply(self, values, /, *, key=None, **kwargs):
        return self._apply(values, key=key, **kwargs)

    def linear_action(self, values, /, *, key=None, **kwargs):
        return self._apply(values, key=key, **kwargs)

    def adjoint_action(self, value, /, *, key=None, **kwargs):
        del value, key, kwargs
        raise TypeError(
            "PointJetAction function-space adjoints require a representation provider."
        )

    def linearize(self, values, /, *, key=None, **kwargs):
        del values, key, kwargs
        raise TypeError("A globally linear PointJetAction does not need linearization.")


class LinearReductionAction(AbstractConditionOperator):
    """Certified finite linear action through one prepared integration reduction."""

    field: str = eqx.field(static=True)
    reduction: PreparedLinearReduction
    coefficients: Array
    capabilities: OperatorCapabilities = eqx.field(static=True)
    action_id: str = eqx.field(static=True)

    def __init__(
        self,
        field: str,
        reduction: PreparedLinearReduction,
        coefficients: ArrayLike,
        /,
    ):
        name = str(field)
        if not name or not isinstance(reduction, PreparedLinearReduction):
            raise TypeError(
                "LinearReductionAction requires a field and PreparedLinearReduction."
            )
        rows = jnp.asarray(coefficients)
        if rows.ndim != 1 or int(rows.shape[0]) <= 0:
            raise ValueError(
                "Reduction coefficients must contain one value per equation row."
            )
        if not jnp.issubdtype(rows.dtype, jnp.inexact):
            rows = rows.astype(float)
        self.field = name
        self.reduction = reduction
        self.coefficients = eqx.error_if(
            rows, jnp.any(~jnp.isfinite(rows)), "Reduction coefficients must be finite."
        )
        self.capabilities = OperatorCapabilities(is_linear=True)
        self.action_id = canonical_fingerprint(
            {
                "kind": "linear-reduction-action",
                "field": name,
                "reduction": reduction.realization_id,
                "coefficients": array_tree_fingerprint(rows),
            }
        )

    def _apply(self, values: Mapping[str, Any], /, *, key=None, **kwargs: Any) -> Array:
        if self.field not in values:
            raise KeyError(f"Missing reduction field {self.field!r}.")
        function = values[self.field]
        if not isinstance(function, DomainFunction):
            raise TypeError("LinearReductionAction acts on DomainFunction values.")
        reduced = self.reduction.apply(function, key=key, **kwargs)
        if isinstance(reduced, cx.Field):
            if reduced.named_dims:
                raise ValueError("Finite LinearReductionAction cannot retain named axes.")
            data = jnp.asarray(reduced.data)
        else:
            data = jnp.asarray(reduced)
        return self.coefficients.reshape((-1,) + (1,) * data.ndim) * data

    def apply(self, values, /, *, key=None, **kwargs):
        return self._apply(values, key=key, **kwargs)

    def linear_action(self, values, /, *, key=None, **kwargs):
        return self._apply(values, key=key, **kwargs)

    def adjoint_action(self, value, /, *, key=None, **kwargs):
        del value, key, kwargs
        raise TypeError("Reduction adjoints require a representation or metric provider.")

    def linearize(self, values, /, *, key=None, **kwargs):
        del values, key, kwargs
        raise TypeError("A globally linear reduction does not need linearization.")


class MatrixLinearFunctional(AbstractConditionOperator):
    """Finite joint linear action over flattened array-valued source fields."""

    field_names: tuple[str, ...] = eqx.field(static=True)
    input_shapes: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    matrices: tuple[Array, ...]
    output_shape: tuple[int, ...] = eqx.field(static=True)
    capabilities: OperatorCapabilities = eqx.field(static=True)
    operator_id: str = eqx.field(static=True)

    def __init__(
        self,
        field_names: Sequence[str],
        input_shapes: Sequence[Sequence[int]],
        matrices: Sequence[ArrayLike],
        /,
        *,
        output_shape: Sequence[int] | None = None,
    ):
        names = tuple(str(name) for name in field_names)
        shapes = tuple(tuple(int(size) for size in shape) for shape in input_shapes)
        blocks = tuple(jnp.asarray(matrix) for matrix in matrices)
        if (
            not names
            or len(set(names)) != len(names)
            or len(names) != len(shapes)
            or len(names) != len(blocks)
        ):
            raise ValueError(
                "MatrixLinearFunctional needs aligned unique field names, shapes, and matrices."
            )
        if any(any(size <= 0 for size in shape) for shape in shapes):
            raise ValueError("MatrixLinearFunctional input dimensions must be positive.")
        row_count: int | None = None
        normalized = []
        for shape, block in zip(shapes, blocks, strict=True):
            value = (
                block.astype(float)
                if not jnp.issubdtype(block.dtype, jnp.inexact)
                else block
            )
            size = 1
            for dimension in shape:
                size *= dimension
            if value.ndim != 2 or int(value.shape[1]) != size:
                raise ValueError(
                    "Each matrix block must have shape (output_size, flattened_input_size)."
                )
            if row_count is None:
                row_count = int(value.shape[0])
            elif int(value.shape[0]) != row_count:
                raise ValueError("MatrixLinearFunctional blocks must share row count.")
            normalized.append(
                eqx.error_if(
                    value,
                    jnp.any(~jnp.isfinite(value)),
                    "MatrixLinearFunctional blocks must be finite.",
                )
            )
        if row_count is None or row_count <= 0:
            raise ValueError("MatrixLinearFunctional output size must be positive.")
        output = (
            (row_count,)
            if output_shape is None
            else tuple(int(size) for size in output_shape)
        )
        output_size = 1
        for dimension in output:
            if dimension <= 0:
                raise ValueError(
                    "MatrixLinearFunctional output dimensions must be positive."
                )
            output_size *= dimension
        if output_size != row_count:
            raise ValueError("output_shape size must match the matrix row count.")
        self.field_names = names
        self.input_shapes = shapes
        self.matrices = tuple(normalized)
        self.output_shape = output
        self.capabilities = OperatorCapabilities(
            is_linear=True,
            has_adjoint=True,
        )
        self.operator_id = canonical_fingerprint(
            {
                "kind": "matrix-linear-functional",
                "fields": names,
                "input_shapes": shapes,
                "output_shape": output,
                "matrices": tuple(array_tree_fingerprint(value) for value in normalized),
            }
        )

    def _apply(self, values: Mapping[str, Any], /) -> Array:
        if tuple(values.keys()) != self.field_names:
            missing = tuple(name for name in self.field_names if name not in values)
            extra = tuple(name for name in values if name not in self.field_names)
            if missing or extra:
                raise ValueError(
                    f"Matrix functional sources mismatch; missing={missing}, extra={extra}."
                )
        result = None
        for name, shape, matrix in zip(
            self.field_names, self.input_shapes, self.matrices, strict=True
        ):
            value = jnp.asarray(values[name])
            if value.shape != shape:
                raise ValueError(
                    f"Matrix functional field {name!r} has shape {value.shape}; expected {shape}."
                )
            contribution = oe.contract("oi,i->o", matrix, value.reshape((-1,)))
            result = contribution if result is None else result + contribution
        if result is None:
            raise RuntimeError("MatrixLinearFunctional lost every source block.")
        return result.reshape(self.output_shape)

    def apply(self, values, /, *, key=None, **kwargs):
        del key, kwargs
        return self._apply(values)

    def linear_action(self, values, /, *, key=None, **kwargs):
        del key, kwargs
        return self._apply(values)

    def adjoint_action(self, value, /, *, key=None, **kwargs):
        del key, kwargs
        covector = jnp.asarray(value)
        if covector.shape != self.output_shape:
            raise ValueError("Matrix functional cotangent has the wrong shape.")
        flat = covector.reshape((-1,))
        return {
            name: oe.contract("oi,o->i", jnp.conj(matrix), flat).reshape(shape)
            for name, shape, matrix in zip(
                self.field_names, self.input_shapes, self.matrices, strict=True
            )
        }

    def linearize(self, values, /, *, key=None, **kwargs):
        del values, key, kwargs
        raise TypeError(
            "A globally linear matrix functional does not need linearization."
        )


class LinearFunctional(AbstractConditionOperator):
    """Row-aligned sum of complete certified linear condition actions."""

    terms: tuple[AbstractConditionOperator, ...]
    capabilities: OperatorCapabilities = eqx.field(static=True)
    operator_id: str = eqx.field(static=True)

    def __init__(self, terms: Sequence[AbstractConditionOperator], /):
        values = tuple(terms)
        if not values or any(
            not isinstance(term, AbstractConditionOperator) for term in values
        ):
            raise TypeError("LinearFunctional terms must be condition operators.")
        if any(not term.capabilities.is_linear for term in values):
            raise TypeError("Every LinearFunctional term must certify linearity.")
        self.terms = values
        self.capabilities = OperatorCapabilities(is_linear=True)
        term_ids = tuple(
            (
                term.action_id
                if isinstance(term, (PointJetAction, LinearReductionAction))
                else type(term).__qualname__
            )
            for term in values
        )
        self.operator_id = canonical_fingerprint(
            {"kind": "linear-functional", "terms": term_ids}
        )

    def _apply(self, values: Mapping[str, Any], /, *, key=None, **kwargs: Any) -> Any:
        outputs = tuple(
            term.linear_action(values, key=key, **kwargs) for term in self.terms
        )
        reference = jnp.asarray(outputs[0])
        if any(jnp.asarray(output).shape != reference.shape for output in outputs[1:]):
            raise ValueError("LinearFunctional terms must return exactly equal shapes.")
        result = reference
        for output in outputs[1:]:
            result = result + jnp.asarray(output)
        return result

    def apply(self, values, /, *, key=None, **kwargs):
        return self._apply(values, key=key, **kwargs)

    def linear_action(self, values, /, *, key=None, **kwargs):
        return self._apply(values, key=key, **kwargs)

    def adjoint_action(self, value, /, *, key=None, **kwargs):
        del value, key, kwargs
        raise TypeError(
            "LinearFunctional adjoints require a representation or metric provider."
        )

    def linearize(self, values, /, *, key=None, **kwargs):
        del values, key, kwargs
        raise TypeError("A globally linear functional does not need linearization.")


def linear_functional_condition(
    condition_id: str,
    fields: ProductFieldSpec,
    terms: Sequence[AbstractConditionOperator],
    codomain: ConditionCodomain,
    target: Any,
    /,
    *,
    quantifier: ConditionQuantifier | str = ConditionQuantifier.deterministic,
    label: str | None = None,
) -> Condition:
    """Construct one typed equality from row-aligned linear actions."""
    return Condition(
        condition_id,
        fields,
        LinearFunctional(terms),
        codomain,
        Equality(target),
        quantifier=quantifier,
        label=label,
    )


__all__ = [
    "EventLinearMap",
    "LinearFunctional",
    "LinearReductionAction",
    "MatrixLinearFunctional",
    "PointJetAction",
    "linear_functional_condition",
]
