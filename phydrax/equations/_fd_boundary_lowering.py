#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization import (
    BoundaryStageContext,
    BoundaryWorkspace,
    CellGhostBoundary,
    ConformingInterfaceRuntime,
    CornerPolicy,
    NodalBoundaryRuntime,
    PreparedTensorGrid,
)
from ..discretization.finite_difference._boundary_runtime import GhostConditionKind
from ._ir import PDECondition, PDEExpression, PDEProblemIR, PDERegion


BoundarySide: TypeAlias = Literal["lower", "upper"]
BoundaryTargetKind: TypeAlias = Literal["constant", "parameter", "expression"]


def _evaluate_target_expression(
    expression: PDEExpression,
    context: BoundaryStageContext,
    boundary_axis: str,
    side: BoundarySide,
    field_shape: tuple[int, ...],
    /,
) -> Array:
    op = expression.op
    if op == "constant":
        return jnp.asarray(float(expression.value))
    if op == "parameter":
        if not isinstance(context.args, Mapping) or expression.symbol not in context.args:
            raise ValueError(
                f"Boundary target requires runtime parameter {expression.symbol!r}."
            )
        return jnp.asarray(context.args[str(expression.symbol)])
    boundary_index = context.axis_names.index(boundary_axis)
    if op == "coordinate":
        name = str(expression.symbol)
        if name not in context.axis_names:
            return context.time
        coordinate_index = context.axis_names.index(name)
        coordinate = context.coordinates[coordinate_index]
        if coordinate_index == boundary_index:
            return coordinate[0] if side == "lower" else coordinate[-1]
        target_shape = field_shape[:boundary_index] + field_shape[boundary_index + 1 :]
        target_axis = (
            coordinate_index
            if coordinate_index < boundary_index
            else coordinate_index - 1
        )
        reshape = [1] * len(target_shape)
        reshape[target_axis] = int(coordinate.size)
        return jnp.broadcast_to(coordinate.reshape(reshape), target_shape)
    if op == "field":
        if (
            not isinstance(context.state, Mapping)
            or expression.symbol not in context.state
        ):
            raise ValueError(
                f"Boundary target requires runtime field {expression.symbol!r}."
            )
        value = jnp.asarray(context.state[str(expression.symbol)])
        if value.shape != field_shape:
            raise ValueError("Boundary target field has incompatible spatial shape.")
        index = 0 if side == "lower" else field_shape[boundary_index] - 1
        return jnp.take(value, index, axis=boundary_index)
    values = tuple(
        _evaluate_target_expression(
            argument,
            context,
            boundary_axis,
            side,
            field_shape,
        )
        for argument in expression.args
    )
    if op == "negate":
        return -values[0]
    if op == "add":
        return values[0] + values[1]
    if op == "multiply":
        return values[0] * values[1]
    if op == "divide":
        return values[0] / values[1]
    if op == "power":
        return values[0] ** values[1]
    if op == "sin":
        return jnp.sin(values[0])
    if op == "cos":
        return jnp.cos(values[0])
    if op == "exp":
        return jnp.exp(values[0])
    if op == "log":
        return jnp.log(values[0])
    if op == "sqrt":
        return jnp.sqrt(values[0])
    raise ValueError(f"Unsupported dynamic FD boundary target operation {op!r}.")


class BoundaryTarget(StrictModule, NonTrainableState):
    """Runtime-evaluable boundary target expression with explicit context."""

    kind: BoundaryTargetKind = eqx.field(static=True)
    value: float | None = eqx.field(static=True)
    parameter: str | None = eqx.field(static=True)
    expression: PDEExpression = eqx.field(static=True)
    target_id: str = eqx.field(static=True)

    def __init__(self, expression: PDEExpression, /):
        if not isinstance(expression, PDEExpression):
            raise TypeError("Boundary target must be a PDEExpression.")
        if expression.op == "constant" and expression.value is not None:
            kind: BoundaryTargetKind = "constant"
            value = float(expression.value)
            parameter = None
        elif expression.op == "parameter" and expression.symbol:
            kind = "parameter"
            value = None
            parameter = expression.symbol
        else:
            kind = "expression"
            value = None
            parameter = None
        self.kind = kind
        self.value = value
        self.parameter = parameter
        self.expression = expression
        self.target_id = canonical_fingerprint(
            {
                "kind": "fd-boundary-target",
                "target_kind": kind,
                "value": value,
                "parameter": parameter,
                "expression": repr(expression),
            }
        )

    def evaluate(
        self,
        context: BoundaryStageContext,
        axis: str,
        side: BoundarySide,
        field_shape: tuple[int, ...],
        /,
    ) -> Array:
        if not isinstance(context, BoundaryStageContext):
            raise TypeError("Boundary target requires BoundaryStageContext.")
        return _evaluate_target_expression(
            self.expression,
            context,
            axis,
            side,
            field_shape,
        )


class FDBoundaryBinding(StrictModule, NonTrainableState):
    """One field/axis/side scientific condition lowered to numerical coefficients."""

    condition_name: str = eqx.field(static=True)
    field_name: str = eqx.field(static=True)
    axis: str = eqx.field(static=True)
    side: BoundarySide = eqx.field(static=True)
    kind: GhostConditionKind = eqx.field(static=True)
    alpha: float = eqx.field(static=True)
    beta: float = eqx.field(static=True)
    target: BoundaryTarget
    binding_id: str = eqx.field(static=True)

    def __init__(
        self,
        condition_name: str,
        field_name: str,
        axis: str,
        side: BoundarySide,
        kind: GhostConditionKind,
        target: BoundaryTarget,
        /,
        *,
        alpha: float,
        beta: float,
    ):
        if side not in ("lower", "upper") or kind not in (
            "periodic",
            "dirichlet",
            "neumann",
            "robin",
        ):
            raise ValueError("Invalid FD boundary side or kind.")
        self.condition_name = str(condition_name)
        self.field_name = str(field_name)
        self.axis = str(axis)
        self.side = side
        self.kind = kind
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.target = target
        self.binding_id = canonical_fingerprint(
            {
                "kind": "fd-boundary-binding",
                "condition": condition_name,
                "field": field_name,
                "axis": axis,
                "side": side,
                "boundary_kind": kind,
                "alpha": float(alpha),
                "beta": float(beta),
                "target": target.target_id,
            }
        )


def _scaled_atom(
    expression: PDEExpression,
    coordinate: str,
    /,
) -> tuple[str, float, str]:
    coefficient = 1.0
    atom = expression
    if expression.op == "multiply" and len(expression.args) == 2:
        left, right = expression.args
        if left.op == "constant" and left.value is not None:
            coefficient = float(left.value)
            atom = right
        elif right.op == "constant" and right.value is not None:
            coefficient = float(right.value)
            atom = left
    if atom.op == "field" and atom.symbol:
        return atom.symbol, coefficient, "field"
    if (
        atom.op == "derivative"
        and atom.coordinate == coordinate
        and atom.order == 1
        and len(atom.args) == 1
        and atom.args[0].op == "field"
        and atom.args[0].symbol
    ):
        return atom.args[0].symbol, coefficient, "derivative"
    raise ValueError("Unsupported FD boundary expression atom.")


def _condition_form(
    condition: PDECondition,
    coordinate: str,
    /,
) -> tuple[str, GhostConditionKind, float, float]:
    expression = condition.expression
    if expression.op != "add":
        field_name, coefficient, atom = _scaled_atom(expression, coordinate)
        return (
            field_name,
            "dirichlet" if atom == "field" else "neumann",
            coefficient if atom == "field" else 0.0,
            coefficient if atom == "derivative" else 0.0,
        )
    terms = tuple(_scaled_atom(value, coordinate) for value in expression.args)
    fields = {value[0] for value in terms}
    if len(fields) != 1:
        raise ValueError("Robin boundary terms must reference one field.")
    alpha = sum(value[1] for value in terms if value[2] == "field")
    beta = sum(value[1] for value in terms if value[2] == "derivative")
    if alpha == 0.0 or beta == 0.0:
        raise ValueError("Robin boundary requires nonzero field and derivative terms.")
    return next(iter(fields)), "robin", alpha, beta


def _condition_sides(region: PDERegion, /) -> tuple[BoundarySide, ...]:
    if region.component is None:
        return ("lower", "upper")
    component = region.component.lower()
    if component in ("left", "lower", "minimum"):
        return ("lower",)
    if component in ("right", "upper", "maximum"):
        return ("upper",)
    raise ValueError(f"Unsupported structured boundary component {region.component!r}.")


class PreparedFDBoundaryPair(StrictModule, NonTrainableState):
    """Executable lower/upper boundary pair for one field and structured axis."""

    field_name: str = eqx.field(static=True)
    axis: str = eqx.field(static=True)
    lower: FDBoundaryBinding
    upper: FDBoundaryBinding
    cell_runtime: CellGhostBoundary | None
    nodal_runtime: NodalBoundaryRuntime | None
    runtime_id: str = eqx.field(static=True)

    def __init__(
        self,
        grid: PreparedTensorGrid,
        field_name: str,
        axis: str,
        lower: FDBoundaryBinding,
        upper: FDBoundaryBinding,
        /,
        *,
        lower_width: int = 1,
        upper_width: int = 1,
    ):
        if lower.side != "lower" or upper.side != "upper":
            raise ValueError("Prepared boundary pair requires lower and upper bindings.")
        if lower.field_name != field_name or upper.field_name != field_name:
            raise ValueError("Prepared boundary pair field names do not align.")
        axis_index = grid.axis_names.index(axis)
        structured_axis = grid.structured_axes[axis_index]
        if structured_axis.primary_entity == "interval" or structured_axis.periodic:
            widths = structured_axis.interval_widths
            if not bool(jnp.allclose(widths, widths[0])):
                raise ValueError("Cell ghost runtime requires uniform spacing.")
            cell_runtime = CellGhostBoundary(
                axis_index,
                lower.kind,
                upper.kind,
                float(widths[0]),
                lower_width=lower_width,
                upper_width=upper_width,
                lower_alpha=lower.alpha,
                lower_beta=lower.beta,
                upper_alpha=upper.alpha,
                upper_beta=upper.beta,
            )
            nodal_runtime = None
        else:
            cell_runtime = None
            nodal_runtime = NodalBoundaryRuntime(
                axis_index,
                lower.kind,
                upper.kind,
                lower_alpha=lower.alpha,
                lower_beta=lower.beta,
                upper_alpha=upper.alpha,
                upper_beta=upper.beta,
            )
        runtime_identifier = (
            cell_runtime.runtime_id
            if cell_runtime is not None
            else nodal_runtime.runtime_id
            if nodal_runtime is not None
            else None
        )
        if runtime_identifier is None:
            raise RuntimeError("Prepared FD boundary has no executable runtime.")
        self.field_name = str(field_name)
        self.axis = str(axis)
        self.lower = lower
        self.upper = upper
        self.cell_runtime = cell_runtime
        self.nodal_runtime = nodal_runtime
        self.runtime_id = canonical_fingerprint(
            {
                "kind": "prepared-fd-boundary-pair",
                "grid": grid.prepared_id,
                "field": field_name,
                "axis": axis,
                "lower": lower.binding_id,
                "upper": upper.binding_id,
                "runtime": runtime_identifier,
            }
        )

    def target_values(
        self,
        context: BoundaryStageContext,
        field_shape: tuple[int, ...],
        /,
    ) -> tuple[Array, Array]:
        lower = self.lower.target.evaluate(
            context,
            self.axis,
            "lower",
            field_shape,
        )
        upper = self.upper.target.evaluate(
            context,
            self.axis,
            "upper",
            field_shape,
        )
        return lower, upper

    def apply(
        self,
        values: Array,
        context: BoundaryStageContext,
        /,
    ) -> Array:
        lower, upper = self.target_values(context, values.shape)
        if self.cell_runtime is not None:
            return self.cell_runtime.fill(values, lower, upper)
        if self.nodal_runtime is None:
            raise RuntimeError("Prepared FD boundary pair lost its runtime.")
        return self.nodal_runtime.apply_state(values, lower, upper)


def prepare_fd_boundary_runtime(
    grid: PreparedTensorGrid,
    bindings: tuple[FDBoundaryBinding, ...],
    field_name: str,
    /,
    *,
    ghost_widths: Mapping[str, tuple[int, int]] | None = None,
) -> tuple[PreparedFDBoundaryPair, ...]:
    """Group scientific bindings into complete executable axis pairs."""
    relevant = tuple(value for value in bindings if value.field_name == field_name)
    widths = {} if ghost_widths is None else dict(ghost_widths)
    output = []
    for axis in grid.axis_names:
        axis_bindings = tuple(value for value in relevant if value.axis == axis)
        if not axis_bindings:
            continue
        lower = tuple(value for value in axis_bindings if value.side == "lower")
        upper = tuple(value for value in axis_bindings if value.side == "upper")
        if len(lower) != 1 or len(upper) != 1:
            raise ValueError(
                "Each lowered FD boundary axis requires exactly one condition per side."
            )
        lower_width, upper_width = widths.get(axis, (1, 1))
        output.append(
            PreparedFDBoundaryPair(
                grid,
                field_name,
                axis,
                lower[0],
                upper[0],
                lower_width=lower_width,
                upper_width=upper_width,
            )
        )
    return tuple(output)


class PreparedFDBoundaryProgram(StrictModule, NonTrainableState):
    """All field/axis boundary runtimes with one-evaluation-per-stage workspaces."""

    grid: PreparedTensorGrid
    pairs: tuple[PreparedFDBoundaryPair, ...]
    corner_policy: CornerPolicy = eqx.field(static=True)
    program_id: str = eqx.field(static=True)

    def __init__(
        self,
        grid: PreparedTensorGrid,
        pairs: Sequence[PreparedFDBoundaryPair],
        /,
        *,
        corner_policy: CornerPolicy = "axis_separable",
    ):
        pairs_ = tuple(pairs)
        if not isinstance(grid, PreparedTensorGrid) or not all(
            isinstance(value, PreparedFDBoundaryPair) for value in pairs_
        ):
            raise TypeError("Boundary program requires a grid and prepared pairs.")
        if corner_policy not in ("error", "axis_separable", "tensor_product"):
            raise ValueError("Unknown FD corner policy.")
        keys = tuple((value.field_name, value.axis) for value in pairs_)
        if len(set(keys)) != len(keys):
            raise ValueError("Boundary program field/axis pairs must be unique.")
        if any(value.axis not in grid.axis_names for value in pairs_):
            raise ValueError("Boundary program pair references an unknown grid axis.")
        self.grid = grid
        self.pairs = pairs_
        self.corner_policy = corner_policy
        self.program_id = canonical_fingerprint(
            {
                "kind": "prepared-fd-boundary-program",
                "grid": grid.prepared_id,
                "pairs": [value.runtime_id for value in pairs_],
                "corner_policy": corner_policy,
            }
        )

    def stage_context(
        self,
        time: Array,
        state: Any,
        args: Any,
        /,
        *,
        stage_id: str,
    ) -> BoundaryStageContext:
        layout = self.grid.primary_entity_layout
        return BoundaryStageContext(
            time,
            state,
            args,
            self.grid.axis_names,
            layout.coordinates_by_axis,
            stage_id=stage_id,
        )

    def pairs_for_field(
        self,
        field_name: str,
        /,
    ) -> tuple[PreparedFDBoundaryPair, ...]:
        return tuple(value for value in self.pairs if value.field_name == str(field_name))

    def pair(
        self,
        field_name: str,
        axis: str,
        /,
    ) -> PreparedFDBoundaryPair | None:
        selected = tuple(
            value
            for value in self.pairs
            if value.field_name == field_name and value.axis == axis
        )
        if len(selected) > 1:
            raise RuntimeError("Boundary program contains duplicate field/axis pairs.")
        return None if not selected else selected[0]

    def workspace(
        self,
        field_name: str,
        values: Array,
        context: BoundaryStageContext,
        /,
        *,
        require_tensor: bool = False,
    ) -> BoundaryWorkspace:
        relevant = self.pairs_for_field(field_name)
        if not relevant:
            return BoundaryWorkspace(
                values,
                (),
                (),
                (),
                (),
                values if require_tensor else None,
                (),
                context.stage_id,
            )
        targets = tuple(pair.target_values(context, values.shape) for pair in relevant)
        axis_values = tuple(
            pair.cell_runtime.fill(values, lower, upper)
            if pair.cell_runtime is not None
            else pair.nodal_runtime.apply_state(values, lower, upper)
            if pair.nodal_runtime is not None
            else values
            for pair, (lower, upper) in zip(relevant, targets, strict=True)
        )
        tensor_values = None
        if require_tensor:
            if len(relevant) == 1:
                tensor_values = axis_values[0]
            elif self.corner_policy in ("error", "axis_separable"):
                raise ValueError(
                    "A corner-filled workspace requires tensor_product corner policy."
                )
            else:
                if any(
                    jnp.asarray(value).shape != ()
                    for target in targets
                    for value in target
                ):
                    raise ValueError(
                        "Tensor-product corner filling currently requires scalar side data."
                    )
                tensor_values = values
                for pair, (lower, upper) in zip(relevant, targets, strict=True):
                    tensor_values = (
                        pair.cell_runtime.fill(tensor_values, lower, upper)
                        if pair.cell_runtime is not None
                        else pair.nodal_runtime.apply_state(
                            tensor_values,
                            lower,
                            upper,
                        )
                        if pair.nodal_runtime is not None
                        else tensor_values
                    )
        return BoundaryWorkspace(
            values,
            tuple(value.axis for value in relevant),
            axis_values,
            tuple(value[0] for value in targets),
            tuple(value[1] for value in targets),
            tensor_values,
            tuple(value.runtime_id for value in relevant),
            context.stage_id,
        )

    def constrain_state(
        self,
        field_name: str,
        values: Array,
        context: BoundaryStageContext,
        /,
    ) -> Array:
        result = values
        for pair in self.pairs_for_field(field_name):
            if pair.nodal_runtime is None:
                continue
            lower, upper = pair.target_values(context, values.shape)
            result = pair.nodal_runtime.apply_state(result, lower, upper)
        return result

    def constrain_coordinate_derivative(
        self,
        field_name: str,
        axis: str,
        derivative: Array,
        state: Array,
        context: BoundaryStageContext,
        /,
    ) -> Array:
        pair = self.pair(field_name, axis)
        if pair is None or pair.nodal_runtime is None:
            return derivative
        lower, upper = pair.target_values(context, state.shape)
        return pair.nodal_runtime.apply_coordinate_derivative(
            derivative,
            state,
            lower,
            upper,
        )

    def constrain_time_derivative(
        self,
        field_name: str,
        derivative: Array,
        context: BoundaryStageContext,
        /,
    ) -> Array:
        result = derivative
        for pair in self.pairs_for_field(field_name):
            if pair.nodal_runtime is None:
                continue

            def target_rate(binding: FDBoundaryBinding, side: BoundarySide) -> Array:
                _, rate = jax.jvp(
                    lambda time: binding.target.evaluate(
                        BoundaryStageContext(
                            time,
                            context.state,
                            context.args,
                            context.axis_names,
                            context.coordinates,
                            stage_id=context.stage_id,
                        ),
                        pair.axis,
                        side,
                        derivative.shape,
                    ),
                    (context.time,),
                    (jnp.ones_like(context.time),),
                )
                return rate

            result = pair.nodal_runtime.apply_time_derivative(
                result,
                target_rate(pair.lower, "lower"),
                target_rate(pair.upper, "upper"),
            )
        return result


def prepare_fd_boundary_program(
    grid: PreparedTensorGrid,
    bindings: tuple[FDBoundaryBinding, ...],
    field_names: Sequence[str],
    /,
    *,
    ghost_widths: Mapping[str, tuple[int, int]] | None = None,
    corner_policy: CornerPolicy = "axis_separable",
) -> PreparedFDBoundaryProgram:
    pairs = tuple(
        pair
        for field_name in field_names
        for pair in prepare_fd_boundary_runtime(
            grid,
            bindings,
            str(field_name),
            ghost_widths=ghost_widths,
        )
    )
    return PreparedFDBoundaryProgram(
        grid,
        pairs,
        corner_policy=corner_policy,
    )


def lower_fd_boundaries(
    problem: PDEProblemIR,
    grid: PreparedTensorGrid,
    /,
) -> tuple[FDBoundaryBinding, ...]:
    """Lower typed PDE boundary conditions into field/axis/side bindings."""
    if not isinstance(problem, PDEProblemIR) or not isinstance(grid, PreparedTensorGrid):
        raise TypeError("problem and grid must be PDEProblemIR/PreparedTensorGrid.")
    regions = {region.name: region for region in problem.regions}
    output = []
    for condition in problem.conditions:
        if condition.kind != "boundary":
            continue
        if condition.region not in regions:
            raise ValueError(f"Unknown PDE boundary region {condition.region!r}.")
        region = regions[condition.region]
        if region.kind != "boundary" or len(region.coordinates) != 1:
            raise ValueError("FD boundaries require one-coordinate boundary regions.")
        coordinate = condition.coordinate or region.coordinates[0]
        if coordinate not in grid.axis_names:
            raise ValueError("FD boundary coordinate is not a grid axis.")
        field_name, kind, alpha, beta = _condition_form(condition, coordinate)
        target = BoundaryTarget(condition.target)
        for side in _condition_sides(region):
            output.append(
                FDBoundaryBinding(
                    condition.name,
                    field_name,
                    coordinate,
                    side,
                    kind,
                    target,
                    alpha=alpha,
                    beta=beta,
                )
            )
    identities = tuple(value.binding_id for value in output)
    if len(set(identities)) != len(identities):
        raise ValueError("Duplicate FD boundary bindings are not allowed.")
    return tuple(output)


FDInterfaceConditionKind: TypeAlias = Literal["field_jump", "flux_jump"]


class FDInterfaceBinding(StrictModule, NonTrainableState):
    """One conforming interface field- or coordinate-flux-jump condition."""

    condition_name: str = eqx.field(static=True)
    interface_name: str = eqx.field(static=True)
    field_name: str = eqx.field(static=True)
    axis: str = eqx.field(static=True)
    kind: FDInterfaceConditionKind = eqx.field(static=True)
    coefficient: float = eqx.field(static=True)
    target: BoundaryTarget
    binding_id: str = eqx.field(static=True)

    def __init__(
        self,
        condition_name: str,
        interface_name: str,
        field_name: str,
        axis: str,
        kind: FDInterfaceConditionKind,
        coefficient: float,
        target: BoundaryTarget,
        /,
    ):
        coefficient_ = float(coefficient)
        if (
            not condition_name
            or not interface_name
            or not field_name
            or not axis
            or kind not in ("field_jump", "flux_jump")
            or coefficient_ == 0.0
        ):
            raise ValueError("FD interface binding metadata is invalid.")
        self.condition_name = str(condition_name)
        self.interface_name = str(interface_name)
        self.field_name = str(field_name)
        self.axis = str(axis)
        self.kind = kind
        self.coefficient = coefficient_
        self.target = target
        self.binding_id = canonical_fingerprint(
            {
                "kind": "fd-interface-binding",
                "condition": condition_name,
                "interface": interface_name,
                "field": field_name,
                "axis": axis,
                "interface_kind": kind,
                "coefficient": coefficient_,
                "target": target.target_id,
            }
        )


class PreparedFDInterface(StrictModule, NonTrainableState):
    """Executable conforming field/flux interface jump pair."""

    grid: PreparedTensorGrid
    interface_name: str = eqx.field(static=True)
    field_name: str = eqx.field(static=True)
    axis: str = eqx.field(static=True)
    field_binding: FDInterfaceBinding | None
    flux_binding: FDInterfaceBinding | None
    runtime: ConformingInterfaceRuntime
    interface_id: str = eqx.field(static=True)

    def __init__(
        self,
        grid: PreparedTensorGrid,
        interface_name: str,
        field_name: str,
        axis: str,
        bindings: Sequence[FDInterfaceBinding],
        /,
    ):
        if not isinstance(grid, PreparedTensorGrid):
            raise TypeError("Prepared interface requires a PreparedTensorGrid.")
        bindings_ = tuple(bindings)
        field = tuple(value for value in bindings_ if value.kind == "field_jump")
        flux = tuple(value for value in bindings_ if value.kind == "flux_jump")
        if len(field) > 1 or len(flux) > 1 or not bindings_:
            raise ValueError("Interface permits at most one field and one flux jump.")
        if any(
            value.interface_name != interface_name
            or value.field_name != field_name
            or value.axis != axis
            for value in bindings_
        ):
            raise ValueError("Prepared interface bindings do not share one trace.")
        runtime = ConformingInterfaceRuntime(field_name, axis)
        self.grid = grid
        self.interface_name = str(interface_name)
        self.field_name = str(field_name)
        self.axis = str(axis)
        self.field_binding = None if not field else field[0]
        self.flux_binding = None if not flux else flux[0]
        self.runtime = runtime
        self.interface_id = canonical_fingerprint(
            {
                "kind": "prepared-fd-interface",
                "grid": grid.prepared_id,
                "interface": interface_name,
                "field": field_name,
                "axis": axis,
                "bindings": [value.binding_id for value in bindings_],
                "runtime": runtime.runtime_id,
            }
        )

    def couple(
        self,
        left_value: Array,
        right_value: Array,
        left_outward_flux: Array,
        right_outward_flux: Array,
        context: BoundaryStageContext,
        /,
    ) -> tuple[Array, Array, Array, Array]:
        shape = self.grid.shape
        field_jump = (
            0.0
            if self.field_binding is None
            else self.field_binding.target.evaluate(
                context,
                self.axis,
                "lower",
                shape,
            )
            / self.field_binding.coefficient
        )
        flux_jump = (
            0.0
            if self.flux_binding is None
            else self.flux_binding.target.evaluate(
                context,
                self.axis,
                "lower",
                shape,
            )
            / self.flux_binding.coefficient
        )
        return self.runtime.couple(
            left_value,
            right_value,
            left_outward_flux,
            right_outward_flux,
            field_jump=field_jump,
            flux_jump=flux_jump,
        )


def lower_fd_interfaces(
    problem: PDEProblemIR,
    grid: PreparedTensorGrid,
    /,
) -> tuple[FDInterfaceBinding, ...]:
    """Lower typed interface conditions into explicit field/flux jump bindings."""
    if not isinstance(problem, PDEProblemIR) or not isinstance(grid, PreparedTensorGrid):
        raise TypeError("problem and grid must be PDEProblemIR/PreparedTensorGrid.")
    regions = {region.name: region for region in problem.regions}
    output = []
    for condition in problem.conditions:
        if condition.kind != "interface":
            continue
        if condition.region not in regions:
            raise ValueError(f"Unknown PDE interface region {condition.region!r}.")
        region = regions[condition.region]
        if region.kind != "interface" or len(region.coordinates) != 1:
            raise ValueError("FD interfaces require one-coordinate interface regions.")
        coordinate = condition.coordinate or region.coordinates[0]
        if coordinate not in grid.axis_names:
            raise ValueError("FD interface coordinate is not a grid axis.")
        field_name, coefficient, atom = _scaled_atom(
            condition.expression,
            coordinate,
        )
        output.append(
            FDInterfaceBinding(
                condition.name,
                region.name,
                field_name,
                coordinate,
                "field_jump" if atom == "field" else "flux_jump",
                coefficient,
                BoundaryTarget(condition.target),
            )
        )
    identities = tuple(value.binding_id for value in output)
    if len(set(identities)) != len(identities):
        raise ValueError("Duplicate FD interface bindings are not allowed.")
    return tuple(output)


def prepare_fd_interfaces(
    grid: PreparedTensorGrid,
    bindings: Sequence[FDInterfaceBinding],
    /,
) -> tuple[PreparedFDInterface, ...]:
    """Group lowered interface conditions by interface, field, and axis."""
    bindings_ = tuple(bindings)
    keys = tuple(
        dict.fromkeys(
            (value.interface_name, value.field_name, value.axis) for value in bindings_
        )
    )
    return tuple(
        PreparedFDInterface(
            grid,
            interface_name,
            field_name,
            axis,
            tuple(
                value
                for value in bindings_
                if (
                    value.interface_name,
                    value.field_name,
                    value.axis,
                )
                == (interface_name, field_name, axis)
            ),
        )
        for interface_name, field_name, axis in keys
    )


__all__ = [
    "BoundaryTarget",
    "FDBoundaryBinding",
    "FDInterfaceBinding",
    "FDInterfaceConditionKind",
    "lower_fd_boundaries",
    "lower_fd_interfaces",
    "prepare_fd_boundary_program",
    "prepare_fd_boundary_runtime",
    "prepare_fd_interfaces",
    "PreparedFDBoundaryPair",
    "PreparedFDBoundaryProgram",
    "PreparedFDInterface",
]
