#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization import (
    BoundaryWorkspace,
    ConservativeAdvectionPlan,
    ConservativeBoundaryCondition,
    ConservativeDiffusionPlan,
    CornerPolicy,
    DerivativeRequest,
    DiscretizationBundle,
    DiscretizationKey,
    DiscretizationRecord,
    DiscretizationRole,
    FiniteDifferencePlan,
    fornberg_weights,
    PreparedConservativeAdvection,
    PreparedConservativeDiffusion,
    PreparedFiniteDifferenceDiscretization,
    PreparedTensorGrid,
)
from ._fd_boundary_lowering import (
    lower_fd_boundaries,
    lower_fd_interfaces,
    prepare_fd_boundary_program,
    prepare_fd_interfaces,
    PreparedFDBoundaryProgram,
    PreparedFDInterface,
)
from ._ir import PDEExpression, PDEProblemIR
from ._stencil_compile import StencilStateLayout


class FiniteDifferenceCompilationPolicy(StrictModule):
    """Explicit accuracy, Laplacian, and corner realization for native FD lowering."""

    accuracy_order: int = eqx.field(static=True)
    laplacian: str = eqx.field(static=True)
    corner_policy: CornerPolicy = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        accuracy_order: int = 2,
        laplacian: str = "direct_second_derivative",
        corner_policy: CornerPolicy = "axis_separable",
    ):
        accuracy = int(accuracy_order)
        if accuracy <= 0 or laplacian not in (
            "direct_second_derivative",
            "grad_div",
        ):
            raise ValueError("Invalid FD accuracy or Laplacian policy.")
        if corner_policy not in ("error", "axis_separable", "tensor_product"):
            raise ValueError("Invalid FD corner policy.")
        self.accuracy_order = accuracy
        self.laplacian = laplacian
        self.corner_policy = corner_policy
        self.policy_id = canonical_fingerprint(
            {
                "kind": "fd-compilation-policy",
                "accuracy_order": accuracy,
                "laplacian": laplacian,
                "corner_policy": corner_policy,
            }
        )


def _expression_axes(
    expression: PDEExpression,
    coordinate_axes: tuple[tuple[str, tuple[str, ...]], ...],
    /,
) -> tuple[str, ...]:
    mapping = dict(coordinate_axes)
    coordinate = expression.coordinate
    if coordinate not in mapping:
        raise ValueError(f"FD expression coordinate {coordinate!r} is not spatial.")
    axes = mapping[str(coordinate)]
    if expression.op == "derivative":
        if expression.axis is None:
            if len(axes) != 1:
                raise ValueError(
                    "Derivative over a grouped coordinate requires an explicit axis."
                )
            return axes
        index = int(expression.axis)
        if index < 0 or index >= len(axes):
            raise ValueError("Derivative axis is outside its grouped coordinate.")
        return (axes[index],)
    return axes


def _conservative_diffusion_parts(
    expression: PDEExpression,
    /,
) -> tuple[str, PDEExpression] | None:
    if (
        expression.op != "divergence"
        or len(expression.args) != 1
        or expression.args[0].op != "multiply"
        or len(expression.args[0].args) != 2
    ):
        return None
    left, right = expression.args[0].args
    for gradient, coefficient in ((left, right), (right, left)):
        if (
            gradient.op == "gradient"
            and len(gradient.args) == 1
            and gradient.args[0].op == "field"
            and gradient.args[0].symbol
        ):
            return str(gradient.args[0].symbol), coefficient
    return None


def _conservative_advection_parts(
    expression: PDEExpression,
    /,
) -> tuple[str, PDEExpression] | None:
    if (
        expression.op != "divergence"
        or len(expression.args) != 1
        or expression.args[0].op != "multiply"
        or len(expression.args[0].args) != 2
    ):
        return None
    left, right = expression.args[0].args
    for field, velocity in ((left, right), (right, left)):
        if field.op == "field" and field.symbol:
            return str(field.symbol), velocity
    return None


def _collect_derivatives(
    expression: PDEExpression,
    coordinate_axes: tuple[tuple[str, tuple[str, ...]], ...],
    policy: FiniteDifferenceCompilationPolicy,
    output: set[tuple[str, int]],
    /,
) -> None:
    if expression.op == "derivative":
        output.update(
            (axis, int(expression.order))
            for axis in _expression_axes(expression, coordinate_axes)
        )
    elif expression.op in ("gradient", "divergence", "curl"):
        output.update((axis, 1) for axis in _expression_axes(expression, coordinate_axes))
    elif expression.op == "laplacian":
        axes = _expression_axes(expression, coordinate_axes)
        output.update((axis, 2) for axis in axes)
        if policy.laplacian == "grad_div":
            output.update((axis, 1) for axis in axes)
    for argument in expression.args:
        _collect_derivatives(argument, coordinate_axes, policy, output)


class _GhostDerivativeRule(StrictModule, NonTrainableState):
    """Uniform centered derivative applied to one axis-specific ghost workspace."""

    axis: str = eqx.field(static=True)
    axis_index: int = eqx.field(static=True)
    derivative_order: int = eqx.field(static=True)
    accuracy_order: int = eqx.field(static=True)
    offsets: tuple[int, ...] = eqx.field(static=True)
    weights: Array
    lower_width: int = eqx.field(static=True)
    upper_width: int = eqx.field(static=True)
    rule_id: str = eqx.field(static=True)

    def __init__(
        self,
        axis: str,
        axis_index: int,
        derivative_order: int,
        accuracy_order: int,
        spacing: float,
        /,
    ):
        derivative = int(derivative_order)
        accuracy = int(accuracy_order)
        width = derivative + accuracy - 1
        if width % 2 == 0:
            width += 1
        width = max(width, derivative + 1)
        if width % 2 == 0:
            width += 1
        radius = width // 2
        offsets = tuple(range(-radius, radius + 1))
        nodes = np.asarray(offsets, dtype=float) * float(spacing)
        weights = fornberg_weights(nodes, 0.0, derivative)
        self.axis = str(axis)
        self.axis_index = int(axis_index)
        self.derivative_order = derivative
        self.accuracy_order = accuracy
        self.offsets = offsets
        self.weights = jnp.asarray(weights)
        self.lower_width = radius
        self.upper_width = radius
        self.rule_id = canonical_fingerprint(
            {
                "kind": "native-fd-ghost-derivative-rule",
                "axis": axis,
                "axis_index": int(axis_index),
                "derivative_order": derivative,
                "accuracy_order": accuracy,
                "spacing": float(spacing),
                "offsets": list(offsets),
            }
        )

    def apply(
        self,
        padded: Array,
        original_shape: tuple[int, ...],
        /,
    ) -> Array:
        expected = list(original_shape)
        expected[self.axis_index] += self.lower_width + self.upper_width
        if padded.shape != tuple(expected):
            raise ValueError(
                f"Ghost derivative expected padded shape {tuple(expected)}; "
                f"got {padded.shape}."
            )
        result = jnp.zeros(original_shape, dtype=jnp.result_type(padded, self.weights))
        count = original_shape[self.axis_index]
        for offset, weight in zip(self.offsets, self.weights, strict=True):
            index = [slice(None)] * padded.ndim
            start = self.lower_width + offset
            index[self.axis_index] = slice(start, start + count)
            result = result + weight * padded[tuple(index)]
        return result


def _prepare_ghost_rules(
    grid: PreparedTensorGrid,
    derivatives: set[tuple[str, int]],
    accuracy_order: int,
    /,
) -> tuple[_GhostDerivativeRule, ...]:
    output = []
    for axis, derivative in sorted(derivatives):
        axis_index = grid.axis_names.index(axis)
        structured_axis = grid.structured_axes[axis_index]
        if structured_axis.periodic:
            continue
        widths = np.asarray(structured_axis.interval_widths)
        if not np.allclose(widths, widths[0], rtol=1e-10, atol=1e-12):
            continue
        output.append(
            _GhostDerivativeRule(
                axis,
                axis_index,
                derivative,
                accuracy_order,
                float(widths[0]),
            )
        )
    return tuple(output)


def _evolution_rhs(problem: PDEProblemIR, /) -> tuple[tuple[str, PDEExpression], ...]:
    time_coordinates = {
        coordinate.name for coordinate in problem.coordinates if coordinate.kind == "time"
    }
    output = []
    for equation in problem.equations:
        lhs = equation.lhs
        if (
            lhs.op != "derivative"
            or lhs.coordinate not in time_coordinates
            or int(lhs.order) != 1
            or len(lhs.args) != 1
            or lhs.args[0].op != "field"
            or not lhs.args[0].symbol
        ):
            raise ValueError(
                "Native FD dynamics require equations of the form d(field)/dt = rhs."
            )
        output.append((lhs.args[0].symbol, equation.rhs))
    if not output or len({name for name, _ in output}) != len(output):
        raise ValueError("Native FD dynamics require one equation per evolved field.")
    return tuple(output)


class _FiniteDifferenceExpressionEvaluator(StrictModule):
    problem: PDEProblemIR = eqx.field(static=True)
    discretization: PreparedFiniteDifferenceDiscretization
    layout: StencilStateLayout
    equations: tuple[tuple[str, PDEExpression], ...] = eqx.field(static=True)
    parameter_defaults: tuple[tuple[str, float | None], ...] = eqx.field(static=True)
    boundary_program: PreparedFDBoundaryProgram
    ghost_rules: tuple[_GhostDerivativeRule, ...]
    interfaces: tuple[PreparedFDInterface, ...]
    diffusion_templates: tuple[tuple[str, PreparedConservativeDiffusion], ...]
    advection_templates: tuple[tuple[str, PreparedConservativeAdvection], ...]
    spatial_axes: tuple[str, ...] = eqx.field(static=True)
    coordinate_axes: tuple[tuple[str, tuple[str, ...]], ...] = eqx.field(static=True)
    policy: FiniteDifferenceCompilationPolicy

    def __init__(
        self,
        problem: PDEProblemIR,
        discretization: PreparedFiniteDifferenceDiscretization,
        layout: StencilStateLayout,
        equations: tuple[tuple[str, PDEExpression], ...],
        parameter_defaults: tuple[tuple[str, float | None], ...],
        boundary_program: PreparedFDBoundaryProgram,
        ghost_rules: tuple[_GhostDerivativeRule, ...],
        interfaces: tuple[PreparedFDInterface, ...],
        diffusion_templates: tuple[tuple[str, PreparedConservativeDiffusion], ...],
        advection_templates: tuple[tuple[str, PreparedConservativeAdvection], ...],
        spatial_axes: tuple[str, ...],
        coordinate_axes: tuple[tuple[str, tuple[str, ...]], ...],
        policy: FiniteDifferenceCompilationPolicy,
        /,
    ):
        self.problem = problem
        self.discretization = discretization
        self.layout = layout
        self.equations = equations
        self.parameter_defaults = parameter_defaults
        self.boundary_program = boundary_program
        self.ghost_rules = ghost_rules
        self.interfaces = interfaces
        self.diffusion_templates = diffusion_templates
        self.advection_templates = advection_templates
        self.spatial_axes = spatial_axes
        self.coordinate_axes = coordinate_axes
        self.policy = policy

    def _parameter(self, name: str, args: Any, /) -> Array:
        if isinstance(args, Mapping) and name in args:
            return jnp.asarray(args[name])
        defaults = dict(self.parameter_defaults)
        if name not in defaults or defaults[name] is None:
            raise ValueError(f"FD parameter {name!r} requires a runtime value.")
        return jnp.asarray(defaults[name])

    def _coordinate(self, name: str, /) -> Array:
        mapping = dict(self.coordinate_axes)
        if name not in mapping:
            raise ValueError(f"FD coordinate {name!r} is not spatial.")
        components = []
        for axis_name in mapping[name]:
            axis = self.spatial_axes.index(axis_name)
            values = self.discretization.grid.primary_entity_layout.coordinates_by_axis[
                axis
            ]
            reshape = [1] * len(self.spatial_axes)
            reshape[axis] = int(values.size)
            components.append(
                jnp.broadcast_to(values.reshape(reshape), self.layout.spatial_shape)
            )
        return components[0] if len(components) == 1 else jnp.stack(components, axis=-1)

    def _operator(self, axis: str, order: int):
        return self.discretization.operator(f"d_{axis}_{order}")

    def _diffusion_template(
        self,
        field_name: str,
        /,
    ) -> PreparedConservativeDiffusion | None:
        selected = tuple(
            value for name, value in self.diffusion_templates if name == field_name
        )
        if len(selected) > 1:
            raise RuntimeError("Native FD compiler has duplicate diffusion templates.")
        return None if not selected else selected[0]

    def _advection_template(
        self,
        field_name: str,
        /,
    ) -> PreparedConservativeAdvection | None:
        selected = tuple(
            value for name, value in self.advection_templates if name == field_name
        )
        if len(selected) > 1:
            raise RuntimeError("Native FD compiler has duplicate advection templates.")
        return None if not selected else selected[0]

    def _boundary_values(
        self,
        field_name: str,
        workspaces: Mapping[str, BoundaryWorkspace],
        /,
    ) -> dict[str, tuple[Array, Array]]:
        if field_name not in workspaces:
            return {}
        workspace = workspaces[field_name]
        return {axis: workspace.target_values(axis) for axis in workspace.axis_names}

    def _ghost_rule(
        self,
        axis: str,
        order: int,
        /,
    ) -> _GhostDerivativeRule | None:
        selected = tuple(
            value
            for value in self.ghost_rules
            if value.axis == axis and value.derivative_order == int(order)
        )
        if len(selected) > 1:
            raise RuntimeError("Native FD compiler contains duplicate ghost rules.")
        return None if not selected else selected[0]

    def _differentiate(
        self,
        axis: str,
        order: int,
        value: Array,
        field_name: str | None,
        context: Any,
        workspaces: Mapping[str, BoundaryWorkspace],
        /,
    ) -> Array:
        pair = (
            None if field_name is None else self.boundary_program.pair(field_name, axis)
        )
        if pair is not None and pair.cell_runtime is not None:
            if field_name not in workspaces:
                raise RuntimeError("Cell ghost derivative lost its stage workspace.")
            rule = self._ghost_rule(axis, order)
            if rule is None:
                raise ValueError(
                    f"No centered ghost derivative rule exists for {axis!r}, order {order}."
                )
            padded = workspaces[str(field_name)].for_axis(axis)
            lower_extra = pair.cell_runtime.lower_width - rule.lower_width
            upper_extra = pair.cell_runtime.upper_width - rule.upper_width
            index = [slice(None)] * padded.ndim
            stop = (
                padded.shape[rule.axis_index] - upper_extra
                if upper_extra
                else padded.shape[rule.axis_index]
            )
            index[rule.axis_index] = slice(lower_extra, stop)
            result = rule.apply(padded[tuple(index)], value.shape)
        else:
            result = self._operator(axis, order).mv(value)
        if pair is not None and pair.nodal_runtime is not None:
            result = self.boundary_program.constrain_coordinate_derivative(
                str(field_name),
                axis,
                result,
                value,
                context,
            )
        return result

    def evaluate(
        self,
        expression: PDEExpression,
        fields: Mapping[str, Array],
        args: Any,
        context: Any,
        workspaces: Mapping[str, BoundaryWorkspace],
        /,
    ) -> Array:
        op = expression.op
        if op == "field":
            if expression.symbol not in fields:
                raise ValueError(f"Unknown FD field {expression.symbol!r}.")
            return fields[str(expression.symbol)]
        if op == "parameter":
            return self._parameter(str(expression.symbol), args)
        if op == "coordinate":
            return self._coordinate(str(expression.symbol))
        if op == "constant":
            return jnp.asarray(expression.value)
        if op == "divergence" and set(
            _expression_axes(expression, self.coordinate_axes)
        ) == set(self.spatial_axes):
            diffusion_parts = _conservative_diffusion_parts(expression)
            if diffusion_parts is not None:
                field_name, coefficient_expression = diffusion_parts
                template = self._diffusion_template(field_name)
                if template is not None:
                    coefficient = self.evaluate(
                        coefficient_expression,
                        fields,
                        args,
                        context,
                        workspaces,
                    )
                    return template.apply_with_coefficient(
                        fields[field_name],
                        coefficient,
                        boundary_values=self._boundary_values(
                            field_name,
                            workspaces,
                        ),
                    )
            advection_parts = _conservative_advection_parts(expression)
            if advection_parts is not None:
                field_name, velocity_expression = advection_parts
                template = self._advection_template(field_name)
                if template is not None:
                    velocity = self.evaluate(
                        velocity_expression,
                        fields,
                        args,
                        context,
                        workspaces,
                    )
                    return template.apply_with_velocity(
                        fields[field_name],
                        velocity,
                        boundary_values=self._boundary_values(
                            field_name,
                            workspaces,
                        ),
                    )
        values = tuple(
            self.evaluate(argument, fields, args, context, workspaces)
            for argument in expression.args
        )
        direct_field = (
            str(expression.args[0].symbol)
            if expression.args
            and expression.args[0].op == "field"
            and expression.args[0].symbol
            else None
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
        if op == "derivative":
            axis = _expression_axes(expression, self.coordinate_axes)[0]
            return self._differentiate(
                axis,
                int(expression.order),
                values[0],
                direct_field,
                context,
                workspaces,
            )
        if op == "laplacian":
            axes = _expression_axes(expression, self.coordinate_axes)
            if self.policy.laplacian == "direct_second_derivative":
                result = jnp.zeros_like(values[0])
                for axis in axes:
                    result = result + self._differentiate(
                        axis,
                        2,
                        values[0],
                        direct_field,
                        context,
                        workspaces,
                    )
                return result
            result = jnp.zeros_like(values[0])
            for axis in axes:
                first = self._differentiate(
                    axis,
                    1,
                    values[0],
                    direct_field,
                    context,
                    workspaces,
                )
                result = result + self._differentiate(
                    axis,
                    1,
                    first,
                    None,
                    context,
                    workspaces,
                )
            return result
        if op == "gradient":
            axes = _expression_axes(expression, self.coordinate_axes)
            return jnp.stack(
                [
                    self._differentiate(
                        axis,
                        1,
                        values[0],
                        direct_field,
                        context,
                        workspaces,
                    )
                    for axis in axes
                ],
                axis=-1,
            )
        if op == "divergence":
            axes = _expression_axes(expression, self.coordinate_axes)
            if values[0].shape[-1] != len(axes):
                raise ValueError("FD divergence input must match coordinate rank.")
            result = jnp.zeros_like(values[0][..., 0])
            for index, axis in enumerate(axes):
                result = result + self._differentiate(
                    axis,
                    1,
                    values[0][..., index],
                    None,
                    context,
                    workspaces,
                )
            return result
        if op == "curl":
            axes = _expression_axes(expression, self.coordinate_axes)
            if len(axes) == 2 and values[0].shape[-1] == 2:
                return self._differentiate(
                    axes[0],
                    1,
                    values[0][..., 1],
                    None,
                    context,
                    workspaces,
                ) - self._differentiate(
                    axes[1],
                    1,
                    values[0][..., 0],
                    None,
                    context,
                    workspaces,
                )
            raise ValueError("Initial native FD curl supports two-dimensional vectors.")
        if op == "dot":
            return jnp.sum(values[0] * values[1], axis=-1)
        if op == "component":
            if expression.axis is None:
                raise ValueError("FD component expression requires an axis.")
            return values[0][..., int(expression.axis)]
        raise ValueError(f"Unsupported native FD expression operation {op!r}.")

    def __call__(self, time: Array, state: Array, args: Any) -> Array:
        fields = self.layout.unpack(state)
        context = self.boundary_program.stage_context(
            time,
            fields,
            args,
            stage_id="native-fd-stage",
        )
        workspaces = {
            name: self.boundary_program.workspace(name, value, context)
            for name, value in fields.items()
        }
        mutable = dict(fields)
        for boundary in self.boundary_program.pairs:
            if boundary.nodal_runtime is None:
                continue
            lower, upper = workspaces[boundary.field_name].target_values(boundary.axis)
            mutable[boundary.field_name] = boundary.nodal_runtime.apply_state(
                mutable[boundary.field_name],
                lower,
                upper,
            )
        constrained_context = self.boundary_program.stage_context(
            time,
            mutable,
            args,
            stage_id="native-fd-stage",
        )
        derivatives = {
            name: self.evaluate(
                expression,
                mutable,
                args,
                constrained_context,
                workspaces,
            )
            for name, expression in self.equations
        }
        derivatives = {
            name: self.boundary_program.constrain_time_derivative(
                name,
                derivative,
                constrained_context,
            )
            for name, derivative in derivatives.items()
        }
        return self.layout.pack(derivatives)


class CompiledFiniteDifferenceDynamics(StrictModule):
    """Native FD expression dynamics with boundary and discretization provenance."""

    drift: _FiniteDifferenceExpressionEvaluator
    layout: StencilStateLayout
    spatial_discretization: PreparedFiniteDifferenceDiscretization
    boundary_program: PreparedFDBoundaryProgram
    interfaces: tuple[PreparedFDInterface, ...]
    discretization_bundle: DiscretizationBundle
    compilation_id: str = eqx.field(static=True)

    def __init__(
        self,
        evaluator: _FiniteDifferenceExpressionEvaluator,
        /,
    ):
        discretization = evaluator.discretization
        compilation_id = canonical_fingerprint(
            {
                "kind": "compiled-native-fd-dynamics",
                "problem": evaluator.problem.canonical_hash,
                "discretization": discretization.prepared_id,
                "policy": evaluator.policy.policy_id,
                "boundary_program": evaluator.boundary_program.program_id,
                "interfaces": [value.interface_id for value in evaluator.interfaces],
            }
        )
        residual_key = DiscretizationKey(
            "native_fd_problem",
            DiscretizationRole.RESIDUAL,
            domain_labels=evaluator.spatial_axes,
        )
        self.drift = evaluator
        self.layout = evaluator.layout
        self.spatial_discretization = discretization
        self.boundary_program = evaluator.boundary_program
        self.interfaces = evaluator.interfaces
        self.discretization_bundle = DiscretizationBundle(
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
                    "native-fd-expression-program",
                    compilation_id,
                    dependency_key_ids=(discretization.key.key_id,),
                ),
            )
        )
        self.compilation_id = compilation_id

    def __call__(self, time: Array, state: Array, args: Any) -> Array:
        value = jnp.asarray(state)
        if len(self.layout.field_names) == 1 and value.shape == self.layout.spatial_shape:
            result = self.drift(time, value[..., None], args)
            return result[..., 0]
        return self.drift(time, value, args)


def _conservative_boundaries(
    grid: PreparedTensorGrid,
    program: PreparedFDBoundaryProgram,
    field_name: str,
    /,
) -> dict[
    str,
    tuple[ConservativeBoundaryCondition, ConservativeBoundaryCondition],
]:
    output = {}
    for axis_name, structured_axis in zip(
        grid.axis_names,
        grid.structured_axes,
        strict=True,
    ):
        pair = program.pair(field_name, axis_name)
        if pair is None:
            kind = "periodic" if structured_axis.periodic else "neumann"
            output[axis_name] = (
                ConservativeBoundaryCondition(kind),
                ConservativeBoundaryCondition(kind),
            )
        else:
            output[axis_name] = (
                ConservativeBoundaryCondition(
                    pair.lower.kind,
                    alpha=pair.lower.alpha,
                    beta=pair.lower.beta,
                ),
                ConservativeBoundaryCondition(
                    pair.upper.kind,
                    alpha=pair.upper.alpha,
                    beta=pair.upper.beta,
                ),
            )
    return output


def compile_finite_difference_pde(
    problem: PDEProblemIR,
    grid: PreparedTensorGrid,
    /,
    *,
    policy: FiniteDifferenceCompilationPolicy | None = None,
) -> CompiledFiniteDifferenceDynamics:
    """Compile a scalar/tensor strong PDE directly through prepared FD operators."""
    if not isinstance(problem, PDEProblemIR) or not isinstance(grid, PreparedTensorGrid):
        raise TypeError("problem and grid must be PDEProblemIR/PreparedTensorGrid.")
    policy_ = FiniteDifferenceCompilationPolicy() if policy is None else policy
    if not isinstance(policy_, FiniteDifferenceCompilationPolicy):
        raise TypeError("policy must be FiniteDifferenceCompilationPolicy.")
    equations = _evolution_rhs(problem)
    field_names = tuple(field.name for field in problem.fields)
    if set(name for name, _ in equations) != set(field_names):
        raise ValueError("Native FD compiler requires one evolution equation per field.")
    spatial_coordinates = tuple(
        coordinate for coordinate in problem.coordinates if coordinate.kind == "space"
    )
    coordinate_axes_list = []
    cursor = 0
    for coordinate in spatial_coordinates:
        stop = cursor + int(coordinate.size)
        if stop > len(grid.axis_names):
            raise ValueError("PDE spatial coordinate rank exceeds the tensor grid rank.")
        coordinate_axes_list.append((coordinate.name, grid.axis_names[cursor:stop]))
        cursor = stop
    if cursor != len(grid.axis_names):
        raise ValueError("PDE spatial coordinate rank does not cover the tensor grid.")
    coordinate_axes = tuple(coordinate_axes_list)
    derivatives: set[tuple[str, int]] = set()
    for _, expression in equations:
        _collect_derivatives(expression, coordinate_axes, policy_, derivatives)
    if not derivatives:
        raise ValueError("Native FD compilation requires a spatial derivative.")
    requests = tuple(
        DerivativeRequest(
            f"d_{axis}_{order}",
            grid,
            axis,
            derivative_order=order,
            accuracy_order=policy_.accuracy_order,
        )
        for axis, order in sorted(derivatives)
    )
    discretization = FiniteDifferencePlan(
        grid,
        requests,
        field_name="native_fd_state",
        key=DiscretizationKey(
            "native_finite_difference",
            DiscretizationRole.PHYSICAL,
            domain_labels=grid.axis_names,
        ),
    ).prepare()
    ghost_rules = _prepare_ghost_rules(
        grid,
        derivatives,
        policy_.accuracy_order,
    )
    ghost_widths = {
        axis: (
            max(rule.lower_width for rule in ghost_rules if rule.axis == axis),
            max(rule.upper_width for rule in ghost_rules if rule.axis == axis),
        )
        for axis in {rule.axis for rule in ghost_rules}
    }
    bindings = lower_fd_boundaries(problem, grid)
    boundary_program = prepare_fd_boundary_program(
        grid,
        bindings,
        field_names,
        ghost_widths=ghost_widths,
        corner_policy=policy_.corner_policy,
    )
    interfaces = prepare_fd_interfaces(
        grid,
        lower_fd_interfaces(problem, grid),
    )
    if grid.primary_entity_layout.layout_id == grid.cells().layout_id:
        diffusion_templates = tuple(
            (
                field_name,
                ConservativeDiffusionPlan(
                    grid,
                    boundaries=_conservative_boundaries(
                        grid,
                        boundary_program,
                        field_name,
                    ),
                ).prepare(1.0),
            )
            for field_name in field_names
        )
        zero_velocity = jnp.zeros(
            grid.shape + (len(grid.shape),),
            dtype=grid.points.dtype,
        )
        advection_templates = tuple(
            (
                field_name,
                ConservativeAdvectionPlan(
                    grid,
                    form="conservative",
                    boundaries=_conservative_boundaries(
                        grid,
                        boundary_program,
                        field_name,
                    ),
                ).prepare(zero_velocity),
            )
            for field_name in field_names
        )
    else:
        diffusion_templates = ()
        advection_templates = ()
    layout = StencilStateLayout(field_names, grid.shape)
    evaluator = _FiniteDifferenceExpressionEvaluator(
        problem,
        discretization,
        layout,
        equations,
        tuple((parameter.name, parameter.value) for parameter in problem.parameters),
        boundary_program,
        ghost_rules,
        interfaces,
        diffusion_templates,
        advection_templates,
        grid.axis_names,
        coordinate_axes,
        policy_,
    )
    return CompiledFiniteDifferenceDynamics(evaluator)


__all__ = [
    "CompiledFiniteDifferenceDynamics",
    "FiniteDifferenceCompilationPolicy",
    "compile_finite_difference_pde",
]
