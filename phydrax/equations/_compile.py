#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Literal, TYPE_CHECKING

import jax.numpy as jnp

from phydrax.domain import SamplingPlan

from ._ir import PDECondition, PDEEquation, PDEExpression, PDEProblemIR
from ._validate import infer_expression_type, validate_pde_ir


if TYPE_CHECKING:
    from phydrax.domain import DomainFunction



DifferentialBackend = Literal["ad", "jet", "fd", "basis"]
IntegralCompiler = Callable[[Any, str, PDEProblemIR], Any]


_DIFFERENTIAL_BACKENDS = ("ad", "jet", "fd", "basis")


def _validate_differential_backend(backend: str, /) -> DifferentialBackend:
    if backend not in _DIFFERENTIAL_BACKENDS:
        raise ValueError(
            f"Unknown differential backend {backend!r}; expected one of "
            f"{_DIFFERENTIAL_BACKENDS}."
        )
    return backend


@dataclass(frozen=True, slots=True)
class CompiledPDEEquation:
    name: str
    residual: Any
    source: PDEEquation


@dataclass(frozen=True, slots=True)
class CompiledPDECondition:
    name: str
    kind: str
    region: str
    residual: Any
    source: PDECondition


@dataclass(frozen=True, slots=True)
class CompiledPDEProblem:
    """PDE IR compiled to PhydraX DomainFunction residuals and condition metadata."""

    equations: tuple[CompiledPDEEquation, ...]
    conditions: tuple[CompiledPDECondition, ...]
    canonical_hash: str
    source: PDEProblemIR

    def equation(self, name: str, /) -> CompiledPDEEquation:
        return next(item for item in self.equations if item.name == name)

    def condition(self, name: str, /) -> CompiledPDECondition:
        return next(item for item in self.conditions if item.name == name)


def _map_value(value: Any, function: Callable[[Any], Any], /) -> Any:
    from phydrax.domain import DomainFunction

    if not isinstance(value, DomainFunction):
        return function(value)

    def transformed(*args: Any, key: Any = None) -> Any:
        return function(value.func(*args, key=key))

    return DomainFunction(
        domain=value.domain,
        deps=value.deps,
        func=transformed,
        metadata=value.metadata,
    )


def _require_domain_function(value: Any, operation: str, /) -> DomainFunction:
    from phydrax.domain import DomainFunction

    if not isinstance(value, DomainFunction):
        raise TypeError(f"PDE {operation} compilation requires a DomainFunction operand.")
    return value


def compile_pde_expression(
    expression: PDEExpression,
    problem: PDEProblemIR,
    /,
    *,
    fields: Mapping[str, DomainFunction],
    parameters: Mapping[str, Any] | None = None,
    coordinates: Mapping[str, DomainFunction] | None = None,
    differential_backend: DifferentialBackend = "ad",
    integral_compiler: IntegralCompiler | None = None,
) -> Any:
    """Compile a validated expression DAG to native PhydraX operations."""
    differential_backend = _validate_differential_backend(differential_backend)
    infer_expression_type(expression, problem)
    parameter_values: dict[str, Any] = {
        item.name: item.value for item in problem.parameters if item.value is not None
    }
    if parameters is not None:
        parameter_values.update(parameters)
    coordinate_values = {} if coordinates is None else dict(coordinates)

    def compile_node(node: PDEExpression) -> Any:
        if node.op == "constant":
            assert node.value is not None
            return node.value
        if node.op == "field":
            assert node.symbol is not None
            if node.symbol not in fields:
                raise KeyError(f"No DomainFunction supplied for PDE field {node.symbol!r}.")
            return fields[node.symbol]
        if node.op == "parameter":
            assert node.symbol is not None
            if node.symbol not in parameter_values:
                raise KeyError(f"No value supplied for PDE parameter {node.symbol!r}.")
            return parameter_values[node.symbol]
        if node.op == "coordinate":
            assert node.symbol is not None
            if node.symbol not in coordinate_values:
                raise KeyError(
                    f"No DomainFunction supplied for PDE coordinate {node.symbol!r}."
                )
            return coordinate_values[node.symbol]

        args = tuple(compile_node(argument) for argument in node.args)
        if node.op == "add":
            result = args[0]
            for argument in args[1:]:
                result = result + argument
            return result
        if node.op == "multiply":
            result = args[0]
            for argument in args[1:]:
                result = result * argument
            return result
        if node.op == "divide":
            return args[0] / args[1]
        if node.op == "negate":
            return -args[0]
        if node.op == "power":
            return args[0] ** args[1]
        if node.op == "sin":
            return _map_value(args[0], jnp.sin)
        if node.op == "cos":
            return _map_value(args[0], jnp.cos)
        if node.op == "exp":
            return _map_value(args[0], jnp.exp)
        if node.op == "log":
            return _map_value(args[0], jnp.log)
        if node.op == "sqrt":
            return _map_value(args[0], jnp.sqrt)
        if node.op == "component":
            assert node.axis is not None
            return _map_value(args[0], lambda value: value[..., node.axis])
        if node.op == "dot":
            product = args[0] * args[1]
            return _map_value(product, lambda value: jnp.sum(value, axis=-1))
        if node.op in (
            "derivative",
            "gradient",
            "divergence",
            "curl",
            "laplacian",
        ):
            from ..operators.differential import curl, div, grad, laplacian, partial_n

        if node.op == "derivative":
            assert node.coordinate is not None
            return partial_n(
                _require_domain_function(args[0], node.op),
                var=node.coordinate,
                axis=node.axis,
                order=node.order,
                backend=differential_backend,
            )
        if node.op == "gradient":
            assert node.coordinate is not None
            return grad(
                _require_domain_function(args[0], node.op),
                var=node.coordinate,
                backend=differential_backend,
            )
        if node.op == "divergence":
            assert node.coordinate is not None
            return div(
                _require_domain_function(args[0], node.op),
                var=node.coordinate,
                backend=differential_backend,
            )
        if node.op == "curl":
            assert node.coordinate is not None
            return curl(
                _require_domain_function(args[0], node.op),
                var=node.coordinate,
                backend=differential_backend,
            )
        if node.op == "laplacian":
            assert node.coordinate is not None
            return laplacian(
                _require_domain_function(args[0], node.op),
                var=node.coordinate,
                backend=differential_backend,
            )
        if node.op == "integral":
            if integral_compiler is None:
                raise ValueError(
                    "Integral expressions require an integral_compiler bound to a "
                    "concrete sampling or quadrature contract."
                )
            assert node.region is not None
            return integral_compiler(args[0], node.region, problem)
        raise ValueError(f"Unsupported PDE expression operation {node.op!r}.")

    return compile_node(expression)


def make_pde_operator(
    expression: PDEExpression,
    problem: PDEProblemIR,
    /,
    *,
    field_names: tuple[str, ...] | None = None,
    parameters: Mapping[str, Any] | None = None,
    coordinates: Mapping[str, DomainFunction] | None = None,
    differential_backend: DifferentialBackend = "ad",
    integral_compiler: IntegralCompiler | None = None,
) -> Callable[..., Any]:
    """Adapt an expression to the operator signature used by PhydraX constraints."""
    differential_backend = _validate_differential_backend(differential_backend)
    names = (
        tuple(field.name for field in problem.fields)
        if field_names is None
        else tuple(field_names)
    )
    known = {field.name for field in problem.fields}
    unknown = set(names) - known
    if unknown:
        raise ValueError(f"Unknown PDE constraint fields {sorted(unknown)}.")
    if len(names) != len(set(names)):
        raise ValueError("PDE constraint field names must be unique.")

    def operator(*field_values: DomainFunction) -> Any:
        if len(field_values) != len(names):
            raise ValueError(
                f"PDE operator expected {len(names)} fields, got {len(field_values)}."
            )
        return compile_pde_expression(
            expression,
            problem,
            fields=dict(zip(names, field_values, strict=True)),
            parameters=parameters,
            coordinates=coordinates,
            differential_backend=differential_backend,
            integral_compiler=integral_compiler,
        )

    return operator


def compile_pde_functional_constraint(
    expression: PDEExpression,
    problem: PDEProblemIR,
    /,
    *,
    component: Any,
    sampling: SamplingPlan,
    field_names: tuple[str, ...] | None = None,
    parameters: Mapping[str, Any] | None = None,
    coordinates: Mapping[str, DomainFunction] | None = None,
    differential_backend: DifferentialBackend = "ad",
    integral_compiler: IntegralCompiler | None = None,
    weight: Any = 1.0,
    label: str | None = None,
    over: str | tuple[str, ...] | None = None,
    reduction: Literal["mean", "integral"] = "mean",
    sampling_mode: Literal["resample", "fixed"] = "resample",
) -> Any:
    """Compile an IR residual directly into a native FunctionalConstraint."""
    from ..constraints._functional import FunctionalConstraint

    names = (
        tuple(field.name for field in problem.fields)
        if field_names is None
        else tuple(field_names)
    )
    operator = make_pde_operator(
        expression,
        problem,
        field_names=names,
        parameters=parameters,
        coordinates=coordinates,
        differential_backend=differential_backend,
        integral_compiler=integral_compiler,
    )
    return FunctionalConstraint.from_operator(
        component=component,
        operator=operator,
        constraint_vars=names,
        sampling=sampling,
        weight=weight,
        label=label,
        over=over,
        reduction=reduction,
        sampling_mode=sampling_mode,
    )


def compile_pde_problem(
    problem: PDEProblemIR,
    /,
    *,
    fields: Mapping[str, DomainFunction],
    parameters: Mapping[str, Any] | None = None,
    coordinates: Mapping[str, DomainFunction] | None = None,
    differential_backend: DifferentialBackend = "ad",
    integral_compiler: IntegralCompiler | None = None,
) -> CompiledPDEProblem:
    """Compile every equation and restriction to executable residuals."""
    differential_backend = _validate_differential_backend(differential_backend)
    validate_pde_ir(problem)

    def compile_expression(expression: PDEExpression) -> Any:
        return compile_pde_expression(
            expression,
            problem,
            fields=fields,
            parameters=parameters,
            coordinates=coordinates,
            differential_backend=differential_backend,
            integral_compiler=integral_compiler,
        )

    equations = tuple(
        CompiledPDEEquation(item.name, compile_expression(item.residual), item)
        for item in problem.equations
    )
    conditions = tuple(
        CompiledPDECondition(
            item.name,
            item.kind,
            item.region,
            compile_expression(item.residual),
            item,
        )
        for item in problem.conditions
    )
    return CompiledPDEProblem(
        equations=equations,
        conditions=conditions,
        canonical_hash=problem.canonical_hash,
        source=problem,
    )


__all__ = [
    "CompiledPDECondition",
    "CompiledPDEEquation",
    "CompiledPDEProblem",
    "DifferentialBackend",
    "IntegralCompiler",
    "compile_pde_expression",
    "compile_pde_functional_constraint",
    "compile_pde_problem",
    "make_pde_operator",
]
