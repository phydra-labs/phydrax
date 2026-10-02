#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, TYPE_CHECKING, TypeAlias

import jax.numpy as jnp

from ..typing import parse
from ._ir import PDECondition, PDEEquation, PDEExpression, PDEProblemIR
from ._validate import infer_expression_type, validate_pde_ir


if TYPE_CHECKING:
    from phydrax.domain import DomainFunction
    from phydrax.integration import IntegrationSource
    from phydrax.operators.differential._form_ops import DomainDifferentialForm

    from ..operators.differential._taylor_contracts import TaylorContractionPolicy
    from ..operators.differential._taylor_planning import TaylorContractionPlan
    from ._form_compile import PDEFormGeometry
    from ._linear_differential import LinearDifferentialFamily, NativeContributionPath


DifferentialBackend: TypeAlias = Literal["ad", "jet", "fd", "basis"]
IntegralCompiler = Callable[[Any, str, PDEProblemIR], Any]


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
    from phydrax.domain import DomainFunction, UnaryFieldEvaluator

    from ..domain._evaluation import FunctionBinding, PointwiseEvaluator

    if not isinstance(value, DomainFunction):
        return function(value)

    return DomainFunction(
        domain=value.domain,
        deps=value.deps,
        func=PointwiseEvaluator(
            UnaryFieldEvaluator(value.func, function),
            binding=FunctionBinding(pass_key=True),
        ),
        metadata=value.metadata,
    )


def _require_domain_function(value: Any, operation: str, /) -> DomainFunction:
    from phydrax.domain import DomainFunction

    if not isinstance(value, DomainFunction):
        raise TypeError(f"PDE {operation} compilation requires a DomainFunction operand.")
    return value


@dataclass
class _PDEExpressionCompiler:
    """Own chart bindings while lowering one expression DAG in scientific order."""

    problem: PDEProblemIR
    fields: Mapping[str, DomainFunction | DomainDifferentialForm]
    parameter_values: Mapping[str, Any]
    coordinate_values: Mapping[str, DomainFunction]
    differential_backend: DifferentialBackend
    taylor_policy: TaylorContractionPolicy | None
    integral_compiler: IntegralCompiler | None
    form_geometry: PDEFormGeometry | None
    taylor_plans: Mapping[LinearDifferentialFamily, TaylorContractionPlan] | None
    native_paths: (
        Mapping[LinearDifferentialFamily, tuple[NativeContributionPath, ...] | None]
        | None
    )
    form_bases: dict[PDEExpression, DomainDifferentialForm] = field(default_factory=dict)

    def as_form(self, node: PDEExpression) -> DomainDifferentialForm:
        if node not in self.form_bases:
            raise ValueError(
                "Exterior lowering requires an explicit charted form binding."
            )
        return self.form_bases[node]

    def compile_form_node(
        self, node: PDEExpression, args: tuple[Any, ...]
    ) -> DomainFunction:
        from ._form_compile import domain_form_proxy, lower_domain_form_operation

        spec = infer_expression_type(node, self.problem).form
        if spec is None:
            raise ValueError("Exterior lowering requires a form-valued expression.")
        operands = tuple(
            self.as_form(argument)
            for argument in node.args
            if infer_expression_type(argument, self.problem).form is not None
        )
        vector = (
            _require_domain_function(args[0], node.op)
            if node.op in ("interior_product", "lie_derivative")
            else None
        )
        output = lower_domain_form_operation(
            node,
            operands,
            vector=vector,
            geometry=self.form_geometry,
            backend=self.differential_backend,
        )
        if output.form_type.form_type_id != spec.form_type.form_type_id:
            raise ValueError(
                "Smooth exterior lowering produced an incompatible form type."
            )
        self.form_bases[node] = output
        return domain_form_proxy(output.coefficients, spec)

    def compile_form_arithmetic(
        self, node: PDEExpression, args: tuple[Any, ...]
    ) -> DomainFunction:
        from ..operators.differential._form_ops import DomainDifferentialForm
        from ._form_compile import domain_form_proxy

        spec = infer_expression_type(node, self.problem).form
        if spec is None:
            raise ValueError("Form arithmetic requires a form-valued expression.")
        bases = tuple(
            self.form_bases[argument]
            for argument in node.args
            if argument in self.form_bases
        )
        if not bases:
            raise ValueError("A form-valued expression lost its chart binding.")
        base = bases[0]
        if any(
            not base.chart.compatible_with(other.chart) or base.var != other.var
            for other in bases[1:]
        ):
            raise ValueError("Form arithmetic requires a common chart and variable.")
        coefficients = tuple(
            self.form_bases[argument].coefficients
            if argument in self.form_bases
            else value
            for argument, value in zip(node.args, args, strict=True)
        )
        match node.op:
            case "add":
                result = coefficients[0]
                for value in coefficients[1:]:
                    result = result + value
            case "multiply":
                result = coefficients[0]
                for value in coefficients[1:]:
                    result = result * value
            case "divide":
                result = coefficients[0] / coefficients[1]
            case "negate":
                result = -coefficients[0]
            case "derivative":
                from ..operators.differential import partial_n

                if node.coordinate is None:
                    raise ValueError("A compiled derivative requires a coordinate.")
                result = partial_n(
                    base.coefficients,
                    var=node.coordinate,
                    axis=node.axis,
                    order=node.order,
                    backend=self.differential_backend,
                )
            case _:
                raise ValueError(f"{node.op} cannot implicitly reinterpret a form.")
        output = DomainDifferentialForm(
            _require_domain_function(result, node.op),
            chart=base.chart,
            degree=spec.form_type.degree,
            twist=spec.form_type.twist,
            fiber_shape=spec.form_type.fiber_shape,
            var=base.var,
        )
        self.form_bases[node] = output
        return domain_form_proxy(output.coefficients, spec)

    def compile_field(self, node: PDEExpression) -> DomainFunction:
        from phydrax.domain import DomainFunction

        from ..operators.differential._form_ops import DomainDifferentialForm
        from ._form_compile import domain_form_proxy

        if node.symbol is None:
            raise ValueError("A compiled field requires a symbol.")
        if node.symbol not in self.fields:
            raise KeyError(f"No DomainFunction supplied for PDE field {node.symbol!r}.")
        binding = self.fields[node.symbol]
        schema = next(item for item in self.problem.fields if item.name == node.symbol)
        if schema.form is None:
            if not isinstance(binding, DomainFunction):
                raise TypeError("Untyped PDE fields require DomainFunction bindings.")
            return binding
        if not isinstance(binding, DomainDifferentialForm):
            raise TypeError(
                "Typed PDE fields require explicit DomainDifferentialForm bindings."
            )
        if binding.form_type.form_type_id != schema.form.form_type.form_type_id:
            raise ValueError("PDE field binding has a different scientific form type.")
        self.form_bases[node] = binding
        return domain_form_proxy(binding.coefficients, schema.form)

    def compile_binding(self, node: PDEExpression) -> Any:
        if node.op == "constant":
            if node.value is None:
                raise ValueError("A compiled constant requires a value.")
            return (
                node.value
                if isinstance(node.value, float)
                else jnp.asarray(float(node.value))
            )
        if node.op == "field":
            return self.compile_field(node)
        if node.op == "parameter":
            if node.symbol is None:
                raise ValueError("A compiled parameter requires a symbol.")
            if node.symbol not in self.parameter_values:
                raise KeyError(f"No value supplied for PDE parameter {node.symbol!r}.")
            return self.parameter_values[node.symbol]
        if node.op == "coordinate":
            if node.symbol is None:
                raise ValueError("A compiled coordinate requires a symbol.")
            if node.symbol not in self.coordinate_values:
                raise KeyError(
                    f"No DomainFunction supplied for PDE coordinate {node.symbol!r}."
                )
            return self.coordinate_values[node.symbol]
        raise ValueError(f"Unsupported PDE expression operation {node.op!r}.")

    def compile_linear_family(self, family: LinearDifferentialFamily) -> Any:
        from phydrax.domain import DomainFunction

        from ._linear_differential import compile_exact_family

        operand = self.compile_node(family.operand)
        if not isinstance(operand, DomainFunction):
            zero = jnp.zeros_like(jnp.asarray(operand))
            if family.component is not None:
                zero = zero[..., family.component]
            if any(item.component for item in family.directions):
                zero = jnp.sum(zero, axis=-1)
            return zero
        prepared = None if self.taylor_plans is None else self.taylor_plans.get(family)
        native_paths = (
            None if self.native_paths is None else self.native_paths.get(family)
        )
        return compile_exact_family(
            family,
            operand,
            self.taylor_policy,
            plan=prepared,
            components=infer_expression_type(family.operand, self.problem).components,
            native_paths=native_paths,
        )

    def compile_arithmetic(self, node: PDEExpression, args: tuple[Any, ...]) -> Any:
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
            return args[0] ** jnp.asarray(args[1])
        raise ValueError(f"Unsupported PDE expression operation {node.op!r}.")

    def compile_pointwise(self, node: PDEExpression, args: tuple[Any, ...]) -> Any:
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
            if node.axis is None:
                raise ValueError("A compiled component requires an axis.")
            return _map_value(args[0], lambda value: value[..., node.axis])
        if node.op == "dot":
            product = args[0] * args[1]
            return _map_value(product, lambda value: jnp.sum(value, axis=-1))
        raise ValueError(f"Unsupported PDE expression operation {node.op!r}.")

    def compile_derivative(
        self, node: PDEExpression, args: tuple[Any, ...]
    ) -> DomainFunction:
        from ..operators.differential import curl, div, grad, laplacian, partial_n

        if node.op == "derivative":
            if node.coordinate is None:
                raise ValueError("A compiled derivative requires a coordinate.")
            return partial_n(
                _require_domain_function(args[0], node.op),
                var=node.coordinate,
                axis=node.axis,
                order=node.order,
                backend=self.differential_backend,
            )
        if node.op == "gradient":
            if node.coordinate is None:
                raise ValueError("A compiled gradient requires a coordinate.")
            return grad(
                _require_domain_function(args[0], node.op),
                var=node.coordinate,
                backend=self.differential_backend,
            )
        if node.op == "divergence":
            if node.coordinate is None:
                raise ValueError("A compiled divergence requires a coordinate.")
            return div(
                _require_domain_function(args[0], node.op),
                var=node.coordinate,
                backend=self.differential_backend,
            )
        if node.op == "curl":
            if node.coordinate is None:
                raise ValueError("A compiled curl requires a coordinate.")
            return curl(
                _require_domain_function(args[0], node.op),
                var=node.coordinate,
                backend=self.differential_backend,
            )
        if node.op == "laplacian":
            if node.coordinate is None:
                raise ValueError("A compiled Laplacian requires a coordinate.")
            return laplacian(
                _require_domain_function(args[0], node.op),
                var=node.coordinate,
                backend=self.differential_backend,
            )
        raise ValueError(f"Unsupported PDE expression operation {node.op!r}.")

    def compile_node(self, node: PDEExpression) -> Any:
        # Bindings precede family peeling; forms precede scalar dispatch. Keep this
        # selector explicit so scientific operation ownership remains inspectable.
        if node.op in ("constant", "field", "parameter", "coordinate"):
            return self.compile_binding(node)
        if (
            self.differential_backend == "jet"
            and infer_expression_type(node, self.problem).form is None
        ):
            from ._linear_differential import extract_linear_differential

            family = extract_linear_differential(node, self.problem)
            if family is not None:
                return self.compile_linear_family(family)
        args = tuple(self.compile_node(argument) for argument in node.args)
        value_type = infer_expression_type(node, self.problem)
        if value_type.form is not None and node.op in (
            "exterior_derivative",
            "codifferential",
            "hodge_star",
            "wedge",
            "interior_product",
            "lie_derivative",
            "trace",
            "gradient",
            "curl",
            "divergence",
            "laplacian",
        ):
            return self.compile_form_node(node, args)
        if value_type.form is not None:
            return self.compile_form_arithmetic(node, args)
        if node.op in ("add", "multiply", "divide", "negate", "power"):
            return self.compile_arithmetic(node, args)
        if node.op in ("sin", "cos", "exp", "log", "sqrt", "component", "dot"):
            return self.compile_pointwise(node, args)
        if node.op in ("derivative", "gradient", "divergence", "curl", "laplacian"):
            return self.compile_derivative(node, args)
        if node.op == "integral":
            if self.integral_compiler is None:
                raise ValueError(
                    "Integral expressions require an integral_compiler bound to a "
                    "concrete sampling or quadrature contract."
                )
            if node.region is None:
                raise ValueError("A compiled integral requires a region.")
            return self.integral_compiler(args[0], node.region, self.problem)
        raise ValueError(f"Unsupported PDE expression operation {node.op!r}.")


def compile_pde_expression(
    expression: PDEExpression,
    problem: PDEProblemIR,
    /,
    *,
    fields: Mapping[str, DomainFunction | DomainDifferentialForm],
    parameters: Mapping[str, Any] | None = None,
    coordinates: Mapping[str, DomainFunction] | None = None,
    differential_backend: DifferentialBackend = "ad",
    taylor_policy: TaylorContractionPolicy | None = None,
    integral_compiler: IntegralCompiler | None = None,
    form_geometry: PDEFormGeometry | None = None,
    _taylor_plans: Mapping[LinearDifferentialFamily, TaylorContractionPlan] | None = None,
    _native_paths: Mapping[
        LinearDifferentialFamily, tuple[NativeContributionPath, ...] | None
    ]
    | None = None,
) -> Any:
    """Compile a validated expression DAG to native PhydraX operations."""
    differential_backend = parse(
        differential_backend, DifferentialBackend, "differential_backend"
    )
    infer_expression_type(expression, problem)
    parameter_values: dict[str, Any] = {
        item.name: item.value for item in problem.parameters if item.value is not None
    }
    if parameters is not None:
        parameter_values.update(parameters)
    coordinate_values = {} if coordinates is None else dict(coordinates)
    compiler = _PDEExpressionCompiler(
        problem=problem,
        fields=fields,
        parameter_values=parameter_values,
        coordinate_values=coordinate_values,
        differential_backend=differential_backend,
        taylor_policy=taylor_policy,
        integral_compiler=integral_compiler,
        form_geometry=form_geometry,
        taylor_plans=_taylor_plans,
        native_paths=_native_paths,
    )
    if differential_backend == "jet":
        from ._linear_differential import normalize_linear_expression

        expression = normalize_linear_expression(expression)
    return compiler.compile_node(expression)


def make_pde_operator(
    expression: PDEExpression,
    problem: PDEProblemIR,
    /,
    *,
    field_names: tuple[str, ...] | None = None,
    parameters: Mapping[str, Any] | None = None,
    coordinates: Mapping[str, DomainFunction] | None = None,
    differential_backend: DifferentialBackend = "ad",
    taylor_policy: TaylorContractionPolicy | None = None,
    integral_compiler: IntegralCompiler | None = None,
    form_geometry: PDEFormGeometry | None = None,
) -> Callable[..., Any]:
    """Adapt an expression to the operator signature used by PhydraX constraints."""
    differential_backend = parse(
        differential_backend, DifferentialBackend, "differential_backend"
    )
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
    taylor_plans: dict[LinearDifferentialFamily, TaylorContractionPlan] = {}
    native_paths: dict[
        LinearDifferentialFamily, tuple[NativeContributionPath, ...] | None
    ] = {}
    if differential_backend == "jet":
        from ._linear_differential import (
            extract_linear_differential,
            normalize_linear_expression,
            prepare_family,
            prepare_native_contribution_paths,
        )

        expression = normalize_linear_expression(expression)

        def prepare(node: PDEExpression) -> None:
            if infer_expression_type(node, problem).form is None:
                family = extract_linear_differential(node, problem)
                if family is not None:
                    if family.operand.op != "constant" and family not in taylor_plans:
                        taylor_plans[family] = prepare_family(family, taylor_policy)
                        native_paths[family] = prepare_native_contribution_paths(
                            family, taylor_policy
                        )
                    prepare(family.operand)
                    return
            for argument in node.args:
                prepare(argument)

        prepare(expression)

    def operator(*field_values: DomainFunction | DomainDifferentialForm) -> Any:
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
            taylor_policy=taylor_policy,
            integral_compiler=integral_compiler,
            form_geometry=form_geometry,
            _taylor_plans=taylor_plans,
            _native_paths=native_paths,
        )

    return operator


def compile_pde_residual_term(
    expression: PDEExpression,
    problem: PDEProblemIR,
    /,
    *,
    component: Any,
    source: IntegrationSource,
    field_names: tuple[str, ...] | None = None,
    parameters: Mapping[str, Any] | None = None,
    coordinates: Mapping[str, DomainFunction] | None = None,
    differential_backend: DifferentialBackend = "ad",
    taylor_policy: TaylorContractionPolicy | None = None,
    integral_compiler: IntegralCompiler | None = None,
    scale: Any = 1.0,
    label: str | None = None,
    form_geometry: PDEFormGeometry | None = None,
) -> Any:
    """Compile an IR equation into a declarative residual and numerical penalty."""
    from ..conditions import Residual
    from ..terms import ResidualPenalty

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
        taylor_policy=taylor_policy,
        integral_compiler=integral_compiler,
        form_geometry=form_geometry,
    )
    condition = Residual(names, component, operator, label=label)
    return ResidualPenalty(condition, source, scale=scale)


def compile_pde_problem(
    problem: PDEProblemIR,
    /,
    *,
    fields: Mapping[str, DomainFunction | DomainDifferentialForm],
    parameters: Mapping[str, Any] | None = None,
    coordinates: Mapping[str, DomainFunction] | None = None,
    differential_backend: DifferentialBackend = "ad",
    taylor_policy: TaylorContractionPolicy | None = None,
    integral_compiler: IntegralCompiler | None = None,
    form_geometry: PDEFormGeometry | None = None,
) -> CompiledPDEProblem:
    """Compile every equation and restriction to executable residuals."""
    differential_backend = parse(
        differential_backend, DifferentialBackend, "differential_backend"
    )
    validate_pde_ir(problem)

    def compile_expression(expression: PDEExpression) -> Any:
        return compile_pde_expression(
            expression,
            problem,
            fields=fields,
            parameters=parameters,
            coordinates=coordinates,
            differential_backend=differential_backend,
            taylor_policy=taylor_policy,
            integral_compiler=integral_compiler,
            form_geometry=form_geometry,
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
    "compile_pde_residual_term",
    "compile_pde_problem",
    "make_pde_operator",
]
