#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
from typing import Any, Literal, TYPE_CHECKING, TypeAlias

import coordax as cx
import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Key

from .._strict import StrictModule


if TYPE_CHECKING:
    from phydrax.domain import (
        DomainComponent,
        DomainFunction,
        GridBatch,
        PointBatch,
        SamplingPlan,
    )

    from ..operators.differential._dimension_estimators import (
        DimensionSamplingPolicy,
    )
    from ..operators.differential._stochastic_estimators import (
        StochasticTracePolicy,
    )
    from ..terms._randomized_residual import (
        RandomizedResidualLossMode,
        RandomizedResidualSamples,
        RandomizedResidualTerm,
    )
from ._compile import compile_pde_expression
from ._ir import PDEEquation, PDEExpression, PDEProblemIR
from ._validate import infer_expression_type, validate_pde_ir


RandomizedDifferentialMethod: TypeAlias = Literal["hutchinson", "dimension"]
RandomizedNodeCoupling: TypeAlias = Literal["independent", "common"]


def _plan_identity(
    method: RandomizedDifferentialMethod,
    trace_policy: StochasticTracePolicy | None,
    dimension_policy: DimensionSamplingPolicy | None,
    loss_mode: RandomizedResidualLossMode,
    node_coupling: RandomizedNodeCoupling,
    prefer_exact: bool,
    /,
) -> tuple[Any, ...]:
    trace = (
        None
        if trace_policy is None
        else (trace_policy.num_probes, trace_policy.distribution)
    )
    dimension = (
        None
        if dimension_policy is None
        else (
            dimension_policy.total_dimension,
            dimension_policy.subset_size,
            dimension_policy.sampling,
            dimension_policy.replace,
            dimension_policy.policy_id,
        )
    )
    return method, trace, dimension, loss_mode, node_coupling, bool(prefer_exact)


def _stable_id(identity: tuple[Any, ...], /) -> str:
    digest = sha256(b"phydrax-randomized-differential-plan\0")
    digest.update(repr(identity).encode("utf-8"))
    return digest.hexdigest()


class RandomizedDifferentialPlan(StrictModule):
    """Immutable randomization and squared-loss policy for one PDE residual."""

    trace_policy: StochasticTracePolicy | None
    dimension_policy: DimensionSamplingPolicy | None
    method: RandomizedDifferentialMethod = eqx.field(static=True)
    loss_mode: RandomizedResidualLossMode = eqx.field(static=True)
    node_coupling: RandomizedNodeCoupling = eqx.field(static=True)
    prefer_exact: bool = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        method: RandomizedDifferentialMethod = "hutchinson",
        /,
        *,
        trace_policy: StochasticTracePolicy | None = None,
        dimension_policy: DimensionSamplingPolicy | None = None,
        loss_mode: RandomizedResidualLossMode = "u_statistic",
        node_coupling: RandomizedNodeCoupling = "independent",
        prefer_exact: bool = True,
        plan_id: str | None = None,
    ):
        from ..operators.differential._dimension_estimators import (
            DimensionSamplingPolicy,
        )
        from ..operators.differential._stochastic_estimators import (
            StochasticTracePolicy,
        )

        if method not in ("hutchinson", "dimension"):
            raise ValueError("method must be 'hutchinson' or 'dimension'.")
        if loss_mode not in ("u_statistic", "independent_product", "plug_in"):
            raise ValueError("Unknown randomized residual loss_mode.")
        if node_coupling not in ("independent", "common"):
            raise ValueError("node_coupling must be 'independent' or 'common'.")
        if method == "hutchinson":
            if dimension_policy is not None:
                raise ValueError("Hutchinson plans do not accept dimension_policy.")
            trace = StochasticTracePolicy() if trace_policy is None else trace_policy
            if not isinstance(trace, StochasticTracePolicy):
                raise TypeError("trace_policy must be a StochasticTracePolicy.")
            dimension = None
        else:
            if trace_policy is not None:
                raise ValueError("Dimension plans do not accept trace_policy.")
            if not isinstance(dimension_policy, DimensionSamplingPolicy):
                raise TypeError("Dimension plans require a DimensionSamplingPolicy.")
            if loss_mode == "u_statistic" and not dimension_policy.replace:
                raise ValueError(
                    "u_statistic requires independent coordinate draws; use replacement "
                    "or loss_mode='independent_product'."
                )
            trace = None
            dimension = dimension_policy
        identity = _plan_identity(
            method,
            trace,
            dimension,
            loss_mode,
            node_coupling,
            bool(prefer_exact),
        )
        resolved_id = _stable_id(identity) if plan_id is None else str(plan_id)
        if not resolved_id:
            raise ValueError("plan_id must be non-empty.")
        self.trace_policy = trace
        self.dimension_policy = dimension
        self.method = method
        self.loss_mode = loss_mode
        self.node_coupling = node_coupling
        self.prefer_exact = bool(prefer_exact)
        self.plan_id = resolved_id

    @property
    def num_realizations(self) -> int:
        if self.trace_policy is not None:
            return self.trace_policy.num_probes
        if self.dimension_policy is None:
            raise RuntimeError("Randomized differential policy is unavailable.")
        return self.dimension_policy.subset_size


@dataclass(frozen=True, slots=True)
class RandomizedCompilationReport:
    supported: bool
    problem_hash: str
    equation_name: str
    plan_id: str
    randomized_node_paths: tuple[str, ...]
    exact_node_paths: tuple[str, ...]
    node_methods: tuple[tuple[str, str], ...]
    rejection_reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CompiledRandomizedPDETerm:
    term: RandomizedResidualTerm
    report: RandomizedCompilationReport
    source: PDEEquation


def _coordinate(problem: PDEProblemIR, name: str, /):
    return next(item for item in problem.coordinates if item.name == name)


def _has_unknown_dependence(expression: PDEExpression, problem: PDEProblemIR, /) -> bool:
    if expression.op == "field":
        return True
    if expression.op == "parameter":
        assert expression.symbol is not None
        parameter = next(item for item in problem.parameters if item.name == expression.symbol)
        return parameter.functional
    return any(_has_unknown_dependence(argument, problem) for argument in expression.args)


def _analyze_expression(
    expression: PDEExpression,
    problem: PDEProblemIR,
    plan: RandomizedDifferentialPlan,
    /,
) -> tuple[
    tuple[str, ...],
    tuple[str, ...],
    tuple[tuple[str, str], ...],
    tuple[str, ...],
]:
    randomized: list[str] = []
    exact: list[str] = []
    methods: list[tuple[str, str]] = []
    reasons: list[str] = []

    def visit(node: PDEExpression, path: str) -> bool:
        children = tuple(
            visit(argument, f"{path}.args[{index}]")
            for index, argument in enumerate(node.args)
        )
        randomized_children = sum(children)
        if node.op in ("laplacian", "divergence"):
            if randomized_children:
                reasons.append(f"{path}: nested randomized differential operators are unsupported.")
                return True
            assert node.coordinate is not None
            coordinate = _coordinate(problem, node.coordinate)
            if plan.method == "dimension":
                assert plan.dimension_policy is not None
                if plan.dimension_policy.total_dimension != coordinate.size:
                    reasons.append(
                        f"{path}: dimension policy size "
                        f"{plan.dimension_policy.total_dimension} does not match coordinate "
                        f"{node.coordinate!r} size {coordinate.size}."
                    )
            if plan.prefer_exact and not _has_unknown_dependence(node.args[0], problem):
                exact.append(path)
                methods.append((path, "exact-ad"))
                return False
            randomized.append(path)
            methods.append((path, plan.method))
            return True
        if node.op in ("derivative", "gradient", "curl") and randomized_children:
            reasons.append(
                f"{path}: differentiation of a randomized intermediate is unsupported."
            )
        if node.op in ("sin", "cos", "exp", "log", "sqrt", "power") and randomized_children:
            reasons.append(
                f"{path}: nonlinear transformation of a randomized estimator is biased."
            )
        if node.op == "multiply" and randomized_children > 1:
            reasons.append(f"{path}: product of randomized intermediates is biased.")
        if node.op == "divide" and len(children) == 2 and children[1]:
            reasons.append(f"{path}: randomized denominators are unsupported.")
        if node.op == "dot" and randomized_children > 1:
            reasons.append(f"{path}: dot product of randomized intermediates is biased.")
        if node.op == "integral":
            reasons.append(f"{path}: randomized expressions under integral nodes are unsupported.")
        return randomized_children > 0

    visit(expression, "root")
    return tuple(randomized), tuple(exact), tuple(methods), tuple(dict.fromkeys(reasons))


def analyze_randomized_compilation(
    problem: PDEProblemIR,
    equation: str | PDEEquation,
    plan: RandomizedDifferentialPlan,
    /,
) -> RandomizedCompilationReport:
    """Classify every trace-like node before any executable objective is built."""
    if not isinstance(plan, RandomizedDifferentialPlan):
        raise TypeError("plan must be a RandomizedDifferentialPlan.")
    validate_pde_ir(problem)
    source = _resolve_equation(problem, equation)
    value_type = infer_expression_type(source.residual, problem)
    randomized, exact, methods, reasons = _analyze_expression(
        source.residual,
        problem,
        plan,
    )
    rejected = list(reasons)
    if not value_type.is_scalar:
        rejected.append("root: randomized residual objectives currently require a scalar equation.")
    if not randomized:
        rejected.append(
            "root: no stochastic differential node remains after exact-first lowering; "
            "use the deterministic PDE compiler."
        )
    return RandomizedCompilationReport(
        supported=not rejected,
        problem_hash=problem.canonical_hash,
        equation_name=source.name,
        plan_id=plan.plan_id,
        randomized_node_paths=randomized,
        exact_node_paths=exact,
        node_methods=methods,
        rejection_reasons=tuple(rejected),
    )


def _resolve_equation(
    problem: PDEProblemIR,
    equation: str | PDEEquation,
    /,
) -> PDEEquation:
    if isinstance(equation, PDEEquation):
        if equation not in problem.equations:
            raise ValueError("equation must belong to problem.equations.")
        return equation
    name = str(equation)
    matches = tuple(item for item in problem.equations if item.name == name)
    if len(matches) != 1:
        raise KeyError(f"Expected exactly one PDE equation named {name!r}.")
    return matches[0]


def _promoted_fields(
    functions: Mapping[str, DomainFunction],
    problem: PDEProblemIR,
    /,
) -> tuple[Any, dict[str, DomainFunction]]:
    from phydrax.domain import DomainFunction

    names = tuple(field.name for field in problem.fields)
    missing = tuple(name for name in names if name not in functions)
    if missing:
        raise KeyError(f"Missing PDE fields {missing!r}.")
    selected = {name: functions[name] for name in names}
    if any(not isinstance(value, DomainFunction) for value in selected.values()):
        raise TypeError("Every compiled PDE field must be a DomainFunction.")
    iterator = iter(selected.values())
    first = next(iterator)
    domain = first.domain
    for value in iterator:
        if value.domain.labels != domain.labels:
            domain = domain.join(value.domain)
    return domain, {name: value.promote(domain) for name, value in selected.items()}


def _coordinate_functions(domain: Any, problem: PDEProblemIR, /) -> dict[str, DomainFunction]:
    from phydrax.domain import DomainFunction

    coordinates: dict[str, DomainFunction] = {}
    for coordinate in problem.coordinates:
        name = coordinate.name

        def identity(value, *, key=None, iter=None):
            del key, iter
            return value

        coordinates[name] = DomainFunction(
            domain=domain,
            deps=(name,),
            func=identity,
        )
    return coordinates


def _evaluate_domain_value(
    value: Any,
    labels: tuple[str, ...],
    args: tuple[Any, ...],
    key: Key[Array, ""],
    /,
) -> Array:
    from phydrax.domain import DomainFunction

    if not isinstance(value, DomainFunction):
        return jnp.asarray(value)
    positions = {label: index for index, label in enumerate(labels)}
    local_args = tuple(args[positions[label]] for label in value.deps)
    return jnp.asarray(value.func(*local_args, key=key))


class _RandomizedPointCallable(StrictModule):
    fields: Mapping[str, DomainFunction]
    parameters: Mapping[str, Any]
    plan: RandomizedDifferentialPlan
    problem: PDEProblemIR = eqx.field(static=True)
    expression: PDEExpression = eqx.field(static=True)
    randomized_paths: tuple[str, ...] = eqx.field(static=True)
    node_indices: tuple[tuple[str, int], ...] = eqx.field(static=True)
    labels: tuple[str, ...] = eqx.field(static=True)

    def _node_key(self, key: Key[Array, ""], path: str, /) -> Key[Array, ""]:
        if self.plan.node_coupling == "common":
            return key
        index = dict(self.node_indices)[path]
        return jr.fold_in(key, index)

    def _exact(
        self,
        node: PDEExpression,
        args: tuple[Any, ...],
        key: Key[Array, ""],
        /,
    ) -> Array:
        coordinates = _coordinate_functions(next(iter(self.fields.values())).domain, self.problem)
        compiled = compile_pde_expression(
            node,
            self.problem,
            fields=self.fields,
            parameters=self.parameters,
            coordinates=coordinates,
            differential_backend="ad",
        )
        return _evaluate_domain_value(compiled, self.labels, args, key)

    def _random_operator(
        self,
        node: PDEExpression,
        path: str,
        args: tuple[Any, ...],
        key: Key[Array, ""],
        /,
    ) -> Array:
        from phydrax.domain import DomainFunction

        from ..operators.differential._dimension_estimators import (
            coordinate_divergence_samples,
            coordinate_second_derivative_samples,
        )
        from ..operators.differential._stochastic_estimators import (
            stochastic_divergence_samples,
            stochastic_trace_samples,
        )

        assert node.coordinate is not None
        coordinate_position = self.labels.index(node.coordinate)
        state = jnp.asarray(args[coordinate_position])
        coordinates = _coordinate_functions(next(iter(self.fields.values())).domain, self.problem)
        operand = compile_pde_expression(
            node.args[0],
            self.problem,
            fields=self.fields,
            parameters=self.parameters,
            coordinates=coordinates,
            differential_backend="ad",
        )
        if not isinstance(operand, DomainFunction):
            raise TypeError(f"{path}: randomized differential operand must be a DomainFunction.")
        operand_positions = {label: index for index, label in enumerate(self.labels)}
        local_args = [args[operand_positions[label]] for label in operand.deps]
        if node.coordinate not in operand.deps:
            return jnp.zeros((self.plan.num_realizations,))
        local_position = operand.deps.index(node.coordinate)
        node_key = self._node_key(key, path)

        def evaluate(local_state):
            current = list(local_args)
            current[local_position] = local_state
            return operand.func(*current, key=key)

        if node.op == "laplacian":
            if self.plan.method == "hutchinson":
                if self.plan.trace_policy is None:
                    raise RuntimeError("Hutchinson trace policy is unavailable.")
                samples = stochastic_trace_samples(
                    evaluate,
                    state,
                    lambda current, direction: direction,
                    node_key,
                    policy=self.plan.trace_policy,
                )
            else:
                if self.plan.dimension_policy is None:
                    raise RuntimeError("Dimension sampling policy is unavailable.")
                samples = coordinate_second_derivative_samples(
                    evaluate,
                    state,
                    node_key,
                    self.plan.dimension_policy,
                )
            return samples.values
        if node.op == "divergence":
            if self.plan.method == "hutchinson":
                if self.plan.trace_policy is None:
                    raise RuntimeError("Hutchinson trace policy is unavailable.")
                samples = stochastic_divergence_samples(
                    evaluate,
                    state,
                    node_key,
                    policy=self.plan.trace_policy,
                )
            else:
                if self.plan.dimension_policy is None:
                    raise RuntimeError("Dimension sampling policy is unavailable.")
                samples = coordinate_divergence_samples(
                    evaluate,
                    state,
                    node_key,
                    self.plan.dimension_policy,
                )
            return samples.values
        raise RuntimeError("Internal randomized differential dispatch failed.")

    def _evaluate(
        self,
        node: PDEExpression,
        path: str,
        args: tuple[Any, ...],
        key: Key[Array, ""],
        /,
    ) -> tuple[Array, bool]:
        randomized_path_set = frozenset(self.randomized_paths)
        if path in randomized_path_set:
            return self._random_operator(node, path, args, key), True
        descendants = tuple(
            item == path or item.startswith(path + ".")
            for item in self.randomized_paths
        )
        if not any(descendants):
            return self._exact(node, args, key), False
        evaluated = tuple(
            self._evaluate(argument, f"{path}.args[{index}]", args, key)
            for index, argument in enumerate(node.args)
        )
        values = tuple(item[0] for item in evaluated)
        randomized = tuple(item[1] for item in evaluated)
        if node.op == "add":
            result = values[0]
            for value in values[1:]:
                result = result + value
            return result, True
        if node.op == "multiply":
            result = values[0]
            for value in values[1:]:
                result = result * value
            return result, True
        if node.op == "divide":
            return values[0] / values[1], True
        if node.op == "negate":
            return -values[0], True
        if node.op == "component":
            assert node.axis is not None
            return values[0][..., node.axis], True
        if node.op == "dot":
            left, right = values
            if randomized[0] and not randomized[1]:
                right = right[None, ...]
            elif randomized[1] and not randomized[0]:
                left = left[None, ...]
            return jnp.sum(left * right, axis=-1), True
        raise RuntimeError(f"Unsupported randomized expression node {node.op!r} at {path}.")

    def __call__(self, *args: Any, key=None, iter=None, **kwargs: Any) -> Array:
        del iter, kwargs
        resolved_key = jr.key(0) if key is None else key
        result, randomized = self._evaluate(
            self.expression,
            "root",
            tuple(args),
            resolved_key,
        )
        if not randomized:
            raise RuntimeError("Randomized expression unexpectedly lowered to an exact value.")
        return result


class _RandomizedPDEEvaluator(StrictModule):
    parameters: Mapping[str, Any]
    plan: RandomizedDifferentialPlan
    problem: PDEProblemIR = eqx.field(static=True)
    expression: PDEExpression = eqx.field(static=True)
    randomized_paths: tuple[str, ...] = eqx.field(static=True)

    def __call__(
        self,
        functions: Mapping[str, DomainFunction],
        collocation: Any,
        key: Key[Array, ""],
        /,
    ) -> RandomizedResidualSamples:
        from phydrax.domain import DomainFunction, GridBatch, PointBatch

        from ..terms._randomized_residual import RandomizedResidualSamples

        if isinstance(collocation, tuple):
            raise TypeError("Randomized PDE objectives do not support ComponentSum batches.")
        if not isinstance(collocation, (PointBatch, GridBatch)):
            raise TypeError("Randomized PDE collocation must be a structured point batch.")
        domain, fields = _promoted_fields(functions, self.problem)
        node_indices = tuple(
            (path, index) for index, path in enumerate(self.randomized_paths)
        )
        residual = DomainFunction(
            domain=domain,
            deps=domain.labels,
            func=_RandomizedPointCallable(
                fields=fields,
                parameters=self.parameters,
                plan=self.plan,
                problem=self.problem,
                expression=self.expression,
                randomized_paths=self.randomized_paths,
                node_indices=node_indices,
                labels=domain.labels,
            ),
        )
        evaluated = residual(collocation, key=key)
        named_positions = tuple(
            index for index, dim in enumerate(evaluated.dims) if dim is not None
        )
        output_positions = tuple(
            index for index, dim in enumerate(evaluated.dims) if dim is None
        )
        if len(output_positions) < 1:
            raise ValueError("Randomized residual output is missing its realization axis.")
        permutation = named_positions + output_positions
        data = jnp.transpose(jnp.asarray(evaluated.data), permutation)
        sample_shape = tuple(int(data.shape[index]) for index in range(len(named_positions)))
        if int(data.shape[len(sample_shape)]) != self.plan.num_realizations:
            raise ValueError("Randomized residual realization count does not match its plan.")
        values = jnp.moveaxis(data, len(sample_shape), 0)
        event_shape = tuple(int(size) for size in values.shape[1 + len(sample_shape) :])
        mask = jnp.ones(sample_shape, dtype=bool)
        weights = jnp.ones(sample_shape, dtype=float)
        if isinstance(collocation, GridBatch):
            named_dims = tuple(evaluated.dims[index] for index in named_positions)
            mask_field = cx.Field(mask, dims=named_dims)
            weight_field = cx.Field(weights, dims=named_dims)
            for current in collocation.coord_mask_by_label.values():
                mask_field = mask_field * current
            for current in collocation.coord_geometry_weight_by_label.values():
                weight_field = weight_field * current
            mask = jnp.asarray(mask_field.data, dtype=bool)
            weights = jnp.asarray(weight_field.data, dtype=float)
        return RandomizedResidualSamples(
            values,
            sample_shape=sample_shape,
            event_shape=event_shape,
            mask=mask,
            weights=weights,
            estimator_id=f"pde-{self.plan.plan_id}",
        )


class _RandomizedCollocationSampler(StrictModule):
    component: DomainComponent
    sampling: SamplingPlan

    def __call__(self, key: Key[Array, ""], /):
        return self.component.sample(self.sampling, key=key)


def compile_pde_randomized_term(
    problem: PDEProblemIR,
    equation: str | PDEEquation,
    plan: RandomizedDifferentialPlan,
    /,
    *,
    component: DomainComponent,
    sampling: SamplingPlan,
    parameters: Mapping[str, Any] | None = None,
    weight: Any = 1.0,
    label: str | None = None,
    sampling_mode: Literal["resample", "fixed"] = "resample",
    fixed_batch: PointBatch | GridBatch | None = None,
    fixed_batch_key: Key[Array, ""] = jr.key(0),
) -> CompiledRandomizedPDETerm:
    """Compile one scalar IR equation to an estimator-aware sampled term."""
    from phydrax.domain import ComponentSum, DomainComponent

    from ..terms._randomized_residual import RandomizedResidualTerm

    if isinstance(component, ComponentSum):
        raise TypeError("Randomized PDE terms do not support ComponentSum.")
    if not isinstance(component, DomainComponent):
        raise TypeError("component must be a DomainComponent.")
    report = analyze_randomized_compilation(problem, equation, plan)
    source = _resolve_equation(problem, equation)
    if not report.supported:
        details = "; ".join(report.rejection_reasons)
        raise ValueError(f"Randomized PDE compilation rejected: {details}")
    collocation_sampler = _RandomizedCollocationSampler(
        component=component,
        sampling=sampling,
    )
    mode = str(sampling_mode).lower()
    if mode not in ("resample", "fixed"):
        raise ValueError("sampling_mode must be 'resample' or 'fixed'.")
    if mode == "fixed":
        collocation = (
            collocation_sampler(fixed_batch_key) if fixed_batch is None else fixed_batch
        )
    else:
        if fixed_batch is not None:
            raise ValueError("fixed_batch is only valid with sampling_mode='fixed'.")
        collocation = collocation_sampler
    evaluator = _RandomizedPDEEvaluator(
        parameters={} if parameters is None else dict(parameters),
        plan=plan,
        problem=problem,
        expression=source.residual,
        randomized_paths=report.randomized_node_paths,
    )
    term = RandomizedResidualTerm(
        evaluator,
        collocation=collocation,
        loss_mode=plan.loss_mode,
        sampling_mode=mode,
        scalar_weight=weight,
        label=source.name if label is None else label,
    )
    return CompiledRandomizedPDETerm(term, report, source)




__all__ = [
    "analyze_randomized_compilation",
    "CompiledRandomizedPDETerm",
    "compile_pde_randomized_term",
    "RandomizedCompilationReport",
    "RandomizedDifferentialMethod",
    "RandomizedDifferentialPlan",
    "RandomizedNodeCoupling",
]
