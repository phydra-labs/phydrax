#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from hashlib import sha256
from math import prod
from typing import Any, assert_never, final, Literal, TYPE_CHECKING, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jax import Array

import phydrax.axes as cx

from .._strict import StrictModule


if TYPE_CHECKING:
    from phydrax.domain import (
        DomainComponent,
        DomainFunction,
        GraphBatch,
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
    from ..operators.differential._taylor_contracts import TaylorContractionPolicy
    from ..operators.differential._taylor_planning import TaylorContractionPlan
    from ..terms._randomized_residual import (
        RandomizedResidualSamples,
        RandomizedResidualSamplingMode,
        RandomizedResidualTerm,
    )
from .._randomized_residual_modes import (
    RandomizedResidualLossMode,
    RealizationSamplingDesign,
)
from .._sampling._addressing import derive_key, SampleAddress
from ..typing import parse, PRNGKey
from ._compile import compile_pde_expression
from ._ir import PDEEquation, PDEExpression, PDEProblemIR
from ._linear_differential import (
    extract_linear_differential,
    family_contribution,
    LinearDifferentialFamily,
    NativeContributionPath,
    NativeFamilyCallable,
    normalize_linear_expression,
    prepare_family,
    prepare_native_contribution_paths,
    try_owned_family,
)
from ._validate import infer_expression_type, validate_pde_ir


RandomizedDifferentialMethod: TypeAlias = Literal[
    "hutchinson", "dimension", "gaussian_bilaplacian"
]
RandomizedNodeCoupling: TypeAlias = Literal["independent", "common"]
RandomizedExecutionBackend: TypeAlias = Literal["ad", "jet"]
RandomizedPopulation: TypeAlias = Literal["coordinate", "terms"]


def _plan_identity(
    method: RandomizedDifferentialMethod,
    trace_policy: StochasticTracePolicy | None,
    dimension_policy: DimensionSamplingPolicy | None,
    loss_mode: RandomizedResidualLossMode,
    node_coupling: RandomizedNodeCoupling,
    prefer_exact: bool,
    backend: RandomizedExecutionBackend,
    population: RandomizedPopulation,
    taylor_policy: TaylorContractionPolicy | None,
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
    base = method, trace, dimension, loss_mode, node_coupling, bool(prefer_exact)
    if backend == "ad" and population == "coordinate" and taylor_policy is None:
        return base
    taylor = None
    if taylor_policy is not None:
        resources = taylor_policy.resources
        taylor = (
            taylor_policy.strategy,
            resources.max_order,
            resources.max_certificate_states,
            resources.max_linear_terms,
            resources.max_candidates,
            resources.workset_size,
            resources.max_logical_buffer_elements,
        )
    return (*base, backend, population, taylor)


def _stable_id(identity: tuple[Any, ...], /) -> str:
    digest = sha256(b"phydrax-randomized-differential-plan\0")
    digest.update(repr(identity).encode("utf-8"))
    return digest.hexdigest()


@final
class RandomizedDifferentialPlan(StrictModule):
    """Immutable randomization and squared-loss policy for one PDE residual."""

    trace_policy: StochasticTracePolicy | None
    dimension_policy: DimensionSamplingPolicy | None
    method: RandomizedDifferentialMethod = eqx.field(static=True)
    loss_mode: RandomizedResidualLossMode = eqx.field(static=True)
    node_coupling: RandomizedNodeCoupling = eqx.field(static=True)
    prefer_exact: bool = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)
    backend: RandomizedExecutionBackend = eqx.field(static=True)
    population: RandomizedPopulation = eqx.field(static=True)
    taylor_policy: TaylorContractionPolicy | None = eqx.field(static=True)

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
        backend: RandomizedExecutionBackend = "ad",
        population: RandomizedPopulation = "coordinate",
        taylor_policy: TaylorContractionPolicy | None = None,
    ) -> None:
        from ..operators.differential._dimension_estimators import (
            DimensionSamplingPolicy,
        )
        from ..operators.differential._stochastic_estimators import (
            StochasticTracePolicy,
        )

        method = parse(method, RandomizedDifferentialMethod, "method")
        loss_mode = parse(loss_mode, RandomizedResidualLossMode, "loss_mode")
        node_coupling = parse(node_coupling, RandomizedNodeCoupling, "node_coupling")
        backend = parse(backend, RandomizedExecutionBackend, "backend")
        population = parse(population, RandomizedPopulation, "population")
        if population == "terms" and method != "dimension":
            raise ValueError("Term populations require method='dimension'.")
        if taylor_policy is not None:
            from ..operators.differential._taylor_contracts import TaylorContractionPolicy

            if not isinstance(taylor_policy, TaylorContractionPolicy):
                raise TypeError("taylor_policy must be a TaylorContractionPolicy.")
        match method:
            case "hutchinson" | "gaussian_bilaplacian":
                if dimension_policy is not None:
                    raise ValueError("Probe plans do not accept dimension_policy.")
                trace = (
                    StochasticTracePolicy(
                        distribution="normal"
                        if method == "gaussian_bilaplacian"
                        else "rademacher"
                    )
                    if trace_policy is None
                    else trace_policy
                )
                if not isinstance(trace, StochasticTracePolicy):
                    raise TypeError("trace_policy must be a StochasticTracePolicy.")
                if method == "gaussian_bilaplacian" and trace.distribution != "normal":
                    raise ValueError("Gaussian bilaplacian requires normal probes.")
                dimension = None
            case "dimension":
                if trace_policy is not None:
                    raise ValueError("Dimension plans do not accept trace_policy.")
                if not isinstance(dimension_policy, DimensionSamplingPolicy):
                    raise TypeError("Dimension plans require a DimensionSamplingPolicy.")
                if (
                    loss_mode == "u_statistic"
                    and not dimension_policy.replace
                    and dimension_policy.subset_size != dimension_policy.total_dimension
                ):
                    raise ValueError(
                        "u_statistic requires independent coordinate draws; use replacement "
                        "or loss_mode='independent_product'."
                    )
                trace = None
                dimension = dimension_policy
            case _:
                assert_never(method)
        identity = _plan_identity(
            method,
            trace,
            dimension,
            loss_mode,
            node_coupling,
            bool(prefer_exact),
            backend,
            population,
            taylor_policy,
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
        self.backend = backend
        self.population = population
        self.taylor_policy = taylor_policy

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
    execution_backend: RandomizedExecutionBackend = "ad"
    population: RandomizedPopulation = "coordinate"
    population_size: int | None = None
    term_identities: tuple[tuple[str, tuple[object, ...], int], ...] = ()
    contraction_certificates: tuple[tuple[str, str], ...] = ()
    contraction_plans: tuple[tuple[str, TaylorContractionPlan], ...] = ()
    native_contribution_paths: tuple[
        tuple[str, tuple[NativeContributionPath, ...] | None], ...
    ] = ()
    sampling_design: RealizationSamplingDesign = "unknown"


@dataclass(frozen=True, slots=True)
class CompiledRandomizedPDETerm:
    term: RandomizedResidualTerm
    report: RandomizedCompilationReport
    source: PDEEquation


def _has_unknown_dependence(expression: PDEExpression, problem: PDEProblemIR, /) -> bool:
    if expression.op == "field":
        return True
    if expression.op == "parameter":
        if expression.symbol is None:
            raise ValueError("A PDE parameter dependence requires a symbol.")
        parameter = next(
            item for item in problem.parameters if item.name == expression.symbol
        )
        return parameter.functional
    return any(_has_unknown_dependence(argument, problem) for argument in expression.args)


def _admit_probe_family(
    family: LinearDifferentialFamily,
    path: str,
    method: Literal["hutchinson", "gaussian_bilaplacian"],
    reasons: list[str],
    /,
) -> None:
    if method == "hutchinson":
        summed = tuple(item for item in family.directions if item.axis is None)
        if len(summed) != 1:
            reasons.append(
                f"{path}: Hutchinson requires one trace contraction; use Gaussian bilaplacian or a dimension population."
            )
    elif (
        len(family.directions) != 2
        or any(
            item.order != 2 or item.axis is not None or item.component
            for item in family.directions
        )
        or family.directions[0].coordinate != family.directions[1].coordinate
    ):
        reasons.append(
            f"{path}: Gaussian bilaplacian requires two Laplacians on the same declared coordinate."
        )


def _admit_randomized_family(
    family: LinearDifferentialFamily,
    path: str,
    plan: RandomizedDifferentialPlan,
    reasons: list[str],
    /,
) -> None:
    match plan.method:
        case "dimension":
            if plan.population == "coordinate":
                if plan.dimension_policy is None:
                    raise RuntimeError("Dimension policy is unavailable.")
                if plan.dimension_policy.total_dimension != family.population_size:
                    reasons.append(
                        f"{path}: dimension policy size {plan.dimension_policy.total_dimension} does not match coordinate population size {family.population_size}."
                    )
        case "hutchinson" | "gaussian_bilaplacian":
            _admit_probe_family(family, path, plan.method, reasons)
        case _:
            assert_never(plan.method)


def _reject_randomized_intermediates(
    node: PDEExpression,
    path: str,
    children: tuple[bool, ...],
    reasons: list[str],
    /,
) -> bool:
    randomized_children = sum(children)
    if (
        node.op in ("derivative", "gradient", "curl", "laplacian", "divergence")
        and randomized_children
    ):
        reasons.append(
            f"{path}: differentiation of a randomized intermediate is unsupported; use a native linear contraction chain."
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
        reasons.append(
            f"{path}: randomized expressions under integral nodes are unsupported."
        )
    return randomized_children > 0


def _analyze_expression(
    expression: PDEExpression,
    problem: PDEProblemIR,
    plan: RandomizedDifferentialPlan,
    /,
) -> tuple[
    tuple[str, ...], tuple[str, ...], tuple[tuple[str, str], ...], tuple[str, ...]
]:
    randomized: list[str] = []
    exact: list[str] = []
    methods: list[tuple[str, str]] = []
    reasons: list[str] = []

    def visit(node: PDEExpression, path: str) -> bool:
        try:
            family = extract_linear_differential(node, problem)
        except ValueError as error:
            reasons.append(f"{path}: {error}")
            return False
        candidate = family is not None and (family.has_sum or plan.population == "terms")
        if candidate and family is not None:
            if infer_expression_type(family.operand, problem).form is not None:
                reasons.append(
                    f"{path}: randomized native contractions cannot implicitly reinterpret an exterior form layout."
                )
                return False
            if plan.prefer_exact and not _has_unknown_dependence(family.operand, problem):
                exact.append(path)
                methods.append((path, "exact-ad"))
                return False
            _admit_randomized_family(family, path, plan, reasons)
            randomized.append(path)
            methods.append((path, plan.method))
            return True
        children = tuple(
            visit(argument, f"{path}.args[{index}]")
            for index, argument in enumerate(node.args)
        )
        return _reject_randomized_intermediates(node, path, children, reasons)

    visit(expression, "root")
    return tuple(randomized), tuple(exact), tuple(methods), tuple(dict.fromkeys(reasons))


@final
@dataclass(frozen=True, slots=True)
class _DifferentialWorkset:
    path: str
    family: LinearDifferentialFamily
    plan: TaylorContractionPlan | None
    native_paths: tuple[NativeContributionPath, ...] | None


@final
@dataclass(frozen=True, slots=True)
class _PopulationTerm:
    workset: _DifferentialWorkset | None
    exact: PDEExpression | None
    contexts: tuple[tuple[PDEExpression, int], ...] = ()


def _worksets(
    expression: PDEExpression,
    problem: PDEProblemIR,
    plan: RandomizedDifferentialPlan,
    paths: tuple[str, ...],
    /,
    *,
    prepared_plans: Mapping[str, TaylorContractionPlan] | None = None,
    prepared_native_paths: Mapping[str, tuple[NativeContributionPath, ...] | None]
    | None = None,
) -> tuple[_DifferentialWorkset, ...]:
    result: list[_DifferentialWorkset] = []

    def visit(node: PDEExpression, path: str) -> None:
        if path in paths:
            family = extract_linear_differential(node, problem)
            if family is None:
                raise RuntimeError("Randomized contraction lost its native family.")
            prepared = None
            if plan.backend == "jet":
                if prepared_plans is None:
                    prepared = prepare_family(
                        family,
                        plan.taylor_policy,
                        gaussian=plan.method == "gaussian_bilaplacian",
                    )
                else:
                    prepared = prepared_plans[path]
            native_paths = (
                prepare_native_contribution_paths(
                    family, plan.taylor_policy, backend=plan.backend
                )
                if prepared_native_paths is None
                else prepared_native_paths[path]
            )
            result.append(_DifferentialWorkset(path, family, prepared, native_paths))
            return
        for index, argument in enumerate(node.args):
            visit(argument, f"{path}.args[{index}]")

    visit(expression, "root")
    return tuple(result)


def _population_terms(
    expression: PDEExpression,
    worksets: tuple[_DifferentialWorkset, ...],
    /,
) -> tuple[_PopulationTerm, ...]:
    by_path = {item.path: item for item in worksets}

    def visit(node: PDEExpression, path: str) -> tuple[_PopulationTerm, ...]:
        if path in by_path:
            return (_PopulationTerm(by_path[path], None),)
        descendants = tuple(item for item in by_path if item.startswith(path + "."))
        if not descendants:
            return (_PopulationTerm(None, node),)
        if node.op == "add":
            return tuple(
                term
                for index, argument in enumerate(node.args)
                for term in visit(argument, f"{path}.args[{index}]")
            )
        active = tuple(
            index
            for index in range(len(node.args))
            if any(item.startswith(f"{path}.args[{index}]") for item in descendants)
        )
        if len(active) != 1:
            raise ValueError(
                "A term population requires a linear combination of native families."
            )
        index = active[0]
        if node.op not in ("negate", "multiply", "divide", "dot", "component"):
            raise ValueError(
                f"{path}: {node.op} cannot lower a finite linear population."
            )
        return tuple(
            _PopulationTerm(term.workset, term.exact, term.contexts + ((node, index),))
            for term in visit(node.args[index], f"{path}.args[{index}]")
        )

    return visit(expression, "root")


def _sampling_law(
    plan: RandomizedDifferentialPlan, /
) -> tuple[RealizationSamplingDesign, int | None]:
    if plan.dimension_policy is None:
        return "iid", None
    policy = plan.dimension_policy
    if policy.replace:
        return "iid", None
    if policy.subset_size == policy.total_dimension:
        return "exact", policy.total_dimension
    return "finite_population", policy.total_dimension


def _admit_worksets(
    expression: PDEExpression,
    problem: PDEProblemIR,
    plan: RandomizedDifferentialPlan,
    randomized: tuple[str, ...],
    rejected: list[str],
    /,
) -> tuple[tuple[_DifferentialWorkset, ...], int]:
    worksets: tuple[_DifferentialWorkset, ...] = ()
    try:
        worksets = _worksets(expression, problem, plan, randomized)
    except (ValueError, TypeError, RuntimeError) as error:
        rejected.append(f"root: native contraction admission refused: {error}")
    family_limit = (
        256
        if plan.taylor_policy is None
        else plan.taylor_policy.resources.max_linear_terms
    )
    if len(worksets) > family_limit:
        rejected.append(
            f"root: heterogeneous differential workset exceeds the {family_limit}-family resource bound."
        )
    population_size = sum(item.family.population_size for item in worksets)
    if (
        plan.population == "terms"
        and plan.dimension_policy is not None
        and plan.dimension_policy.total_dimension != population_size
    ):
        rejected.append(
            f"root: dimension policy size {plan.dimension_policy.total_dimension} does not match term population size {population_size}."
        )
    if plan.population == "terms" and not rejected:
        try:
            _population_terms(expression, worksets)
        except ValueError as error:
            rejected.append(str(error))
    return worksets, population_size


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
    expression = normalize_linear_expression(source.residual)
    value_type = infer_expression_type(source.residual, problem)
    randomized, exact, methods, reasons = _analyze_expression(
        expression,
        problem,
        plan,
    )
    rejected = list(reasons)
    worksets, population_size = _admit_worksets(
        expression, problem, plan, randomized, rejected
    )
    law, _ = _sampling_law(plan)
    if not value_type.is_scalar:
        rejected.append(
            "root: randomized residual objectives currently require a scalar equation."
        )
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
        execution_backend=plan.backend,
        population=plan.population,
        population_size=population_size
        if plan.population == "terms"
        else (
            plan.dimension_policy.total_dimension
            if plan.dimension_policy is not None
            else None
        ),
        term_identities=tuple(
            (item.path, item.family.identity, item.family.population_size)
            for item in worksets
        ),
        contraction_certificates=tuple(
            (
                item.path,
                item.plan.plan_id if item.plan is not None else "exact-directional-ad",
            )
            for item in worksets
        ),
        contraction_plans=tuple(
            (item.path, item.plan) for item in worksets if item.plan is not None
        ),
        native_contribution_paths=tuple(
            (item.path, item.native_paths) for item in worksets
        ),
        sampling_design=law,
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


def _coordinate_functions(
    domain: Any, problem: PDEProblemIR, /
) -> dict[str, DomainFunction]:
    from phydrax.domain import DomainFunction

    coordinates: dict[str, DomainFunction] = {}
    for coordinate in problem.coordinates:
        name = coordinate.name

        def identity(
            value: object,
            *,
            key: PRNGKey | None = None,
            iter: object = None,
        ) -> object:
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
    key: PRNGKey,
    /,
) -> Array:
    from phydrax.domain import DomainFunction

    if not isinstance(value, DomainFunction):
        return jnp.asarray(value)
    positions = {label: index for index, label in enumerate(labels)}
    local_args = tuple(args[positions[label]] for label in value.deps)
    return jnp.asarray(value.func(*local_args, key=key))


def _realization_aligned(
    values: tuple[Array, ...],
    randomized: tuple[bool, ...],
    /,
) -> tuple[Array, ...]:
    """Give every operand a leading realization axis and a common event rank.

    Randomized operands carry realizations on axis 0 and deterministic operands
    carry only their event shape. Event shapes broadcast right-aligned, so
    singleton axes are inserted between the realization axis and each event
    shape; deterministic operands get a singleton realization axis.
    """
    events = tuple(
        value.shape[1:] if is_randomized else value.shape
        for value, is_randomized in zip(values, randomized, strict=True)
    )
    rank = max(len(event) for event in events)
    return tuple(
        value.reshape(
            (value.shape[:1] if is_randomized else (1,))
            + (1,) * (rank - len(event))
            + event
        )
        for value, is_randomized, event in zip(values, randomized, events, strict=True)
    )


@final
class _RandomizedPointCallable(StrictModule):
    fields: Mapping[str, DomainFunction]
    parameters: Mapping[str, Any]
    plan: RandomizedDifferentialPlan
    problem: PDEProblemIR = eqx.field(static=True)
    expression: PDEExpression = eqx.field(static=True)
    randomized_paths: tuple[str, ...] = eqx.field(static=True)
    node_indices: tuple[tuple[str, int], ...] = eqx.field(static=True)
    labels: tuple[str, ...] = eqx.field(static=True)
    worksets: tuple[_DifferentialWorkset, ...] = eqx.field(static=True)
    population_terms: tuple[_PopulationTerm, ...] = eqx.field(static=True)
    operands: Mapping[str, DomainFunction | Array]
    owned: Mapping[str, DomainFunction]
    exact_only: bool = eqx.field(static=True)

    def _node_key(self, key: PRNGKey, path: str, /) -> PRNGKey:
        if self.plan.node_coupling == "common":
            return key
        index = dict(self.node_indices)[path]
        return jr.fold_in(key, index)

    def _exact(
        self,
        node: PDEExpression,
        args: tuple[Any, ...],
        key: PRNGKey,
        /,
    ) -> Array:
        coordinates = _coordinate_functions(
            next(iter(self.fields.values())).domain, self.problem
        )
        compiled = compile_pde_expression(
            node,
            self.problem,
            fields=self.fields,
            parameters=self.parameters,
            coordinates=coordinates,
            differential_backend="ad",
        )
        return _evaluate_domain_value(compiled, self.labels, args, key)

    def _contribution(
        self,
        workset: _DifferentialWorkset,
        index: Array,
        args: tuple[Any, ...],
        key: PRNGKey,
        /,
        *,
        probe: Array | None = None,
    ) -> Array:
        from ..domain._evaluation import PointwiseEvaluator

        if workset.path in self.owned:
            owned = self.owned[workset.path]
            function = owned.func
            if isinstance(function, PointwiseEvaluator):
                function = function.function
            if (
                isinstance(function, NativeFamilyCallable)
                and function.num_contributions == workset.family.population_size
            ):
                return function.contribution(
                    index, tuple(jnp.asarray(value) for value in args), key
                )
            return _evaluate_domain_value(owned, self.labels, args, key)
        primals = tuple(jnp.asarray(value) for value in args)
        operand = self.operands[workset.path]
        components = infer_expression_type(
            workset.family.operand, self.problem
        ).components

        def function(*values: Array) -> Array:
            value = _evaluate_domain_value(operand, self.labels, tuple(values), key)
            if value.size != components:
                raise ValueError(
                    "The differential operand event shape conflicts with its PDE IR component type."
                )
            return value

        return family_contribution(
            function,
            primals,
            self.labels,
            workset.family,
            index,
            backend=self.plan.backend,
            plan=workset.plan,
            probe=probe,
            gaussian=self.plan.method == "gaussian_bilaplacian",
        )

    def _random_operator(
        self,
        node: PDEExpression,
        path: str,
        args: tuple[Any, ...],
        key: PRNGKey,
        /,
    ) -> Array:
        if path in self.owned:
            value = _evaluate_domain_value(self.owned[path], self.labels, args, key)
            count = 1 if self.exact_only else self.plan.num_realizations
            return jnp.broadcast_to(value, (count, *value.shape))
        workset = next(item for item in self.worksets if item.path == path)
        if (
            self.plan.backend == "ad"
            and len(workset.family.directions) == 1
            and node.op in ("laplacian", "divergence")
            and self.plan.method != "gaussian_bilaplacian"
            and jnp.asarray(
                args[self.labels.index(workset.family.directions[0].coordinate)]
            ).ndim
            > 0
        ):
            return self._legacy_random_operator(node, path, args, key)
        node_key = self._node_key(key, path)
        match self.plan.method:
            case "dimension":
                return self._dimension_operator(workset, args, key, node_key)
            case "hutchinson" | "gaussian_bilaplacian":
                return self._probe_operator(workset, args, key, node_key)
            case _:
                assert_never(self.plan.method)

    def _dimension_operator(
        self,
        workset: _DifferentialWorkset,
        args: tuple[Any, ...],
        key: PRNGKey,
        node_key: PRNGKey,
        /,
    ) -> Array:
        from ..operators.differential._dimension_estimators import dimension_sum_samples

        if self.plan.dimension_policy is None:
            raise RuntimeError("Dimension policy is unavailable.")

        def contribution(index: Array) -> Array:
            return self._contribution(workset, index, args, key)

        return dimension_sum_samples(
            contribution, node_key, self.plan.dimension_policy
        ).values

    def _probe_operator(
        self,
        workset: _DifferentialWorkset,
        args: tuple[Any, ...],
        key: PRNGKey,
        node_key: PRNGKey,
        /,
    ) -> Array:
        from ..operators.differential._stochastic_estimators import _probes

        if self.plan.trace_policy is None:
            raise RuntimeError("Probe policy is unavailable.")
        summed = next(item for item in workset.family.directions if item.axis is None)
        state = jnp.asarray(args[self.labels.index(summed.coordinate)])
        if self.plan.method == "gaussian_bilaplacian":
            return self._gaussian_operator(
                workset, args, key, node_key, state, self.plan.trace_policy
            )
        probes = _probes(node_key, state.shape, state.dtype, self.plan.trace_policy)

        def contribution(probe: Array) -> Array:
            return self._contribution(
                workset, jnp.asarray(0, dtype=jnp.int32), args, key, probe=probe
            )

        return jax.vmap(contribution)(probes)

    def _gaussian_operator(
        self,
        workset: _DifferentialWorkset,
        args: tuple[Any, ...],
        key: PRNGKey,
        node_key: PRNGKey,
        state: Array,
        policy: StochasticTracePolicy,
        /,
    ) -> Array:
        if not jnp.issubdtype(state.dtype, jnp.floating):
            raise ValueError(
                "Gaussian bilaplacian requires real floating coordinate values."
            )
        count = policy.num_probes
        if count > jnp.iinfo(jnp.int32).max:
            raise ValueError(
                "Gaussian probe count exceeds native int32 addressing capacity."
            )
        if workset.plan is not None:
            components = (
                1
                if workset.family.component is not None
                else infer_expression_type(
                    workset.family.operand, self.problem
                ).components
            )
            retained = (count + 3) * components + count
            if retained > workset.plan.policy.resources.max_logical_buffer_elements:
                raise ValueError(
                    "Gaussian bilaplacian realizations exceed retained-buffer resources."
                )
        probe_root = derive_key(
            node_key, SampleAddress("differential", "bilaplacian", role="probe")
        )

        def one(index: Array) -> Array:
            probe = jr.normal(
                jr.fold_in(probe_root, index), state.shape, dtype=state.dtype
            )
            return self._contribution(
                workset, jnp.asarray(0, dtype=jnp.int32), args, key, probe=probe
            )

        return jax.lax.map(jax.checkpoint(one), jnp.arange(count, dtype=jnp.int32))

    def _apply_contexts(
        self,
        value: Array,
        contexts: tuple[tuple[PDEExpression, int], ...],
        args: tuple[Any, ...],
        key: PRNGKey,
        /,
    ) -> Array:
        for node, selected in contexts:
            match node.op:
                case "negate":
                    value = -value
                case "component":
                    if node.axis is None:
                        raise RuntimeError("A component lost its scientific axis.")
                    value = value[..., node.axis]
                case "multiply":
                    for index, argument in enumerate(node.args):
                        if index != selected:
                            value = value * self._exact(argument, args, key)
                case "divide":
                    value = value / self._exact(node.args[1], args, key)
                case "dot":
                    value = jnp.sum(
                        value * self._exact(node.args[1 - selected], args, key), axis=-1
                    )
                case _:
                    raise RuntimeError("Invalid linear population context.")
        return value

    def _population_sample(self, args: tuple[Any, ...], key: PRNGKey, /) -> Array:
        from ..operators.differential._dimension_estimators import dimension_sum_samples

        policy = self.plan.dimension_policy
        if policy is None:
            raise RuntimeError("Term population policy is unavailable.")
        stochastic = tuple(
            term for term in self.population_terms if term.workset is not None
        )
        exact = tuple(term for term in self.population_terms if term.exact is not None)
        starts: list[int] = []
        end = 0
        for term in stochastic:
            starts.append(end)
            if term.workset is None:
                raise RuntimeError("A population term lost its family.")
            end += term.workset.family.population_size
        # Branches execute only their own signature. No padded realizations or pair tables.
        prototypes = tuple(
            self._exact(item.family.operand, args, key) for item in self.worksets
        )
        coefficients = tuple(
            self._exact(argument, args, key)
            for term in stochastic
            for node, selected in term.contexts
            for index, argument in enumerate(node.args)
            if index != selected
        )
        dtype = jnp.result_type(*prototypes, *coefficients)
        branches: list[Callable[[Array], Array]] = []
        for term, start in zip(stochastic, starts, strict=True):

            def branch(
                index: Array, term: _PopulationTerm = term, start: int = start
            ) -> Array:
                if term.workset is None:
                    raise RuntimeError("A population term lost its family.")
                value = self._contribution(term.workset, index - start, args, key)
                return jnp.asarray(
                    self._apply_contexts(value, term.contexts, args, key), dtype=dtype
                )

            branches.append(branch)

        def contribution(index: Array) -> Array:
            branch_index = jnp.asarray(0, dtype=jnp.int32)
            for start in starts[1:]:
                branch_index = branch_index + (index >= start).astype(jnp.int32)
            return jax.lax.switch(branch_index, tuple(branches), index)

        values = dimension_sum_samples(
            contribution, key, policy, evaluation="sequential"
        ).values
        for term in exact:
            if term.exact is None:
                raise RuntimeError("An exact population offset lost its expression.")
            values = values + self._apply_contexts(
                self._exact(term.exact, args, key), term.contexts, args, key
            )
        return values

    def _legacy_random_operator(
        self,
        node: PDEExpression,
        path: str,
        args: tuple[Any, ...],
        key: PRNGKey,
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
        coordinates = _coordinate_functions(
            next(iter(self.fields.values())).domain, self.problem
        )
        operand = compile_pde_expression(
            node.args[0],
            self.problem,
            fields=self.fields,
            parameters=self.parameters,
            coordinates=coordinates,
            differential_backend="ad",
        )
        if not isinstance(operand, DomainFunction):
            raise TypeError(
                f"{path}: randomized differential operand must be a DomainFunction."
            )
        operand_positions = {label: index for index, label in enumerate(self.labels)}
        local_args = [args[operand_positions[label]] for label in operand.deps]
        if node.coordinate not in operand.deps:
            return jnp.zeros((self.plan.num_realizations,))
        local_position = operand.deps.index(node.coordinate)
        node_key = self._node_key(key, path)

        def evaluate(local_state: Array) -> Array:
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
        key: PRNGKey,
        /,
    ) -> tuple[Array, bool]:
        randomized_path_set = frozenset(self.randomized_paths)
        if path in randomized_path_set:
            return self._random_operator(node, path, args, key), True
        descendants = tuple(
            item == path or item.startswith(path + ".") for item in self.randomized_paths
        )
        if not any(descendants):
            return self._exact(node, args, key), False
        evaluated = tuple(
            self._evaluate(argument, f"{path}.args[{index}]", args, key)
            for index, argument in enumerate(node.args)
        )
        values = _realization_aligned(
            tuple(item[0] for item in evaluated),
            tuple(item[1] for item in evaluated),
        )
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
            return jnp.sum(left * right, axis=-1), True
        raise RuntimeError(
            f"Unsupported randomized expression node {node.op!r} at {path}."
        )

    def __call__(
        self,
        *args: Any,
        key: PRNGKey | None = None,
        iter: object = None,
        **kwargs: Any,
    ) -> Array:
        del iter, kwargs
        resolved_key = jr.key(0) if key is None else key
        if self.plan.population == "terms" and not self.exact_only:
            return self._population_sample(tuple(args), resolved_key)
        result, randomized = self._evaluate(
            self.expression,
            "root",
            tuple(args),
            resolved_key,
        )
        if not randomized:
            raise RuntimeError(
                "Randomized expression unexpectedly lowered to an exact value."
            )
        return result


@final
class _RandomizedPDEEvaluator(StrictModule):
    parameters: Mapping[str, Any]
    plan: RandomizedDifferentialPlan
    problem: PDEProblemIR = eqx.field(static=True)
    expression: PDEExpression = eqx.field(static=True)
    randomized_paths: tuple[str, ...] = eqx.field(static=True)
    worksets: tuple[_DifferentialWorkset, ...] = eqx.field(static=True)
    population_terms: tuple[_PopulationTerm, ...] = eqx.field(static=True)

    def _prepare_operands(
        self,
        domain: Any,
        fields: Mapping[str, DomainFunction],
        /,
    ) -> tuple[dict[str, DomainFunction | Array], dict[str, DomainFunction]]:
        from phydrax.domain import DomainFunction

        from ..operators.differential._requests import admit_direct_derivative

        coordinates = _coordinate_functions(domain, self.problem)
        operands: dict[str, DomainFunction | Array] = {}
        owned: dict[str, DomainFunction] = {}
        for workset in self.worksets:
            operand = compile_pde_expression(
                workset.family.operand,
                self.problem,
                fields=fields,
                parameters=self.parameters,
                coordinates=coordinates,
                differential_backend="ad",
            )
            operands[workset.path] = (
                operand if isinstance(operand, DomainFunction) else jnp.asarray(operand)
            )
            if isinstance(operand, DomainFunction):
                native = try_owned_family(
                    workset.family,
                    operand,
                    backend=self.plan.backend,
                    native_paths=workset.native_paths,
                )
                if native is not None:
                    owned[workset.path] = native
                else:
                    order = (
                        workset.plan.required_regularity_order
                        if workset.plan is not None
                        else workset.family.order
                    )
                    for coordinate in dict.fromkeys(
                        item.coordinate for item in workset.family.directions
                    ):
                        admit_direct_derivative(operand, coordinate, order)
        return operands, owned

    def _admit_owned_population(
        self,
        owned: Mapping[str, DomainFunction],
        exact_only: bool,
        /,
    ) -> None:
        from ..domain._evaluation import PointwiseEvaluator

        if not (self.plan.population == "terms" and owned and not exact_only):
            return
        unavailable: list[str] = []
        for item in self.worksets:
            if item.path not in owned or item.family.population_size == 1:
                continue
            function = owned[item.path].func
            if isinstance(function, PointwiseEvaluator):
                function = function.function
            if not (
                isinstance(function, NativeFamilyCallable)
                and function.num_contributions == item.family.population_size
            ):
                unavailable.append(item.path)
        if unavailable:
            raise ValueError(
                "Native derivative ownership provides only full-family sums at "
                f"{unavailable!r}; a mixed term population requires actual per-term "
                "contributions, not redistributed sums or padded zeros."
            )

    def _evaluate_realizations(
        self,
        domain: Any,
        fields: Mapping[str, DomainFunction],
        operands: Mapping[str, DomainFunction | Array],
        owned: Mapping[str, DomainFunction],
        exact_only: bool,
        collocation: Any,
        key: PRNGKey,
        /,
    ) -> cx.AxisArray:
        from phydrax.domain import DomainFunction

        from ..domain._evaluation import FunctionBinding, PointwiseEvaluator

        node_indices = tuple(
            (path, index) for index, path in enumerate(self.randomized_paths)
        )
        residual = DomainFunction(
            domain=domain,
            deps=domain.labels,
            func=PointwiseEvaluator(
                _RandomizedPointCallable(
                    fields=fields,
                    parameters=self.parameters,
                    plan=self.plan,
                    problem=self.problem,
                    expression=self.expression,
                    randomized_paths=self.randomized_paths,
                    node_indices=node_indices,
                    labels=domain.labels,
                    worksets=self.worksets,
                    population_terms=self.population_terms,
                    operands=operands,
                    owned=owned,
                    exact_only=exact_only,
                ),
                binding=FunctionBinding(pass_key=True),
            ),
        )
        return residual(collocation, key=key)

    def _realization_layout(
        self,
        evaluated: cx.AxisArray,
        exact_only: bool,
        /,
    ) -> tuple[Array, tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        named_positions = tuple(
            index for index, dim in enumerate(evaluated.dims) if dim is not None
        )
        output_positions = tuple(
            index for index, dim in enumerate(evaluated.dims) if dim is None
        )
        if len(output_positions) < 1:
            raise ValueError(
                "Randomized residual output is missing its realization axis."
            )
        permutation = named_positions + output_positions
        data = jnp.transpose(jnp.asarray(evaluated.data), permutation)
        sample_shape = tuple(data.shape[index] for index in range(len(named_positions)))
        if data.shape[len(sample_shape)] != (
            1 if exact_only else self.plan.num_realizations
        ):
            raise ValueError(
                "Randomized residual realization count does not match its plan."
            )
        values = jnp.moveaxis(data, len(sample_shape), 0)
        event_shape = tuple(values.shape[1 + len(sample_shape) :])
        if prod(event_shape) != 1:
            raise ValueError(
                "A scalar PDE residual cannot acquire an incompatible event shape from its runtime bindings."
            )
        return values, sample_shape, event_shape, named_positions

    @staticmethod
    def _collocation_measure(
        collocation: Any,
        evaluated: cx.AxisArray,
        sample_shape: tuple[int, ...],
        named_positions: tuple[int, ...],
        /,
    ) -> tuple[Array, Array]:
        from phydrax.domain import GridBatch

        mask = jnp.ones(sample_shape, dtype=jnp.bool_)
        weights = jnp.ones(sample_shape, dtype=jnp.float64)
        if isinstance(collocation, GridBatch):
            named_dims = tuple(evaluated.dims[index] for index in named_positions)
            mask_field = cx.AxisArray(mask, dims=named_dims)
            weight_field = cx.AxisArray(weights, dims=named_dims)
            for current in collocation.coord_mask_by_label.values():
                mask_field = mask_field * current
            for current in collocation.coord_geometry_weight_by_label.values():
                weight_field = weight_field * current
            mask = jnp.asarray(mask_field.data, dtype=jnp.bool_)
            weights = jnp.asarray(weight_field.data, dtype=jnp.float64)
        return mask, weights

    def __call__(
        self,
        functions: Mapping[str, DomainFunction],
        collocation: Any,
        key: PRNGKey,
        /,
    ) -> RandomizedResidualSamples:
        from phydrax.domain import GridBatch, PointBatch

        from ..terms._randomized_residual import RandomizedResidualSamples

        if isinstance(collocation, tuple):
            raise TypeError(
                "Randomized PDE objectives do not support ComponentSum batches."
            )
        if not isinstance(collocation, (PointBatch, GridBatch)):
            raise TypeError(
                "Randomized PDE collocation must be a structured point batch."
            )
        domain, fields = _promoted_fields(functions, self.problem)
        operands, owned = self._prepare_operands(domain, fields)
        exact_only = len(owned) == len(self.worksets)
        self._admit_owned_population(owned, exact_only)
        evaluated = self._evaluate_realizations(
            domain, fields, operands, owned, exact_only, collocation, key
        )
        values, sample_shape, event_shape, named_positions = self._realization_layout(
            evaluated, exact_only
        )
        mask, weights = self._collocation_measure(
            collocation, evaluated, sample_shape, named_positions
        )
        law, population_size = _sampling_law(self.plan)
        if exact_only:
            law, population_size = "exact", 1
        return RandomizedResidualSamples(
            values,
            sample_shape=sample_shape,
            event_shape=event_shape,
            mask=mask,
            weights=weights,
            estimator_id=f"pde-{self.plan.plan_id}",
            sampling_design=law,
            population_size=population_size,
        )


@final
class _RandomizedCollocationSampler(StrictModule):
    component: DomainComponent
    sampling: SamplingPlan

    def __call__(self, key: PRNGKey, /) -> PointBatch | GridBatch | GraphBatch:
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
    sampling_mode: RandomizedResidualSamplingMode = "resample",
    fixed_batch: PointBatch | GridBatch | None = None,
    fixed_batch_key: PRNGKey = jr.key(0),
) -> CompiledRandomizedPDETerm:
    """Compile one scalar IR equation to an estimator-aware sampled term."""
    from phydrax.domain import ComponentSum, DomainComponent

    from ..terms._randomized_residual import (
        RandomizedResidualSamplingMode,
        RandomizedResidualTerm,
    )

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
    mode = parse(sampling_mode, RandomizedResidualSamplingMode, "sampling_mode")
    match mode:
        case "fixed":
            collocation = (
                collocation_sampler(fixed_batch_key)
                if fixed_batch is None
                else fixed_batch
            )
        case "resample":
            if fixed_batch is not None:
                raise ValueError("fixed_batch is only valid with sampling_mode='fixed'.")
            collocation = collocation_sampler
        case _:
            assert_never(mode)
    expression = normalize_linear_expression(source.residual)
    worksets = _worksets(
        expression,
        problem,
        plan,
        report.randomized_node_paths,
        prepared_plans=dict(report.contraction_plans),
        prepared_native_paths=dict(report.native_contribution_paths),
    )
    evaluator = _RandomizedPDEEvaluator(
        parameters={} if parameters is None else dict(parameters),
        plan=plan,
        problem=problem,
        expression=expression,
        randomized_paths=report.randomized_node_paths,
        worksets=worksets,
        population_terms=_population_terms(expression, worksets)
        if plan.population == "terms"
        else (),
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
    "RandomizedExecutionBackend",
    "RandomizedPopulation",
]
