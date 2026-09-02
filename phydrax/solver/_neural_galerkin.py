#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from math import isclose, isfinite
from typing import Any, Literal, NamedTuple, TypeAlias

import coordax as cx
import diffrax as dfx
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jax import core as jax_core
from jaxtyping import Array, ArrayLike, Key

from phydrax.domain import ComponentSum, DomainFunction

from .._doc import DOC_KEY0
from .._fingerprint import canonical_fingerprint
from .._frozendict import frozendict
from .._strict import StrictModule
from .._trainable import partition_trainable
from ..dynamics import TimeGrid
from ..enforcement import EnforcementProgram
from ..integration import (
    ComponentTarget,
    DensityTarget,
    IntegrationRealization,
    PointIntegrationBatch,
    SeparableIntegrationBatch,
)
from ..integration._fixed import _target_reduction_weights
from ..linalg import (
    AbstractPreconditionerBuilder,
    ArraySpace,
    DifferentiationPolicy,
    FunctionLinearOperator,
    GeneralizedLSMR,
    IdentityLinearOperator,
    JacobianLinearOperator,
    LeastSquaresProblem,
    LinearSolvePolicy,
    LinearSystem,
    OperatorProperties,
    PCG,
    PreconditioningPolicy,
    prepare_linearization,
    RandomizedNystromPreconditionerBuilder,
    ScaledLinearOperator,
    solve,
)
from ..nn.parameters import ParameterSubspace
from ._differential import DifferentialProblem, DifferentialSolution
from ._diffrax_backend import solve_diffrax
from ._hybrid_event import HybridReplayPolicy
from ._temporal_precision import TemporalPrecisionPolicy


TangentFormulation: TypeAlias = Literal["rectangular", "gram"]
RateFunction: TypeAlias = Callable[
    [Array, Mapping[str, DomainFunction], Any], Mapping[str, DomainFunction]
]


def _norm(value: Array, /) -> Array:
    array = jnp.asarray(value)
    return jnp.sqrt(jnp.maximum(jnp.real(jnp.vdot(array, array)), 0.0))


def _require_matching_domain_support(
    rate_function: DomainFunction,
    current_function: DomainFunction,
    /,
) -> None:
    rate_domain = rate_function.domain
    current_domain = current_function.domain
    if rate_domain.labels != current_domain.labels:
        raise ValueError("Neural Galerkin rate and field domains must agree.")
    leaves = jax.tree_util.tree_leaves((rate_domain, current_domain))
    if any(isinstance(leaf, jax_core.Tracer) for leaf in leaves):
        return
    if not rate_domain.same_support(current_domain):
        raise ValueError("Neural Galerkin rate and field domains must agree.")


def _qualified_type_name(value: Any, /) -> str:
    cls = type(value)
    return f"{cls.__module__}.{cls.__qualname__}"


def _default_parameter_subspace(functions: frozendict[str, DomainFunction]):
    trainable, _ = partition_trainable(functions)
    paths = tuple(
        jax.tree_util.keystr(path)
        for path, leaf in jax.tree_util.tree_flatten_with_path(trainable)[0]
        if eqx.is_inexact_array(leaf)
    )
    if not paths:
        raise ValueError("Neural Galerkin evolution requires trainable function leaves.")
    return ParameterSubspace.from_leaf_paths(functions, paths)


def _copy_linear_policy(
    policy: LinearSolvePolicy,
    /,
    *,
    preconditioning: PreconditioningPolicy | None,
) -> LinearSolvePolicy:
    return LinearSolvePolicy(
        policy.method,
        tolerance=policy.tolerance,
        rank=policy.rank,
        materialization=policy.materialization,
        preconditioning=preconditioning,
        recycling=policy.recycling,
        differentiation=policy.differentiation,
        failure=policy.failure,
        resources=policy.resources,
        precision=policy.precision,
        require_device_binding=policy.require_device_binding,
    )


class FieldProjectionMetric(StrictModule):
    """One named field norm realized on a fixed physical measure."""

    realization: IntegrationRealization
    scale: Array
    field: str = eqx.field(static=True)
    label: str = eqx.field(static=True)

    def __init__(
        self,
        field: str,
        realization: IntegrationRealization,
        /,
        *,
        scale: ArrayLike = 1.0,
        label: str | None = None,
    ):
        name = str(field)
        if not name:
            raise ValueError("field must be non-empty.")
        if not isinstance(realization, IntegrationRealization):
            raise TypeError("realization must be an IntegrationRealization.")
        if not isinstance(realization.target, (ComponentTarget, DensityTarget)):
            raise TypeError(
                "Field projection requires a component or density integration target."
            )
        batch = realization.batch
        if isinstance(batch, tuple):
            if not batch or any(
                not isinstance(item, (PointIntegrationBatch, SeparableIntegrationBatch))
                for item in batch
            ):
                raise TypeError(
                    "Field projection component sums require fixed point or separable batches."
                )
        elif not isinstance(batch, (PointIntegrationBatch, SeparableIntegrationBatch)):
            raise TypeError(
                "Field projection requires a fixed point or separable integration batch."
            )
        scale_ = jnp.asarray(scale, dtype=float)
        if scale_.shape != () or not bool(jnp.isfinite(scale_)) or float(scale_) <= 0.0:
            raise ValueError("scale must be a finite strictly positive scalar.")
        resolved_label = name if label is None else str(label)
        if not resolved_label:
            raise ValueError("label must be non-empty.")
        self.field = name
        self.realization = realization
        self.scale = scale_.reshape(())
        self.label = resolved_label


class NeuralTangentSolvePolicy(StrictModule):
    """Rectangular or Gram formulation for one neural tangent projection."""

    linear_policy: LinearSolvePolicy
    preconditioner: AbstractPreconditionerBuilder | None
    damping: float = eqx.field(static=True)
    maximum_relative_defect: float | None = eqx.field(static=True)
    formulation: TangentFormulation = eqx.field(static=True)

    def __init__(
        self,
        formulation: TangentFormulation = "rectangular",
        /,
        *,
        linear_policy: LinearSolvePolicy | None = None,
        damping: float = 1e-6,
        maximum_relative_defect: float | None = None,
        preconditioner: AbstractPreconditionerBuilder | None = None,
    ):
        if formulation not in ("rectangular", "gram"):
            raise ValueError("formulation must be 'rectangular' or 'gram'.")
        damping_ = float(damping)
        if not isfinite(damping_) or damping_ < 0.0:
            raise ValueError("damping must be finite and non-negative.")
        if formulation == "gram" and damping_ <= 0.0:
            raise ValueError(
                "Gram tangent projection requires strictly positive damping."
            )
        if maximum_relative_defect is None:
            defect_limit = None
        else:
            defect_limit = float(maximum_relative_defect)
            if not isfinite(defect_limit) or defect_limit < 0.0:
                raise ValueError(
                    "maximum_relative_defect must be finite and non-negative or None."
                )
        if preconditioner is not None and not isinstance(
            preconditioner, AbstractPreconditionerBuilder
        ):
            raise TypeError(
                "preconditioner must be an AbstractPreconditionerBuilder or None."
            )
        if formulation == "rectangular" and preconditioner is not None:
            raise ValueError(
                "Rectangular tangent projection does not accept a preconditioner."
            )
        if isinstance(
            preconditioner, RandomizedNystromPreconditionerBuilder
        ) and not isclose(
            preconditioner.shift,
            damping_,
            rel_tol=32.0 * jnp.finfo(float).eps,
            abs_tol=32.0 * jnp.finfo(float).eps,
        ):
            raise ValueError(
                "Randomized Nyström shift must equal neural tangent damping."
            )
        if linear_policy is None:
            method = GeneralizedLSMR() if formulation == "rectangular" else PCG()
            linear_policy = LinearSolvePolicy(
                method,
                differentiation=DifferentiationPolicy("mathematical"),
            )
        if not isinstance(linear_policy, LinearSolvePolicy):
            raise TypeError("linear_policy must be a LinearSolvePolicy or None.")
        if linear_policy.preconditioning is not None and preconditioner is not None:
            raise ValueError(
                "Supply neural tangent preconditioning through preconditioner, not both policies."
            )
        self.formulation = formulation
        self.linear_policy = linear_policy
        self.preconditioner = preconditioner
        self.damping = damping_
        self.maximum_relative_defect = defect_limit


class NeuralGalerkinAdjointPolicy(StrictModule):
    """Explicit recursive-checkpoint or certified implicit-backsolve contract."""

    mode: Literal["recursive_checkpoint", "certified_backsolve"] = eqx.field(static=True)
    maximum_primal_residual: float = eqx.field(static=True)
    maximum_adjoint_residual: float = eqx.field(static=True)

    def __init__(
        self,
        mode: Literal[
            "recursive_checkpoint", "certified_backsolve"
        ] = "recursive_checkpoint",
        *,
        maximum_primal_residual: float = 1.0e-6,
        maximum_adjoint_residual: float = 1.0e-6,
    ):
        if mode not in ("recursive_checkpoint", "certified_backsolve"):
            raise ValueError("Unknown neural Galerkin adjoint policy.")
        primal = float(maximum_primal_residual)
        adjoint = float(maximum_adjoint_residual)
        if min(primal, adjoint) <= 0.0 or not all(
            isfinite(value) for value in (primal, adjoint)
        ):
            raise ValueError("Adjoint residual tolerances must be finite and positive.")
        self.mode = mode
        self.maximum_primal_residual = primal
        self.maximum_adjoint_residual = adjoint


class NeuralGalerkinEpoch(StrictModule):
    """One fixed-population neural Galerkin execution epoch."""

    problem: NeuralGalerkinProblem
    grid: TimeGrid
    population_id: str = eqx.field(static=True)

    def __init__(
        self, problem: NeuralGalerkinProblem, grid: TimeGrid, /, *, population_id: str
    ):
        if not isinstance(problem, NeuralGalerkinProblem) or not isinstance(
            grid, TimeGrid
        ):
            raise TypeError("Epoch requires NeuralGalerkinProblem and TimeGrid.")
        if not population_id:
            raise ValueError("population_id must be nonempty.")
        self.problem = problem
        self.grid = grid
        self.population_id = str(population_id)


class NeuralGalerkinEpochPlan(StrictModule):
    """Ordered fixed epochs with frozen population identities for replay."""

    epochs: tuple[NeuralGalerkinEpoch, ...]
    replay_id: str = eqx.field(static=True)

    def __init__(self, epochs: Sequence[NeuralGalerkinEpoch], /):
        values = tuple(epochs)
        if not values or any(
            not isinstance(value, NeuralGalerkinEpoch) for value in values
        ):
            raise TypeError("epochs must contain NeuralGalerkinEpoch values.")
        paths = values[0].problem.parameter_subspace.leaf_paths
        for left, right in zip(values, values[1:], strict=False):
            if not bool(jnp.isclose(left.grid.t1, right.grid.t0)):
                raise ValueError("Neural Galerkin epochs must be contiguous.")
            if right.problem.parameter_subspace.leaf_paths != paths:
                raise ValueError("Neural Galerkin epoch parameter subspaces must match.")
        self.epochs = values
        self.replay_id = canonical_fingerprint(
            {
                "kind": "neural-galerkin-epochs",
                "epochs": tuple(
                    (value.population_id, value.grid.time_id) for value in values
                ),
                "parameter_paths": paths,
            }
        )


class NeuralGalerkinReplayJournal(StrictModule):
    """Frozen DCD-capacity population sequence and segment-boundary states."""

    boundary_parameters: Array
    population_ids: tuple[str, ...] = eqx.field(static=True)
    replay_id: str = eqx.field(static=True)
    replay_policy_id: str = eqx.field(static=True)
    selection_stopped: bool = eqx.field(static=True, default=True)


class NeuralGalerkinEpochResult(StrictModule):
    segments: tuple[NeuralFieldEvolutionResult, ...]
    population_ids: tuple[str, ...] = eqx.field(static=True)
    replay_id: str = eqx.field(static=True)
    replay_journal: NeuralGalerkinReplayJournal

    @property
    def successful(self) -> Array:
        return jnp.all(jnp.stack(tuple(segment.successful for segment in self.segments)))


class NeuralGalerkinProblem(StrictModule):
    """Named model fields and one deterministic tangent-projected evolution law."""

    functions: frozendict[str, DomainFunction]
    rate: RateFunction
    metrics: tuple[FieldProjectionMetric, ...]
    parameter_subspace: ParameterSubspace
    enforcement: EnforcementProgram | None
    args: Any
    evaluation_key: Array
    problem_id: str = eqx.field(static=True)

    def __init__(
        self,
        functions: Mapping[str, DomainFunction],
        rate: RateFunction,
        metrics: Sequence[FieldProjectionMetric],
        /,
        *,
        parameter_subspace: ParameterSubspace | None = None,
        enforcement: EnforcementProgram | None = None,
        args: Any = None,
        evaluation_key: Key[Array, ""] = DOC_KEY0,
        problem_id: str | None = None,
    ):
        fields = frozendict(functions)
        if not fields or any(
            not isinstance(value, DomainFunction) for value in fields.values()
        ):
            raise TypeError(
                "functions must be a non-empty mapping of DomainFunction values."
            )
        if not callable(rate):
            raise TypeError("rate must be callable.")
        metric_values = tuple(metrics)
        if not metric_values or any(
            not isinstance(metric, FieldProjectionMetric) for metric in metric_values
        ):
            raise TypeError("metrics must contain at least one FieldProjectionMetric.")
        names = tuple(metric.field for metric in metric_values)
        if len(set(names)) != len(names):
            raise ValueError("Each evolved field may have exactly one projection metric.")
        missing = tuple(name for name in names if name not in fields)
        if missing:
            raise KeyError(f"Projection metrics reference missing fields {missing!r}.")
        subspace = (
            _default_parameter_subspace(fields)
            if parameter_subspace is None
            else parameter_subspace
        )
        if not isinstance(subspace, ParameterSubspace):
            raise TypeError("parameter_subspace must be a ParameterSubspace or None.")
        subspace.validate_root(fields)
        if enforcement is not None and not isinstance(enforcement, EnforcementProgram):
            raise TypeError("enforcement must be an EnforcementProgram or None.")
        identifier = (
            canonical_fingerprint(
                {
                    "kind": "neural-galerkin-problem",
                    "fields": tuple(fields),
                    "metrics": tuple(
                        (metric.field, metric.label) for metric in metric_values
                    ),
                    "parameter_paths": subspace.leaf_paths,
                    "rate": _qualified_type_name(rate),
                }
            )
            if problem_id is None
            else str(problem_id)
        )
        if not identifier:
            raise ValueError("problem_id must be non-empty.")
        self.functions = fields
        self.rate = rate
        self.metrics = metric_values
        self.parameter_subspace = subspace
        self.enforcement = enforcement
        self.args = args
        self.evaluation_key = jnp.asarray(evaluation_key)
        self.problem_id = identifier

    @property
    def evolved_fields(self) -> tuple[str, ...]:
        return tuple(metric.field for metric in self.metrics)

    def ansatz(self, flat_parameters: Array, /) -> frozendict[str, DomainFunction]:
        functions = self.parameter_subspace.reconstruct_vector(flat_parameters)
        if self.enforcement is not None:
            functions = self.enforcement.apply(functions)
        return frozendict(functions)


class NeuralGalerkinAudit(StrictModule):
    """Saved-node tangent projection evidence."""

    status: Array
    iterations: Array
    matvec_count: Array
    adjoint_matvec_count: Array
    linear_residual_norm: Array
    rate_norm: Array
    parameter_rate_norm: Array
    projection_defect: Array
    relative_projection_defect: Array
    accepted: Array


class _TangentEvaluation(NamedTuple):
    rate: Array
    status: Array
    iterations: Array
    matvec_count: Array
    adjoint_matvec_count: Array
    linear_residual_norm: Array
    rate_norm: Array
    parameter_rate_norm: Array
    projection_defect: Array
    relative_projection_defect: Array
    accepted: Array


def _component_batches_and_keys(metric: FieldProjectionMetric):
    target = metric.realization.target
    base = target.base if isinstance(target, DensityTarget) else target
    if not isinstance(base, ComponentTarget):
        raise TypeError("Neural field metrics require component targets.")
    batch = metric.realization.batch
    key = problem_key(metric.realization.key)
    if isinstance(base.component, ComponentSum):
        if not isinstance(batch, tuple):
            raise TypeError("Component-sum metric realization must contain batch tuples.")
        return batch, tuple(jr.split(key, len(batch)))
    if isinstance(batch, tuple):
        raise TypeError(
            "Single-component metric realization cannot contain batch tuples."
        )
    return (batch,), (key,)


def problem_key(key: Any, /) -> Array:
    return DOC_KEY0 if key is None else jnp.asarray(key)


def _checked_coefficient(coefficient: cx.Field, /) -> cx.Field:
    if any(dim is None for dim in coefficient.dims):
        raise ValueError("Projection coefficients may not contain event axes.")
    data = jnp.asarray(coefficient.data)
    if jnp.iscomplexobj(data):
        raise TypeError("Projection coefficients must be real.")
    data = eqx.error_if(
        data.astype(float),
        jnp.any(~jnp.isfinite(data)) | jnp.any(data < 0.0),
        "Projection coefficients must be finite and non-negative.",
    )
    return cx.Field(data, dims=coefficient.dims)


def _metric_vector(
    field: DomainFunction,
    metric: FieldProjectionMetric,
    /,
) -> Array:
    realization = metric.realization
    weights = _target_reduction_weights(
        realization.target,
        realization.batch,
        key=problem_key(realization.key),
        kwargs={},
    )
    batches, keys = _component_batches_and_keys(metric)
    coefficients = weights if isinstance(weights, tuple) else (weights,)
    pieces = []
    for batch, coefficient, key in zip(batches, coefficients, keys, strict=True):
        value = field(batch.points, key=key)
        if not isinstance(value, cx.Field):
            raise TypeError("Projected fields must evaluate to coordax.Field.")
        checked = _checked_coefficient(coefficient)
        square_root = cx.Field(
            jnp.sqrt(jnp.asarray(metric.scale) * jnp.asarray(checked.data)),
            dims=checked.dims,
        )
        pieces.append(jnp.asarray((value * square_root).data).reshape((-1,)))
    return pieces[0] if len(pieces) == 1 else jnp.concatenate(tuple(pieces))


class _NeuralGalerkinVectorField(StrictModule):
    problem: NeuralGalerkinProblem
    policy: NeuralTangentSolvePolicy
    implicit_adjoint: bool = eqx.field(static=True, default=False)

    def _sampled_fields(self, parameters: Array, /) -> Array:
        functions = self.problem.ansatz(parameters)
        pieces = tuple(
            _metric_vector(functions[metric.field], metric)
            for metric in self.problem.metrics
        )
        return pieces[0] if len(pieces) == 1 else jnp.concatenate(pieces)

    def _target(self, time: Array, parameters: Array, /) -> Array:
        current = self.problem.ansatz(parameters)
        rates = frozendict(self.problem.rate(time, current, self.problem.args))
        if set(rates) != set(self.problem.evolved_fields):
            raise ValueError(
                "Neural Galerkin rate must return exactly the evolved field names."
            )
        for name in self.problem.evolved_fields:
            if not isinstance(rates[name], DomainFunction):
                raise TypeError("Neural Galerkin rates must be DomainFunction values.")
            _require_matching_domain_support(rates[name], current[name])
        targets = tuple(
            _metric_vector(rates[metric.field], metric) for metric in self.problem.metrics
        )
        return targets[0] if len(targets) == 1 else jnp.concatenate(targets)

    def evaluate(self, time: Array, parameters: Array, /) -> _TangentEvaluation:
        target = self._target(time, parameters)
        source = ArraySpace(parameters.shape, dtype=parameters.dtype)
        target_space = ArraySpace(target.shape, dtype=target.dtype)
        linearization = prepare_linearization(
            self._sampled_fields,
            parameters,
            source=source,
            target=target_space,
            linearization_id=f"{self.problem.problem_id}:field-jacobian",
        )
        jacobian = JacobianLinearOperator(
            linearization,
            operator_id=f"{self.problem.problem_id}:field-jacobian-operator",
        )
        if self.policy.formulation == "rectangular":
            regularizer = (
                None
                if self.policy.damping == 0.0
                else ScaledLinearOperator(
                    IdentityLinearOperator(source),
                    jnp.sqrt(
                        jnp.asarray(self.policy.damping, dtype=parameters.real.dtype)
                    ),
                )
            )
            result = solve(
                LeastSquaresProblem(
                    jacobian,
                    regularizer=regularizer,
                    problem_id=f"{self.problem.problem_id}:tangent-least-squares",
                ),
                target,
                policy=self.policy.linear_policy,
            )
        else:
            normal_properties = OperatorProperties(
                self_adjoint=True,
                positive_semidefinite=True,
                evidence={
                    "self_adjoint": "construction",
                    "positive_semidefinite": "construction",
                },
            )
            normal = FunctionLinearOperator(
                lambda vector: jacobian.adjoint_mv(jacobian.mv(vector)),
                source=source,
                target=source,
                properties=normal_properties,
                operator_id=f"{self.problem.problem_id}:tangent-gram",
            )
            damping = jnp.asarray(self.policy.damping, dtype=parameters.real.dtype)
            shifted_properties = OperatorProperties(
                self_adjoint=True,
                positive_semidefinite=True,
                positive_definite=True,
                evidence={
                    "self_adjoint": "construction",
                    "positive_semidefinite": "construction",
                    "positive_definite": "construction",
                },
            )
            shifted = FunctionLinearOperator(
                lambda vector: normal.mv(vector) + damping * vector,
                source=source,
                target=source,
                properties=shifted_properties,
                operator_id=f"{self.problem.problem_id}:damped-tangent-gram",
            )
            if self.policy.preconditioner is None:
                linear_policy = self.policy.linear_policy
            else:
                linear_policy = _copy_linear_policy(
                    self.policy.linear_policy,
                    preconditioning=PreconditioningPolicy(
                        self.policy.preconditioner,
                        setup_operator=normal,
                    ),
                )
            result = solve(
                LinearSystem(
                    shifted,
                    problem_id=f"{self.problem.problem_id}:tangent-normal-system",
                ),
                jacobian.adjoint_mv(target),
                policy=linear_policy,
            )
        rate = jnp.asarray(result.value)
        defect = jnp.asarray(jacobian.mv(rate)) - target
        defect_norm = _norm(defect)
        target_norm = _norm(target)
        relative_defect = defect_norm / jnp.maximum(target_norm, 1e-30)
        finite = (
            jnp.all(jnp.isfinite(rate))
            & jnp.isfinite(defect_norm)
            & jnp.isfinite(relative_defect)
        )
        accepted = result.successful & finite
        if self.policy.maximum_relative_defect is not None:
            accepted = accepted & (relative_defect <= self.policy.maximum_relative_defect)
        safe_rate = jnp.where(accepted, rate, jnp.full_like(rate, jnp.nan))
        diagnostics = result.diagnostics
        return _TangentEvaluation(
            rate=safe_rate,
            status=result.status,
            iterations=diagnostics.iterations,
            matvec_count=diagnostics.matvec_count,
            adjoint_matvec_count=diagnostics.adjoint_matvec_count,
            linear_residual_norm=diagnostics.residual_norm,
            rate_norm=target_norm,
            parameter_rate_norm=_norm(rate),
            projection_defect=defect_norm,
            relative_projection_defect=relative_defect,
            accepted=accepted,
        )

    def __call__(self, time: Array, parameters: Array, args: Any) -> Array:
        del args
        if self.implicit_adjoint:
            return _implicit_tangent_rate((time, parameters), self)
        return self.evaluate(time, parameters).rate


@eqx.filter_custom_vjp
def _implicit_tangent_rate(
    inputs: tuple[Array, Array],
    field: _NeuralGalerkinVectorField,
    /,
) -> Array:
    time, parameters = inputs
    return field.evaluate(time, parameters).rate


@_implicit_tangent_rate.def_fwd
def _implicit_tangent_rate_fwd(
    perturbed,
    inputs: tuple[Array, Array],
    field: _NeuralGalerkinVectorField,
):
    del perturbed
    time, parameters = inputs
    rate = field.evaluate(time, parameters).rate
    return rate, (time, parameters, rate)


@_implicit_tangent_rate.def_bwd
def _implicit_tangent_rate_bwd(
    residual,
    rate_cotangent: Array,
    perturbed,
    inputs: tuple[Array, Array],
    field: _NeuralGalerkinVectorField,
):
    del perturbed, inputs
    time, parameters, rate = residual
    damping = jnp.asarray(field.policy.damping, dtype=parameters.real.dtype)

    def sampled(candidate):
        return field._sampled_fields(candidate)

    _, pullback = jax.vjp(sampled, parameters)

    def normal_action(vector):
        tangent = jax.jvp(sampled, (parameters,), (vector,))[1]
        return pullback(tangent)[0] + damping * vector

    source = ArraySpace(parameters.shape, dtype=parameters.dtype)
    properties = OperatorProperties(
        self_adjoint=True,
        positive_semidefinite=True,
        positive_definite=field.policy.damping > 0.0,
        evidence={
            "self_adjoint": "implicit-normal-equation",
            "positive_semidefinite": "construction",
        },
    )
    normal = FunctionLinearOperator(
        normal_action,
        source=source,
        target=source,
        properties=properties,
        operator_id=f"{field.problem.problem_id}:implicit-adjoint-normal",
    )
    adjoint_result = solve(
        LinearSystem(
            normal,
            problem_id=f"{field.problem.problem_id}:implicit-adjoint",
        ),
        rate_cotangent,
        policy=field.policy.linear_policy,
    )
    multiplier = eqx.error_if(
        adjoint_result.value,
        ~adjoint_result.successful | (adjoint_result.diagnostics.residual_norm > 1.0e-6),
        "Neural Galerkin implicit adjoint solve failed its residual audit.",
    )

    def stationarity(t, candidate):
        target = field._target(t, candidate)

        def sampled_candidate(value):
            return field._sampled_fields(value)

        sampled_value, sampled_pullback = jax.vjp(
            sampled_candidate,
            candidate,
        )
        del sampled_value
        tangent = jax.jvp(
            sampled_candidate,
            (candidate,),
            (jax.lax.stop_gradient(rate),),
        )[1]
        return sampled_pullback(tangent - target)[0] + damping * jax.lax.stop_gradient(
            rate
        )

    _, stationarity_pullback = jax.vjp(stationarity, time, parameters)
    time_gradient, parameter_gradient = stationarity_pullback(-multiplier)
    return time_gradient, parameter_gradient


class NeuralFieldEvolutionResult(StrictModule):
    """Diffrax parameter trajectory plus independently audited field projection."""

    problem: NeuralGalerkinProblem
    parameter_solution: DifferentialSolution
    audit: NeuralGalerkinAudit

    @property
    def successful(self) -> Array:
        return (
            jnp.asarray(self.parameter_solution.backend_successful, dtype=bool)
            & jnp.all(self.parameter_solution.valid)
            & jnp.all(self.audit.accepted)
        )

    def functions_at(self, index: int, /) -> frozendict[str, DomainFunction]:
        node = int(index)
        count = int(self.parameter_solution.states.shape[0])
        if node < 0 or node >= count:
            raise IndexError("Neural field node index is out of range.")
        if not bool(self.parameter_solution.valid[node]) or not bool(
            self.audit.accepted[node]
        ):
            raise ValueError("Cannot reconstruct an invalid neural field node.")
        return self.problem.ansatz(self.parameter_solution.states[node])

    def field_at(self, index: int, name: str, /) -> DomainFunction:
        functions = self.functions_at(index)
        if name not in functions:
            raise KeyError(f"Unknown neural field {name!r}.")
        return functions[name]

    def functions_at_time(
        self,
        time: ArrayLike,
        /,
    ) -> frozendict[str, DomainFunction]:
        if self.parameter_solution.interpolation is None:
            raise ValueError("Dense parameter interpolation was not requested.")
        query = jnp.asarray(time)
        if query.shape != ():
            raise ValueError("functions_at_time requires one scalar time.")
        parameters = self.parameter_solution.interpolation.evaluate(query)
        return self.problem.ansatz(parameters)


def _invalid_audit(parameters: Array, /) -> _TangentEvaluation:
    nan = jnp.asarray(jnp.nan, dtype=jnp.asarray(parameters).real.dtype)
    zero = jnp.asarray(0, dtype=jnp.int32)
    return _TangentEvaluation(
        rate=jnp.full_like(parameters, jnp.nan),
        status=jnp.asarray(-1, dtype=jnp.int32),
        iterations=zero,
        matvec_count=zero,
        adjoint_matvec_count=zero,
        linear_residual_norm=nan,
        rate_norm=nan,
        parameter_rate_norm=nan,
        projection_defect=nan,
        relative_projection_defect=nan,
        accepted=jnp.asarray(False),
    )


def _stack_audits(values: Sequence[_TangentEvaluation], /) -> NeuralGalerkinAudit:
    return NeuralGalerkinAudit(
        status=jnp.stack(tuple(value.status for value in values)),
        iterations=jnp.stack(tuple(value.iterations for value in values)),
        matvec_count=jnp.stack(tuple(value.matvec_count for value in values)),
        adjoint_matvec_count=jnp.stack(
            tuple(value.adjoint_matvec_count for value in values)
        ),
        linear_residual_norm=jnp.stack(
            tuple(value.linear_residual_norm for value in values)
        ),
        rate_norm=jnp.stack(tuple(value.rate_norm for value in values)),
        parameter_rate_norm=jnp.stack(
            tuple(value.parameter_rate_norm for value in values)
        ),
        projection_defect=jnp.stack(tuple(value.projection_defect for value in values)),
        relative_projection_defect=jnp.stack(
            tuple(value.relative_projection_defect for value in values)
        ),
        accepted=jnp.stack(tuple(value.accepted for value in values)),
    )


def solve_neural_galerkin(
    problem: NeuralGalerkinProblem,
    time_grid: TimeGrid,
    /,
    *,
    tangent: NeuralTangentSolvePolicy | None = None,
    solver: Any | None = None,
    stepsize_controller: Any | None = None,
    adjoint: Any | None = None,
    adjoint_policy: NeuralGalerkinAdjointPolicy | None = None,
    dt0: ArrayLike | None = None,
    rtol: float = 1e-6,
    atol: float = 1e-8,
    dense: bool = False,
    max_steps: int | None = 4096,
    throw: bool = False,
    precision: TemporalPrecisionPolicy | None = None,
) -> NeuralFieldEvolutionResult:
    """Evolve a fixed-measure neural field manifold through Diffrax."""
    if not isinstance(problem, NeuralGalerkinProblem):
        raise TypeError("problem must be a NeuralGalerkinProblem.")
    if not isinstance(time_grid, TimeGrid):
        raise TypeError("time_grid must be a TimeGrid.")
    selected_tangent = NeuralTangentSolvePolicy() if tangent is None else tangent
    if not isinstance(selected_tangent, NeuralTangentSolvePolicy):
        raise TypeError("tangent must be a NeuralTangentSolvePolicy or None.")
    selected_adjoint_policy = (
        NeuralGalerkinAdjointPolicy() if adjoint_policy is None else adjoint_policy
    )
    if not isinstance(selected_adjoint_policy, NeuralGalerkinAdjointPolicy):
        raise TypeError("adjoint_policy must be NeuralGalerkinAdjointPolicy or None.")
    if isinstance(adjoint, dfx.BacksolveAdjoint) and (
        selected_adjoint_policy.mode != "certified_backsolve"
    ):
        raise ValueError(
            "BacksolveAdjoint requires certified_backsolve policy and audited tangent solves."
        )
    vector_field = _NeuralGalerkinVectorField(
        problem,
        selected_tangent,
        selected_adjoint_policy.mode == "certified_backsolve",
    )
    initial = problem.parameter_subspace.pack()
    vector_field._target(time_grid.t0, initial)
    differential = DifferentialProblem(
        vector_field,
        initial,
        t0=time_grid.t0,
        t1=time_grid.t1,
        args=None,
        problem_id=problem.problem_id,
    )
    solution = solve_diffrax(
        differential,
        save_times=time_grid.times,
        solver=solver,
        stepsize_controller=stepsize_controller,
        adjoint=adjoint,
        dt0=dt0,
        rtol=rtol,
        atol=atol,
        dense=dense,
        max_steps=max_steps,
        throw=throw,
        precision=precision,
    )
    audits = []
    for time, parameters, valid in zip(
        tuple(solution.times),
        tuple(solution.states),
        tuple(solution.valid),
        strict=True,
    ):
        node_valid = bool(jax.device_get(valid)) and bool(
            jax.device_get(jnp.isfinite(time) & jnp.all(jnp.isfinite(parameters)))
        )
        audits.append(
            vector_field.evaluate(time, parameters)
            if node_valid
            else _invalid_audit(parameters)
        )
    return NeuralFieldEvolutionResult(
        problem=problem,
        parameter_solution=solution,
        audit=_stack_audits(audits),
    )


def solve_neural_galerkin_epochs(
    plan: NeuralGalerkinEpochPlan,
    /,
    **solve_options: Any,
) -> NeuralGalerkinEpochResult:
    """Execute fixed populations segment by segment and freeze their replay identity."""
    if not isinstance(plan, NeuralGalerkinEpochPlan):
        raise TypeError("plan must be NeuralGalerkinEpochPlan.")
    segments: list[NeuralFieldEvolutionResult] = []
    previous: NeuralFieldEvolutionResult | None = None
    paths = plan.epochs[0].problem.parameter_subspace.leaf_paths
    for epoch in plan.epochs:
        problem = epoch.problem
        if previous is not None:
            functions = previous.problem.ansatz(previous.parameter_solution.states[-1])
            subspace = ParameterSubspace.from_leaf_paths(functions, paths)
            problem = NeuralGalerkinProblem(
                functions,
                epoch.problem.rate,
                epoch.problem.metrics,
                parameter_subspace=subspace,
                enforcement=epoch.problem.enforcement,
                args=epoch.problem.args,
                evaluation_key=epoch.problem.evaluation_key,
                problem_id=epoch.problem.problem_id,
            )
        result = solve_neural_galerkin(
            problem,
            epoch.grid,
            **solve_options,
        )
        segments.append(result)
        previous = result
    population_ids = tuple(epoch.population_id for epoch in plan.epochs)
    replay_policy = HybridReplayPolicy(max(1, len(plan.epochs) - 1))
    journal = NeuralGalerkinReplayJournal(
        jax.lax.stop_gradient(
            jnp.stack(
                tuple(segment.parameter_solution.states[-1] for segment in segments)
            )
        ),
        population_ids,
        plan.replay_id,
        replay_policy.policy_id,
        True,
    )
    return NeuralGalerkinEpochResult(
        tuple(segments),
        population_ids,
        plan.replay_id,
        journal,
    )


def replay_neural_galerkin_epochs(
    plan: NeuralGalerkinEpochPlan,
    journal: NeuralGalerkinReplayJournal,
    /,
    **solve_options: Any,
) -> NeuralGalerkinEpochResult:
    """Replay an identical frozen population journal; selection remains stopped."""
    if not isinstance(journal, NeuralGalerkinReplayJournal):
        raise TypeError("journal must be NeuralGalerkinReplayJournal.")
    expected = tuple(epoch.population_id for epoch in plan.epochs)
    policy = HybridReplayPolicy(max(1, len(plan.epochs) - 1))
    if (
        journal.replay_id != plan.replay_id
        or journal.population_ids != expected
        or journal.replay_policy_id != policy.policy_id
    ):
        raise ValueError("Neural Galerkin replay journal identity does not match.")
    replayed = solve_neural_galerkin_epochs(plan, **solve_options)
    return replayed


__all__ = [
    "FieldProjectionMetric",
    "NeuralFieldEvolutionResult",
    "NeuralGalerkinAudit",
    "NeuralGalerkinAdjointPolicy",
    "NeuralGalerkinEpoch",
    "NeuralGalerkinEpochPlan",
    "NeuralGalerkinReplayJournal",
    "NeuralGalerkinEpochResult",
    "NeuralGalerkinProblem",
    "NeuralTangentSolvePolicy",
    "RateFunction",
    "TangentFormulation",
    "solve_neural_galerkin",
    "replay_neural_galerkin_epochs",
    "solve_neural_galerkin_epochs",
]
