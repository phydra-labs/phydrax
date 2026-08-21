#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import abc
from collections.abc import Callable
from math import isfinite
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
from jaxtyping import Array, PyTree

from .._strict import StrictModule
from ._iterative._types import (
    _tree_add_scaled,
    _tree_allfinite,
    _tree_inner,
    _tree_norm,
    _validate_real_inexact_tree,
    IterativeStepMetrics,
    MinimizationProblem,
    OptimizationDiagnostics,
    OptimizationProvenance,
    OptimizationStatus,
    OptimizationTermination,
)


def _tree_sum_squares(tree: PyTree[Any], /) -> Array:
    return _tree_inner(tree, tree)


def _tree_subtract(left: PyTree[Any], right: PyTree[Any], /) -> PyTree[Array]:
    return jax.tree.map(lambda x, y: x - y, left, right)


def _soft_threshold(value: Array, threshold: Any, /) -> Array:
    threshold_ = jnp.asarray(threshold, dtype=value.dtype)
    return jnp.sign(value) * jnp.maximum(jnp.abs(value) - threshold_, 0.0)


def _broadcast_bound(bound: Any, parameters: PyTree[Any], /) -> PyTree[Array]:
    parameter_definition = jax.tree.structure(parameters)
    if jax.tree.structure(bound) == parameter_definition:
        return jax.tree.map(jnp.asarray, bound)
    try:
        scalar = jnp.asarray(bound)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "A bound must be scalar or match the parameter PyTree."
        ) from error
    if scalar.shape != ():
        raise ValueError("A bound must be scalar or match the parameter PyTree.")
    return jax.tree.map(lambda leaf: jnp.asarray(scalar, dtype=leaf.dtype), parameters)


class AbstractProximalFunctional(StrictModule):
    """Closed extended-real functional with a Euclidean proximal map."""

    @abc.abstractmethod
    def value(self, parameters: PyTree[Any], /) -> Array:
        raise NotImplementedError

    @abc.abstractmethod
    def proximal(
        self,
        parameters: PyTree[Any],
        step_size: Any,
        /,
    ) -> PyTree[Array]:
        raise NotImplementedError


class L1Functional(AbstractProximalFunctional):
    """Weighted elementwise L1 functional and soft-threshold proximal map."""

    weight: float = eqx.field(static=True)

    def __init__(self, weight: float = 1.0, /):
        weight_ = float(weight)
        if not isfinite(weight_) or weight_ < 0.0:
            raise ValueError("weight must be finite and non-negative.")
        self.weight = weight_

    def value(self, parameters: PyTree[Any], /) -> Array:
        leaves = jax.tree.leaves(parameters)
        if not leaves:
            raise ValueError("parameters must contain at least one array leaf.")
        return self.weight * sum(jnp.sum(jnp.abs(leaf)) for leaf in leaves)

    def proximal(self, parameters: PyTree[Any], step_size: Any, /) -> PyTree[Array]:
        rate = jnp.asarray(step_size)
        return jax.tree.map(
            lambda leaf: _soft_threshold(leaf, rate * self.weight),
            parameters,
        )


class ElasticNetFunctional(AbstractProximalFunctional):
    """L1 plus squared-L2 functional with its exact separable proximal map."""

    l1_weight: float = eqx.field(static=True)
    l2_weight: float = eqx.field(static=True)

    def __init__(self, l1_weight: float = 1.0, l2_weight: float = 1.0, /):
        l1 = float(l1_weight)
        l2 = float(l2_weight)
        if any(not isfinite(value) or value < 0.0 for value in (l1, l2)):
            raise ValueError("Elastic-net weights must be finite and non-negative.")
        self.l1_weight = l1
        self.l2_weight = l2

    def value(self, parameters: PyTree[Any], /) -> Array:
        leaves = jax.tree.leaves(parameters)
        if not leaves:
            raise ValueError("parameters must contain at least one array leaf.")
        l1 = sum(jnp.sum(jnp.abs(leaf)) for leaf in leaves)
        return self.l1_weight * l1 + 0.5 * self.l2_weight * _tree_sum_squares(parameters)

    def proximal(self, parameters: PyTree[Any], step_size: Any, /) -> PyTree[Array]:
        rate = jnp.asarray(step_size)
        denominator = 1.0 + rate * self.l2_weight
        return jax.tree.map(
            lambda leaf: _soft_threshold(leaf, rate * self.l1_weight) / denominator,
            parameters,
        )


class IndicatorFunctional(AbstractProximalFunctional):
    """Indicator of a closed set supplied through projection and membership maps."""

    projection: Callable[[PyTree[Any]], PyTree[Any]]
    contains: Callable[[PyTree[Any]], Any]

    def __init__(
        self,
        projection: Callable[[PyTree[Any]], PyTree[Any]],
        contains: Callable[[PyTree[Any]], Any],
        /,
    ):
        if not callable(projection) or not callable(contains):
            raise TypeError("projection and contains must be callable.")
        self.projection = projection
        self.contains = contains

    def value(self, parameters: PyTree[Any], /) -> Array:
        leaves = jax.tree.leaves(parameters)
        if not leaves:
            raise ValueError("parameters must contain at least one array leaf.")
        dtype = jnp.result_type(*(jnp.asarray(leaf).dtype for leaf in leaves), float)
        return jnp.where(
            jnp.asarray(self.contains(parameters), dtype=bool),
            jnp.asarray(0.0, dtype=dtype),
            jnp.asarray(jnp.inf, dtype=dtype),
        )

    def proximal(self, parameters: PyTree[Any], step_size: Any, /) -> PyTree[Array]:
        del step_size
        projected = self.projection(parameters)
        if jax.tree.structure(projected) != jax.tree.structure(parameters):
            raise ValueError("projection must preserve the parameter PyTree structure.")
        return projected


class BoxIndicator(AbstractProximalFunctional):
    """Box-set indicator whose proximal map is elementwise clipping."""

    lower: PyTree[Any]
    upper: PyTree[Any]

    def __init__(self, lower: Any = -jnp.inf, upper: Any = jnp.inf, /):
        self.lower = lower
        self.upper = upper

    def _bounds(self, parameters: PyTree[Any], /):
        lower = _broadcast_bound(self.lower, parameters)
        upper = _broadcast_bound(self.upper, parameters)
        valid = jax.tree.reduce(
            jnp.logical_and,
            jax.tree.map(lambda lo, hi: jnp.all(lo <= hi), lower, upper),
        )
        return lower, upper, valid

    def value(self, parameters: PyTree[Any], /) -> Array:
        lower, upper, valid = self._bounds(parameters)
        contained = valid & jax.tree.reduce(
            jnp.logical_and,
            jax.tree.map(
                lambda value, lo, hi: jnp.all((value >= lo) & (value <= hi)),
                parameters,
                lower,
                upper,
            ),
        )
        dtype = jnp.result_type(
            *(jnp.asarray(leaf).dtype for leaf in jax.tree.leaves(parameters)),
            float,
        )
        return jnp.where(contained, jnp.asarray(0.0, dtype=dtype), jnp.inf)

    def proximal(self, parameters: PyTree[Any], step_size: Any, /) -> PyTree[Array]:
        del step_size
        lower, upper, valid = self._bounds(parameters)
        first_leaf = jax.tree.leaves(parameters)[0]
        valid = eqx.error_if(
            jnp.asarray(valid),
            ~jnp.asarray(valid),
            "BoxIndicator lower bounds must not exceed upper bounds.",
        )
        del first_leaf, valid
        return jax.tree.map(jnp.clip, parameters, lower, upper)


class SimplexIndicator(AbstractProximalFunctional):
    """Probability-simplex indicator over all leaves of one parameter PyTree."""

    mass: float = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)

    def __init__(self, mass: float = 1.0, /, *, tolerance: float = 1e-7):
        mass_ = float(mass)
        tolerance_ = float(tolerance)
        if not isfinite(mass_) or mass_ <= 0.0:
            raise ValueError("mass must be finite and positive.")
        if not isfinite(tolerance_) or tolerance_ < 0.0:
            raise ValueError("tolerance must be finite and non-negative.")
        self.mass = mass_
        self.tolerance = tolerance_

    def value(self, parameters: PyTree[Any], /) -> Array:
        flat, _ = ravel_pytree(parameters)
        contained = jnp.all(flat >= -self.tolerance) & (
            jnp.abs(jnp.sum(flat) - self.mass) <= self.tolerance
        )
        return jnp.where(contained, jnp.zeros((), dtype=flat.dtype), jnp.inf)

    def proximal(self, parameters: PyTree[Any], step_size: Any, /) -> PyTree[Array]:
        del step_size
        flat, unravel = ravel_pytree(parameters)
        ordered = jnp.sort(flat)[::-1]
        cumulative = jnp.cumsum(ordered) - self.mass
        indices = jnp.arange(1, flat.size + 1, dtype=flat.dtype)
        active = ordered - cumulative / indices > 0.0
        rho = jnp.maximum(jnp.sum(active, dtype=jnp.int32) - 1, 0)
        threshold = cumulative[rho] / (rho.astype(flat.dtype) + 1.0)
        return unravel(jnp.maximum(flat - threshold, 0.0))


class GroupLassoFunctional(AbstractProximalFunctional):
    """Leafwise group-L2 penalty along a chosen array axis."""

    weight: float = eqx.field(static=True)
    axis: int = eqx.field(static=True)

    def __init__(self, weight: float = 1.0, /, *, axis: int = -1):
        weight_ = float(weight)
        if not isfinite(weight_) or weight_ < 0.0:
            raise ValueError("weight must be finite and non-negative.")
        self.weight = weight_
        self.axis = int(axis)

    def _norm(self, leaf: Array, /, *, keepdims: bool) -> Array:
        if leaf.ndim == 0:
            return jnp.abs(leaf)
        axis = self.axis % leaf.ndim
        return jnp.linalg.norm(leaf, axis=axis, keepdims=keepdims)

    def value(self, parameters: PyTree[Any], /) -> Array:
        leaves = jax.tree.leaves(parameters)
        if not leaves:
            raise ValueError("parameters must contain at least one array leaf.")
        return self.weight * sum(
            jnp.sum(self._norm(leaf, keepdims=False)) for leaf in leaves
        )

    def proximal(self, parameters: PyTree[Any], step_size: Any, /) -> PyTree[Array]:
        rate = jnp.asarray(step_size)

        def shrink(leaf):
            norm = self._norm(leaf, keepdims=True)
            scale = jnp.maximum(0.0, 1.0 - rate * self.weight / jnp.maximum(norm, 1e-30))
            return scale * leaf

        return jax.tree.map(shrink, parameters)


class NuclearNormFunctional(AbstractProximalFunctional):
    """Sum of leaf nuclear norms with singular-value thresholding."""

    weight: float = eqx.field(static=True)

    def __init__(self, weight: float = 1.0, /):
        weight_ = float(weight)
        if not isfinite(weight_) or weight_ < 0.0:
            raise ValueError("weight must be finite and non-negative.")
        self.weight = weight_

    def _validate(self, parameters: PyTree[Any], /):
        leaves = jax.tree.leaves(parameters)
        if not leaves:
            raise ValueError("parameters must contain at least one array leaf.")
        if any(jnp.asarray(leaf).ndim < 2 for leaf in leaves):
            raise ValueError("Every nuclear-norm leaf must have at least two axes.")
        return leaves

    def value(self, parameters: PyTree[Any], /) -> Array:
        leaves = self._validate(parameters)
        return self.weight * sum(
            jnp.sum(jnp.linalg.svd(leaf, compute_uv=False)) for leaf in leaves
        )

    def proximal(self, parameters: PyTree[Any], step_size: Any, /) -> PyTree[Array]:
        self._validate(parameters)
        rate = jnp.asarray(step_size)

        def shrink(matrix):
            left, singular, right = jnp.linalg.svd(matrix, full_matrices=False)
            singular = jnp.maximum(singular - rate * self.weight, 0.0)
            return (left * singular[..., None, :]) @ right

        return jax.tree.map(shrink, parameters)


class ProximalProblem(StrictModule):
    """Smooth scalar objective plus one closed proximable functional."""

    smooth: MinimizationProblem
    nonsmooth: AbstractProximalFunctional

    def __init__(
        self,
        smooth: MinimizationProblem | Callable[[PyTree[Any], Any], Any],
        nonsmooth: AbstractProximalFunctional,
        /,
        *,
        has_aux: bool = False,
        problem_id: str = "callable-proximal-minimization",
    ):
        smooth_ = (
            smooth
            if isinstance(smooth, MinimizationProblem)
            else MinimizationProblem(smooth, has_aux=has_aux, problem_id=problem_id)
        )
        if smooth_.bounds is not None or smooth_.constraints:
            raise ValueError(
                "ProximalProblem smooth terms must be unconstrained; encode a closed "
                "set with an indicator functional."
            )
        if not isinstance(nonsmooth, AbstractProximalFunctional):
            raise TypeError("nonsmooth must be an AbstractProximalFunctional.")
        self.smooth = smooth_
        self.nonsmooth = nonsmooth

    @property
    def problem_id(self) -> str:
        return self.smooth.problem_id

    def value(
        self,
        parameters: PyTree[Any],
        args: Any = None,
        /,
    ) -> tuple[Array, Array, Array, Any]:
        smooth, auxiliary = self.smooth.value(parameters, args)
        nonsmooth = self.nonsmooth.value(parameters)
        return smooth + nonsmooth, smooth, nonsmooth, auxiliary

    def value_and_gradient(
        self,
        parameters: PyTree[Any],
        args: Any = None,
        /,
    ):
        return self.smooth.value_and_gradient(parameters, args)

    def stationarity(
        self,
        parameters: PyTree[Any],
        gradient: PyTree[Any],
        step_size: Any,
        /,
    ) -> Array:
        trial = self.nonsmooth.proximal(
            _tree_add_scaled(parameters, gradient, -jnp.asarray(step_size)),
            step_size,
        )
        mapping = jax.tree.map(
            lambda current, prox: (current - prox) / jnp.asarray(step_size),
            parameters,
            trial,
        )
        return _tree_norm(mapping)


class ProximalResult(StrictModule):
    """Accepted composite point with explicit gradient-mapping stationarity."""

    parameters: PyTree[Array]
    objective: Array
    smooth_objective: Array
    nonsmooth_objective: Array
    auxiliary: Any
    composite_stationarity: Array
    status: Array
    diagnostics: OptimizationDiagnostics
    provenance: OptimizationProvenance

    def __init__(
        self,
        *,
        parameters: PyTree[Any],
        objective: Any,
        smooth_objective: Any,
        nonsmooth_objective: Any,
        auxiliary: Any,
        composite_stationarity: Any,
        status: Any,
        diagnostics: OptimizationDiagnostics,
        provenance: OptimizationProvenance,
    ):
        if not isinstance(diagnostics, OptimizationDiagnostics):
            raise TypeError("diagnostics must be OptimizationDiagnostics.")
        if not isinstance(provenance, OptimizationProvenance):
            raise TypeError("provenance must be OptimizationProvenance.")
        self.parameters = parameters
        self.objective = jnp.asarray(objective)
        self.smooth_objective = jnp.asarray(smooth_objective)
        self.nonsmooth_objective = jnp.asarray(nonsmooth_objective)
        self.auxiliary = auxiliary
        self.composite_stationarity = jnp.asarray(composite_stationarity)
        self.status = jnp.asarray(status, dtype=jnp.int32)
        self.diagnostics = diagnostics
        self.provenance = provenance

    @property
    def successful(self) -> Array:
        return self.status == int(OptimizationStatus.SUCCESS)


class ProximalState(StrictModule):
    """Array carry for proximal accepted-point iterations."""

    iteration: Array
    extrapolated: PyTree[Array]
    momentum: Array
    step_size: Array
    initial_stationarity: Array
    stationarity: Array
    objective: Array
    accepted_steps: Array
    rejected_steps: Array
    objective_evaluations: Array
    gradient_evaluations: Array
    hvp_evaluations: Array
    globalization_evaluations: Array
    final_step_norm: Array
    accepted_rate: Array
    status: Array
    metrics: IterativeStepMetrics

    def __init__(
        self,
        *,
        iteration: Any,
        extrapolated: PyTree[Any],
        momentum: Any,
        step_size: Any,
        initial_stationarity: Any,
        stationarity: Any,
        objective: Any,
        accepted_steps: Any = 0,
        rejected_steps: Any = 0,
        objective_evaluations: Any = 0,
        gradient_evaluations: Any = 0,
        hvp_evaluations: Any = 0,
        globalization_evaluations: Any = 0,
        final_step_norm: Any = 0.0,
        accepted_rate: Any = 0.0,
        status: Any = OptimizationStatus.ITERATING,
        metrics: IterativeStepMetrics | None = None,
    ):
        self.iteration = jnp.asarray(iteration, dtype=jnp.int32)
        self.extrapolated = extrapolated
        self.momentum = jnp.asarray(momentum)
        self.step_size = jnp.asarray(step_size)
        self.initial_stationarity = jnp.asarray(initial_stationarity)
        self.stationarity = jnp.asarray(stationarity)
        self.objective = jnp.asarray(objective)
        self.accepted_steps = jnp.asarray(accepted_steps, dtype=jnp.int32)
        self.rejected_steps = jnp.asarray(rejected_steps, dtype=jnp.int32)
        self.objective_evaluations = jnp.asarray(objective_evaluations, dtype=jnp.int32)
        self.gradient_evaluations = jnp.asarray(gradient_evaluations, dtype=jnp.int32)
        self.hvp_evaluations = jnp.asarray(hvp_evaluations, dtype=jnp.int32)
        self.globalization_evaluations = jnp.asarray(
            globalization_evaluations, dtype=jnp.int32
        )
        self.final_step_norm = jnp.asarray(final_step_norm)
        self.accepted_rate = jnp.asarray(accepted_rate)
        self.status = jnp.asarray(status, dtype=jnp.int32)
        self.metrics = IterativeStepMetrics() if metrics is None else metrics


class AbstractProximalMethod(StrictModule):
    """Method implementing prepare, step, and solve for ProximalProblem."""

    initial_step_size: float = eqx.field(static=True)
    contraction: float = eqx.field(static=True)
    minimum_step_size: float = eqx.field(static=True)
    maximum_backtracking_steps: int = eqx.field(static=True)

    @property
    @abc.abstractmethod
    def method_id(self) -> str:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def accelerated(self) -> bool:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def proximal_newton(self) -> bool:
        raise NotImplementedError

    def _initialize_policy(
        self,
        *,
        initial_step_size: float,
        contraction: float,
        minimum_step_size: float,
        maximum_backtracking_steps: int,
    ):
        initial = float(initial_step_size)
        contraction_ = float(contraction)
        minimum = float(minimum_step_size)
        steps = int(maximum_backtracking_steps)
        if not isfinite(initial) or initial <= 0.0:
            raise ValueError("initial_step_size must be finite and positive.")
        if not isfinite(contraction_) or not 0.0 < contraction_ < 1.0:
            raise ValueError("contraction must lie in (0, 1).")
        if not isfinite(minimum) or minimum <= 0.0 or minimum > initial:
            raise ValueError(
                "minimum_step_size must be positive and no larger than initial_step_size."
            )
        if steps < 1:
            raise ValueError("maximum_backtracking_steps must be positive.")
        self.initial_step_size = initial
        self.contraction = contraction_
        self.minimum_step_size = minimum
        self.maximum_backtracking_steps = steps

    def prepare_state(
        self,
        problem: ProximalProblem,
        initial_parameters: PyTree[Any],
        /,
        *,
        args: Any,
    ) -> ProximalState:
        if not isinstance(problem, ProximalProblem):
            raise TypeError("problem must be a ProximalProblem.")
        parameters = _validate_real_inexact_tree(
            initial_parameters, name="initial_parameters"
        )
        (smooth, _), gradient = problem.value_and_gradient(parameters, args)
        nonsmooth = problem.nonsmooth.value(parameters)
        stationarity = problem.stationarity(parameters, gradient, self.initial_step_size)
        finite = (
            _tree_allfinite(parameters)
            & _tree_allfinite(gradient)
            & jnp.isfinite(smooth)
            & jnp.isfinite(stationarity)
        )
        status = jnp.where(
            finite,
            int(OptimizationStatus.ITERATING),
            int(OptimizationStatus.NONFINITE_INPUT),
        )
        return ProximalState(
            iteration=0,
            extrapolated=parameters,
            momentum=jnp.asarray(1.0, dtype=jnp.asarray(smooth).dtype),
            step_size=jnp.asarray(
                self.initial_step_size, dtype=jnp.asarray(smooth).dtype
            ),
            initial_stationarity=stationarity,
            stationarity=stationarity,
            objective=smooth + nonsmooth,
            objective_evaluations=1,
            gradient_evaluations=1,
            status=status,
            metrics=IterativeStepMetrics(
                objective=smooth + nonsmooth,
                optimality_norm=stationarity,
                status=status,
            ),
        )

    def step(
        self,
        problem: ProximalProblem,
        parameters: PyTree[Any],
        state: ProximalState,
        /,
        *,
        termination: OptimizationTermination,
        args: Any,
    ) -> tuple[PyTree[Array], ProximalState]:
        return _proximal_step(
            self,
            problem,
            parameters,
            state,
            termination=termination,
            args=args,
        )

    def solve(
        self,
        problem: ProximalProblem,
        initial_parameters: PyTree[Any],
        /,
        *,
        termination: OptimizationTermination,
        args: Any,
    ) -> ProximalResult:
        return _solve_proximal(
            self,
            problem,
            initial_parameters,
            termination=termination,
            args=args,
        )


class ProximalGradient(AbstractProximalMethod):
    """Composite proximal-gradient method with majorization backtracking."""

    def __init__(
        self,
        *,
        initial_step_size: float = 1.0,
        contraction: float = 0.5,
        minimum_step_size: float = 1e-12,
        maximum_backtracking_steps: int = 30,
    ):
        self._initialize_policy(
            initial_step_size=initial_step_size,
            contraction=contraction,
            minimum_step_size=minimum_step_size,
            maximum_backtracking_steps=maximum_backtracking_steps,
        )

    @property
    def method_id(self) -> str:
        return "proximal-gradient"

    @property
    def accelerated(self) -> bool:
        return False

    @property
    def proximal_newton(self) -> bool:
        return False


class AcceleratedProximalGradient(AbstractProximalMethod):
    """FISTA recurrence with monotone objective restart and backtracking."""

    def __init__(
        self,
        *,
        initial_step_size: float = 1.0,
        contraction: float = 0.5,
        minimum_step_size: float = 1e-12,
        maximum_backtracking_steps: int = 30,
    ):
        self._initialize_policy(
            initial_step_size=initial_step_size,
            contraction=contraction,
            minimum_step_size=minimum_step_size,
            maximum_backtracking_steps=maximum_backtracking_steps,
        )

    @property
    def method_id(self) -> str:
        return "accelerated-proximal-gradient/fista-restart"

    @property
    def accelerated(self) -> bool:
        return True

    @property
    def proximal_newton(self) -> bool:
        return False


class ProximalNewton(AbstractProximalMethod):
    """Dense proximal Newton method solving each regularized quadratic model."""

    minimum_curvature: float = eqx.field(static=True)
    inner_steps: int = eqx.field(static=True)
    sufficient_decrease: float = eqx.field(static=True)
    max_dense_dimension: int = eqx.field(static=True)

    def __init__(
        self,
        *,
        initial_step_size: float = 1.0,
        contraction: float = 0.5,
        minimum_step_size: float = 1e-12,
        maximum_backtracking_steps: int = 30,
        minimum_curvature: float = 1e-6,
        inner_steps: int = 50,
        sufficient_decrease: float = 1e-4,
        max_dense_dimension: int = 256,
    ):
        self._initialize_policy(
            initial_step_size=initial_step_size,
            contraction=contraction,
            minimum_step_size=minimum_step_size,
            maximum_backtracking_steps=maximum_backtracking_steps,
        )
        curvature = float(minimum_curvature)
        decrease = float(sufficient_decrease)
        inner = int(inner_steps)
        dimension = int(max_dense_dimension)
        if not isfinite(curvature) or curvature <= 0.0:
            raise ValueError("minimum_curvature must be positive and finite.")
        if not isfinite(decrease) or not 0.0 < decrease < 1.0:
            raise ValueError("sufficient_decrease must lie in (0, 1).")
        if inner < 1 or dimension < 1:
            raise ValueError("inner_steps and max_dense_dimension must be positive.")
        self.minimum_curvature = curvature
        self.inner_steps = inner
        self.sufficient_decrease = decrease
        self.max_dense_dimension = dimension

    @property
    def method_id(self) -> str:
        return "proximal-newton/quadratic-subproblem"

    @property
    def accelerated(self) -> bool:
        return False

    @property
    def proximal_newton(self) -> bool:
        return True


def _within_evaluation_budget(
    state: ProximalState,
    evaluations: Array,
    termination: OptimizationTermination,
    /,
) -> Array:
    if termination.maximum_evaluations is None:
        return jnp.asarray(True)
    return state.objective_evaluations + 1 + evaluations < termination.maximum_evaluations


def _first_order_proposal(
    method: AbstractProximalMethod,
    problem: ProximalProblem,
    reference: PyTree[Any],
    state: ProximalState,
    termination: OptimizationTermination,
    args: Any,
    /,
):
    (reference_value, _), reference_gradient = problem.value_and_gradient(reference, args)

    def condition(carry):
        evaluations, rate, accepted, *_ = carry
        return (
            (evaluations < method.maximum_backtracking_steps)
            & (~accepted)
            & (rate >= method.minimum_step_size)
            & _within_evaluation_budget(state, evaluations, termination)
        )

    def body(carry):
        (
            evaluations,
            rate,
            _,
            accepted_parameters,
            accepted_smooth,
            accepted_nonsmooth,
            accepted_gradient,
        ) = carry
        candidate = problem.nonsmooth.proximal(
            _tree_add_scaled(reference, reference_gradient, -rate), rate
        )
        (candidate_smooth, _), candidate_gradient = problem.value_and_gradient(
            candidate, args
        )
        candidate_nonsmooth = problem.nonsmooth.value(candidate)
        difference = _tree_subtract(candidate, reference)
        majorization = (
            reference_value
            + _tree_inner(reference_gradient, difference)
            + 0.5 * _tree_sum_squares(difference) / rate
        )
        accepted = (
            _tree_allfinite(candidate)
            & _tree_allfinite(candidate_gradient)
            & jnp.isfinite(candidate_smooth)
            & jnp.isfinite(candidate_nonsmooth)
            & (candidate_smooth <= majorization)
        )
        return (
            evaluations + 1,
            jnp.where(accepted, rate, rate * method.contraction),
            accepted,
            jax.tree.map(
                lambda new, old: jnp.where(accepted, new, old),
                candidate,
                accepted_parameters,
            ),
            jnp.where(accepted, candidate_smooth, accepted_smooth),
            jnp.where(accepted, candidate_nonsmooth, accepted_nonsmooth),
            jax.tree.map(
                lambda new, old: jnp.where(accepted, new, old),
                candidate_gradient,
                accepted_gradient,
            ),
        )

    return jax.lax.while_loop(
        condition,
        body,
        (
            jnp.asarray(0, dtype=jnp.int32),
            state.step_size,
            jnp.asarray(False),
            reference,
            reference_value,
            problem.nonsmooth.value(reference),
            reference_gradient,
        ),
    ) + (
        reference_value,
        reference_gradient,
    )


def _proximal_newton_proposal(
    method: ProximalNewton,
    problem: ProximalProblem,
    parameters: PyTree[Any],
    state: ProximalState,
    termination: OptimizationTermination,
    args: Any,
    /,
):
    flat_parameters, unravel = ravel_pytree(parameters)
    if int(flat_parameters.size) > method.max_dense_dimension:
        raise ValueError(
            f"ProximalNewton has {flat_parameters.size} variables, exceeding "
            f"max_dense_dimension={method.max_dense_dimension}."
        )

    def flat_smooth(candidate):
        return problem.smooth.value(unravel(candidate), args)[0]

    smooth, gradient = jax.value_and_grad(flat_smooth)(flat_parameters)
    hessian = jax.hessian(flat_smooth)(flat_parameters)
    hessian = 0.5 * (hessian + hessian.T)
    eigenvalues = jnp.linalg.eigvalsh(hessian)
    shift = jnp.maximum(method.minimum_curvature - jnp.min(eigenvalues), 0.0)
    model_hessian = hessian + shift * jnp.eye(flat_parameters.size, dtype=hessian.dtype)
    model_lipschitz = jnp.maximum(
        jnp.max(jnp.linalg.eigvalsh(model_hessian)), method.minimum_curvature
    )
    inner_rate = 1.0 / model_lipschitz

    def inner_body(_, candidate_flat):
        model_gradient = gradient + model_hessian @ (candidate_flat - flat_parameters)
        candidate = unravel(candidate_flat)
        model_gradient_tree = unravel(model_gradient)
        next_candidate = problem.nonsmooth.proximal(
            _tree_add_scaled(candidate, model_gradient_tree, -inner_rate),
            inner_rate,
        )
        return ravel_pytree(next_candidate)[0]

    model_candidate = jax.lax.fori_loop(
        0, method.inner_steps, inner_body, flat_parameters
    )
    direction = model_candidate - flat_parameters
    current_nonsmooth = problem.nonsmooth.value(parameters)
    model_nonsmooth = problem.nonsmooth.value(unravel(model_candidate))
    predicted_reduction = -(
        jnp.vdot(gradient, direction).real
        + 0.5 * jnp.vdot(direction, model_hessian @ direction).real
        + model_nonsmooth
        - current_nonsmooth
    )

    def condition(carry):
        evaluations, rate, accepted, *_ = carry
        return (
            (evaluations < method.maximum_backtracking_steps)
            & (~accepted)
            & (rate >= method.minimum_step_size)
            & _within_evaluation_budget(state, evaluations, termination)
        )

    def body(carry):
        (
            evaluations,
            rate,
            _,
            accepted_flat,
            accepted_smooth,
            accepted_nonsmooth,
            accepted_gradient,
        ) = carry
        candidate_flat = flat_parameters + rate * direction
        (candidate_smooth, _), candidate_gradient = problem.smooth.value_and_gradient(
            unravel(candidate_flat), args
        )
        candidate_gradient_flat = ravel_pytree(candidate_gradient)[0]
        candidate_nonsmooth = problem.nonsmooth.value(unravel(candidate_flat))
        actual_reduction = (
            smooth + current_nonsmooth - candidate_smooth - candidate_nonsmooth
        )
        accepted = (
            jnp.all(jnp.isfinite(candidate_flat))
            & jnp.all(jnp.isfinite(candidate_gradient_flat))
            & jnp.isfinite(candidate_smooth)
            & jnp.isfinite(candidate_nonsmooth)
            & jnp.isfinite(predicted_reduction)
            & (predicted_reduction > 0.0)
            & (
                actual_reduction
                >= method.sufficient_decrease * rate * predicted_reduction
            )
        )
        return (
            evaluations + 1,
            jnp.where(accepted, rate, rate * method.contraction),
            accepted,
            jnp.where(accepted, candidate_flat, accepted_flat),
            jnp.where(accepted, candidate_smooth, accepted_smooth),
            jnp.where(accepted, candidate_nonsmooth, accepted_nonsmooth),
            jnp.where(accepted, candidate_gradient_flat, accepted_gradient),
        )

    output = jax.lax.while_loop(
        condition,
        body,
        (
            jnp.asarray(0, dtype=jnp.int32),
            state.step_size,
            jnp.asarray(False),
            flat_parameters,
            smooth,
            current_nonsmooth,
            gradient,
        ),
    )
    (
        evaluations,
        rate,
        accepted,
        candidate_flat,
        candidate_smooth,
        candidate_nonsmooth,
        candidate_gradient,
    ) = output
    return (
        evaluations,
        rate,
        accepted,
        unravel(candidate_flat),
        candidate_smooth,
        candidate_nonsmooth,
        unravel(candidate_gradient),
        smooth,
        unravel(gradient),
        jnp.asarray(flat_parameters.size, dtype=jnp.int32),
    )


def _proximal_step(
    method: AbstractProximalMethod,
    problem: ProximalProblem,
    parameters: PyTree[Any],
    state: ProximalState,
    /,
    *,
    termination: OptimizationTermination,
    args: Any,
):
    converged = state.stationarity <= termination.optimality_threshold(
        state.initial_stationarity
    )

    def finish(_):
        return parameters, eqx.tree_at(
            lambda value: (value.status, value.metrics),
            state,
            (
                jnp.asarray(int(OptimizationStatus.SUCCESS), dtype=jnp.int32),
                IterativeStepMetrics(
                    objective=state.objective,
                    optimality_norm=state.stationarity,
                    accepted=True,
                    status=OptimizationStatus.SUCCESS,
                ),
            ),
        )

    def take_step(_):
        reference = state.extrapolated if method.accelerated else parameters
        if method.proximal_newton:
            proposal = _proximal_newton_proposal(
                method,
                problem,
                parameters,
                state,
                termination,
                args,
            )
            (
                evaluations,
                rate,
                accepted,
                candidate,
                candidate_smooth,
                candidate_nonsmooth,
                candidate_gradient,
                _,
                _,
                hessian_actions,
            ) = proposal
            restart = jnp.asarray(False)
            reference_evaluations = jnp.asarray(1, dtype=jnp.int32)
        else:
            proposal = _first_order_proposal(
                method,
                problem,
                reference,
                state,
                termination,
                args,
            )
            (
                first_evaluations,
                first_rate,
                first_accepted,
                first_candidate,
                first_smooth,
                first_nonsmooth,
                first_gradient,
                _,
                _,
            ) = proposal
            first_objective = first_smooth + first_nonsmooth
            restart = (
                jnp.asarray(method.accelerated)
                & first_accepted
                & jnp.isfinite(state.objective)
                & (first_objective > state.objective)
            )
            restart_state = eqx.tree_at(
                lambda value: (
                    value.objective_evaluations,
                    value.gradient_evaluations,
                ),
                state,
                (
                    state.objective_evaluations + 1 + first_evaluations,
                    state.gradient_evaluations + 1 + first_evaluations,
                ),
            )

            def recompute_from_accepted(_):
                (
                    retry_evaluations,
                    retry_rate,
                    retry_accepted,
                    retry_candidate,
                    retry_smooth,
                    retry_nonsmooth,
                    retry_gradient,
                    _,
                    _,
                ) = _first_order_proposal(
                    method,
                    problem,
                    parameters,
                    restart_state,
                    termination,
                    args,
                )
                retry_objective = retry_smooth + retry_nonsmooth
                retry_accepted = retry_accepted & (retry_objective <= state.objective)
                return (
                    first_evaluations + retry_evaluations,
                    retry_rate,
                    retry_accepted,
                    retry_candidate,
                    retry_smooth,
                    retry_nonsmooth,
                    retry_gradient,
                    jnp.asarray(2, dtype=jnp.int32),
                )

            def keep_extrapolated(_):
                return (
                    first_evaluations,
                    first_rate,
                    first_accepted,
                    first_candidate,
                    first_smooth,
                    first_nonsmooth,
                    first_gradient,
                    jnp.asarray(1, dtype=jnp.int32),
                )

            (
                evaluations,
                rate,
                accepted,
                candidate,
                candidate_smooth,
                candidate_nonsmooth,
                candidate_gradient,
                reference_evaluations,
            ) = jax.lax.cond(
                restart,
                recompute_from_accepted,
                keep_extrapolated,
                None,
            )
            hessian_actions = jnp.asarray(0, dtype=jnp.int32)
        candidate_objective = candidate_smooth + candidate_nonsmooth
        stationarity = problem.stationarity(candidate, candidate_gradient, rate)
        stationarity = jnp.where(accepted, stationarity, state.stationarity)
        step = _tree_subtract(candidate, parameters)
        step_norm = _tree_norm(step)
        next_momentum = 0.5 * (
            1.0 + jnp.sqrt(1.0 + 4.0 * state.momentum * state.momentum)
        )
        momentum = jnp.where(
            method.accelerated & accepted & (~restart),
            next_momentum,
            1.0,
        )
        factor = jnp.where(
            method.accelerated & accepted & (~restart),
            (state.momentum - 1.0) / next_momentum,
            0.0,
        )
        extrapolated = _tree_add_scaled(candidate, step, factor)
        extrapolated = jax.tree.map(
            lambda new, old: jnp.where(accepted, new, old),
            extrapolated,
            state.extrapolated,
        )
        stagnated = (
            accepted
            & (step_norm <= termination.step_threshold(_tree_norm(candidate)))
            & (
                stationarity
                > termination.optimality_threshold(state.initial_stationarity)
            )
        )
        budget_exhausted = (
            jnp.asarray(False)
            if termination.maximum_evaluations is None
            else state.objective_evaluations + reference_evaluations + evaluations
            >= termination.maximum_evaluations
        )
        status = jnp.where(
            stagnated,
            int(OptimizationStatus.STAGNATION),
            jnp.where(
                accepted,
                jnp.where(
                    budget_exhausted,
                    int(OptimizationStatus.MAXIMUM_EVALUATIONS_REACHED),
                    int(OptimizationStatus.ITERATING),
                ),
                jnp.where(
                    budget_exhausted,
                    int(OptimizationStatus.MAXIMUM_EVALUATIONS_REACHED),
                    int(OptimizationStatus.LINE_SEARCH_FAILED),
                ),
            ),
        )
        objective = jnp.where(accepted, candidate_objective, state.objective)
        next_parameters = jax.tree.map(
            lambda new, old: jnp.where(accepted, new, old),
            candidate,
            parameters,
        )
        updated = ProximalState(
            iteration=state.iteration + 1,
            extrapolated=extrapolated,
            momentum=momentum,
            step_size=jnp.where(accepted, rate, state.step_size),
            initial_stationarity=state.initial_stationarity,
            stationarity=stationarity,
            objective=objective,
            accepted_steps=state.accepted_steps + accepted.astype(jnp.int32),
            rejected_steps=state.rejected_steps + (~accepted).astype(jnp.int32),
            objective_evaluations=(
                state.objective_evaluations + reference_evaluations + evaluations
            ),
            gradient_evaluations=(
                state.gradient_evaluations + reference_evaluations + evaluations
            ),
            hvp_evaluations=state.hvp_evaluations + hessian_actions,
            globalization_evaluations=state.globalization_evaluations + evaluations,
            final_step_norm=jnp.where(accepted, step_norm, state.final_step_norm),
            accepted_rate=jnp.where(accepted, rate, 0.0),
            status=status,
            metrics=IterativeStepMetrics(
                objective=objective,
                optimality_norm=stationarity,
                step_norm=step_norm,
                accepted_step_size=jnp.where(accepted, rate, 0.0),
                globalization_evaluations=evaluations,
                accepted=accepted,
                damping=jnp.where(method.proximal_newton, rate, 0.0),
                direction_fallback=restart & accepted,
                status=status,
            ),
        )
        return next_parameters, updated

    return jax.lax.cond(converged, finish, take_step, None)


def _solve_proximal(
    method: AbstractProximalMethod,
    problem: ProximalProblem,
    initial_parameters: PyTree[Any],
    /,
    *,
    termination: OptimizationTermination,
    args: Any,
) -> ProximalResult:
    if not isinstance(problem, ProximalProblem):
        raise TypeError("problem must be a ProximalProblem.")
    if not isinstance(termination, OptimizationTermination):
        raise TypeError("termination must be an OptimizationTermination.")
    parameters = _validate_real_inexact_tree(
        initial_parameters, name="initial_parameters"
    )
    state = method.prepare_state(problem, parameters, args=args)
    dynamic_state, static_state = eqx.partition(state, eqx.is_array)

    def condition(carry):
        _, current = carry
        within_evaluations = (
            jnp.asarray(True)
            if termination.maximum_evaluations is None
            else current.objective_evaluations < termination.maximum_evaluations
        )
        return (
            (current.status == int(OptimizationStatus.ITERATING))
            & (current.iteration < termination.maximum_steps)
            & within_evaluations
        )

    def body(carry):
        current_parameters, dynamic = carry
        current_state = eqx.combine(dynamic, static_state)
        next_parameters, next_state = method.step(
            problem,
            current_parameters,
            current_state,
            termination=termination,
            args=args,
        )
        dynamic_next, _ = eqx.partition(next_state, eqx.is_array)
        return next_parameters, dynamic_next

    parameters, dynamic_state = jax.lax.while_loop(
        condition,
        body,
        (parameters, dynamic_state),
    )
    state = eqx.combine(dynamic_state, static_state)
    exhausted = (
        int(OptimizationStatus.MAXIMUM_STEPS_REACHED)
        if termination.maximum_evaluations is None
        else jnp.where(
            state.objective_evaluations >= termination.maximum_evaluations,
            int(OptimizationStatus.MAXIMUM_EVALUATIONS_REACHED),
            int(OptimizationStatus.MAXIMUM_STEPS_REACHED),
        )
    )
    status = jnp.where(
        state.status == int(OptimizationStatus.ITERATING),
        exhausted,
        state.status,
    )
    (smooth, auxiliary), gradient = problem.value_and_gradient(parameters, args)
    nonsmooth = problem.nonsmooth.value(parameters)
    stationarity = problem.stationarity(parameters, gradient, state.step_size)
    finite = (
        _tree_allfinite(parameters)
        & _tree_allfinite(gradient)
        & jnp.isfinite(smooth)
        & jnp.isfinite(nonsmooth)
        & jnp.isfinite(stationarity)
    )
    success_eligible = (
        (status == int(OptimizationStatus.ITERATING))
        | (status == int(OptimizationStatus.MAXIMUM_STEPS_REACHED))
        | (status == int(OptimizationStatus.MAXIMUM_EVALUATIONS_REACHED))
        | (status == int(OptimizationStatus.STAGNATION))
    )
    status = jnp.where(
        ~finite,
        jnp.where(
            status == int(OptimizationStatus.NONFINITE_INPUT),
            status,
            int(OptimizationStatus.NONFINITE_EVALUATION),
        ),
        jnp.where(
            success_eligible
            & (
                stationarity
                <= termination.optimality_threshold(state.initial_stationarity)
            ),
            int(OptimizationStatus.SUCCESS),
            status,
        ),
    )
    diagnostics = OptimizationDiagnostics(
        iterations=state.iteration,
        accepted_steps=state.accepted_steps,
        rejected_steps=state.rejected_steps,
        objective_evaluations=state.objective_evaluations + 1,
        gradient_evaluations=state.gradient_evaluations + 1,
        hvp_evaluations=state.hvp_evaluations,
        globalization_evaluations=state.globalization_evaluations,
        initial_optimality_norm=state.initial_stationarity,
        final_optimality_norm=stationarity,
        final_step_norm=state.final_step_norm,
        accepted_step_size=state.accepted_rate,
        damping=jnp.where(method.proximal_newton, state.step_size, 0.0),
    )
    return ProximalResult(
        parameters=parameters,
        objective=smooth + nonsmooth,
        smooth_objective=smooth,
        nonsmooth_objective=nonsmooth,
        auxiliary=auxiliary,
        composite_stationarity=stationarity,
        status=status,
        diagnostics=diagnostics,
        provenance=OptimizationProvenance(
            problem_id=problem.problem_id,
            method=method.method_id,
            backend="phydrax-native",
            globalization=(
                "composite-armijo" if method.proximal_newton else "smooth-majorization"
            ),
            matrix_free=not method.proximal_newton,
            implicit_differentiation=False,
            notes="final_optimality_norm is the composite proximal-gradient mapping norm.",
        ),
    )


def proximal_minimize(
    problem_or_smooth: ProximalProblem | Callable[[PyTree[Any], Any], Any],
    initial_parameters: PyTree[Any],
    /,
    *,
    nonsmooth: AbstractProximalFunctional | None = None,
    method: AbstractProximalMethod | None = None,
    termination: OptimizationTermination | None = None,
    args: Any = None,
    has_aux: bool = False,
) -> ProximalResult:
    """Minimize one smooth-plus-proximable composite objective."""

    if isinstance(problem_or_smooth, ProximalProblem):
        if nonsmooth is not None:
            raise ValueError(
                "nonsmooth must be omitted when a ProximalProblem is supplied."
            )
        problem = problem_or_smooth
    else:
        if nonsmooth is None:
            raise TypeError("nonsmooth is required with a smooth callable.")
        problem = ProximalProblem(
            problem_or_smooth,
            nonsmooth,
            has_aux=has_aux,
        )
    method_ = ProximalGradient() if method is None else method
    termination_ = OptimizationTermination() if termination is None else termination
    if not isinstance(method_, AbstractProximalMethod):
        raise TypeError("method must be an AbstractProximalMethod or None.")
    return method_.solve(
        problem,
        initial_parameters,
        termination=termination_,
        args=args,
    )


__all__ = [
    "AbstractProximalFunctional",
    "AbstractProximalMethod",
    "AcceleratedProximalGradient",
    "BoxIndicator",
    "ElasticNetFunctional",
    "GroupLassoFunctional",
    "IndicatorFunctional",
    "L1Functional",
    "NuclearNormFunctional",
    "ProximalGradient",
    "ProximalNewton",
    "ProximalProblem",
    "ProximalResult",
    "ProximalState",
    "SimplexIndicator",
    "proximal_minimize",
]
