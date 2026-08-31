#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import core as jax_core
from jaxtyping import Array, PyTree

from .._strict import StrictModule
from ._method import AbstractLinearCombinatorialMethod, solve_combinatorial
from ._problem import AbstractCombinatorialSpace, LinearCombinatorialProblem
from ._types import (
    CombinatorialCertification,
    CombinatorialResult,
)


class BlackboxInterpolation(StrictModule):
    """Loss-dependent first-order surrogate pullback through a hard oracle."""

    lambda_: Array
    certification: CombinatorialCertification

    def __init__(
        self,
        lambda_: Any,
        /,
        *,
        certification: CombinatorialCertification | None = None,
    ):
        scale = jnp.asarray(lambda_, dtype=float)
        if scale.ndim != 0:
            raise ValueError("blackbox interpolation lambda_ must be scalar.")
        invalid = ~jnp.isfinite(scale) | (scale <= 0.0)
        if isinstance(invalid, jax_core.Tracer):
            scale = eqx.error_if(
                scale,
                invalid,
                "blackbox interpolation lambda_ must be finite and positive.",
            )
        elif bool(invalid):
            raise ValueError(
                "blackbox interpolation lambda_ must be finite and positive."
            )
        selected = (
            CombinatorialCertification() if certification is None else certification
        )
        if not isinstance(selected, CombinatorialCertification):
            raise TypeError("certification must be a CombinatorialCertification or None.")
        self.lambda_ = scale
        self.certification = selected


class BlackboxPullbackResult(StrictModule):
    """Surrogate cost gradient and complete forward/perturbed solve evidence."""

    gradient: PyTree[Array]
    forward: CombinatorialResult
    perturbed: CombinatorialResult
    lambda_: Array
    cost_norm: Array
    perturbation_norm: Array
    relative_perturbation: Array
    feature_change_norm: Array
    zero_gradient: Array
    exact_theory_applicable: Array
    valid: Array


def _validate_method(method: AbstractLinearCombinatorialMethod, /) -> None:
    if not isinstance(method, AbstractLinearCombinatorialMethod):
        raise TypeError("method must be an AbstractLinearCombinatorialMethod.")
    capabilities = method.capabilities
    if not capabilities.exact:
        raise ValueError("blackbox interpolation requires an exact combinatorial method.")
    if not capabilities.deterministic_ties:
        raise ValueError("blackbox interpolation requires deterministic tie handling.")
    if not capabilities.signed_costs:
        raise ValueError("additive blackbox interpolation requires signed-cost support.")
    if not capabilities.surrogate_pullback:
        raise ValueError("method does not declare surrogate-pullback compatibility.")


def _validated_cotangent(
    cotangent: PyTree[Any],
    features: PyTree[Array],
    /,
) -> PyTree[Array]:
    raw_leaves, raw_tree = jax.tree_util.tree_flatten(cotangent)
    feature_leaves, feature_tree = jax.tree_util.tree_flatten(features)
    if raw_tree != feature_tree:
        raise ValueError(
            "cotangent and objective features must share one PyTree structure."
        )
    arrays: list[Array] = []
    for raw, feature in zip(raw_leaves, feature_leaves, strict=True):
        value = jnp.asarray(raw, dtype=feature.dtype)
        if value.shape != feature.shape:
            raise ValueError(
                f"cotangent leaf must have shape {feature.shape}; got {value.shape}."
            )
        arrays.append(value)
    return feature_tree.unflatten(arrays)


def _tree_norm(
    tree: PyTree[Array],
    feature_spec: PyTree[jax.ShapeDtypeStruct],
    batch_shape: tuple[int, ...],
    /,
) -> Array:
    leaves, tree_definition = jax.tree_util.tree_flatten(tree)
    specs, spec_definition = jax.tree_util.tree_flatten(feature_spec)
    if tree_definition != spec_definition:
        raise ValueError("norm inputs must preserve the objective feature structure.")
    squared: Array | None = None
    for value, spec in zip(leaves, specs, strict=True):
        feature_rank = len(spec.shape)
        contribution = jnp.abs(value) ** 2
        if feature_rank:
            contribution = jnp.sum(
                contribution,
                axis=tuple(range(-feature_rank, 0)),
            )
        squared = contribution if squared is None else squared + contribution
    if squared is None:
        return jnp.zeros(batch_shape)
    return jnp.sqrt(squared)


def _masked_tree(
    tree: PyTree[Array],
    valid: Array,
    batch_shape: tuple[int, ...],
    /,
) -> PyTree[Array]:
    return jax.tree_util.tree_map(
        lambda value: jnp.where(
            valid.reshape(batch_shape + (1,) * (value.ndim - len(batch_shape))),
            value,
            jnp.zeros_like(value),
        ),
        tree,
    )


def _perturbed_problem(
    problem: LinearCombinatorialProblem,
    cotangent: PyTree[Array],
    policy: BlackboxInterpolation,
    /,
) -> LinearCombinatorialProblem:
    costs = jax.tree_util.tree_map(
        lambda cost, gradient: cost + policy.lambda_.astype(cost.dtype) * gradient,
        problem.costs,
        cotangent,
    )
    return problem.with_costs(costs)


def estimate_blackbox_pullback(
    problem: LinearCombinatorialProblem,
    method: AbstractLinearCombinatorialMethod,
    cotangent: PyTree[Any],
    /,
    *,
    policy: BlackboxInterpolation,
) -> BlackboxPullbackResult:
    """Evaluate the explicit one-extra-solve blackbox surrogate pullback."""

    if not isinstance(problem, LinearCombinatorialProblem):
        raise TypeError("problem must be a LinearCombinatorialProblem.")
    if not isinstance(policy, BlackboxInterpolation):
        raise TypeError("policy must be a BlackboxInterpolation.")
    _validate_method(method)
    forward = solve_combinatorial(
        problem,
        method,
        certification=policy.certification,
    )
    gradient_output = _validated_cotangent(cotangent, forward.features)
    perturbed_problem = _perturbed_problem(problem, gradient_output, policy)
    perturbed = solve_combinatorial(
        perturbed_problem,
        method,
        certification=policy.certification,
    )
    raw_gradient = jax.tree_util.tree_map(
        lambda changed, original: (
            (changed - original) / policy.lambda_.astype(changed.dtype)
        ),
        perturbed.features,
        forward.features,
    )
    cotangent_finite = jnp.ones(problem.batch_shape, dtype=bool)
    for value, spec in zip(
        jax.tree_util.tree_leaves(gradient_output),
        jax.tree_util.tree_leaves(problem.space.feature_spec()),
        strict=True,
    ):
        finite = jnp.isfinite(value)
        if len(spec.shape):
            finite = jnp.all(finite, axis=tuple(range(-len(spec.shape), 0)))
        cotangent_finite = cotangent_finite & finite
    exact = forward.success & perturbed.success & cotangent_finite
    gradient = _masked_tree(raw_gradient, exact, problem.batch_shape)
    feature_spec = problem.space.feature_spec()
    cost_norm = _tree_norm(problem.costs, feature_spec, problem.batch_shape)
    perturbation = jax.tree_util.tree_map(
        lambda value: policy.lambda_.astype(value.dtype) * value,
        gradient_output,
    )
    perturbation_norm = _tree_norm(
        perturbation,
        feature_spec,
        problem.batch_shape,
    )
    feature_change = jax.tree_util.tree_map(
        lambda changed, original: changed - original,
        perturbed.features,
        forward.features,
    )
    feature_change_norm = _tree_norm(
        feature_change,
        feature_spec,
        problem.batch_shape,
    )
    epsilon = jnp.finfo(cost_norm.dtype).eps
    relative = perturbation_norm / jnp.maximum(cost_norm, epsilon)
    zero_gradient = _tree_norm(gradient, feature_spec, problem.batch_shape) == 0.0
    return BlackboxPullbackResult(
        gradient=gradient,
        forward=forward,
        perturbed=perturbed,
        lambda_=policy.lambda_,
        cost_norm=cost_norm,
        perturbation_norm=perturbation_norm,
        relative_perturbation=relative,
        feature_change_norm=feature_change_norm,
        zero_gradient=zero_gradient,
        exact_theory_applicable=exact,
        valid=exact,
    )


def _require_exact_gradient(
    gradient: PyTree[Array],
    valid: Array,
    /,
) -> PyTree[Array]:
    leaves, tree_definition = jax.tree_util.tree_flatten(gradient)
    if not leaves:
        raise ValueError("blackbox gradients must contain at least one array leaf.")
    first = eqx.error_if(
        leaves[0],
        ~jnp.all(valid),
        "blackbox interpolation requires certified forward and perturbed optima.",
    )
    return tree_definition.unflatten((first, *leaves[1:]))


@eqx.filter_custom_vjp
def _blackbox_features(
    costs: PyTree[Array],
    space: AbstractCombinatorialSpace,
    method: AbstractLinearCombinatorialMethod,
    policy: BlackboxInterpolation,
    problem_id: str,
    /,
) -> PyTree[Array]:
    problem = LinearCombinatorialProblem(space, costs, problem_id=problem_id)
    result = solve_combinatorial(
        problem,
        method,
        certification=policy.certification,
    )
    return _require_exact_gradient(result.features, result.success)


@_blackbox_features.def_fwd
def _blackbox_features_forward(
    perturbed: PyTree[bool],
    costs: PyTree[Array],
    space: AbstractCombinatorialSpace,
    method: AbstractLinearCombinatorialMethod,
    policy: BlackboxInterpolation,
    problem_id: str,
    /,
) -> tuple[PyTree[Array], tuple[PyTree[Array], Array]]:
    del perturbed
    problem = LinearCombinatorialProblem(space, costs, problem_id=problem_id)
    result = solve_combinatorial(
        problem,
        method,
        certification=policy.certification,
    )
    features = _require_exact_gradient(result.features, result.success)
    return features, (features, result.success)


@_blackbox_features.def_bwd
def _blackbox_features_backward(
    residuals: tuple[PyTree[Array], Array],
    grad_obj: PyTree[Array | None],
    perturbed: PyTree[bool],
    costs: PyTree[Array],
    space: AbstractCombinatorialSpace,
    method: AbstractLinearCombinatorialMethod,
    policy: BlackboxInterpolation,
    problem_id: str,
    /,
) -> PyTree[Array | None]:
    del perturbed
    forward_features, forward_success = residuals
    cotangent = jax.tree_util.tree_map(
        lambda gradient, feature: (
            jnp.zeros_like(feature) if gradient is None else gradient
        ),
        grad_obj,
        forward_features,
    )
    problem = LinearCombinatorialProblem(space, costs, problem_id=problem_id)
    perturbed_problem = _perturbed_problem(problem, cotangent, policy)
    changed = solve_combinatorial(
        perturbed_problem,
        method,
        certification=policy.certification,
    )
    gradient = jax.tree_util.tree_map(
        lambda changed_feature, original: (
            (changed_feature - original) / policy.lambda_.astype(changed_feature.dtype)
        ),
        changed.features,
        forward_features,
    )
    return _require_exact_gradient(gradient, forward_success & changed.success)


def blackbox_solution(
    problem: LinearCombinatorialProblem,
    method: AbstractLinearCombinatorialMethod,
    /,
    *,
    policy: BlackboxInterpolation,
) -> PyTree[Array]:
    """Return hard objective features with a first-order DBB surrogate pullback."""

    if not isinstance(problem, LinearCombinatorialProblem):
        raise TypeError("problem must be a LinearCombinatorialProblem.")
    if not isinstance(policy, BlackboxInterpolation):
        raise TypeError("policy must be a BlackboxInterpolation.")
    _validate_method(method)
    return _blackbox_features(
        problem.costs,
        problem.space,
        method,
        policy,
        problem.problem_id,
    )


__all__ = [
    "BlackboxInterpolation",
    "BlackboxPullbackResult",
    "blackbox_solution",
    "estimate_blackbox_pullback",
]
