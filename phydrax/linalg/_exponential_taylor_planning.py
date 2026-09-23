#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Operator-only preparation and bounded coordinate 1-norm evidence for Taylor actions."""

from __future__ import annotations

from numbers import Integral
from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from ._assembly import assemble_diagonal
from ._operators import (
    AbstractLinearOperator,
    DenseLinearOperator,
    DiagonalLinearOperator,
    IdentityLinearOperator,
)
from ._policies import DifferentiationPolicy, FailurePolicy


class TaylorExponentialResourcePolicy(StrictModule):
    """Finite setup, action, and memory budgets; all capacities are host-static."""

    max_degree: int = eqx.field(static=True)
    max_power: int = eqx.field(static=True)
    max_scaling_count: int = eqx.field(static=True)
    max_setup_matvec_count: int = eqx.field(static=True)
    max_action_matvec_count: int = eqx.field(static=True)
    block_size: int = eqx.field(static=True)
    estimator_iterations: int = eqx.field(static=True)
    estimator_retries: int = eqx.field(static=True)
    max_retained_storage_bytes: int = eqx.field(static=True)
    max_workspace_bytes: int = eqx.field(static=True)

    def __init__(
        self,
        *,
        max_degree: int = 55,
        max_power: int = 8,
        max_scaling_count: int = 1024,
        max_setup_matvec_count: int = 4096,
        max_action_matvec_count: int = 65536,
        block_size: int = 4,
        estimator_iterations: int = 5,
        estimator_retries: int = 2,
        max_retained_storage_bytes: int = 64 * 1024 * 1024,
        max_workspace_bytes: int = 512 * 1024 * 1024,
    ):
        values = {
            "max_degree": max_degree,
            "max_power": max_power,
            "max_scaling_count": max_scaling_count,
            "max_setup_matvec_count": max_setup_matvec_count,
            "max_action_matvec_count": max_action_matvec_count,
            "block_size": block_size,
            "estimator_iterations": estimator_iterations,
            "estimator_retries": estimator_retries,
            "max_retained_storage_bytes": max_retained_storage_bytes,
            "max_workspace_bytes": max_workspace_bytes,
        }
        for name, value in values.items():
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise TypeError(f"{name} must be a host integer.")
            if value < 1:
                raise ValueError(f"{name} must be positive.")
        if max_degree > 55:
            raise ValueError("max_degree exceeds the available Taylor threshold table.")
        if max_power > 8:
            raise ValueError("max_power exceeds the supported alpha_p degree range.")
        int_limit = np.iinfo(np.int32).max
        if any(value > int_limit for value in values.values()):
            raise ValueError("Taylor resources must fit int32 JAX execution counters.")
        if 2 * max_scaling_count * (max_degree + 1) > int_limit:
            raise ValueError("Taylor action capacity exceeds int32 JAX work counters.")
        setup_work = (
            2
            * block_size
            * estimator_iterations
            * estimator_retries
            * sum(range(1, max_power + 2))
        )
        if setup_work > int_limit:
            raise ValueError("Taylor setup capacity exceeds int32 JAX work counters.")
        for name, value in values.items():
            setattr(self, name, value)


class TaylorExponentialPolicy(StrictModule):
    """Accuracy and norm-evidence choice for the scaled Taylor exponential."""

    error_tolerance: float = eqx.field(static=True)
    norm_mode: Literal["auto", "exact", "estimate"] = eqx.field(static=True)
    resources: TaylorExponentialResourcePolicy = eqx.field(static=True)
    differentiation: DifferentiationPolicy
    failure: FailurePolicy = eqx.field(static=True)

    def __init__(
        self,
        *,
        error_tolerance: float = 1e-8,
        norm_mode: Literal["auto", "exact", "estimate"] = "auto",
        resources: TaylorExponentialResourcePolicy | None = None,
        differentiation: DifferentiationPolicy | None = None,
        failure: FailurePolicy | None = None,
    ):
        if not isinstance(error_tolerance, (int, float, np.floating)) or isinstance(
            error_tolerance, bool
        ):
            raise TypeError("error_tolerance must be a positive host real scalar.")
        tolerance = float(error_tolerance)
        if not np.isfinite(tolerance) or not (2.0**-53 <= tolerance <= 0.5):
            raise ValueError("error_tolerance must be finite and in [2**-53, 0.5].")
        if norm_mode not in ("auto", "exact", "estimate"):
            raise ValueError("norm_mode must be 'auto', 'exact', or 'estimate'.")
        resource_policy = (
            TaylorExponentialResourcePolicy() if resources is None else resources
        )
        differentiation_ = (
            DifferentiationPolicy("algorithmic")
            if differentiation is None
            else differentiation
        )
        failure_ = FailurePolicy() if failure is None else failure
        if not isinstance(resource_policy, TaylorExponentialResourcePolicy):
            raise TypeError(
                "resources must be a TaylorExponentialResourcePolicy or None."
            )
        if not isinstance(differentiation_, DifferentiationPolicy):
            raise TypeError("differentiation must be a DifferentiationPolicy or None.")
        if differentiation_.mode == "mathematical":
            raise ValueError(
                "Taylor exponential actions do not yet provide a mathematical "
                "Frechet derivative rule."
            )
        if not isinstance(failure_, FailurePolicy):
            raise TypeError("failure must be a FailurePolicy or None.")
        self.error_tolerance = tolerance
        self.norm_mode = norm_mode
        self.resources = resource_policy
        self.differentiation = differentiation_
        self.failure = failure_


class TaylorExponentialPlan(StrictModule):
    """Static operator structure and executable resource envelope."""

    policy: TaylorExponentialPolicy = eqx.field(static=True)
    operator_id: str = eqx.field(static=True)
    space_id: str = eqx.field(static=True)
    dimension: int = eqx.field(static=True)
    norm_source: Literal[
        "exact-stored-coordinates", "bounded-sparse-columns", "estimated-block-1-norm"
    ] = eqx.field(static=True)
    trace_source: Literal["exact-diagonal", "none"] = eqx.field(static=True)
    feasible: bool = eqx.field(static=True)
    setup_matvec_count: int = eqx.field(static=True)
    transpose_matvec_count: int = eqx.field(static=True)
    retained_storage_bytes: int = eqx.field(static=True)
    workspace_bytes: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)


class PreparedTaylorExponentialAction(StrictModule):
    """Operator and norm evidence reusable across right-hand sides and scales."""

    operator: AbstractLinearOperator
    norm_one: Array
    trace: Array
    alpha_p: Array
    norm_finite: Array
    numeric_version: Array
    plan: TaylorExponentialPlan = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)


def _validate_operator(operator: AbstractLinearOperator, /) -> None:
    if not isinstance(operator, AbstractLinearOperator):
        raise TypeError("operator must be an AbstractLinearOperator.")
    if operator.batch_shape or not operator.source.compatible(operator.target):
        raise ValueError("Taylor action requires an unbatched endomorphism.")
    if operator.source.size < 1:
        raise ValueError("Taylor action requires a nonempty vector space.")
    dtypes = {
        np.dtype(leaf.dtype) for leaf in jax.tree.leaves(operator.source.structure())
    }
    supported = {
        np.dtype(np.float32),
        np.dtype(np.float64),
        np.dtype(np.complex64),
        np.dtype(np.complex128),
    }
    if len(dtypes) != 1 or not dtypes.issubset(supported):
        raise TypeError(
            "Taylor action requires one float32, float64, complex64, or "
            "complex128 coordinate dtype."
        )


def plan_taylor_exponential_action(
    operator: AbstractLinearOperator,
    policy: TaylorExponentialPolicy | None = None,
    /,
) -> TaylorExponentialPlan:
    """Plan stored-coordinate bounds or bounded matrix-free norm estimation."""
    _validate_operator(operator)
    selected = TaylorExponentialPolicy() if policy is None else policy
    if not isinstance(selected, TaylorExponentialPolicy):
        raise TypeError("policy must be a TaylorExponentialPolicy or None.")
    resource = selected.resources
    n = operator.source.size
    from ..sparse._linear import SparseCoordinateOperator

    stored = isinstance(
        operator, (DenseLinearOperator, DiagonalLinearOperator, IdentityLinearOperator)
    )
    sparse = isinstance(operator, SparseCoordinateOperator)
    if selected.norm_mode == "exact" and not (stored or sparse):
        raise ValueError(
            "Exact norm mode requires stored dense, diagonal, identity, or sparse coordinates."
        )
    use_exact = selected.norm_mode != "estimate" and (stored or sparse)
    if not use_exact and not operator.capabilities.transpose:
        raise ValueError(
            "Estimated coordinate 1-norm requires an algebraic transpose action."
        )
    source = (
        "bounded-sparse-columns"
        if sparse and use_exact
        else "exact-stored-coordinates"
        if use_exact
        else "estimated-block-1-norm"
    )
    trace_source = "exact-diagonal" if operator.capabilities.diagonal_assembly else "none"
    power_work = sum(range(1, resource.max_power + 2))
    setup_count = (
        0
        if use_exact
        else resource.block_size
        * resource.estimator_iterations
        * resource.estimator_retries
        * power_work
    )
    transpose_count = setup_count
    dtype = np.dtype(jax.tree.leaves(operator.source.structure())[0].dtype)
    itemsize = dtype.itemsize
    retained = (resource.max_power + 1) * itemsize + 4
    workspace = itemsize * n * (16 + 8 * resource.block_size)
    if stored and isinstance(operator, DenseLinearOperator):
        workspace += operator.matrix.size * itemsize
    if sparse:
        workspace += operator.coefficients.size * (itemsize + 8) * 2
    feasible = (
        setup_count + transpose_count <= resource.max_setup_matvec_count
        and retained <= resource.max_retained_storage_bytes
        and workspace <= resource.max_workspace_bytes
    )
    identifier = canonical_fingerprint(
        {
            "kind": "taylor-exponential-plan",
            "operator": operator.operator_id,
            "space": operator.source.space_id,
            "dimension": n,
            "norm_source": source,
            "trace_source": trace_source,
            "policy": {
                "tolerance": selected.error_tolerance,
                "norm_mode": selected.norm_mode,
                "resources": {
                    name: getattr(resource, name)
                    for name in (
                        "max_degree",
                        "max_power",
                        "max_scaling_count",
                        "max_setup_matvec_count",
                        "max_action_matvec_count",
                        "block_size",
                        "estimator_iterations",
                        "estimator_retries",
                        "max_retained_storage_bytes",
                        "max_workspace_bytes",
                    )
                },
            },
        }
    )
    return TaylorExponentialPlan(
        policy=selected,
        operator_id=operator.operator_id,
        space_id=operator.source.space_id,
        dimension=n,
        norm_source=source,
        trace_source=trace_source,
        feasible=feasible,
        setup_matvec_count=setup_count,
        transpose_matvec_count=transpose_count,
        retained_storage_bytes=retained,
        workspace_bytes=workspace,
        plan_id=identifier,
    )


def _check_plan(operator: AbstractLinearOperator, plan: TaylorExponentialPlan, /) -> None:
    _validate_operator(operator)
    if not isinstance(plan, TaylorExponentialPlan):
        raise TypeError("plan must be a TaylorExponentialPlan.")
    if (
        operator.operator_id != plan.operator_id
        or operator.source.space_id != plan.space_id
        or operator.source.size != plan.dimension
    ):
        raise ValueError(
            "Operator identity or vector space does not match the Taylor plan."
        )


def _stored_norm_one(operator: AbstractLinearOperator, /) -> Array:
    from ..sparse._linear import SparseCoordinateOperator

    if isinstance(operator, DenseLinearOperator):
        return jnp.max(jnp.sum(jnp.abs(operator.matrix), axis=0))
    if isinstance(operator, DiagonalLinearOperator):
        return jnp.max(jnp.abs(operator.diagonal))
    if isinstance(operator, IdentityLinearOperator):
        dtype = jax.tree.leaves(operator.source.structure())[0].dtype
        return jnp.asarray(1.0, dtype=jnp.real(jnp.zeros((), dtype=dtype)).dtype)
    if isinstance(operator, SparseCoordinateOperator):
        relation, coefficients = operator._edge_form()
        safe_sources = jnp.where(relation.valid, relation.source_indices, 0)
        absolute = jnp.where(relation.valid, jnp.abs(coefficients), 0.0)
        columns = (
            jnp.zeros((relation.source_size,), dtype=absolute.dtype)
            .at[safe_sources]
            .add(absolute)
        )
        # Duplicate routes only enlarge this bound, never silently cancel.
        return jnp.max(columns)
    raise TypeError("Stored norm requires a supported coordinate operator.")


def _estimated_power_norm_one(
    operator: AbstractLinearOperator,
    plan: TaylorExponentialPlan,
    key: Array,
    power: int,
    /,
) -> Array:
    resource = plan.policy.resources
    n, width = plan.dimension, resource.block_size
    dtype = jax.tree.leaves(operator.source.structure())[0].dtype
    real_dtype = jnp.real(jnp.zeros((), dtype=dtype)).dtype
    real_one = jnp.asarray(1.0, dtype=real_dtype)

    def attempt(retry: int, best: Array) -> Array:
        sample_key = jax.random.fold_in(jax.random.fold_in(key, power), retry)
        probes = (
            jax.random.rademacher(sample_key, (n, width), dtype=real_dtype).astype(dtype)
            / n
        )

        def iteration(_: int, state: tuple[Array, Array]) -> tuple[Array, Array]:
            vectors, maximum = state
            images = jax.lax.fori_loop(
                0, power, lambda _, x: operator.mv_block(x), vectors
            )
            maximum = jnp.maximum(maximum, jnp.max(jnp.sum(jnp.abs(images), axis=0)))
            magnitudes = jnp.abs(images)
            phases = images / jnp.where(magnitudes > 0, magnitudes, real_one)
            # Coordinate A^H, not the possibly non-Euclidean pairing adjoint.
            dual = jnp.conj(
                jax.lax.fori_loop(
                    0,
                    power,
                    lambda _, x: operator.transpose_mv_block(x),
                    jnp.conj(phases),
                )
            )
            columns = jnp.argmax(jnp.abs(dual), axis=0)
            basis = jax.nn.one_hot(columns, n, dtype=dtype).T
            return basis, maximum

        _, estimate = jax.lax.fori_loop(
            0,
            resource.estimator_iterations,
            iteration,
            (probes, jnp.asarray(0.0, dtype=real_dtype)),
        )
        return jnp.maximum(best, estimate)

    return jax.lax.fori_loop(
        0,
        resource.estimator_retries,
        attempt,
        jnp.asarray(0.0, dtype=real_dtype),
    )


def _prepare(
    operator: AbstractLinearOperator,
    plan: TaylorExponentialPlan,
    key: Array | None,
    numeric_version: Array,
    prepared_id: str,
    /,
) -> PreparedTaylorExponentialAction:
    if plan.norm_source == "estimated-block-1-norm" and key is None:
        raise ValueError(
            "Estimated Taylor norm planning requires an explicit JAX PRNG key."
        )
    if key is None:
        selected_key = None
    else:
        selected_key = jnp.asarray(key)
        typed_key = jax.dtypes.issubdtype(selected_key.dtype, jax.dtypes.prng_key)
        legacy_key = selected_key.shape == (2,) and selected_key.dtype == jnp.uint32
        if not ((typed_key and selected_key.shape == ()) or legacy_key):
            raise TypeError("key must be a typed JAX PRNG key or legacy uint32[2] key.")
    if plan.feasible:
        if plan.norm_source == "estimated-block-1-norm":
            assert selected_key is not None
            powers = jnp.stack(
                tuple(
                    _estimated_power_norm_one(operator, plan, selected_key, power)
                    for power in range(1, plan.policy.resources.max_power + 2)
                )
            )
            indices = jnp.arange(
                1, plan.policy.resources.max_power + 2, dtype=powers.dtype
            )
            roots = powers ** (1.0 / indices)
            norm = roots[0]
            alpha_p = jnp.maximum(roots[1:-1], roots[2:])
        else:
            norm = _stored_norm_one(operator)
            alpha_p = jnp.empty((0,), dtype=norm.dtype)
    else:
        norm = jnp.asarray(jnp.inf, dtype=jnp.float32)
        alpha_p = jnp.empty((0,), dtype=norm.dtype)
    coordinate_dtype = jax.tree.leaves(operator.source.structure())[0].dtype
    trace = (
        jnp.sum(assemble_diagonal(operator))
        if plan.feasible and plan.trace_source == "exact-diagonal"
        else jnp.asarray(0.0, dtype=coordinate_dtype)
    )
    finite = jnp.isfinite(norm) & jnp.all(jnp.isfinite(alpha_p))
    return PreparedTaylorExponentialAction(
        operator=operator,
        norm_one=jax.lax.stop_gradient(norm),
        alpha_p=jax.lax.stop_gradient(alpha_p),
        trace=jax.lax.stop_gradient(trace),
        norm_finite=finite & jnp.isfinite(trace),
        numeric_version=numeric_version,
        plan=plan,
        prepared_id=prepared_id,
    )


def prepare_taylor_exponential_action(
    operator: AbstractLinearOperator,
    policy_or_plan: TaylorExponentialPolicy | TaylorExponentialPlan | None = None,
    /,
    *,
    key: Array | None = None,
) -> PreparedTaylorExponentialAction:
    """Bind one operator and its finite-work norm evidence, without an RHS."""
    plan = (
        policy_or_plan
        if isinstance(policy_or_plan, TaylorExponentialPlan)
        else plan_taylor_exponential_action(operator, policy_or_plan)
    )
    _check_plan(operator, plan)
    identifier = canonical_fingerprint(
        {
            "kind": "prepared-taylor-exponential",
            "plan": plan.plan_id,
            "operator": operator.operator_id,
        }
    )
    return _prepare(operator, plan, key, jnp.asarray(0, dtype=jnp.int32), identifier)


def refresh_taylor_exponential_action(
    prepared: PreparedTaylorExponentialAction,
    operator: AbstractLinearOperator,
    /,
    *,
    key: Array | None = None,
) -> PreparedTaylorExponentialAction:
    """Rebind numerical operator coefficients while retaining structural identity."""
    if not isinstance(prepared, PreparedTaylorExponentialAction):
        raise TypeError("prepared must be a PreparedTaylorExponentialAction.")
    _check_plan(operator, prepared.plan)
    return _prepare(
        operator,
        prepared.plan,
        key,
        prepared.numeric_version + jnp.asarray(1, dtype=jnp.int32),
        prepared.prepared_id,
    )
