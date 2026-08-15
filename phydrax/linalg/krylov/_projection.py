#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from typing import Any, Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, PyTree

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from .._certificates import _operator_numeric_fingerprint
from .._operators import AbstractLinearOperator, estimate_operator_action_cost
from .._spaces import _coordinate_dtype
from ._decompositions import arnoldi, lanczos, Orthogonalization
from ._results import KrylovDecomposition


KrylovProjectionMethod: TypeAlias = Literal["auto", "arnoldi", "lanczos"]


class KrylovProjectionResourcePolicy(StrictModule):
    """Optional hard budgets for one reusable Krylov projection."""

    max_matvec_count: int | None = eqx.field(static=True)
    max_storage_bytes: int | None = eqx.field(static=True)
    max_workspace_bytes: int | None = eqx.field(static=True)

    def __init__(
        self,
        *,
        max_matvec_count: int | None = None,
        max_storage_bytes: int | None = None,
        max_workspace_bytes: int | None = None,
    ):
        self.max_matvec_count = _optional_nonnegative_int(
            max_matvec_count, "max_matvec_count"
        )
        self.max_storage_bytes = _optional_nonnegative_int(
            max_storage_bytes, "max_storage_bytes"
        )
        self.max_workspace_bytes = _optional_nonnegative_int(
            max_workspace_bytes, "max_workspace_bytes"
        )


class KrylovProjectionPolicy(StrictModule):
    """Algorithmic and resource policy for a bound Krylov basis."""

    method: KrylovProjectionMethod = eqx.field(static=True)
    max_dimension: int = eqx.field(static=True)
    orthogonalization: Orthogonalization = eqx.field(static=True)
    breakdown_tolerance: float | None = eqx.field(static=True)
    resources: KrylovProjectionResourcePolicy = eqx.field(static=True)

    def __init__(
        self,
        method: KrylovProjectionMethod = "auto",
        /,
        *,
        max_dimension: int = 32,
        orthogonalization: Orthogonalization = "selective",
        breakdown_tolerance: float | None = None,
        resources: KrylovProjectionResourcePolicy | None = None,
    ):
        if method not in ("auto", "arnoldi", "lanczos"):
            raise ValueError("Unknown Krylov projection method.")
        dimension = int(max_dimension)
        if dimension < 1:
            raise ValueError("max_dimension must be positive.")
        if orthogonalization not in ("modified", "double", "selective", "full"):
            raise ValueError("Unknown orthogonalization policy.")
        if breakdown_tolerance is not None:
            tolerance = float(breakdown_tolerance)
            if not math.isfinite(tolerance) or tolerance < 0.0:
                raise ValueError("breakdown_tolerance must be finite and non-negative.")
        else:
            tolerance = None
        if resources is None:
            resources = KrylovProjectionResourcePolicy()
        if not isinstance(resources, KrylovProjectionResourcePolicy):
            raise TypeError("resources must be a KrylovProjectionResourcePolicy or None.")
        self.method = method
        self.max_dimension = dimension
        self.orthogonalization = orthogonalization
        self.breakdown_tolerance = tolerance
        self.resources = resources


class KrylovProjectionCostEstimate(StrictModule):
    """Conservative setup and retained-state cost for one projection."""

    method: str = eqx.field(static=True)
    dimension: int = eqx.field(static=True)
    matvec_count: int = eqx.field(static=True)
    storage_bytes: int = eqx.field(static=True)
    workspace_bytes: int = eqx.field(static=True)
    operator_action_workspace_bytes: int = eqx.field(static=True)
    exact: bool = eqx.field(static=True)


class KrylovProjectionPlan(StrictModule):
    """Immutable symbolic plan for a fixed-capacity Krylov projection."""

    policy: KrylovProjectionPolicy = eqx.field(static=True)
    selected_method: str = eqx.field(static=True)
    dimension: int = eqx.field(static=True)
    cost: KrylovProjectionCostEstimate = eqx.field(static=True)
    operator_id: str = eqx.field(static=True)
    source_space_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)


class PreparedKrylovProjection(StrictModule):
    """Numerically bound, reusable Krylov basis and projected operator."""

    operator: AbstractLinearOperator
    initial_coordinates: Array
    decomposition: KrylovDecomposition
    plan: KrylovProjectionPlan = eqx.field(static=True)
    projection_id: str = eqx.field(static=True)
    operator_fingerprint: str = eqx.field(static=True)
    initial_fingerprint: str = eqx.field(static=True)
    numeric_version: Array
    refresh_count: Array

    @property
    def method(self) -> str:
        return self.plan.selected_method

    @property
    def capacity(self) -> int:
        return self.plan.dimension

    @property
    def effective_dimension(self) -> Array:
        return self.decomposition.effective_dimension

    @property
    def basis(self) -> Array:
        """Return fixed-capacity basis columns with shape ``(n, capacity)``."""
        return jnp.swapaxes(self.decomposition.basis[:-1], -1, -2)

    @property
    def projected_operator(self) -> Array:
        """Return the fixed-capacity projected operator."""
        return self.decomposition.projected[:-1]

    @property
    def initial(self) -> PyTree[Array]:
        return self.operator.source.unflatten(self.initial_coordinates)

    def coefficients(self, vector: PyTree[Any], /) -> Array:
        """Project a source-space vector to active basis coefficients."""
        value = self.operator.source.validate(vector)
        basis_rows = self.decomposition.basis[:-1]
        coefficients = jax.vmap(
            lambda row: self.operator.source.inner(
                self.operator.source.unflatten(row), value
            )
        )(basis_rows)
        active = jnp.arange(self.capacity) < self.effective_dimension
        return jnp.where(active, coefficients, 0)

    def lift(self, coefficients: Array, /) -> PyTree[Array]:
        """Lift fixed-capacity coefficients to the source space."""
        values = jnp.asarray(coefficients)
        if values.shape != (self.capacity,):
            raise ValueError("coefficients must have shape (projection.capacity,).")
        if values.dtype != _coordinate_dtype(self.operator.source):
            raise TypeError("coefficient dtype must match the source coordinate dtype.")
        active = jnp.arange(self.capacity) < self.effective_dimension
        coordinates = self.basis @ jnp.where(active, values, 0)
        return self.operator.source.unflatten(coordinates)

    def project(self, vector: PyTree[Any], /) -> PyTree[Array]:
        """Return the orthogonal projection of a source-space vector."""
        return self.lift(self.coefficients(vector))

    def residual(self, vector: PyTree[Any], /) -> PyTree[Array]:
        """Remove the represented Krylov component from a vector."""
        value = self.operator.source.validate(vector)
        projected = self.project(value)
        return jax.tree.map(lambda left, right: left - right, value, projected)


def plan_krylov_projection(
    operator: AbstractLinearOperator,
    policy: KrylovProjectionPolicy | None = None,
    /,
) -> KrylovProjectionPlan:
    """Plan one reusable Arnoldi or Lanczos projection before applying the operator."""
    _validate_operator(operator)
    selected_policy = KrylovProjectionPolicy() if policy is None else policy
    if not isinstance(selected_policy, KrylovProjectionPolicy):
        raise TypeError("policy must be a KrylovProjectionPolicy or None.")
    method = selected_policy.method
    if method == "auto":
        method = "lanczos" if operator.properties.certifies("self_adjoint") else "arnoldi"
    if method == "lanczos" and not operator.properties.certifies("self_adjoint"):
        raise ValueError("Lanczos projection requires certified self-adjoint structure.")
    dimension = min(selected_policy.max_dimension, operator.source.size)
    cost = _projection_cost(operator, method, dimension)
    _validate_resources(cost, selected_policy.resources)
    payload = {
        "kind": "krylov-projection-plan",
        "operator": operator.operator_id,
        "source": operator.source.space_id,
        "method": method,
        "dimension": dimension,
        "orthogonalization": selected_policy.orthogonalization,
        "breakdown_tolerance": selected_policy.breakdown_tolerance,
        "resources": {
            "max_matvec_count": selected_policy.resources.max_matvec_count,
            "max_storage_bytes": selected_policy.resources.max_storage_bytes,
            "max_workspace_bytes": selected_policy.resources.max_workspace_bytes,
        },
    }
    return KrylovProjectionPlan(
        policy=selected_policy,
        selected_method=method,
        dimension=dimension,
        cost=cost,
        operator_id=operator.operator_id,
        source_space_id=operator.source.space_id,
        plan_id=canonical_fingerprint(payload),
    )


def prepare_krylov_projection(
    operator: AbstractLinearOperator,
    initial: PyTree[Any],
    policy: KrylovProjectionPolicy | KrylovProjectionPlan | None = None,
    /,
) -> PreparedKrylovProjection:
    """Build and bind a reusable projection to one operator and starting vector."""
    plan = (
        policy
        if isinstance(policy, KrylovProjectionPlan)
        else plan_krylov_projection(operator, policy)
    )
    _validate_plan(operator, plan)
    coordinates = operator.source.flatten(operator.source.validate(initial))
    if coordinates.ndim != 1:
        raise ValueError("Krylov projection preparation requires one starting vector.")
    decomposition = _decompose(operator, coordinates, plan)
    return _prepared(
        operator,
        coordinates,
        decomposition,
        plan,
        numeric_version=0,
        refresh_count=0,
    )


def refresh_krylov_projection(
    prepared: PreparedKrylovProjection,
    operator: AbstractLinearOperator,
    initial: PyTree[Any] | None = None,
    /,
) -> PreparedKrylovProjection:
    """Rebuild numerical projection state under one unchanged symbolic plan."""
    if not isinstance(prepared, PreparedKrylovProjection):
        raise TypeError("prepared must be a PreparedKrylovProjection.")
    _validate_plan(operator, prepared.plan)
    coordinates = (
        prepared.initial_coordinates
        if initial is None
        else operator.source.flatten(operator.source.validate(initial))
    )
    decomposition = _decompose(operator, coordinates, prepared.plan)
    return _prepared(
        operator,
        coordinates,
        decomposition,
        prepared.plan,
        numeric_version=prepared.numeric_version + jnp.asarray(1, dtype=jnp.int32),
        refresh_count=prepared.refresh_count + jnp.asarray(1, dtype=jnp.int32),
        projection_id=prepared.projection_id,
    )


def _decompose(
    operator: AbstractLinearOperator,
    coordinates: Array,
    plan: KrylovProjectionPlan,
    /,
) -> KrylovDecomposition:
    action = lambda value: operator.target.flatten(
        operator.mv(operator.source.unflatten(value))
    )
    inner = lambda left, right: operator.source.inner(
        operator.source.unflatten(left), operator.source.unflatten(right)
    )
    kwargs = {
        "max_dimension": plan.dimension,
        "inner": inner,
        "orthogonalization": plan.policy.orthogonalization,
        "breakdown_tolerance": plan.policy.breakdown_tolerance,
    }
    if plan.selected_method == "lanczos":
        decomposition = lanczos(action, coordinates, **kwargs)
    else:
        decomposition = arnoldi(action, coordinates, **kwargs)
    return jax.tree.map(
        lambda value: jax.lax.stop_gradient(value) if eqx.is_array(value) else value,
        decomposition,
    )


def _prepared(
    operator: AbstractLinearOperator,
    coordinates: Array,
    decomposition: KrylovDecomposition,
    plan: KrylovProjectionPlan,
    *,
    numeric_version: Any,
    refresh_count: Any,
    projection_id: str | None = None,
) -> PreparedKrylovProjection:
    operator_fingerprint = _operator_numeric_fingerprint(operator)
    initial_fingerprint = canonical_fingerprint(array_tree_fingerprint(coordinates))
    identifier = (
        canonical_fingerprint(
            {
                "kind": "krylov-projection",
                "plan": plan.plan_id,
                "operator": operator.operator_id,
            }
        )
        if projection_id is None
        else projection_id
    )
    return PreparedKrylovProjection(
        operator=operator,
        initial_coordinates=jax.lax.stop_gradient(coordinates),
        decomposition=decomposition,
        plan=plan,
        projection_id=identifier,
        operator_fingerprint=operator_fingerprint,
        initial_fingerprint=initial_fingerprint,
        numeric_version=jnp.asarray(numeric_version, dtype=jnp.int32),
        refresh_count=jnp.asarray(refresh_count, dtype=jnp.int32),
    )


def _projection_cost(
    operator: AbstractLinearOperator,
    method: str,
    dimension: int,
    /,
) -> KrylovProjectionCostEstimate:
    itemsize = _coordinate_dtype(operator.source).itemsize
    size = operator.source.size
    storage_entries = (dimension + 3) * size + (dimension + 1) * dimension
    action_cost = estimate_operator_action_cost(operator)
    workspace_entries = 3 * size + 2 * dimension
    return KrylovProjectionCostEstimate(
        method=method,
        dimension=dimension,
        matvec_count=dimension,
        storage_bytes=storage_entries * itemsize,
        workspace_bytes=workspace_entries * itemsize
        + action_cost.apply_workspace_bytes_per_rhs,
        operator_action_workspace_bytes=action_cost.apply_workspace_bytes_per_rhs,
        exact=action_cost.exact,
    )


def _validate_operator(operator: AbstractLinearOperator, /) -> None:
    if not isinstance(operator, AbstractLinearOperator):
        raise TypeError("operator must be an AbstractLinearOperator.")
    if operator.batch_shape or not operator.source.compatible(operator.target):
        raise ValueError("Krylov projections require an unbatched endomorphism.")
    if not jnp.issubdtype(_coordinate_dtype(operator.source), jnp.inexact):
        raise TypeError("Krylov projections require real or complex coordinates.")


def _validate_plan(
    operator: AbstractLinearOperator,
    plan: KrylovProjectionPlan,
    /,
) -> None:
    _validate_operator(operator)
    if not isinstance(plan, KrylovProjectionPlan):
        raise TypeError("plan must be a KrylovProjectionPlan.")
    if operator.operator_id != plan.operator_id:
        raise ValueError("Krylov projection plan belongs to a different operator.")
    if operator.source.space_id != plan.source_space_id:
        raise ValueError("Krylov projection plan source space changed.")
    if plan.selected_method == "lanczos" and not operator.properties.certifies(
        "self_adjoint"
    ):
        raise ValueError("Refreshed operator lacks the plan's self-adjoint certificate.")


def _validate_resources(
    cost: KrylovProjectionCostEstimate,
    resources: KrylovProjectionResourcePolicy,
    /,
) -> None:
    limits = (
        ("matvec count", cost.matvec_count, resources.max_matvec_count),
        ("storage", cost.storage_bytes, resources.max_storage_bytes),
        ("workspace", cost.workspace_bytes, resources.max_workspace_bytes),
    )
    violations = [
        f"{name} estimate {value} exceeds budget {limit}"
        for name, value, limit in limits
        if limit is not None and value > limit
    ]
    if violations:
        raise ValueError("Krylov projection resource rejection: " + "; ".join(violations))


def _optional_nonnegative_int(value: int | None, name: str, /) -> int | None:
    if value is None:
        return None
    integer = int(value)
    if integer < 0:
        raise ValueError(f"{name} must be non-negative or None.")
    return integer


__all__ = [
    "KrylovProjectionCostEstimate",
    "KrylovProjectionMethod",
    "KrylovProjectionPlan",
    "KrylovProjectionPolicy",
    "KrylovProjectionResourcePolicy",
    "PreparedKrylovProjection",
    "plan_krylov_projection",
    "prepare_krylov_projection",
    "refresh_krylov_projection",
]
