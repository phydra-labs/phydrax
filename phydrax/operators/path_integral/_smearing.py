#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ...graph._gauge_transport import GaugeStaplePlan
from ...metrix._complex_matrix_manifold import SpecialUnitaryGroup


class StoutSmearingResourcePolicy(StrictModule):
    """Hard fixed-shape and iteration budgets for stout smearing."""

    maximum_links: int = eqx.field(static=True)
    maximum_matrix_dimension: int = eqx.field(static=True)
    maximum_iterations: int = eqx.field(static=True)
    maximum_workspace_bytes: int = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        maximum_links: int = 16_777_216,
        maximum_matrix_dimension: int = 64,
        maximum_iterations: int = 64,
        maximum_workspace_bytes: int = 2_147_483_648,
    ):
        values = tuple(
            (
                maximum_links,
                maximum_matrix_dimension,
                maximum_iterations,
                maximum_workspace_bytes,
            )
        )
        if any(value < 1 for value in values):
            raise ValueError("Stout-smearing resource limits must be positive.")
        (
            self.maximum_links,
            self.maximum_matrix_dimension,
            self.maximum_iterations,
            self.maximum_workspace_bytes,
        ) = values
        self.policy_id = canonical_fingerprint(
            {
                "kind": "stout-smearing-resource-policy",
                "maximum_links": values[0],
                "maximum_matrix_dimension": values[1],
                "maximum_iterations": values[2],
                "maximum_workspace_bytes": values[3],
            }
        )


class StoutSmearingPlan(StrictModule):
    """Immutable differentiable stout update structure."""

    staples: GaugeStaplePlan
    rho: float = eqx.field(static=True)
    iterations: int = eqx.field(static=True)
    link_count: int = eqx.field(static=True)
    matrix_dimension: int = eqx.field(static=True)
    dtype: np.dtype = eqx.field(static=True)
    workspace_bytes: int = eqx.field(static=True)
    policy: StoutSmearingResourcePolicy = eqx.field(static=True)
    link_space_id: str = eqx.field(static=True)
    boundary_id: str = eqx.field(static=True)
    group_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        staples: GaugeStaplePlan,
        /,
        *,
        rho: float,
        iterations: int = 1,
        resources: StoutSmearingResourcePolicy | None = None,
        dtype: Any = jnp.complex64,
    ):
        if not isinstance(staples, GaugeStaplePlan):
            raise TypeError("staples must be GaugeStaplePlan.")
        rho_ = float(rho)
        count = int(iterations)
        if not math.isfinite(rho_):
            raise ValueError("rho must be finite.")
        if count < 1:
            raise ValueError("iterations must be positive.")
        policy = StoutSmearingResourcePolicy() if resources is None else resources
        if not isinstance(policy, StoutSmearingResourcePolicy):
            raise TypeError("resources must be StoutSmearingResourcePolicy or None.")
        link_count = staples.link_space.num_edges
        matrix_dimension = staples.link_space.point_shape[0]
        if link_count > policy.maximum_links:
            raise ValueError("Stout link count exceeds the resource policy.")
        if matrix_dimension > policy.maximum_matrix_dimension:
            raise ValueError("Stout matrix dimension exceeds the resource policy.")
        if count > policy.maximum_iterations:
            raise ValueError("Stout iteration count exceeds the resource policy.")
        dtype_ = np.dtype(jax.dtypes.canonicalize_dtype(np.dtype(dtype)))
        # Current links, staples, algebra generator, exponential, and next links.
        workspace = 5 * link_count * matrix_dimension * matrix_dimension * dtype_.itemsize
        if workspace > policy.maximum_workspace_bytes:
            raise ValueError("Stout transient workspace exceeds the resource policy.")
        self.staples = staples
        self.rho = rho_
        self.iterations = count
        self.link_count = link_count
        self.matrix_dimension = matrix_dimension
        self.dtype = dtype_
        self.workspace_bytes = workspace
        self.policy = policy
        self.link_space_id = staples.link_space.link_space_id
        self.boundary_id = staples.boundaries.boundary_plan_id
        self.group_id = staples.link_space.group.group_id
        self.plan_id = canonical_fingerprint(
            {
                "kind": "stout-smearing-plan",
                "staples": staples.plan_id,
                "link_space": staples.link_space.link_space_id,
                "boundary": staples.boundaries.boundary_plan_id,
                "group": staples.link_space.group.group_id,
                "rho": rho_,
                "iterations": count,
                "dtype": dtype_.str,
                "workspace_bytes": workspace,
                "policy": policy.policy_id,
                "update": "left-exponential-projected-staple-link-inverse",
            }
        )


class StoutSmearingEvidence(StrictModule):
    """Runtime finite/group-membership evidence for every stout iteration."""

    finite_by_iteration: Array
    member_by_iteration: Array
    maximum_unitarity_defect: Array
    maximum_determinant_defect: Array
    successful: Array
    plan_id: str = eqx.field(static=True)
    staple_plan_id: str = eqx.field(static=True)
    workspace_bytes: int = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)


class StoutSmearingResult(StrictModule):
    links: Array
    evidence: StoutSmearingEvidence


def _membership_defects(links: Array, /) -> tuple[Array, Array]:
    dimension = links.shape[-1]
    identity = jnp.eye(dimension, dtype=links.dtype)
    gram = jnp.swapaxes(jnp.conj(links), -1, -2) @ links
    unitarity = jnp.max(jnp.abs(gram - identity))
    determinant = jnp.max(jnp.abs(jnp.linalg.det(links) - 1.0))
    return unitarity, determinant


def _stout_step(plan: StoutSmearingPlan, links: Array, /) -> Array:
    group = plan.staples.link_space.group
    staples = plan.staples.staples(links)
    inverse_links = group.inverse(links)
    omega = group.compose(staples, inverse_links)
    generator = plan.rho * group.project_algebra(omega)
    rotation = group.exp(generator)
    return group.compose(rotation, links)


def stout_smear(plan: StoutSmearingPlan, links: ArrayLike, /) -> StoutSmearingResult:
    """Apply a fixed number of differentiable stout updates without projection."""

    if not isinstance(plan, StoutSmearingPlan):
        raise TypeError("plan must be StoutSmearingPlan.")
    values = jnp.asarray(links)
    expected = plan.staples.link_space.configuration_shape
    if values.shape != expected:
        raise ValueError(f"links must have shape {expected}; got {values.shape}.")
    if not jnp.issubdtype(values.dtype, jnp.complexfloating):
        raise TypeError("Stout smearing requires complex matrix links.")
    if np.dtype(values.dtype) != plan.dtype:
        raise TypeError(
            f"links must have planned dtype {plan.dtype}; got {values.dtype}."
        )

    group = plan.staples.link_space.group
    tolerance = jnp.asarray(
        max(float(group.tolerance), 64.0 * np.finfo(plan.dtype).eps),
        dtype=values.real.dtype,
    )
    requires_special = isinstance(group, SpecialUnitaryGroup)

    def iteration(current, _):
        updated = _stout_step(plan, current)
        finite = jnp.all(jnp.isfinite(updated))
        unitarity, determinant = _membership_defects(updated)
        member = (
            finite
            & (unitarity <= tolerance)
            & ((determinant <= tolerance) if requires_special else jnp.asarray(True))
        )
        diagnostics = jnp.stack(
            (
                finite.astype(updated.real.dtype),
                member.astype(updated.real.dtype),
                unitarity,
                determinant,
            )
        )
        return updated, diagnostics

    smeared, diagnostics = jax.lax.scan(
        iteration,
        values,
        xs=None,
        length=plan.iterations,
    )
    finite = diagnostics[:, 0].astype("bool")
    member = diagnostics[:, 1].astype("bool")
    unitarity = jnp.max(diagnostics[:, 2])
    determinant = jnp.max(diagnostics[:, 3])
    successful = jnp.all(finite & member)
    evidence_id = canonical_fingerprint(
        {
            "kind": "stout-smearing-evidence",
            "plan": plan.plan_id,
            "staples": plan.staples.plan_id,
            "iterations": plan.iterations,
            "workspace_bytes": plan.workspace_bytes,
            "criteria": ["finite", "group-membership"],
        }
    )
    return StoutSmearingResult(
        links=smeared,
        evidence=StoutSmearingEvidence(
            finite_by_iteration=finite,
            member_by_iteration=member,
            maximum_unitarity_defect=unitarity,
            maximum_determinant_defect=determinant,
            successful=successful,
            plan_id=plan.plan_id,
            staple_plan_id=plan.staples.plan_id,
            workspace_bytes=plan.workspace_bytes,
            evidence_id=evidence_id,
        ),
    )


__all__ = [
    "StoutSmearingEvidence",
    "StoutSmearingPlan",
    "StoutSmearingResourcePolicy",
    "StoutSmearingResult",
    "stout_smear",
]
