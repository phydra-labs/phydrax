#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Literal, TypeAlias

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState


SplatAccumulation: TypeAlias = Literal["fast", "deterministic", "compensated"]
SplatBoundaryPolicy: TypeAlias = Literal["reject", "drop"]
SplatGeometryAD: TypeAlias = Literal["piecewise", "frozen"]


class SplatExecutionPolicy(StrictModule, NonTrainableState):
    """Reduction and geometry-differentiation semantics for one splat transfer."""

    accumulation: SplatAccumulation = eqx.field(static=True)
    geometry_ad: SplatGeometryAD = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        accumulation: SplatAccumulation = "deterministic",
        geometry_ad: SplatGeometryAD = "piecewise",
    ):
        if accumulation not in ("fast", "deterministic", "compensated"):
            raise ValueError(
                "accumulation must be 'fast', 'deterministic', or 'compensated'."
            )
        if geometry_ad not in ("piecewise", "frozen"):
            raise ValueError("geometry_ad must be 'piecewise' or 'frozen'.")
        self.accumulation = accumulation
        self.geometry_ad = geometry_ad
        self.policy_id = canonical_fingerprint(
            {
                "kind": "splat-execution-policy",
                "accumulation": accumulation,
                "geometry_ad": geometry_ad,
            }
        )


class ParticleGridSplatBudget(StrictModule, NonTrainableState):
    """Static source, route, relation, and scalar-workspace resource bounds."""

    maximum_sources: int = eqx.field(static=True)
    maximum_routes: int = eqx.field(static=True)
    maximum_relation_bytes: int = eqx.field(static=True)
    maximum_scalar_workspace_bytes: int = eqx.field(static=True)
    budget_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        maximum_sources: int = 2_000_000,
        maximum_routes: int = 32_000_000,
        maximum_relation_bytes: int = 1024**3,
        maximum_scalar_workspace_bytes: int = 1024**3,
    ):
        values = tuple(
            int(value)
            for value in (
                maximum_sources,
                maximum_routes,
                maximum_relation_bytes,
                maximum_scalar_workspace_bytes,
            )
        )
        if any(value <= 0 for value in values):
            raise ValueError("Splat resource limits must be positive integers.")
        (
            self.maximum_sources,
            self.maximum_routes,
            self.maximum_relation_bytes,
            self.maximum_scalar_workspace_bytes,
        ) = values
        self.budget_id = canonical_fingerprint(
            {
                "kind": "particle-grid-splat-budget",
                "maximum_sources": values[0],
                "maximum_routes": values[1],
                "maximum_relation_bytes": values[2],
                "maximum_scalar_workspace_bytes": values[3],
            }
        )

    def admit(
        self,
        *,
        sources: int,
        routes: int,
        relation_bytes: int,
        scalar_workspace_bytes: int,
    ) -> None:
        """Reject a prepared transfer whose exact static resources exceed this budget."""
        counts = {
            "sources": (int(sources), self.maximum_sources),
            "routes": (int(routes), self.maximum_routes),
            "relation_bytes": (int(relation_bytes), self.maximum_relation_bytes),
            "scalar_workspace_bytes": (
                int(scalar_workspace_bytes),
                self.maximum_scalar_workspace_bytes,
            ),
        }
        for name, (value, limit) in counts.items():
            if value < 0:
                raise ValueError(f"Splat resource count {name} must be non-negative.")
            if value > limit:
                raise ValueError(
                    f"Splat resource {name} requires {value}, exceeding budget {limit}."
                )


class SplatBalanceEvidence(StrictModule):
    """Per-call extensive-balance and partition evidence."""

    active_source_total: Array
    supported_source_total: Array
    dropped_source_total: Array
    dropped_source_absolute_total: Array
    target_total: Array
    balance_defect: Array
    maximum_absolute_balance_defect: Array
    maximum_partition_defect: Array
    minimum_route_weight: Array
    valid_route_count: Array
    tolerance: Array
    closed_domain_conservation_valid: Array
    source_support_id: str = eqx.field(static=True)
    target_measure_id: str = eqx.field(static=True)
    execution_policy_id: str = eqx.field(static=True)
    precision_policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        active_source_total: ArrayLike,
        supported_source_total: ArrayLike,
        dropped_source_total: ArrayLike,
        dropped_source_absolute_total: ArrayLike,
        target_total: ArrayLike,
        balance_defect: ArrayLike,
        maximum_absolute_balance_defect: ArrayLike,
        maximum_partition_defect: ArrayLike,
        minimum_route_weight: ArrayLike,
        valid_route_count: ArrayLike,
        tolerance: ArrayLike,
        closed_domain_conservation_valid: ArrayLike,
        source_support_id: str,
        target_measure_id: str,
        execution_policy_id: str,
        precision_policy_id: str,
    ):
        self.active_source_total = jnp.asarray(active_source_total)
        self.supported_source_total = jnp.asarray(supported_source_total)
        self.dropped_source_total = jnp.asarray(dropped_source_total)
        self.dropped_source_absolute_total = jnp.asarray(dropped_source_absolute_total)
        self.target_total = jnp.asarray(target_total)
        self.balance_defect = jnp.asarray(balance_defect)
        self.maximum_absolute_balance_defect = jnp.asarray(
            maximum_absolute_balance_defect
        )
        self.maximum_partition_defect = jnp.asarray(maximum_partition_defect)
        self.minimum_route_weight = jnp.asarray(minimum_route_weight)
        self.valid_route_count = jnp.asarray(valid_route_count, dtype=jnp.int32)
        self.tolerance = jnp.asarray(tolerance)
        self.closed_domain_conservation_valid = jnp.asarray(
            closed_domain_conservation_valid, dtype=bool
        )
        identifiers = (
            str(source_support_id),
            str(target_measure_id),
            str(execution_policy_id),
            str(precision_policy_id),
        )
        if any(not identifier for identifier in identifiers):
            raise ValueError("Splat balance evidence IDs must be non-empty.")
        (
            self.source_support_id,
            self.target_measure_id,
            self.execution_policy_id,
            self.precision_policy_id,
        ) = identifiers

    def require_closed_conservation(self, value: ArrayLike, /) -> Array:
        """Return ``value`` or fail unless closed-domain conservation is valid."""
        return eqx.error_if(
            jnp.asarray(value),
            ~self.closed_domain_conservation_valid,
            "Particle-grid splat did not satisfy closed-domain conservation.",
        )


class SplatDepositResult(StrictModule):
    """Extensive target content, derived density, and balance evidence."""

    content: Array
    density: Array
    balance: SplatBalanceEvidence
    successful: Array

    def __init__(
        self,
        content: ArrayLike,
        density: ArrayLike,
        balance: SplatBalanceEvidence,
        successful: ArrayLike,
        /,
    ):
        if not isinstance(balance, SplatBalanceEvidence):
            raise TypeError("balance must be SplatBalanceEvidence.")
        content_ = jnp.asarray(content)
        density_ = jnp.asarray(density)
        if content_.shape != density_.shape:
            raise ValueError("Splat content and density must have identical shapes.")
        self.content = content_
        self.density = density_
        self.balance = balance
        self.successful = jnp.asarray(successful, dtype=bool)

    def require_success(self, value: ArrayLike, /) -> Array:
        """Return ``value`` or fail unless deposition completed successfully."""
        return eqx.error_if(
            jnp.asarray(value),
            ~self.successful,
            "Particle-grid splat deposition failed.",
        )


class SplatReconstructionResult(StrictModule):
    """Normalized intensive reconstruction with explicit numerator and coverage."""

    values: Array
    numerator: Array
    denominator: Array
    support: Array
    denominator_tolerance: Array
    zero_coverage_count: Array
    successful: Array

    def __init__(
        self,
        *,
        values: ArrayLike,
        numerator: ArrayLike,
        denominator: ArrayLike,
        support: ArrayLike,
        denominator_tolerance: ArrayLike,
        zero_coverage_count: ArrayLike,
        successful: ArrayLike,
    ):
        values_ = jnp.asarray(values)
        numerator_ = jnp.asarray(numerator)
        denominator_ = jnp.asarray(denominator)
        support_ = jnp.asarray(support, dtype=bool)
        if values_.shape != numerator_.shape:
            raise ValueError("Reconstruction values and numerator must match.")
        if values_.shape[: support_.ndim] != support_.shape:
            raise ValueError("Reconstruction support must prefix the value shape.")
        if denominator_.shape != support_.shape:
            raise ValueError("Reconstruction denominator must match support shape.")
        self.values = values_
        self.numerator = numerator_
        self.denominator = denominator_
        self.denominator_tolerance = jnp.asarray(denominator_tolerance)
        self.support = support_
        self.zero_coverage_count = jnp.asarray(zero_coverage_count, dtype=jnp.int32)
        self.successful = jnp.asarray(successful, dtype=bool)

    def require_success(self, value: ArrayLike, /) -> Array:
        """Return ``value`` or fail unless reconstruction completed successfully."""
        return eqx.error_if(
            jnp.asarray(value),
            ~self.successful,
            "Particle-grid splat reconstruction failed.",
        )


__all__ = [
    "ParticleGridSplatBudget",
    "SplatAccumulation",
    "SplatBalanceEvidence",
    "SplatBoundaryPolicy",
    "SplatDepositResult",
    "SplatExecutionPolicy",
    "SplatGeometryAD",
    "SplatReconstructionResult",
]
