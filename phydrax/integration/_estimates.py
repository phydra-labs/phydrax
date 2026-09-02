#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array

from .._numerics import WeightedMomentsDiagnostics
from .._precision import PrecisionEvidenceEnvelope
from .._strict import StrictModule
from ._status import IntegrationStatus


class IntegrationProvenance(StrictModule):
    """Static method, target, and realization identity for an estimate."""

    method: str = eqx.field(static=True)
    target: str = eqx.field(static=True)
    realization: str = eqx.field(static=True)

    def __init__(self, method: str, target: str, realization: str = "direct"):
        self.method = str(method)
        self.target = str(target)
        self.realization = str(realization)


class AdaptivePartition(StrictModule):
    """Static-capacity adaptive interval diagnostics with an active count."""

    count: Array
    lower_bounds: Array
    upper_bounds: Array
    integral_estimates: Array
    estimated_errors: Array
    active: Array


class DiscoveredBreakpoints(StrictModule):
    """Fixed-capacity numerical breakpoint candidates and non-proof evidence."""

    points: Array
    active: Array
    scores: Array
    kinds: Array
    status: Array


class AdaptiveCubaturePartition(StrictModule):
    """Static-capacity hyperrectangle partition."""

    count: Array
    lower_bounds: Array
    upper_bounds: Array
    integral_estimates: Array
    estimated_errors: Array
    active: Array


class AdaptiveCubatureDiagnostics(StrictModule):
    status: Array
    num_evaluations: Array
    estimated_error: Array
    partition: AdaptiveCubaturePartition | None
    dimension: int = eqx.field(static=True)
    low_rule: str = eqx.field(static=True)
    high_rule: str = eqx.field(static=True)


class AdaptiveTrianglePartition(StrictModule):
    """Static-capacity adaptive triangle diagnostics with an active count."""

    count: Array
    vertices: Array
    integral_estimates: Array
    estimated_errors: Array
    active: Array


class AdaptiveTriangleDiagnostics(StrictModule):
    status: Array
    num_evaluations: Array
    estimated_error: Array
    partition: AdaptiveTrianglePartition | None
    low_rule: str = eqx.field(static=True)
    high_rule: str = eqx.field(static=True)


class FixedQuadratureDiagnostics(StrictModule):
    status: Array
    num_evaluations: Array
    target_mass: Array | None
    rule: str = eqx.field(static=True)


class BayesianQuadratureDiagnostics(StrictModule):
    """Posterior-integral and delegated linear-solve evidence."""

    status: Array
    num_evaluations: Array
    posterior_variance: Array
    variance_roundoff_envelope: Array
    kernel_mean: Array
    kernel_double_mean: Array
    observation_noise: Array
    solve_regularization: Array
    solve: Any
    target_id: str = eqx.field(static=True)
    kernel_id: str = eqx.field(static=True)


class AdaptiveQuadratureDiagnostics(StrictModule):
    status: Array
    num_evaluations: Array
    estimated_error: Array
    partition: AdaptivePartition | None
    discovery: DiscoveredBreakpoints | None
    discovery_count: Array
    discovery_overflow: Array
    rule: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        status: Array,
        num_evaluations: Array,
        estimated_error: Array,
        partition: AdaptivePartition | None,
        rule: str,
        discovery: DiscoveredBreakpoints | None = None,
        discovery_count: Array | int = 0,
        discovery_overflow: Array | bool = False,
    ):
        self.status = jnp.asarray(status, dtype=jnp.int32)
        self.num_evaluations = jnp.asarray(num_evaluations, dtype=jnp.int32)
        self.estimated_error = jnp.asarray(estimated_error)
        self.partition = partition
        self.discovery = discovery
        self.discovery_count = jnp.asarray(discovery_count, dtype=jnp.int32)
        self.discovery_overflow = jnp.asarray(discovery_overflow, dtype=bool)
        self.rule = str(rule)


class MonteCarloDiagnostics(StrictModule):
    status: Array
    num_evaluations: Array
    standard_error: Array | None
    num_samples: Array
    num_independent_replicates: Array
    target_mass: Array | None


class StratifiedDiagnostics(StrictModule):
    status: Array
    num_evaluations: Array
    standard_error: Array
    samples_per_stratum: Array
    stratum_estimates: Array
    stratum_variances: Array
    stratum_contributions: Array


class AntitheticDiagnostics(StrictModule):
    status: Array
    num_evaluations: Array
    standard_error: Array | None
    num_pairs: Array
    pair_covariance: Array | None
    variance_reduction_factor: Array | None


class RandomizedQMCDiagnostics(StrictModule):
    status: Array
    num_evaluations: Array
    standard_error: Array | None
    num_samples_per_replicate: Array
    num_independent_replicates: Array
    replicate_estimates: Array
    scrambled: bool = eqx.field(static=True)
    sequence: str = eqx.field(static=True)


class WeightedSampleDiagnostics(StrictModule):
    """Per-slice diagnostics for weighted empirical or importance reductions."""

    status: Array
    num_evaluations: Array
    active_samples: Array
    standard_error: Array | None
    normalizer_estimate: Array
    normalizer_standard_error: Array | None
    weights: WeightedMomentsDiagnostics
    stratum_ids: Array | None
    pair_ids: Array | None
    replicate_ids: Array | None
    ancestry_ids: Array | None
    normalized: bool = eqx.field(static=True)
    independent: bool = eqx.field(static=True)


class SparseGridDiagnostics(StrictModule):
    status: Array
    num_evaluations: Array
    level_difference: Array | None
    level: int = eqx.field(static=True)
    num_unique_nodes: int = eqx.field(static=True)
    previous_num_unique_nodes: int = eqx.field(static=True)
    num_terms: int = eqx.field(static=True)
    axis_rules: tuple[str, ...] = eqx.field(static=True)


class MappedIntegrationDiagnostics(StrictModule):
    status: Array
    num_evaluations: Array
    target_mass: Array
    num_active_points: Array
    cell: str = eqx.field(static=True)


class ProductIntegrationDiagnostics(StrictModule):
    status: Array
    num_evaluations: Array
    error_estimate: Array | None
    factors: tuple[Any, ...]


class IntegrationEstimate(StrictModule):
    """Reduced value with method-correct uncertainty and diagnostics."""

    value: Any
    status: Array
    num_evaluations: Array
    error_estimate: Array | None
    diagnostics: Any
    provenance: IntegrationProvenance
    precision_evidence: PrecisionEvidenceEnvelope | None = eqx.field(static=True)
    error_kind: str | None = eqx.field(static=True)

    def __init__(
        self,
        value: Any,
        /,
        *,
        status: Array | int = IntegrationStatus.CONVERGED,
        num_evaluations: Array | int,
        error_estimate: Array | None,
        error_kind: str | None,
        diagnostics: Any,
        provenance: IntegrationProvenance,
        precision_evidence: PrecisionEvidenceEnvelope | None = None,
    ):
        self.value = value
        self.status = jnp.asarray(status, dtype=jnp.int32)
        self.num_evaluations = jnp.asarray(num_evaluations, dtype=jnp.int32)
        self.error_estimate = error_estimate
        self.error_kind = error_kind
        self.diagnostics = diagnostics
        self.provenance = provenance
        self.precision_evidence = precision_evidence

    @property
    def successful(self) -> Array:
        return self.status == int(IntegrationStatus.CONVERGED)


__all__ = [
    "AdaptivePartition",
    "AdaptiveCubatureDiagnostics",
    "AdaptiveCubaturePartition",
    "AdaptiveTriangleDiagnostics",
    "AdaptiveTrianglePartition",
    "AdaptiveQuadratureDiagnostics",
    "DiscoveredBreakpoints",
    "AntitheticDiagnostics",
    "BayesianQuadratureDiagnostics",
    "FixedQuadratureDiagnostics",
    "WeightedSampleDiagnostics",
    "IntegrationEstimate",
    "IntegrationProvenance",
    "MappedIntegrationDiagnostics",
    "MonteCarloDiagnostics",
    "ProductIntegrationDiagnostics",
    "RandomizedQMCDiagnostics",
    "SparseGridDiagnostics",
    "StratifiedDiagnostics",
]
