#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Literal, TypeAlias

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array

from .._strict import StrictModule


SolverGraduationLevel: TypeAlias = Literal["internal", "validated", "production"]


class SolverGraduationEvidence(StrictModule):
    false_successes: Array
    certified_cases: Array
    total_cases: Array
    peer_best_cases: Array
    profile_fraction_tau2: Array
    maximum_derivative_error: Array
    jit_verified: Array
    vmap_verified: Array
    refresh_verified: Array
    documentation_complete: Array
    benchmark_artifact_present: Array


class SolverGraduationPolicy(StrictModule):
    minimum_certified_fraction: float = eqx.field(static=True)
    maximum_peer_gap: float = eqx.field(static=True)
    minimum_profile_fraction_tau2: float = eqx.field(static=True)
    maximum_derivative_error: float = eqx.field(static=True)

    def __init__(
        self,
        *,
        minimum_certified_fraction: float = 0.99,
        maximum_peer_gap: float = 0.01,
        minimum_profile_fraction_tau2: float = 0.8,
        maximum_derivative_error: float = 1e-6,
    ):
        self.minimum_certified_fraction = float(minimum_certified_fraction)
        self.maximum_peer_gap = float(maximum_peer_gap)
        self.minimum_profile_fraction_tau2 = float(minimum_profile_fraction_tau2)
        self.maximum_derivative_error = float(maximum_derivative_error)


class SolverGraduationResult(StrictModule):
    level: Array
    correctness_passed: Array
    robustness_passed: Array
    differentiation_passed: Array
    execution_passed: Array
    product_passed: Array

    @property
    def production_ready(self):
        return self.level == 2


def evaluate_solver_graduation(
    evidence: SolverGraduationEvidence,
    /,
    *,
    policy: SolverGraduationPolicy | None = None,
) -> SolverGraduationResult:
    if not isinstance(evidence, SolverGraduationEvidence):
        raise TypeError("evidence must be SolverGraduationEvidence.")
    policy_ = SolverGraduationPolicy() if policy is None else policy
    if not isinstance(policy_, SolverGraduationPolicy):
        raise TypeError("policy must be SolverGraduationPolicy or None.")
    certified_fraction = evidence.certified_cases / jnp.maximum(evidence.total_cases, 1)
    peer_fraction = evidence.peer_best_cases / jnp.maximum(evidence.total_cases, 1)
    correctness = (evidence.false_successes == 0) & (
        certified_fraction >= policy_.minimum_certified_fraction
    )
    robustness = (certified_fraction + policy_.maximum_peer_gap >= peer_fraction) & (
        evidence.profile_fraction_tau2 >= policy_.minimum_profile_fraction_tau2
    )
    differentiation = (
        evidence.maximum_derivative_error <= policy_.maximum_derivative_error
    )
    execution = evidence.jit_verified & evidence.vmap_verified & evidence.refresh_verified
    product = evidence.documentation_complete & evidence.benchmark_artifact_present
    validated = correctness & differentiation & product
    production = validated & robustness & execution
    level = jnp.where(production, 2, jnp.where(validated, 1, 0)).astype(jnp.int32)
    return SolverGraduationResult(
        level,
        correctness,
        robustness,
        differentiation,
        execution,
        product,
    )


class SolverRegressionEvidence(StrictModule):
    new_false_successes: Array
    certified_fraction_change: Array
    profile_fraction_tau2_change: Array
    derivative_error_ratio: Array
    dense_materialization_regression: Array
    refresh_recompilation_regression: Array
    work_completeness_regression: Array


class SolverRegressionResult(StrictModule):
    passed: Array
    correctness_passed: Array
    robustness_passed: Array
    derivative_passed: Array
    execution_passed: Array


def evaluate_solver_regression(
    evidence: SolverRegressionEvidence,
    /,
    *,
    maximum_certified_fraction_drop: float = 0.01,
    maximum_profile_drop: float = 0.05,
    maximum_derivative_error_ratio: float = 2.0,
) -> SolverRegressionResult:
    correctness = (evidence.new_false_successes == 0) & (
        evidence.certified_fraction_change >= -maximum_certified_fraction_drop
    )
    robustness = evidence.profile_fraction_tau2_change >= -maximum_profile_drop
    derivative = evidence.derivative_error_ratio <= maximum_derivative_error_ratio
    execution = ~jnp.asarray(
        evidence.dense_materialization_regression
        | evidence.refresh_recompilation_regression
        | evidence.work_completeness_regression
    )
    return SolverRegressionResult(
        correctness & robustness & derivative & execution,
        correctness,
        robustness,
        derivative,
        execution,
    )


__all__ = [
    "SolverGraduationEvidence",
    "SolverGraduationLevel",
    "SolverGraduationPolicy",
    "SolverGraduationResult",
    "SolverRegressionEvidence",
    "SolverRegressionResult",
    "evaluate_solver_graduation",
    "evaluate_solver_regression",
]
