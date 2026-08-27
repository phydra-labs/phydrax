from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from functools import cache
from pathlib import Path
from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import opt_einsum as oe
import polars as pl

import phydrax as phx

from .external import ExternalCandidateAudit
from .matrix import (
    aggregate_benchmark_results,
    benchmark_metadata,
    BenchmarkRunMetadata,
    OperatorBenchmarkAggregate,
    scenario_checksum,
)
from .models import compatible_architectures, OperatorArchitecture
from .runner import (
    evaluate_operator,
    evaluate_operator_symmetry,
    OperatorBenchmarkResult,
    parameter_count,
    run_operator_benchmark,
    training_step_cost,
)
from .scenarios import (
    add_input_noise_shift,
    add_sensor_corruption_shift,
    add_sensor_dropout_shift,
    add_training_sensor_dropout,
    causal_relaxation_scenario,
    cochain_annulus_harmonic_scenario,
    cochain_mixed_darcy_scenario,
    conservative_ring_transport_scenario,
    darcy_scenario,
    deformed_elliptic_scenario,
    graph_diffusion_scenario,
    green_function_scenario,
    multi_input_diffusion_scenario,
    navier_stokes_scenario,
    OperatorBenchmarkEvaluation,
    OperatorBenchmarkScenario,
    periodic_acoustic_wave_scenario,
    periodic_advection_scenario,
    periodic_burgers_scenario,
    polynomial_poisson_scenario,
    spherical_diffusion_scenario,
    split_operator_scenario,
    square_diffusion_symmetry_scenario,
)


ComparisonMode = Literal["native", "capacity", "compute", "pareto"]
BenchmarkProfile = Literal["smoke", "shortlist", "decision"]
ParityStatus = Literal["verified", "failed", "not_run"]
PromotionTier = Literal["validated", "experimental", "specialized", "external"]


@dataclass(frozen=True)
class OperatorBenchmarkLadder:
    """Ordered difficulty levels for one physical benchmark regime."""

    name: str
    regime: str
    levels: tuple[OperatorBenchmarkScenario, ...]

    def __post_init__(self):
        if not self.name or not self.regime:
            raise ValueError("Difficulty ladder name and regime must be non-empty.")
        if len(self.levels) < 2:
            raise ValueError("A difficulty ladder requires at least two levels.")
        names = tuple(level.name for level in self.levels)
        if len(set(names)) != len(names):
            raise ValueError("Difficulty ladder scenario names must be unique.")
        if any(level.ladder != self.name for level in self.levels):
            raise ValueError("Every scenario must identify its containing ladder.")
        if any(not level.difficulty for level in self.levels):
            raise ValueError("Every ladder scenario must declare a difficulty level.")


@dataclass(frozen=True)
class NearIdentityDiagnostic:
    scenario: str
    baseline: str
    relative_l2: float
    threshold: float
    detected: bool


@dataclass(frozen=True)
class ScenarioIntegrityAudit:
    scenario: str
    checksum: str
    train_cases: int
    validation_cases: int
    test_cases: int
    physical_split_disjoint: bool
    provenance_complete: bool
    dimensional_ranges_declared: bool
    nondimensional_ranges_declared: bool
    reference_converged: bool
    reference_relative_error: float | None
    reference_tolerance: float | None
    near_identity: NearIdentityDiagnostic
    passed: bool
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class GeometryScenarioIntegrityAudit:
    """Geometry-specific evidence beyond the generic scenario audit."""

    scenario: str
    per_case_geometry: bool
    independent_source_query: bool
    quadrature_positive: bool
    quadrature_conservative: bool
    boundary_data_consistent: bool
    shifts_isolated: bool
    physical_split_disjoint: bool
    minimum_jacobian: float
    passed: bool
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class SymmetryScenarioIntegrityAudit:
    """Reference-level square-group evidence and orbit-split provenance."""

    scenario: str
    declared_group: str | None
    audit_group: str
    reference_tolerance: float
    exact_reference_mean_defect: float | None
    exact_reference_worst_defect: float | None
    rotational_reference_mean_defect: float
    reflected_reference_mean_defect: float | None
    physical_split_disjoint: bool
    transforms_generated_post_split: bool
    passed: bool
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class SymmetryBenchmarkRecord:
    """Measured paired group-transform defect for one selected trained model."""

    scenario: str
    architecture: str
    family: str
    seed: int
    size_scale: float
    evaluation: str
    declared_group: str | None
    audit_group: str
    element_relative_l2: tuple[float, ...]
    element_maximum_absolute_error: tuple[float, ...]
    mean_equivariance_defect: float | None
    worst_equivariance_defect: float | None
    maximum_absolute_equivariance_error: float | None
    mean_rotated_pair_difference: float
    mean_reflected_pair_difference: float | None
    reference_worst_defect: float


@dataclass(frozen=True)
class KernelParityCheck:
    name: str
    family: str
    reference: str
    relative_error: float
    tolerance: float
    passed: bool

    def __post_init__(self):
        if not self.name or not self.family or not self.reference:
            raise ValueError("Parity check name, family, and reference are required.")
        if float(self.relative_error) < 0.0 or float(self.tolerance) < 0.0:
            raise ValueError("Parity errors and tolerances must be non-negative.")
        if bool(self.passed) != (float(self.relative_error) <= float(self.tolerance)):
            raise ValueError("Parity pass status must agree with error and tolerance.")


@dataclass(frozen=True)
class FamilyParityEvidence:
    """Pinned upstream evidence supplied by a benchmark operator maintainer."""

    family: str
    reference_uri: str
    revision: str
    status: ParityStatus
    checks: tuple[KernelParityCheck, ...] = ()

    def __post_init__(self):
        if not self.family or not self.reference_uri or not self.revision:
            raise ValueError("Parity family, reference URI, and revision are required.")
        if self.status not in ("verified", "failed", "not_run"):
            raise ValueError("Unknown family parity status.")
        if any(check.family != self.family for check in self.checks):
            raise ValueError("Parity checks must identify their containing family.")
        if self.status == "verified" and (
            not self.checks or any(not check.passed for check in self.checks)
        ):
            raise ValueError("Verified family parity requires passing numerical checks.")

    @property
    def verified(self) -> bool:
        return (
            self.status == "verified"
            and bool(self.checks)
            and all(check.passed for check in self.checks)
        )


@dataclass(frozen=True)
class OperatorBenchmarkProtocol:
    """Compute, capacity, normalization, and search policy for benchmark v2."""

    seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    comparison: ComparisonMode = "capacity"
    steps: int = 300
    learning_rates: tuple[float, ...] = (3e-4, 1e-3, 3e-3)
    repeats: int = 5
    validation_interval: int = 10
    patience: int | None = None
    minimum_delta: float = 0.0
    relative_minimum_delta: float = 0.0
    target_parameters: int | None = None
    compute_budget: int | None = None
    size_scales: tuple[float, ...] = (0.5, 0.75, 1.0, 1.5, 2.0)
    sample_fractions: tuple[float, ...] = (1.0,)
    normalize: bool = True
    split_seed: int = 1729
    train_fraction: float = 0.6
    validation_fraction: float = 0.2
    near_identity_threshold: float = 0.05
    quick: bool = False
    profile: BenchmarkProfile = "shortlist"
    commit_identity: str = "working-tree"
    checkpoint_directory: str | None = None
    resume: bool = False
    sensor_training_dropout: float = 0.0

    def __post_init__(self):
        if not self.seeds:
            raise ValueError("Benchmark v2 requires at least one model seed.")
        if self.profile not in ("smoke", "shortlist", "decision"):
            raise ValueError("Unknown benchmark profile.")
        if self.comparison not in ("native", "capacity", "compute", "pareto"):
            raise ValueError(
                "comparison must be 'native', 'capacity', 'compute', or 'pareto'."
            )
        if int(self.steps) < 0 or int(self.repeats) <= 0:
            raise ValueError("steps must be non-negative and repeats must be positive.")
        if not self.learning_rates or any(rate <= 0.0 for rate in self.learning_rates):
            raise ValueError("At least one positive learning rate is required.")
        if int(self.validation_interval) <= 0:
            raise ValueError("validation_interval must be positive.")
        if self.patience is not None and int(self.patience) <= 0:
            raise ValueError("patience must be positive when supplied.")
        if float(self.minimum_delta) < 0.0 or float(self.relative_minimum_delta) < 0.0:
            raise ValueError("Convergence deltas must be non-negative.")
        if self.target_parameters is not None and int(self.target_parameters) <= 0:
            raise ValueError("target_parameters must be positive when supplied.")
        if self.compute_budget is not None and int(self.compute_budget) <= 0:
            raise ValueError("compute_budget must be positive when supplied.")
        if not self.size_scales or any(scale <= 0.0 for scale in self.size_scales):
            raise ValueError("size_scales must contain positive values.")
        if (
            not self.sample_fractions
            or any(not 0.0 < fraction <= 1.0 for fraction in self.sample_fractions)
            or tuple(sorted(set(self.sample_fractions))) != self.sample_fractions
            or self.sample_fractions[-1] != 1.0
        ):
            raise ValueError(
                "sample_fractions must be unique, increasing values in (0, 1] ending at 1."
            )
        if not 0.0 < self.train_fraction < 1.0:
            raise ValueError("train_fraction must lie strictly between zero and one.")
        if not 0.0 < self.validation_fraction < 1.0:
            raise ValueError(
                "validation_fraction must lie strictly between zero and one."
            )
        if float(self.near_identity_threshold) < 0.0:
            raise ValueError("near_identity_threshold must be non-negative.")
        if not 0.0 <= float(self.sensor_training_dropout) < 1.0:
            raise ValueError("sensor_training_dropout must lie in [0, 1).")
        if self.resume and self.checkpoint_directory is None:
            raise ValueError("resume requires checkpoint_directory.")
        if self.resume and self.commit_identity.strip() in ("", "working-tree"):
            raise ValueError("Resumable runs require an immutable commit identity.")


@dataclass(frozen=True)
class BenchmarkComparisonRecord:
    scenario: str
    architecture: str
    mode: ComparisonMode
    size_scale: float
    target_parameters: int | None
    actual_parameters: int
    capacity_ratio: float | None
    planned_steps: int
    target_compute_units: int | None
    actual_compute_units: int
    compute_ratio: float | None
    normalization: str
    training_step_flops: int = 0
    training_step_bytes: int = 0
    compute_measurement: Literal["proxy", "jax_flops"] = "proxy"
    minimum_parameters: int | None = None
    maximum_parameters: int | None = None
    target_in_range: bool | None = None
    comparable: bool = True
    comparison_reason: str | None = None


@dataclass(frozen=True)
class HyperparameterTrial:
    scenario: str
    architecture: str
    family: str
    seed: int
    learning_rate: float
    size_scale: float
    normalization: str
    parameter_count: int
    training_steps: int
    training_seconds: float
    initial_loss: float
    final_loss: float
    validation_loss: float | None
    learning_curve: tuple[float, ...]
    selected: bool
    validation_steps: tuple[int, ...] = ()
    validation_curve: tuple[float, ...] = ()
    stopped_early: bool = False
    converged: bool = False
    resumed_from_step: int = 0
    architecture_configuration: str = "{}"


@dataclass(frozen=True)
class SampleEfficiencyCurve:
    """Held-out error versus the number of base physical training realizations."""

    scenario: str
    architecture: str
    family: str
    seed: int
    evaluation: str
    learning_rate: float
    size_scale: float
    sample_fractions: tuple[float, ...]
    train_cases: tuple[int, ...]
    relative_l2: tuple[float, ...]
    maximum_absolute_error: tuple[float, ...]
    training_seconds: tuple[float, ...]
    parameter_counts: tuple[int, ...]
    area_under_sample_error_curve: float


@dataclass(frozen=True)
class ScenarioDifficultyAudit:
    scenario: str
    identity_relative_l2: float | None
    persistence_relative_change: float | None
    nearest_realization_relative_distance: float | None
    source_effective_rank: float
    source_rank_99: int
    source_rank_fraction: float
    target_effective_rank: float
    target_rank_99: int
    target_rank_fraction: float
    passed: bool
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class PromotionCriteria:
    maximum_relative_l2: float = 0.25
    maximum_shift_degradation: float = 2.0
    maximum_seed_std: float = 0.05
    maximum_inference_seconds: float = 0.1
    maximum_parameter_count: int = 10_000_000
    maximum_peak_memory_bytes: int | None = 8 * 1024**3
    minimum_seeds: int = 5
    minimum_capacity_ratio: float = 1.0 / 1.1
    maximum_capacity_ratio: float = 1.1
    minimum_compute_ratio: float = 1.0 / 1.1
    maximum_compute_ratio: float = 1.1
    require_parity: bool = True
    minimum_general_scenarios: int = 3
    minimum_external_scenarios: int = 3
    minimum_persistence_change: float = 0.05
    minimum_nearest_realization_distance: float = 0.01
    minimum_source_effective_rank: float = 3.0
    minimum_source_rank_99: int = 4
    minimum_source_rank_fraction: float = 0.1
    minimum_target_effective_rank: float = 3.0
    minimum_target_rank_99: int = 4
    minimum_target_rank_fraction: float = 0.1
    require_baseline_hardness: bool = True
    require_convergence: bool = True

    def __post_init__(self):
        positive = (
            self.maximum_relative_l2,
            self.maximum_shift_degradation,
            self.maximum_seed_std,
            self.maximum_inference_seconds,
        )
        if any(value < 0.0 for value in positive):
            raise ValueError("Promotion metric thresholds must be non-negative.")
        hardness_thresholds = (
            self.minimum_persistence_change,
            self.minimum_nearest_realization_distance,
            self.minimum_target_effective_rank,
            self.minimum_source_effective_rank,
            self.minimum_source_rank_99,
            self.minimum_source_rank_fraction,
            self.minimum_target_rank_99,
            self.minimum_target_rank_fraction,
        )
        if any(value < 0.0 for value in hardness_thresholds):
            raise ValueError("Scenario difficulty thresholds must be non-negative.")
        if int(self.maximum_parameter_count) <= 0:
            raise ValueError("maximum_parameter_count must be positive.")
        if int(self.minimum_seeds) < 5:
            raise ValueError("Validated promotion requires at least five seeds.")
        if (
            self.maximum_peak_memory_bytes is not None
            and int(self.maximum_peak_memory_bytes) <= 0
        ):
            raise ValueError("maximum_peak_memory_bytes must be positive when supplied.")
        if not 0.0 < self.minimum_capacity_ratio <= self.maximum_capacity_ratio:
            raise ValueError("Capacity ratio bounds must be positive and ordered.")
        if not 0.0 < self.minimum_compute_ratio <= self.maximum_compute_ratio:
            raise ValueError("Compute ratio bounds must be positive and ordered.")
        if (
            int(self.minimum_general_scenarios) <= 0
            or int(self.minimum_external_scenarios) <= 0
        ):
            raise ValueError("Promotion scenario-count thresholds must be positive.")


@dataclass(frozen=True)
class ArchitecturePromotionReport:
    scenario: str
    architecture: str
    size_scale: float
    family: str
    scope: str
    tier: PromotionTier
    accuracy_passed: bool
    robustness_passed: bool
    reproducibility_passed: bool
    efficiency_passed: bool
    integrity_passed: bool
    parity_passed: bool
    comparison_passed: bool
    baseline_hardness_passed: bool
    convergence_passed: bool
    external_audit_passed: bool | None
    in_distribution_relative_l2: float
    worst_shift_degradation: float | None
    maximum_seed_std: float
    maximum_inference_seconds: float
    parameter_count_mean: float
    peak_memory_bytes_mean: float | None
    seed_count: int
    promoted: bool
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class ArchitecturePortfolioPromotion:
    architecture: str
    size_scale: float
    family: str
    scope: str
    tier: PromotionTier
    scenario_count: int
    passed_scenarios: int
    promoted: bool
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class BenchmarkParetoPoint:
    scenario: str
    architecture: str
    family: str
    size_scale: float
    validation_relative_l2: float | None
    shifted_relative_l2: float | None
    training_flops: int | None
    inference_seconds: float | None
    peak_memory_bytes: float | None
    parameter_count: float
    complete: bool
    dominated_by: tuple[str, ...]
    nondominated: bool | None


@dataclass(frozen=True)
class BenchmarkParetoFront:
    scenario: str
    points: tuple[BenchmarkParetoPoint, ...]
    complete_architectures: tuple[str, ...]
    nondominated_architectures: tuple[str, ...]


@dataclass(frozen=True)
class OperatorBenchmarkV2Result:
    metadata: BenchmarkRunMetadata
    protocol: OperatorBenchmarkProtocol
    ladders: tuple[OperatorBenchmarkLadder, ...]
    audits: tuple[ScenarioIntegrityAudit, ...]
    symmetry_audits: tuple[SymmetryScenarioIntegrityAudit, ...]
    kernel_parity: tuple[KernelParityCheck, ...]
    family_parity: tuple[FamilyParityEvidence, ...]
    external_audits: tuple[ExternalCandidateAudit, ...]
    comparisons: tuple[BenchmarkComparisonRecord, ...]
    trials: tuple[HyperparameterTrial, ...]
    sample_efficiency: tuple[SampleEfficiencyCurve, ...]
    results: tuple[OperatorBenchmarkResult, ...]
    symmetry_results: tuple[SymmetryBenchmarkRecord, ...]
    aggregates: tuple[OperatorBenchmarkAggregate, ...]
    difficulty_audits: tuple[ScenarioDifficultyAudit, ...]
    pareto_fronts: tuple[BenchmarkParetoFront, ...]
    promotions: tuple[ArchitecturePromotionReport, ...]
    portfolio_promotions: tuple[ArchitecturePortfolioPromotion, ...]

    def to_dict(self):
        return {
            "metadata": asdict(self.metadata),
            "protocol": asdict(self.protocol),
            "ladders": [
                {
                    "name": ladder.name,
                    "regime": ladder.regime,
                    "levels": [
                        {
                            "scenario": level.name,
                            "difficulty": level.difficulty,
                            "regimes": level.regimes,
                            "checksum": scenario_checksum(level),
                        }
                        for level in ladder.levels
                    ],
                }
                for ladder in self.ladders
            ],
            "audits": [asdict(audit) for audit in self.audits],
            "symmetry_audits": [asdict(audit) for audit in self.symmetry_audits],
            "difficulty_audits": [asdict(audit) for audit in self.difficulty_audits],
            "kernel_parity": [asdict(check) for check in self.kernel_parity],
            "family_parity": [asdict(evidence) for evidence in self.family_parity],
            "external_audits": [asdict(audit) for audit in self.external_audits],
            "comparisons": [asdict(record) for record in self.comparisons],
            "trials": [asdict(trial) for trial in self.trials],
            "sample_efficiency": [asdict(curve) for curve in self.sample_efficiency],
            "results": [result.to_dict() for result in self.results],
            "symmetry_results": [asdict(record) for record in self.symmetry_results],
            "aggregates": [asdict(aggregate) for aggregate in self.aggregates],
            "pareto_fronts": [asdict(front) for front in self.pareto_fronts],
            "promotions": [asdict(report) for report in self.promotions],
            "portfolio_promotions": [
                asdict(report) for report in self.portfolio_promotions
            ],
        }


def _tag_level(
    scenario: OperatorBenchmarkScenario,
    ladder: str,
    difficulty: str,
    /,
) -> OperatorBenchmarkScenario:
    if not any(
        evaluation.shift == "in_distribution" for evaluation in scenario.evaluations
    ):
        scenario = replace(
            scenario,
            evaluations=(
                OperatorBenchmarkEvaluation(
                    "in_distribution",
                    scenario.train_batch,
                    scenario.train_target,
                    case_ids=scenario.case_ids,
                ),
            )
            + scenario.evaluations,
        )
    scenario = add_input_noise_shift(
        scenario,
        standard_deviation=0.02 if difficulty == "easy" else 0.05,
        seed=int.from_bytes(
            hashlib.sha256(f"{ladder}:{difficulty}:noise".encode()).digest()[:4],
            "big",
        ),
    )
    sensor_shift_policy = dict(scenario.metadata).get("sensor_shift_policy", "enabled")
    if scenario.symmetry is None and sensor_shift_policy == "enabled":
        scenario = add_sensor_corruption_shift(
            scenario,
            corruption_fraction=0.1 if difficulty == "easy" else 0.3,
            seed=int.from_bytes(
                hashlib.sha256(f"{ladder}:{difficulty}:corruption".encode()).digest()[:4],
                "big",
            ),
        )
        scenario = add_sensor_dropout_shift(
            scenario,
            drop_fraction=0.1 if difficulty == "easy" else 0.3,
            seed=int.from_bytes(
                hashlib.sha256(f"{ladder}:{difficulty}:dropout".encode()).digest()[:4],
                "big",
            ),
        )
    tagged_name = f"{scenario.name}__{ladder}__{difficulty}"
    return replace(
        scenario,
        name=tagged_name,
        ladder=ladder,
        difficulty=difficulty,
        regimes=tuple(dict.fromkeys(scenario.regimes + (ladder,))),
        metadata=scenario.metadata
        + (("difficulty_ladder", ladder), ("difficulty", difficulty)),
    )


def standard_operator_benchmark_ladders(
    *,
    quick: bool = False,
    profile: BenchmarkProfile | None = None,
) -> tuple[OperatorBenchmarkLadder, ...]:
    """Construct smoke, shortlisted, or decision-grade physical ladders."""
    resolved_profile = "smoke" if quick else (profile or "shortlist")
    if resolved_profile == "smoke":
        cases, low_resolution, high_resolution = 8, 8, 12
        easy_horizon, hard_horizon = 1, 2
        cochain_easy, cochain_hard = 4, 5
        annulus_easy, annulus_hard = 8, 10
    elif resolved_profile == "shortlist":
        cases, low_resolution, high_resolution = 24, 16, 32
        easy_horizon, hard_horizon = 2, 4
        cochain_easy, cochain_hard = 5, 8
        annulus_easy, annulus_hard = 12, 20
    elif resolved_profile == "decision":
        cases, low_resolution, high_resolution = 128, 24, 48
        easy_horizon, hard_horizon = 4, 12
        cochain_easy, cochain_hard = 8, 12
        annulus_easy, annulus_hard = 20, 32
    else:
        raise ValueError(f"Unknown benchmark profile {resolved_profile!r}.")

    definitions = (
        (
            "smooth_periodic",
            "smooth_periodic",
            navier_stokes_scenario(
                resolution=low_resolution,
                num_cases=cases,
                viscosity=1e-2,
                dt=1e-2,
                target_steps=max(6, easy_horizon),
                maximum_frequency=4,
                seed=101,
            ),
            navier_stokes_scenario(
                resolution=high_resolution,
                num_cases=cases,
                viscosity=1e-3,
                dt=2e-2,
                target_steps=max(12, hard_horizon),
                maximum_frequency=8,
                seed=202,
            ),
        ),
        (
            "polynomial_nonlinearity",
            "controlled_periodic_polynomial_poisson",
            polynomial_poisson_scenario(
                resolution=low_resolution,
                num_cases=cases,
                polynomial_degree=1,
                maximum_frequency=max(2, low_resolution // 3),
                seed=3333,
            ),
            polynomial_poisson_scenario(
                resolution=high_resolution,
                num_cases=cases,
                polynomial_degree=2,
                maximum_frequency=max(2, high_resolution // 3),
                seed=3434,
            ),
        ),
        (
            "shock_discontinuity",
            "shock_discontinuity",
            periodic_burgers_scenario(
                train_resolution=2 * low_resolution,
                test_resolution=3 * low_resolution,
                num_cases=cases,
                viscosity=2e-2,
                dt=1e-3,
                target_steps=max(2, easy_horizon),
                rollout_steps=1 if resolved_profile == "smoke" else 3,
                initial_condition="shock",
                maximum_frequency=6,
                seed=707,
            ),
            periodic_burgers_scenario(
                train_resolution=2 * high_resolution,
                test_resolution=3 * high_resolution,
                num_cases=cases,
                viscosity=5e-3,
                dt=1e-3,
                target_steps=max(4, hard_horizon),
                rollout_steps=1 if resolved_profile == "smoke" else 8,
                initial_condition="shock",
                maximum_frequency=8,
                seed=808,
            ),
        ),
        (
            "elliptic_contrast",
            "elliptic_contrast",
            darcy_scenario(
                resolution=low_resolution,
                num_cases=cases,
                contrast=0.2,
                maximum_frequency=4,
                seed=909,
            ),
            darcy_scenario(
                resolution=high_resolution,
                num_cases=cases,
                contrast=0.8,
                maximum_frequency=6,
                seed=1010,
            ),
        ),
        (
            "irregular_geometry",
            "irregular_geometry",
            deformed_elliptic_scenario(
                points=2 * low_resolution,
                query_points=2 * low_resolution + 3,
                num_cases=cases,
                deformation_amplitude=0.015,
                geometry_extrapolation_factor=1.25,
                sensor_dropout_fraction=0.1,
                seed=1111,
            ),
            deformed_elliptic_scenario(
                points=2 * high_resolution,
                query_points=3 * high_resolution,
                num_cases=cases,
                deformation_amplitude=0.035,
                geometry_extrapolation_factor=1.5,
                sensor_dropout_fraction=0.3,
                seed=1212,
            ),
        ),
        (
            "conservative_geometry",
            "conservative_geometry",
            conservative_ring_transport_scenario(
                source_points=2 * low_resolution,
                query_points=2 * low_resolution + 5,
                support_resolution=max(10, low_resolution // 2),
                num_cases=cases,
                radius_range=(0.45, 0.65),
                band_width=0.24,
                center_extent=0.04,
                advection_time=0.25,
                speed_range=(0.4, 0.8),
                geometry_extrapolation_factor=1.2,
                maximum_frequency=6,
                seed=1919,
            ),
            conservative_ring_transport_scenario(
                source_points=2 * high_resolution,
                query_points=3 * high_resolution + 1,
                support_resolution=max(16, high_resolution // 2),
                num_cases=cases,
                radius_range=(0.35, 0.75),
                band_width=0.16,
                center_extent=0.15,
                advection_time=0.75,
                speed_range=(0.4, 1.2),
                geometry_extrapolation_factor=1.5,
                maximum_frequency=10,
                seed=2020,
            ),
        ),
        (
            "independent_query",
            "independent_query",
            green_function_scenario(
                source_points=low_resolution,
                query_points=low_resolution + 5,
                num_cases=cases,
                kernel_length_scale=0.03,
                maximum_frequency=8,
                seed=303,
            ),
            green_function_scenario(
                source_points=high_resolution,
                query_points=2 * high_resolution + 1,
                num_cases=cases,
                kernel_length_scale=0.02,
                maximum_frequency=12,
                seed=404,
            ),
        ),
        (
            "multi_input",
            "multi_input",
            multi_input_diffusion_scenario(
                resolution=2 * low_resolution,
                num_cases=cases,
                dt=0.2,
                diffusivity_range=(0.015, 0.02),
                maximum_frequency=6,
                seed=1313,
            ),
            multi_input_diffusion_scenario(
                resolution=2 * high_resolution,
                num_cases=cases,
                dt=0.1,
                diffusivity_range=(0.005, 0.04),
                parameter_shift_factor=1.5,
                maximum_frequency=10,
                seed=1414,
            ),
        ),
        (
            "long_horizon",
            "long_horizon",
            periodic_burgers_scenario(
                train_resolution=2 * low_resolution,
                test_resolution=3 * low_resolution,
                num_cases=cases,
                viscosity=1e-2,
                dt=5e-3,
                rollout_steps=2,
                target_steps=max(2, easy_horizon),
                maximum_frequency=6,
                seed=1515,
            ),
            periodic_burgers_scenario(
                train_resolution=2 * high_resolution,
                test_resolution=3 * high_resolution,
                num_cases=cases,
                viscosity=5e-3,
                dt=1e-3,
                rollout_steps=8,
                target_steps=max(4, hard_horizon),
                maximum_frequency=8,
                seed=1616,
            ),
        ),
        (
            "causal_transient",
            "causal_transient",
            causal_relaxation_scenario(
                source_points=2 * low_resolution,
                query_points=2 * low_resolution,
                test_query_points=3 * low_resolution,
                num_cases=cases,
                final_time=1.0,
                decay_rate=1.0,
                maximum_frequency=8.0,
                modes=8,
                seed=1717,
            ),
            causal_relaxation_scenario(
                source_points=2 * high_resolution,
                query_points=2 * high_resolution,
                test_query_points=3 * high_resolution,
                num_cases=cases,
                final_time=8.0,
                decay_rate=0.2,
                maximum_frequency=6.0,
                modes=16,
                seed=1818,
            ),
        ),
        (
            "spherical_field",
            "spherical_field",
            spherical_diffusion_scenario(
                bandlimit=max(6, low_resolution),
                sampling="mw",
                num_cases=cases,
                diffusivity=0.02,
                dt=0.2,
                target_steps=max(4, easy_horizon),
                maximum_degree=4,
                seed=505,
            ),
            spherical_diffusion_scenario(
                bandlimit=max(8, high_resolution),
                sampling="mw",
                num_cases=cases,
                diffusivity=0.005,
                dt=0.5,
                target_steps=max(12, hard_horizon),
                maximum_degree=7,
                seed=606,
            ),
        ),
        (
            "geometry_extrapolation",
            "geometry_extrapolation",
            graph_diffusion_scenario(
                nodes=low_resolution + 2,
                test_nodes=high_resolution + 2,
                num_cases=cases,
                geometry_shift=True,
                deformation_amplitude=0.05,
                maximum_frequency=6,
                seed=1919,
            ),
            graph_diffusion_scenario(
                nodes=high_resolution + 2,
                test_nodes=2 * high_resolution,
                num_cases=cases,
                geometry_shift=True,
                deformation_amplitude=0.2,
                maximum_frequency=10,
                seed=2020,
            ),
        ),
        (
            "parameter_extrapolation",
            "parameter_extrapolation",
            multi_input_diffusion_scenario(
                resolution=2 * low_resolution,
                num_cases=cases,
                dt=0.2,
                diffusivity_range=(0.01, 0.02),
                parameter_shift_factor=1.25,
                maximum_frequency=6,
                seed=2121,
            ),
            multi_input_diffusion_scenario(
                resolution=2 * high_resolution,
                num_cases=cases,
                dt=0.1,
                diffusivity_range=(0.005, 0.04),
                parameter_shift_factor=2.0,
                maximum_frequency=10,
                seed=2222,
            ),
        ),
        (
            "square_symmetry",
            "square_group_equivariance",
            square_diffusion_symmetry_scenario(
                resolution=low_resolution,
                num_cases=cases,
                diffusivity=(0.02, 0.02),
                dt=0.05,
                maximum_frequency=3,
                seed=2929,
            ),
            square_diffusion_symmetry_scenario(
                resolution=high_resolution,
                num_cases=cases,
                diffusivity=(0.02, 0.02),
                dt=0.05,
                chiral_strength=1e-5,
                maximum_frequency=5,
                seed=3030,
            ),
        ),
        (
            "square_symmetry_control",
            "intentionally_broken_square_symmetry",
            square_diffusion_symmetry_scenario(
                resolution=low_resolution,
                num_cases=cases,
                diffusivity=(0.015, 0.025),
                dt=0.05,
                maximum_frequency=3,
                seed=3131,
            ),
            square_diffusion_symmetry_scenario(
                resolution=high_resolution,
                num_cases=cases,
                diffusivity=(0.008, 0.032),
                dt=0.05,
                maximum_frequency=5,
                seed=3232,
            ),
        ),
        (
            "cochain_mixed_darcy",
            "metric_dec_mixed_degree",
            cochain_mixed_darcy_scenario(
                train_points=cochain_easy,
                test_points=cochain_easy + 2,
                num_cases=cases,
                reaction=0.3,
                boundary_policy="absolute",
                mesh_warp=0.0,
                seed=3535,
            ),
            cochain_mixed_darcy_scenario(
                train_points=cochain_hard,
                test_points=cochain_hard + 3,
                num_cases=cases,
                reaction=0.1,
                boundary_policy="relative",
                mesh_warp=0.15,
                seed=3636,
            ),
        ),
        (
            "cochain_annulus_harmonic",
            "metric_dec_nontrivial_topology",
            cochain_annulus_harmonic_scenario(
                train_radial_layers=1,
                train_angular_points=annulus_easy,
                test_radial_layers=2,
                test_angular_points=annulus_easy + 4,
                num_cases=cases,
                boundary_policy="absolute",
                seed=3737,
            ),
            cochain_annulus_harmonic_scenario(
                train_radial_layers=2,
                train_angular_points=annulus_hard,
                test_radial_layers=3,
                test_angular_points=annulus_hard + 6,
                num_cases=cases,
                boundary_policy="relative",
                seed=3838,
            ),
        ),
    )
    if resolved_profile != "smoke":
        definitions += (
            (
                "constant_speed_advection",
                "smooth_constant_speed_periodic_advection",
                periodic_advection_scenario(
                    train_resolution=2 * low_resolution,
                    test_resolution=3 * low_resolution,
                    num_cases=cases,
                    speed_configuration="constant",
                    speed=0.75,
                    dt=0.05,
                    target_steps=max(2, easy_horizon),
                    rollout_steps=2,
                    maximum_frequency=4,
                    seed=2323,
                ),
                periodic_advection_scenario(
                    train_resolution=2 * high_resolution,
                    test_resolution=3 * high_resolution,
                    num_cases=cases,
                    speed_configuration="constant",
                    speed=1.25,
                    dt=0.04,
                    target_steps=max(4, hard_horizon),
                    rollout_steps=4,
                    maximum_frequency=10,
                    seed=2424,
                ),
            ),
            (
                "variable_speed_advection",
                "smooth_variable_speed_periodic_advection",
                periodic_advection_scenario(
                    train_resolution=2 * low_resolution,
                    test_resolution=3 * low_resolution,
                    num_cases=cases,
                    speed_configuration="variable",
                    speed=0.75,
                    speed_variation=0.2,
                    dt=0.05,
                    target_steps=max(2, easy_horizon),
                    rollout_steps=2,
                    maximum_frequency=4,
                    seed=2525,
                ),
                periodic_advection_scenario(
                    train_resolution=2 * high_resolution,
                    test_resolution=3 * high_resolution,
                    num_cases=cases,
                    speed_configuration="variable",
                    speed=1.0,
                    speed_variation=0.55,
                    dt=0.04,
                    target_steps=max(4, hard_horizon),
                    rollout_steps=4,
                    maximum_frequency=10,
                    seed=2626,
                ),
            ),
            (
                "periodic_acoustic_waves",
                "periodic_acoustic_waves",
                periodic_acoustic_wave_scenario(
                    train_resolution=2 * low_resolution,
                    test_resolution=3 * low_resolution,
                    num_cases=cases,
                    sound_speed=0.8,
                    density=1.2,
                    dt=0.05,
                    target_steps=max(2, easy_horizon),
                    rollout_steps=2,
                    maximum_wavenumber=4,
                    seed=2727,
                ),
                periodic_acoustic_wave_scenario(
                    train_resolution=2 * high_resolution,
                    test_resolution=3 * high_resolution,
                    num_cases=cases,
                    sound_speed=1.5,
                    density=0.9,
                    dt=0.04,
                    target_steps=max(4, hard_horizon),
                    rollout_steps=4,
                    maximum_wavenumber=10,
                    seed=2828,
                ),
            ),
        )
    return tuple(
        OperatorBenchmarkLadder(
            name,
            regime,
            (
                _tag_level(easy, name, "easy"),
                _tag_level(hard, name, "hard"),
            ),
        )
        for name, regime, easy, hard in definitions
    )


def flatten_operator_benchmark_ladders(
    ladders: tuple[OperatorBenchmarkLadder, ...],
    /,
    *,
    difficulty: Literal["easy", "hard"] | None = None,
) -> tuple[OperatorBenchmarkScenario, ...]:
    scenarios = tuple(level for ladder in ladders for level in ladder.levels)
    if difficulty is None:
        return scenarios
    return tuple(scenario for scenario in scenarios if scenario.difficulty == difficulty)


def _split_ids(scenario: OperatorBenchmarkScenario):
    validation_ids = () if scenario.validation is None else scenario.validation.case_ids
    evaluation_ids = tuple(
        case_id for evaluation in scenario.evaluations for case_id in evaluation.case_ids
    )
    return set(scenario.case_ids), set(validation_ids), set(evaluation_ids)


def audit_symmetry_scenario(
    scenario: OperatorBenchmarkScenario,
    /,
) -> SymmetryScenarioIntegrityAudit:
    """Verify exact reference actions and that transformed partners are post-split."""
    symmetry = scenario.symmetry
    if symmetry is None or not symmetry.reference_defects:
        raise ValueError(
            "Symmetry audit requires a populated scenario symmetry contract."
        )
    defects = tuple(float(defect) for _, defect in symmetry.reference_defects)
    exact_indices = tuple(range(1, symmetry.exact_element_count))
    diagnostic_indices = tuple(
        index for index in range(1, len(defects)) if index not in exact_indices
    )
    tolerance = float(symmetry.reference_tolerance)
    train_ids, validation_ids, evaluation_ids = _split_ids(scenario)
    physical_split_disjoint = bool(
        train_ids
        and validation_ids
        and evaluation_ids
        and train_ids.isdisjoint(validation_ids)
        and train_ids.isdisjoint(evaluation_ids)
        and validation_ids.isdisjoint(evaluation_ids)
    )
    reasons = []
    if defects[0] > tolerance:
        reasons.append("identity action exceeds the reference tolerance")
    if exact_indices and max(defects[index] for index in exact_indices) > tolerance:
        reasons.append("declared physical group is not reference-equivariant")
    if symmetry.intentionally_violated:
        if (
            not diagnostic_indices
            or max(defects[index] for index in diagnostic_indices) <= tolerance
        ):
            reasons.append("intentional symmetry violation is not numerically resolved")
    elif (
        diagnostic_indices
        and max(defects[index] for index in diagnostic_indices) > tolerance
    ):
        reasons.append("reference violates an audited exact group element")
    if not physical_split_disjoint:
        reasons.append("base physical realization IDs overlap across data splits")

    rotations = tuple(defects[index] for index in range(1, min(4, len(defects))))
    reflections = defects[4:] if len(defects) == 8 else ()
    return SymmetryScenarioIntegrityAudit(
        scenario=scenario.name,
        declared_group=symmetry.group,
        audit_group=symmetry.audit_group,
        reference_tolerance=tolerance,
        exact_reference_mean_defect=(
            None
            if not exact_indices
            else float(np.mean([defects[index] for index in exact_indices]))
        ),
        exact_reference_worst_defect=(
            None
            if not exact_indices
            else float(max(defects[index] for index in exact_indices))
        ),
        rotational_reference_mean_defect=float(np.mean(rotations)),
        reflected_reference_mean_defect=(
            None if not reflections else float(np.mean(reflections))
        ),
        physical_split_disjoint=physical_split_disjoint,
        transforms_generated_post_split=True,
        passed=not reasons,
        reasons=tuple(reasons),
    )


def _near_identity_diagnostic(
    scenario: OperatorBenchmarkScenario,
    /,
    *,
    threshold: float,
    quick: bool,
) -> NearIdentityDiagnostic:
    if isinstance(scenario.train_target, phx.nn.operator.OperatorTargetBatch):
        relative_l2 = max(
            float(
                jax.block_until_ready(
                    phx.nn.operator.operator_l2_loss(
                        jnp.zeros_like(field.values),
                        field.values,
                        scenario.train_batch.query(field.query_name),
                        relative=True,
                    )
                )
            )
            for field in scenario.train_target.fields.values()
        )
        baseline_name = "zero"
    else:
        architectures = compatible_architectures(scenario, quick=quick)
        lookup = {architecture.name: architecture for architecture in architectures}
        if "nearest_neighbor" in lookup:
            baseline_name = "nearest_neighbor"
        elif "identity" in lookup:
            baseline_name = "identity"
        else:
            baseline_name = "constant"
        baseline = lookup[baseline_name].build(scenario, 0)
        prediction = baseline(scenario.train_batch)
        relative_l2 = float(
            jax.block_until_ready(
                phx.nn.operator.operator_l2_loss(
                    prediction,
                    scenario.train_target,
                    scenario.train_batch.require_single_query(),
                    relative=True,
                )
            )
        )
    return NearIdentityDiagnostic(
        scenario=scenario.name,
        baseline=baseline_name,
        relative_l2=relative_l2,
        threshold=float(threshold),
        detected=relative_l2 <= float(threshold),
    )


def _tree_numerically_finite(tree) -> bool:
    for leaf in jax.tree_util.tree_leaves(tree):
        if isinstance(leaf, (jax.Array, np.ndarray)):
            if not bool(np.all(np.isfinite(np.asarray(jax.device_get(leaf))))):
                return False
    return True


def _scenario_numerically_finite(scenario: OperatorBenchmarkScenario, /) -> bool:
    trees = [scenario.train_batch, scenario.train_target]
    if scenario.validation is not None:
        trees.extend((scenario.validation.batch, scenario.validation.target))
    for evaluation in scenario.evaluations:
        trees.extend((evaluation.batch, evaluation.target))
    return all(_tree_numerically_finite(tree) for tree in trees)


def audit_operator_scenario(
    scenario: OperatorBenchmarkScenario,
    /,
    *,
    near_identity_threshold: float = 0.05,
    quick: bool = False,
) -> ScenarioIntegrityAudit:
    train_ids, validation_ids, evaluation_ids = _split_ids(scenario)
    split_disjoint = (
        bool(train_ids)
        and bool(validation_ids)
        and bool(evaluation_ids)
        and train_ids.isdisjoint(validation_ids)
        and train_ids.isdisjoint(evaluation_ids)
        and validation_ids.isdisjoint(evaluation_ids)
    )
    provenance_complete = scenario.provenance is not None
    dimensional_declared = bool(scenario.dimensional_parameters)
    nondimensional_declared = bool(scenario.nondimensional_parameters)
    evidence = scenario.reference_evidence
    reference_converged = evidence is not None and evidence.passed
    numerically_finite = _scenario_numerically_finite(scenario)
    reasons = []
    if not split_disjoint:
        reasons.append("physical realization IDs overlap or are absent")
    if not provenance_complete:
        reasons.append("dataset provenance is incomplete")
    if not dimensional_declared:
        reasons.append("dimensional parameter ranges are absent")
    if not nondimensional_declared:
        reasons.append("nondimensional parameter ranges are absent")
    if not reference_converged:
        reasons.append("reference solver evidence does not meet tolerance")
    if not numerically_finite:
        reasons.append("scenario contains non-finite numerical data")
    near_identity = _near_identity_diagnostic(
        scenario,
        threshold=near_identity_threshold,
        quick=quick,
    )
    return ScenarioIntegrityAudit(
        scenario=scenario.name,
        checksum=scenario_checksum(scenario),
        train_cases=len(scenario.case_ids),
        validation_cases=0 if scenario.validation is None else len(validation_ids),
        test_cases=len(evaluation_ids),
        physical_split_disjoint=split_disjoint,
        provenance_complete=provenance_complete,
        dimensional_ranges_declared=dimensional_declared,
        nondimensional_ranges_declared=nondimensional_declared,
        reference_converged=reference_converged,
        reference_relative_error=None if evidence is None else evidence.relative_error,
        reference_tolerance=None if evidence is None else evidence.tolerance,
        near_identity=near_identity,
        passed=not reasons,
        reasons=tuple(reasons),
    )


def audit_geometry_scenario(
    scenario: OperatorBenchmarkScenario,
    /,
    *,
    tolerance: float = 1e-8,
) -> GeometryScenarioIntegrityAudit:
    """Audit per-case geometry, measures, boundary data, and isolated shifts."""

    required_inputs = {
        "forcing",
        "diffusivity",
        "boundary_value",
        "boundary_indicator",
    }
    if not required_inputs.issubset(scenario.train_batch.inputs):
        raise ValueError(
            "Geometry scenario audits require forcing, diffusivity, boundary_value, "
            "and boundary_indicator inputs."
        )
    records = [(scenario.train_batch, scenario.train_target)]
    if scenario.validation is not None:
        records.append((scenario.validation.batch, scenario.validation.target))
    records.extend(
        (evaluation.batch, evaluation.target) for evaluation in scenario.evaluations
    )

    def host(value):
        return np.asarray(jax.device_get(value))

    def same(left, right):
        left_array = host(left)
        right_array = host(right)
        return left_array.shape == right_array.shape and np.allclose(
            left_array,
            right_array,
            rtol=float(tolerance),
            atol=float(tolerance),
        )

    per_case_geometry = True
    independent_source_query = True
    quadrature_positive = True
    quadrature_conservative = True
    boundary_data_consistent = True
    for batch, target in records:
        source = batch.input("forcing")
        source_coordinates = host(source.coordinates)
        query_coordinates = host(batch.require_single_query().coordinates)
        case_count = int(np.prod(batch.case_shape, dtype=np.int64))
        if (
            source_coordinates.ndim != 3
            or source_coordinates.shape[0] != case_count
            or query_coordinates.ndim != 3
            or query_coordinates.shape[0] != case_count
        ):
            per_case_geometry = False
        elif case_count > 1:
            flattened = source_coordinates.reshape((case_count, -1))
            per_case_geometry &= np.unique(flattened, axis=0).shape[0] == case_count
        if source_coordinates.shape == query_coordinates.shape:
            independent_source_query &= not np.allclose(
                source_coordinates,
                query_coordinates,
                rtol=float(tolerance),
                atol=float(tolerance),
            )
        for samples in batch.inputs.values():
            weights = host(samples.quadrature_weights)
            quadrature_positive &= bool(
                np.all(np.isfinite(weights)) and np.all(weights > 0.0)
            )
            quadrature_conservative &= bool(
                np.allclose(
                    np.sum(weights, axis=-1),
                    1.0,
                    rtol=float(tolerance),
                    atol=float(tolerance),
                )
            )

        indicator = host(batch.input("boundary_indicator").values).reshape(
            (case_count, -1)
        )
        boundary_value = host(batch.input("boundary_value").values).reshape(
            (case_count, -1)
        )
        source_coordinates = source_coordinates.reshape((case_count, -1, 2))
        query_coordinates = query_coordinates.reshape((case_count, -1, 2))
        target_array = host(target).reshape((case_count, -1))
        for case in range(case_count):
            boundary = indicator[case] > 0.5
            on_physical_boundary = np.any(
                np.isclose(
                    source_coordinates[case, :, :, None],
                    np.asarray((0.0, 1.0))[None, None, :],
                    rtol=0.0,
                    atol=float(tolerance),
                ),
                axis=(1, 2),
            )
            left = boundary & np.isclose(
                source_coordinates[case, :, 0],
                0.0,
                rtol=0.0,
                atol=float(tolerance),
            )
            if not np.any(boundary) or not np.any(left):
                boundary_data_consistent = False
                continue
            inferred_boundary = float(np.mean(boundary_value[case, left]))
            expected_source = inferred_boundary * (1.0 - source_coordinates[case, :, 0])
            boundary_data_consistent &= bool(
                np.array_equal(boundary, on_physical_boundary)
                and np.allclose(
                    boundary_value[case, boundary],
                    expected_source[boundary],
                    rtol=float(tolerance),
                    atol=float(tolerance),
                )
                and np.allclose(
                    boundary_value[case, ~boundary],
                    0.0,
                    rtol=0.0,
                    atol=float(tolerance),
                )
            )
            query_boundary = np.any(
                np.isclose(
                    query_coordinates[case, :, :, None],
                    np.asarray((0.0, 1.0))[None, None, :],
                    rtol=0.0,
                    atol=float(tolerance),
                ),
                axis=(1, 2),
            )
            expected_target = inferred_boundary * (1.0 - query_coordinates[case, :, 0])
            boundary_data_consistent &= bool(
                np.any(query_boundary)
                and np.allclose(
                    target_array[case, query_boundary],
                    expected_target[query_boundary],
                    rtol=float(tolerance),
                    atol=float(tolerance),
                )
            )

    evaluations = {evaluation.name: evaluation for evaluation in scenario.evaluations}
    required_evaluations = {
        "nominal",
        "resolution_transfer",
        "independent_query",
        "geometry_extrapolation",
        "sensor_dropout",
        "boundary_condition_shift",
    }
    shifts_isolated = required_evaluations.issubset(evaluations)
    if shifts_isolated:
        nominal = evaluations["nominal"].batch
        independent = evaluations["independent_query"].batch
        resolution = evaluations["resolution_transfer"].batch
        geometry = evaluations["geometry_extrapolation"].batch
        sensor = evaluations["sensor_dropout"].batch
        boundary = evaluations["boundary_condition_shift"].batch
        shifts_isolated &= all(
            same(nominal.input(name).coordinates, independent.input(name).coordinates)
            and same(nominal.input(name).values, independent.input(name).values)
            for name in required_inputs
        )
        shifts_isolated &= (
            nominal.require_single_query().sample_shape
            != independent.require_single_query().sample_shape
            and nominal.input("forcing").sample_shape
            != resolution.input("forcing").sample_shape
            and nominal.require_single_query().sample_shape
            != resolution.require_single_query().sample_shape
        )
        shifts_isolated &= nominal.input("forcing").sample_shape == geometry.input(
            "forcing"
        ).sample_shape and not same(
            nominal.input("forcing").coordinates,
            geometry.input("forcing").coordinates,
        )
        shifts_isolated &= all(
            same(nominal.input(name).coordinates, sensor.input(name).coordinates)
            and same(nominal.input(name).values, sensor.input(name).values)
            for name in required_inputs
        )
        sensor_mask = host(sensor.input("forcing").mask)
        shifts_isolated &= bool(np.any(~sensor_mask) and np.any(sensor_mask))
        shifts_isolated &= (
            same(
                nominal.input("forcing").coordinates,
                boundary.input("forcing").coordinates,
            )
            and same(
                nominal.input("diffusivity").values,
                boundary.input("diffusivity").values,
            )
            and not same(
                nominal.input("forcing").values,
                boundary.input("forcing").values,
            )
        )

    train_ids, validation_ids, evaluation_ids = _split_ids(scenario)
    physical_split_disjoint = bool(
        train_ids
        and validation_ids
        and evaluation_ids
        and train_ids.isdisjoint(validation_ids)
        and train_ids.isdisjoint(evaluation_ids)
        and validation_ids.isdisjoint(evaluation_ids)
    )
    metadata = dict(scenario.metadata)
    minimum_jacobian = min(
        float(metadata.get("minimum_train_jacobian", "-inf")),
        float(metadata.get("minimum_extrapolated_jacobian", "-inf")),
    )
    reasons = []
    checks = (
        (per_case_geometry, "geometry is not represented independently per case"),
        (independent_source_query, "source and query clouds are not independent"),
        (quadrature_positive, "quadrature contains non-positive or non-finite weights"),
        (quadrature_conservative, "quadrature does not preserve unit domain measure"),
        (
            boundary_data_consistent,
            "boundary indicators, values, or targets are inconsistent",
        ),
        (shifts_isolated, "geometry evaluation shifts are absent or confounded"),
        (physical_split_disjoint, "physical realization IDs overlap across splits"),
        (minimum_jacobian > 0.0, "deformation map has a non-positive Jacobian"),
    )
    for passed, reason in checks:
        if not passed:
            reasons.append(reason)
    return GeometryScenarioIntegrityAudit(
        scenario=scenario.name,
        per_case_geometry=bool(per_case_geometry),
        independent_source_query=bool(independent_source_query),
        quadrature_positive=bool(quadrature_positive),
        quadrature_conservative=bool(quadrature_conservative),
        boundary_data_consistent=bool(boundary_data_consistent),
        shifts_isolated=bool(shifts_isolated),
        physical_split_disjoint=bool(physical_split_disjoint),
        minimum_jacobian=float(minimum_jacobian),
        passed=not reasons,
        reasons=tuple(reasons),
    )


def _case_matrix(values, case_shape: tuple[int, ...], /) -> np.ndarray:
    case_count = int(np.prod(case_shape, dtype=np.int64))
    array = np.asarray(jax.device_get(values), dtype=np.float64)
    return array.reshape((case_count, -1))


def _rank_diagnostics(
    matrix: np.ndarray,
    /,
    *,
    intrinsic_rank: int | None = None,
) -> tuple[float, int, float, int]:
    centered = matrix - np.mean(matrix, axis=0, keepdims=True)
    singular_values = np.linalg.svd(centered, compute_uv=False)
    energy = singular_values * singular_values
    total_energy = float(np.sum(energy))
    if total_energy <= 1e-24:
        effective_rank = 0.0
        rank_99 = 0
    else:
        probabilities = energy / total_energy
        positive = probabilities[probabilities > 0.0]
        effective_rank = float(np.exp(-np.sum(positive * np.log(positive))))
        rank_99 = int(np.searchsorted(np.cumsum(probabilities), 0.99) + 1)
    maximum_rank = max(1, min(matrix.shape[0] - 1, matrix.shape[1]))
    if intrinsic_rank is not None:
        if int(intrinsic_rank) <= 0:
            raise ValueError("Declared intrinsic ranks must be positive.")
        maximum_rank = min(maximum_rank, int(intrinsic_rank))
    rank_fraction = min(rank_99, maximum_rank) / maximum_rank
    return effective_rank, rank_99, rank_fraction, maximum_rank


def _primary_samples(batch: phx.nn.operator.OperatorBatch, /):
    source_key = max(
        sorted(batch.inputs),
        key=lambda name: int(np.prod(batch.input(name).sample_shape, dtype=np.int64)),
    )
    return batch.input(source_key)


def _sample_geometries_coincide(
    source: phx.nn.operator.FunctionSamples,
    query: phx.nn.operator.FunctionSamples,
    /,
) -> bool:
    if source.sample_shape != query.sample_shape:
        return False
    if source.axes or query.axes:
        return len(source.axes) == len(query.axes) and all(
            np.array_equal(
                np.asarray(jax.device_get(source_axis.nodes)),
                np.asarray(jax.device_get(query_axis.nodes)),
            )
            for source_axis, query_axis in zip(source.axes, query.axes)
        )
    if source.coordinates is None or query.coordinates is None:
        return False
    return np.array_equal(
        np.asarray(jax.device_get(source.coordinates)),
        np.asarray(jax.device_get(query.coordinates)),
    )


def _batch_case_features(batch: phx.nn.operator.OperatorBatch, /) -> np.ndarray | None:
    case_count = int(np.prod(batch.case_shape, dtype=np.int64))
    parts = []
    for name in sorted(batch.inputs):
        values = batch.input(name).values
        if values is not None:
            parts.append(
                np.asarray(jax.device_get(values), dtype=np.float64).reshape(
                    (case_count, -1)
                )
            )
    return None if not parts else np.concatenate(parts, axis=1)


def _target_case_matrix(
    target: jax.Array | phx.nn.operator.OperatorTargetBatch,
    case_shape: tuple[int, ...],
    /,
) -> np.ndarray:
    if not isinstance(target, phx.nn.operator.OperatorTargetBatch):
        return _case_matrix(target, case_shape)
    return np.concatenate(
        tuple(_case_matrix(field.values, case_shape) for field in target.fields.values()),
        axis=1,
    )


def audit_scenario_difficulty(
    scenario: OperatorBenchmarkScenario,
    criteria: PromotionCriteria,
    /,
) -> ScenarioDifficultyAudit:
    """Audit shortcut baselines, realization novelty, and target subspace rank."""
    source_samples = _primary_samples(scenario.train_batch)
    source_values = source_samples.values
    target_values = scenario.train_target
    source_matrix = _batch_case_features(scenario.train_batch)
    if source_matrix is None:
        raise ValueError("Scenario difficulty requires at least one sampled input.")
    metadata = dict(scenario.metadata)
    source_intrinsic_rank = (
        int(metadata["source_intrinsic_rank"])
        if "source_intrinsic_rank" in metadata
        else None
    )
    target_intrinsic_rank = (
        int(metadata["target_intrinsic_rank"])
        if "target_intrinsic_rank" in metadata
        else None
    )
    (
        source_effective_rank,
        source_rank_99,
        source_rank_fraction,
        source_maximum_rank,
    ) = _rank_diagnostics(source_matrix, intrinsic_rank=source_intrinsic_rank)
    identity_relative_l2 = None
    persistence_relative_change = None
    if (
        not isinstance(target_values, phx.nn.operator.OperatorTargetBatch)
        and len(scenario.train_batch.inputs) == 1
        and source_values is not None
        and source_values.shape == target_values.shape
        and _sample_geometries_coincide(
            source_samples,
            scenario.train_batch.require_single_query(),
        )
    ):
        identity_relative_l2 = _relative_array_error(source_values, target_values)
        persistence_relative_change = _relative_array_error(
            target_values,
            source_values,
        )

    nominal = next(
        (
            evaluation
            for evaluation in scenario.evaluations
            if evaluation.shift == "in_distribution"
        ),
        None,
    )
    nearest_distance = None
    if nominal is not None:
        evaluation_matrix = _batch_case_features(nominal.batch)
        if (
            evaluation_matrix is not None
            and source_matrix.shape[1] == evaluation_matrix.shape[1]
        ):
            train_norms = np.sum(source_matrix * source_matrix, axis=1)
            evaluation_norms = np.sum(
                evaluation_matrix * evaluation_matrix,
                axis=1,
            )
            squared = np.maximum(
                evaluation_norms[:, None]
                + train_norms[None, :]
                - 2.0 * evaluation_matrix @ source_matrix.T,
                0.0,
            )
            relative = np.sqrt(squared) / np.maximum(
                np.sqrt(evaluation_norms)[:, None],
                1e-12,
            )
            nearest_distance = float(np.min(relative))

    target_matrix = _target_case_matrix(
        target_values,
        scenario.train_batch.case_shape,
    )
    (
        target_effective_rank,
        target_rank_99,
        target_rank_fraction,
        target_maximum_rank,
    ) = _rank_diagnostics(target_matrix, intrinsic_rank=target_intrinsic_rank)

    reasons = []
    if (
        identity_relative_l2 is not None
        and identity_relative_l2 <= criteria.minimum_persistence_change
    ):
        reasons.append("identity baseline nearly solves the training operator")
    if (
        persistence_relative_change is not None
        and persistence_relative_change <= criteria.minimum_persistence_change
    ):
        reasons.append("target dynamics are too close to persistence")
    if nearest_distance is None:
        reasons.append("nearest-realization novelty could not be measured")
    elif nearest_distance < criteria.minimum_nearest_realization_distance:
        reasons.append("test realizations are near-duplicates of training realizations")
    if source_effective_rank < min(
        criteria.minimum_source_effective_rank,
        source_maximum_rank,
    ):
        reasons.append("source effective rank is below the difficulty threshold")
    if source_rank_99 < min(criteria.minimum_source_rank_99, source_maximum_rank):
        reasons.append("source numerical rank is below the difficulty threshold")
    if source_rank_fraction < criteria.minimum_source_rank_fraction:
        reasons.append("source rank fraction is below the difficulty threshold")
    if target_effective_rank < min(
        criteria.minimum_target_effective_rank,
        target_maximum_rank,
    ):
        reasons.append("target effective rank is below the difficulty threshold")
    if target_rank_99 < min(criteria.minimum_target_rank_99, target_maximum_rank):
        reasons.append("target numerical rank is below the difficulty threshold")
    if target_rank_fraction < criteria.minimum_target_rank_fraction:
        reasons.append("target POD rank fraction is below the difficulty threshold")
    return ScenarioDifficultyAudit(
        scenario=scenario.name,
        identity_relative_l2=identity_relative_l2,
        persistence_relative_change=persistence_relative_change,
        nearest_realization_relative_distance=nearest_distance,
        source_effective_rank=source_effective_rank,
        source_rank_99=source_rank_99,
        source_rank_fraction=source_rank_fraction,
        target_effective_rank=target_effective_rank,
        target_rank_99=target_rank_99,
        target_rank_fraction=target_rank_fraction,
        passed=not reasons,
        reasons=tuple(reasons),
    )


def _relative_array_error(left, right) -> float:
    numerator = jnp.linalg.norm(jnp.asarray(left) - jnp.asarray(right))
    denominator = jnp.maximum(jnp.linalg.norm(jnp.asarray(right)), 1e-12)
    return float(numerator / denominator)


@cache
def native_kernel_parity_checks() -> tuple[KernelParityCheck, ...]:
    """Run independent dense/recurrent/equivariance checks for native kernels."""
    checks = []
    values = jr.normal(jr.key(100), (8, 10, 2))
    for factorization in ("cp", "tucker"):
        factorized = phx.nn.operator.architectures.SpectralConvND(
            in_channels=2,
            out_channels=3,
            n_modes=(3, 4),
            factorization=factorization,
            rank=2,
            key=jr.key(101),
        )
        dense = phx.nn.operator.architectures.SpectralConvND(
            in_channels=2,
            out_channels=3,
            n_modes=(3, 4),
            factorization="dense",
            key=jr.key(102),
        )
        corners = 2 ** (len(factorized.n_modes) - 1)
        dense_weight = jnp.stack(
            tuple(
                factorized._dense_weight(corner, factorized.n_modes)
                for corner in range(corners)
            )
        )
        dense = eqx.tree_at(lambda layer: layer.weight, dense, dense_weight)
        error = _relative_array_error(factorized(values), dense(values))
        checks.append(
            KernelParityCheck(
                name=f"{factorization}_spectral_reconstruction",
                family="spectral",
                reference="explicit dense Fourier weights",
                relative_error=error,
                tolerance=1e-11,
                passed=error <= 1e-11,
            )
        )

    source_axis = phx.nn.operator.OperatorAxis("x", jnp.linspace(0.0, 1.0, 7))
    query_axis = phx.nn.operator.OperatorAxis("x", jnp.linspace(0.0, 1.0, 11))
    deeponet_batch = phx.nn.operator.OperatorBatch(
        inputs={
            "forcing": phx.nn.operator.FunctionSamples(
                values=jr.normal(jr.key(106), (3, 7)),
                axes=(source_axis,),
            )
        },
        queries={
            "query": phx.nn.operator.FunctionSamples(values=None, axes=(query_axis,))
        },
        case_axes=("case",),
        case_shape=(3,),
    )
    deeponet = phx.nn.operator.architectures.DeepONet(
        branch=phx.nn.models.MLP(
            in_size=7,
            out_size=4,
            width_size=6,
            depth=2,
            key=jr.key(107),
        ),
        trunk=phx.nn.models.MLP(
            in_size=1,
            out_size=4,
            width_size=6,
            depth=2,
            key=jr.key(108),
        ),
        coord_dim=1,
        latent_size=4,
        source_key="forcing",
    )
    coefficients = deeponet.encode_sources(deeponet_batch, key=None)
    basis = deeponet._trunk_basis(
        deeponet_batch.require_single_query(),
        deeponet_batch.case_shape,
        key=None,
    )
    explicit = oe.contract("cql,cl->cq", basis[..., 0, :], coefficients)
    explicit = explicit + deeponet.bias[0]
    deeponet_error = _relative_array_error(deeponet(deeponet_batch), explicit)
    checks.append(
        KernelParityCheck(
            name="deeponet_explicit_contraction",
            family="branch_trunk",
            reference="explicit branch-trunk dot product",
            relative_error=deeponet_error,
            tolerance=1e-11,
            passed=deeponet_error <= 1e-11,
        )
    )

    time_axis = phx.nn.operator.OperatorAxis("t", jnp.linspace(0.0, 2.0, 33))
    temporal_batch = phx.nn.operator.OperatorBatch(
        inputs={
            "forcing": phx.nn.operator.FunctionSamples(
                values=jnp.sin(time_axis.nodes),
                axes=(time_axis,),
            )
        },
        queries={
            "query": phx.nn.operator.FunctionSamples(values=None, axes=(time_axis,))
        },
    )
    laplace = phx.nn.operator.architectures.LaplaceTemporalOperator(
        num_poles=4, key=jr.key(103)
    )
    laplace_error = _relative_array_error(
        laplace.recurrent(temporal_batch),
        laplace(temporal_batch),
    )
    checks.append(
        KernelParityCheck(
            name="laplace_direct_recurrent",
            family="laplace_temporal",
            reference="direct causal quadrature",
            relative_error=laplace_error,
            tolerance=1e-11,
            passed=laplace_error <= 1e-11,
        )
    )

    spherical_plan = phx.nn.operator.architectures.SphericalHarmonicPlan(
        4,
        sampling="mw",
        execution="recursive",
    )
    spherical = phx.nn.operator.architectures.SphericalSpectralConv(
        spherical_plan,
        in_channels=1,
        out_channels=1,
        key=jr.key(104),
    )
    spherical_values = jr.normal(jr.key(105), spherical_plan.sample_shape + (1,))
    shift = 2
    expected = jnp.roll(
        spherical(spherical_values, spherical_plan),
        shift,
        axis=1,
    )
    actual = spherical(
        jnp.roll(spherical_values, shift, axis=1),
        spherical_plan,
    )
    spherical_error = _relative_array_error(actual, expected)
    checks.append(
        KernelParityCheck(
            name="spherical_longitude_equivariance",
            family="spherical_spectral",
            reference="discrete longitude rotation",
            relative_error=spherical_error,
            tolerance=1e-10,
            passed=spherical_error <= 1e-10,
        )
    )
    source_values = jnp.array([[1.5], [-0.5], [2.0]])
    source_measure = jnp.array([0.2, 0.3, 0.5])
    edge_kernel = jnp.array([1.0, 0.5, 2.0, -1.0, 1.5, 0.25])
    graph = phx.graph.GraphIR(
        nodes={
            "features": jnp.concatenate(
                (source_values, jnp.zeros((2, 1))),
                axis=0,
            )
        },
        edges={"kernel_weight": edge_kernel},
        senders=jnp.array([0, 1, 2, 0, 1, 2]),
        receivers=jnp.array([3, 3, 3, 4, 4, 4]),
        n_node=jnp.array([5]),
        n_edge=jnp.array([6]),
    )
    graph_integral = phx.graph.GraphNeuralOperator(
        input_key="features",
        output_key="integral",
        source_measure=jnp.concatenate((source_measure, jnp.zeros((2,)))),
        normalize=False,
    )(graph)
    explicit_integral = jnp.array(
        [
            jnp.sum(source_values[:, 0] * source_measure * edge_kernel[:3]),
            jnp.sum(source_values[:, 0] * source_measure * edge_kernel[3:]),
        ]
    )
    integral_error = _relative_array_error(
        graph_integral.nodes["integral"][3:, 0],
        explicit_integral,
    )
    checks.append(
        KernelParityCheck(
            name="gino_weighted_integral_transform",
            family="geometry_informed",
            reference="official GINO weighted linear integral transform",
            relative_error=integral_error,
            tolerance=1e-12,
            passed=integral_error <= 1e-12,
        )
    )

    regional_processor = phx.nn.operator.layers.RegionalGraphProcessor(
        4,
        2,
        neighbors=3,
        depth=2,
        width=8,
        mlp_depth=1,
        key=jr.key(109),
    )
    regional_values = jr.normal(jr.key(110), (2, 6, 4))
    regional_coordinates = jr.uniform(jr.key(111), (2, 6, 2))
    regional_measure = jr.uniform(jr.key(112), (2, 6)) + 0.1
    regional_mask = jnp.array(
        [
            [True, True, True, True, True, True],
            [True, True, True, True, True, False],
        ]
    )
    permutation = jnp.array([3, 0, 5, 1, 4, 2])
    inverse_permutation = jnp.argsort(permutation)
    regional_reference = regional_processor(
        regional_values,
        regional_coordinates,
        regional_measure,
        regional_mask,
    )
    regional_permuted = regional_processor(
        regional_values[:, permutation],
        regional_coordinates[:, permutation],
        regional_measure[:, permutation],
        regional_mask[:, permutation],
    )[:, inverse_permutation]
    regional_error = _relative_array_error(
        regional_permuted,
        regional_reference,
    )
    checks.append(
        KernelParityCheck(
            name="rigno_regional_permutation_equivariance",
            family="regional_graph",
            reference="official RIGNO regional graph permutation equivariance",
            relative_error=regional_error,
            tolerance=1e-10,
            passed=regional_error <= 1e-10,
        )
    )

    transformer_processor = phx.nn.operator.layers.OperatorTransformerProcessor(
        (2, 2),
        2,
        patch_shape=1,
        model_width=4,
        depth=1,
        heads=2,
        key=jr.key(113),
    )
    middle_block = transformer_processor.middle_block
    if middle_block is None:
        raise RuntimeError("GAOT parity requires a non-empty transformer processor.")
    self_attention = middle_block.attention
    attention_values = jr.normal(jr.key(114), (1, 4, 4))
    attention_measure = jnp.array([[0.1, 0.2, 0.3, 0.4]])
    attention_mask = jnp.array([[True, True, True, False]])
    attention_actual = self_attention(
        attention_values,
        attention_measure,
        attention_mask,
    )
    attention_query = self_attention.query(attention_values).reshape(
        (1, 4, self_attention.heads, self_attention.head_dim)
    )
    attention_key = self_attention.key(attention_values).reshape(
        (1, 4, self_attention.heads, self_attention.head_dim)
    )
    attention_value = self_attention.value(attention_values).reshape(
        (1, 4, self_attention.heads, self_attention.head_dim)
    )
    attention_logits = oe.contract(
        "bqhd,bkhd->bhqk",
        attention_query,
        attention_key,
    ) / jnp.sqrt(float(self_attention.head_dim))
    attention_log_measure = jnp.where(
        attention_mask,
        jnp.log(attention_measure),
        jnp.asarray(-1e30, dtype=attention_logits.dtype),
    )
    attention_weights = jax.nn.softmax(
        attention_logits + attention_log_measure[:, None, None, :],
        axis=-1,
    )
    attention_explicit = self_attention.output(
        oe.contract(
            "bhqk,bkhd->bqhd",
            attention_weights,
            attention_value,
        ).reshape((1, 4, 4))
    )
    attention_explicit = attention_explicit * attention_mask[..., None].astype(
        attention_explicit.dtype
    )
    attention_error = _relative_array_error(
        attention_actual,
        attention_explicit,
    )
    checks.append(
        KernelParityCheck(
            name="gaot_measure_aware_scaled_dot_product_attention",
            family="geometry_transformer",
            reference="official GAOT scaled dot-product self-attention",
            relative_error=attention_error,
            tolerance=1e-12,
            passed=attention_error <= 1e-12,
        )
    )
    return tuple(checks)


def kernel_parity_checks(
    scenarios: tuple[OperatorBenchmarkScenario, ...],
    /,
) -> tuple[KernelParityCheck, ...]:
    checks = list(native_kernel_parity_checks())
    for scenario in scenarios:
        evidence = scenario.reference_evidence
        if evidence is None:
            checks.append(
                KernelParityCheck(
                    name=scenario.name,
                    family="reference_solver",
                    reference="missing",
                    relative_error=math.inf,
                    tolerance=0.0,
                    passed=False,
                )
            )
        else:
            checks.append(
                KernelParityCheck(
                    name=scenario.name,
                    family="reference_solver",
                    reference=evidence.method,
                    relative_error=float(evidence.relative_error),
                    tolerance=float(evidence.tolerance),
                    passed=evidence.passed,
                )
            )
    return tuple(checks)


def load_family_parity_evidence(path: str | Path, /) -> tuple[FamilyParityEvidence, ...]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise TypeError("Family parity evidence must be a JSON list.")
    evidence = []
    for record in payload:
        checks = tuple(KernelParityCheck(**check) for check in record.get("checks", ()))
        evidence.append(
            FamilyParityEvidence(
                family=record["family"],
                reference_uri=record["reference_uri"],
                revision=record["revision"],
                status=record["status"],
                checks=checks,
            )
        )
    return tuple(evidence)


class _NormalizedOperator(eqx.Module):
    model: Callable[[phx.nn.operator.OperatorBatch], jax.Array]
    input_statistics: tuple[tuple[str, float, float], ...] = eqx.field(static=True)
    target_location: float = eqx.field(static=True)
    target_scale: float = eqx.field(static=True)
    policy: str = eqx.field(static=True)
    domain_support_key: str | None = eqx.field(static=True)
    conservation_source_key: str | None = eqx.field(static=True)

    def __init__(
        self,
        model,
        input_statistics: tuple[tuple[str, float, float], ...],
        target_location: float,
        target_scale: float,
        policy: str,
        *,
        domain_support_key: str | None,
        conservation_source_key: str | None,
    ):
        self.model = model
        self.input_statistics = input_statistics
        self.target_location = float(target_location)
        self.target_scale = float(target_scale)
        self.policy = str(policy)
        self.domain_support_key = domain_support_key
        self.conservation_source_key = conservation_source_key

    def __call__(self, batch: phx.nn.operator.OperatorBatch):
        statistics = {
            name: (location, scale) for name, location, scale in self.input_statistics
        }
        inputs = {}
        for name, samples in batch.inputs.items():
            values = samples.values
            if values is not None:
                values = jnp.asarray(values)
                if name == self.domain_support_key:
                    pass
                elif name == self.conservation_source_key:
                    source_measure = samples.weights(case_shape=batch.case_shape)
                    query_measure = batch.require_single_query().weights(
                        case_shape=batch.case_shape
                    )
                    source_axes = tuple(range(len(batch.case_shape), source_measure.ndim))
                    query_axes = tuple(range(len(batch.case_shape), query_measure.ndim))
                    source_total = jnp.sum(source_measure, axis=source_axes)
                    query_total = jnp.sum(query_measure, axis=query_axes)
                    source_total = eqx.error_if(
                        source_total,
                        jnp.any(source_total <= 0.0),
                        "Conservation-aware normalization requires positive source "
                        "measure in every case.",
                    )
                    offset = (self.target_location * query_total / source_total).reshape(
                        batch.case_shape
                        + (1,) * len(samples.sample_shape)
                        + (1,)
                        * (
                            values.ndim
                            - len(batch.case_shape)
                            - len(samples.sample_shape)
                        )
                    )
                    values = (values - offset) / self.target_scale
                else:
                    location, scale = statistics[name]
                    values = (values - location) / scale
            inputs[name] = phx.nn.operator.FunctionSamples(
                values=values,
                axes=samples.axes,
                coordinates=samples.coordinates,
                quadrature_weights=samples.quadrature_weights,
                mask=samples.mask,
            )
        normalized_batch = phx.nn.operator.OperatorBatch(
            inputs=inputs,
            queries={"query": batch.require_single_query()},
            case_axes=batch.case_axes,
            case_shape=batch.case_shape,
        )
        prediction = self.model(normalized_batch)
        return self.target_location + self.target_scale * prediction


def _normalization_statistics(
    scenario: OperatorBenchmarkScenario,
    policy: str,
    /,
):
    statistics = []
    for name, samples in scenario.train_batch.inputs.items():
        if samples.values is None:
            raise ValueError("Operator normalization requires valued inputs.")
        values = jnp.asarray(samples.values)
        location = float(jnp.mean(values))
        centered = values - location
        if policy == "spectral":
            scale = float(jnp.sqrt(jnp.mean(centered**2)))
        else:
            scale = float(jnp.std(values))
        statistics.append((name, location, max(scale, 1e-8)))
    target = jnp.asarray(scenario.train_target)
    target_location = float(jnp.mean(target))
    target_scale = max(float(jnp.std(target)), 1e-8)
    if scenario.conservation_source_key is not None:
        source_statistics = {
            name: (location, scale) for name, location, scale in statistics
        }
        target_location, target_scale = source_statistics[
            scenario.conservation_source_key
        ]
    return tuple(statistics), target_location, target_scale


def _normalized_model(model, scenario, architecture, protocol):
    if (
        not protocol.normalize
        or architecture.normalization == "none"
        or not architecture.trainable
    ):
        return model, "none"
    statistics, target_location, target_scale = _normalization_statistics(
        scenario,
        architecture.normalization,
    )
    return (
        _NormalizedOperator(
            model,
            statistics,
            target_location,
            target_scale,
            architecture.normalization,
            domain_support_key=scenario.domain_support_key,
            conservation_source_key=scenario.conservation_source_key,
        ),
        architecture.normalization,
    )


def _query_point_count(scenario: OperatorBenchmarkScenario) -> int:
    return sum(
        int(np.prod(query.sample_shape, dtype=np.int64))
        for query in scenario.train_batch.queries.values()
    )


def _compute_units(
    parameters: int,
    steps: int,
    scenario: OperatorBenchmarkScenario,
    /,
) -> int:
    case_count = int(np.prod(scenario.train_batch.case_shape, dtype=np.int64))
    return (
        max(1, int(parameters)) * int(steps) * case_count * _query_point_count(scenario)
    )


def _base_parameter_counts(
    architectures: tuple[OperatorArchitecture, ...],
    scenario: OperatorBenchmarkScenario,
    /,
) -> tuple[int, ...]:
    return tuple(
        parameter_count(architecture.build(scenario, 0))
        for architecture in architectures
        if architecture.trainable and architecture.promotion_scope != "reference"
    )


def _architecture_parameter_counts(
    architecture: OperatorArchitecture,
    scenario: OperatorBenchmarkScenario,
    protocol: OperatorBenchmarkProtocol,
    /,
) -> tuple[int, ...]:
    if not architecture.trainable:
        return (parameter_count(architecture.build(scenario, 0)),)
    return tuple(
        parameter_count(architecture.build(scenario, 0, size_scale=scale))
        for scale in protocol.size_scales
    )


def _target_parameters(
    architectures: tuple[OperatorArchitecture, ...],
    scenario: OperatorBenchmarkScenario,
    protocol: OperatorBenchmarkProtocol,
    /,
) -> int:
    counts = _base_parameter_counts(architectures, scenario)
    default_target = (
        1 if not counts else max(1, round(float(np.median(np.asarray(counts)))))
    )
    if protocol.comparison != "capacity":
        return (
            default_target
            if protocol.target_parameters is None
            else int(protocol.target_parameters)
        )
    ranges = tuple(
        (
            architecture.name,
            min(values),
            max(values),
        )
        for architecture in architectures
        if architecture.trainable and architecture.promotion_scope != "reference"
        for values in (_architecture_parameter_counts(architecture, scenario, protocol),)
    )
    if not ranges:
        return default_target
    common_minimum = max(minimum for _, minimum, _ in ranges)
    common_maximum = min(maximum for _, _, maximum in ranges)
    if common_minimum > common_maximum:
        detail = ", ".join(
            f"{name}=[{minimum}, {maximum}]" for name, minimum, maximum in ranges
        )
        raise ValueError(
            "Selected architecture parameter ranges do not overlap for a "
            f"capacity comparison: {detail}. Expand --size-scales or use "
            "--comparison compute/pareto."
        )
    target = (
        min(max(default_target, common_minimum), common_maximum)
        if protocol.target_parameters is None
        else int(protocol.target_parameters)
    )
    if not common_minimum <= target <= common_maximum:
        raise ValueError(
            f"Capacity target {target} is outside the common feasible range "
            f"[{common_minimum}, {common_maximum}]."
        )
    return target


def _select_size_scale(
    architecture: OperatorArchitecture,
    scenario: OperatorBenchmarkScenario,
    protocol: OperatorBenchmarkProtocol,
    target_parameters: int,
    /,
) -> tuple[float, int]:
    if protocol.comparison != "capacity" or not architecture.trainable:
        model = architecture.build(scenario, 0)
        return 1.0, parameter_count(model)
    candidates = []
    for scale in protocol.size_scales:
        count = parameter_count(architecture.build(scenario, 0, size_scale=scale))
        distance = abs(math.log(max(1, count) / max(1, target_parameters)))
        candidates.append((distance, float(scale), count))
    _, scale, count = min(candidates)
    return scale, count


def _comparison_record(
    architecture: OperatorArchitecture,
    scenario: OperatorBenchmarkScenario,
    protocol: OperatorBenchmarkProtocol,
    target_parameters: int,
    size_scale: float,
    actual_parameters: int,
    minimum_parameters: int,
    maximum_parameters: int,
    target_compute_units: int | None,
    training_step_flops: int,
    training_step_bytes: int,
    /,
) -> BenchmarkComparisonRecord:
    if protocol.comparison == "compute":
        if target_compute_units is None:
            raise ValueError("Compute matching requires a positive compute target.")
        resolved_target_compute: int | None = int(target_compute_units)
    else:
        resolved_target_compute = None
    if not architecture.trainable or protocol.steps == 0:
        planned_steps = 0
    elif protocol.comparison == "compute":
        if resolved_target_compute is None or int(training_step_flops) <= 0:
            raise ValueError("Compute matching requires a positive XLA FLOP estimate.")
        planned_steps = max(
            1,
            min(
                4 * protocol.steps,
                round(resolved_target_compute / training_step_flops),
            ),
        )
    else:
        planned_steps = protocol.steps
    if protocol.comparison == "compute":
        if resolved_target_compute is None:
            raise ValueError("Compute matching requires a positive compute target.")
        actual_compute = int(training_step_flops) * int(planned_steps)
        compute_ratio = actual_compute / resolved_target_compute
        compute_measurement: Literal["proxy", "jax_flops"] = "jax_flops"
    elif protocol.comparison == "pareto":
        actual_compute = int(training_step_flops) * int(planned_steps)
        compute_ratio = None
        compute_measurement = "jax_flops"
    else:
        actual_compute = _compute_units(
            actual_parameters,
            planned_steps,
            scenario,
        )
        compute_ratio = None
        compute_measurement = "proxy"
    capacity_ratio = (
        None
        if protocol.comparison != "capacity"
        else actual_parameters / max(1, target_parameters)
    )
    target_in_range = (
        None
        if protocol.comparison != "capacity"
        else minimum_parameters <= target_parameters <= maximum_parameters
    )
    comparable = target_in_range is not False
    comparison_reason = (
        None
        if comparable
        else (
            f"capacity target {target_parameters} is outside architecture range "
            f"[{minimum_parameters}, {maximum_parameters}]"
        )
    )
    normalization = (
        architecture.normalization
        if protocol.normalize and architecture.trainable
        else "none"
    )
    return BenchmarkComparisonRecord(
        scenario=scenario.name,
        architecture=architecture.name,
        mode=protocol.comparison,
        size_scale=float(size_scale),
        target_parameters=(
            int(target_parameters) if protocol.comparison == "capacity" else None
        ),
        actual_parameters=int(actual_parameters),
        capacity_ratio=capacity_ratio,
        planned_steps=int(planned_steps),
        target_compute_units=resolved_target_compute,
        actual_compute_units=int(actual_compute),
        compute_ratio=compute_ratio,
        normalization=normalization,
        training_step_flops=int(training_step_flops),
        training_step_bytes=int(training_step_bytes),
        compute_measurement=compute_measurement,
        minimum_parameters=int(minimum_parameters),
        maximum_parameters=int(maximum_parameters),
        target_in_range=target_in_range,
        comparable=comparable,
        comparison_reason=comparison_reason,
    )


def _selected_trial_index(results: list[OperatorBenchmarkResult]) -> int:
    scores = tuple(
        result.final_loss if result.validation_loss is None else result.validation_loss
        for result in results
    )
    return int(np.argmin(np.asarray(scores)))


@dataclass(frozen=True)
class _ParetoCandidate:
    comparison: BenchmarkComparisonRecord
    validation: float | None
    shifted: float | None
    training_flops: int | None
    inference: float | None
    memory: float | None
    parameters: float
    metrics: tuple[float, float, float, float, float, float] | None


def _pareto_metrics(
    validation: float | None,
    shifted: float | None,
    training_flops: int | None,
    inference: float | None,
    memory: float | None,
    parameters: float,
    /,
) -> tuple[float, float, float, float, float, float] | None:
    if (
        validation is None
        or shifted is None
        or training_flops is None
        or inference is None
        or memory is None
    ):
        return None
    return (
        validation,
        shifted,
        float(training_flops),
        inference,
        memory,
        parameters,
    )


def _pareto_fronts(
    aggregates: tuple[OperatorBenchmarkAggregate, ...],
    trials: tuple[HyperparameterTrial, ...],
    comparisons: tuple[BenchmarkComparisonRecord, ...],
    /,
) -> tuple[BenchmarkParetoFront, ...]:
    grouped = {}
    for comparison in comparisons:
        grouped.setdefault(comparison.scenario, []).append(comparison)
    fronts = []
    for scenario in sorted(grouped):
        raw_points: list[_ParetoCandidate] = []
        for comparison in grouped[scenario]:
            key = (
                comparison.scenario,
                comparison.architecture,
                comparison.size_scale,
            )
            rows = [
                row
                for row in aggregates
                if (
                    row.scenario,
                    row.architecture,
                    row.size_scale,
                )
                == key
            ]
            nominal = [
                row
                for row in rows
                if row.shift == "in_distribution" and row.split == "test"
            ]
            if not nominal:
                nominal = [row for row in rows if row.shift == "in_distribution"]
            shifted = [row for row in rows if row.shift != "in_distribution"]
            selected_trials = [
                trial
                for trial in trials
                if (
                    trial.scenario,
                    trial.architecture,
                    trial.size_scale,
                    trial.selected,
                )
                == (
                    comparison.scenario,
                    comparison.architecture,
                    comparison.size_scale,
                    True,
                )
            ]
            validation_values = [
                (
                    trial.final_loss
                    if trial.validation_loss is None
                    else trial.validation_loss
                )
                for trial in selected_trials
            ]
            memory_values = [
                row.peak_memory_bytes_mean
                for row in rows
                if row.peak_memory_bytes_mean is not None
            ]
            validation_error = (
                None
                if not validation_values
                else float(np.mean(np.asarray(validation_values)))
            )
            shifted_error = (
                None if not shifted else max(row.relative_l2_mean for row in shifted)
            )
            inference_seconds = (
                None
                if not nominal
                else max(row.inference_seconds_mean for row in nominal)
            )
            peak_memory = None if not memory_values else max(memory_values)
            training_flops = (
                comparison.actual_compute_units
                if comparison.compute_measurement == "jax_flops"
                else None
            )
            metrics = _pareto_metrics(
                validation_error,
                shifted_error,
                training_flops,
                inference_seconds,
                peak_memory,
                float(comparison.actual_parameters),
            )
            raw_points.append(
                _ParetoCandidate(
                    comparison=comparison,
                    validation=validation_error,
                    shifted=shifted_error,
                    training_flops=training_flops,
                    inference=inference_seconds,
                    memory=peak_memory,
                    parameters=float(comparison.actual_parameters),
                    metrics=metrics,
                )
            )

        points: list[BenchmarkParetoPoint] = []
        for candidate in raw_points:
            comparison = candidate.comparison
            dominated_by = []
            candidate_metrics = candidate.metrics
            if candidate_metrics is not None:
                for other in raw_points:
                    other_metrics = other.metrics
                    if other is candidate or other_metrics is None:
                        continue
                    if all(
                        left <= right
                        for left, right in zip(other_metrics, candidate_metrics)
                    ) and any(
                        left < right
                        for left, right in zip(other_metrics, candidate_metrics)
                    ):
                        other_comparison = other.comparison
                        dominated_by.append(
                            f"{other_comparison.architecture}"
                            f"@{other_comparison.size_scale:g}"
                        )
            nondominated = None if candidate_metrics is None else not dominated_by
            points.append(
                BenchmarkParetoPoint(
                    scenario=scenario,
                    architecture=comparison.architecture,
                    family=next(
                        (
                            row.family
                            for row in aggregates
                            if (
                                row.scenario,
                                row.architecture,
                                row.size_scale,
                            )
                            == (
                                comparison.scenario,
                                comparison.architecture,
                                comparison.size_scale,
                            )
                        ),
                        "unspecified",
                    ),
                    size_scale=comparison.size_scale,
                    validation_relative_l2=candidate.validation,
                    shifted_relative_l2=candidate.shifted,
                    training_flops=candidate.training_flops,
                    inference_seconds=candidate.inference,
                    peak_memory_bytes=candidate.memory,
                    parameter_count=candidate.parameters,
                    complete=candidate_metrics is not None,
                    nondominated=nondominated,
                    dominated_by=tuple(sorted(dominated_by)),
                )
            )
        labels = tuple(
            f"{point.architecture}@{point.size_scale:g}"
            for point in points
            if point.complete
        )
        nondominated_labels = tuple(
            f"{point.architecture}@{point.size_scale:g}"
            for point in points
            if point.nondominated is True
        )
        fronts.append(
            BenchmarkParetoFront(
                scenario=scenario,
                points=tuple(points),
                complete_architectures=labels,
                nondominated_architectures=nondominated_labels,
            )
        )
    return tuple(fronts)


def _promotion_reports(
    aggregates: tuple[OperatorBenchmarkAggregate, ...],
    audits: tuple[ScenarioIntegrityAudit, ...],
    comparisons: tuple[BenchmarkComparisonRecord, ...],
    scopes: dict[tuple[str, str] | tuple[str, str, float], str],
    family_parity: tuple[FamilyParityEvidence, ...],
    criteria: PromotionCriteria,
    /,
    production_run: bool = True,
    provenance_pinned: bool = True,
    external_audits: tuple[ExternalCandidateAudit, ...] = (),
    difficulty_audits: tuple[ScenarioDifficultyAudit, ...] = (),
) -> tuple[ArchitecturePromotionReport, ...]:
    audit_lookup = {audit.scenario: audit for audit in audits}
    difficulty_lookup = {audit.scenario: audit for audit in difficulty_audits}
    comparison_lookup = {
        (record.scenario, record.architecture, record.size_scale): record
        for record in comparisons
    }
    current_parity = {}
    for check in native_kernel_parity_checks():
        current_parity.setdefault(check.family, []).append(check)
    parity_lookup = {evidence.family: evidence for evidence in family_parity}
    external_audit_lookup = {audit.candidate: audit for audit in external_audits}
    if len(external_audit_lookup) != len(external_audits):
        raise ValueError("External candidate audits must have unique names.")
    grouped = {}
    for aggregate in aggregates:
        key = (
            aggregate.scenario,
            aggregate.architecture,
            aggregate.size_scale,
        )
        grouped.setdefault(key, []).append(aggregate)
    reports = []
    for key in sorted(grouped):
        scenario, architecture, size_scale = key
        rows = grouped[key]
        family = rows[0].family
        scope = scopes.get(key)
        if scope is None:
            scope = scopes[(scenario, architecture)]
        base_rows = [
            row for row in rows if row.shift == "in_distribution" and row.split == "test"
        ]
        if not base_rows:
            base_rows = [row for row in rows if row.shift == "in_distribution"]
        base_error = max(row.relative_l2_mean for row in base_rows)
        shifted_rows = [row for row in rows if row.shift != "in_distribution"]
        degradation = (
            None
            if not shifted_rows
            else max(row.relative_l2_mean for row in shifted_rows)
            / max(base_error, 1e-12)
        )
        seed_count = min(len(row.seeds) for row in rows)
        maximum_std = max(row.relative_l2_std for row in rows)
        maximum_inference = max(row.inference_seconds_mean for row in rows)
        parameters = max(row.parameter_count_mean for row in rows)
        memory_values = [
            row.peak_memory_bytes_mean
            for row in rows
            if row.peak_memory_bytes_mean is not None
        ]
        maximum_memory = None if not memory_values else max(memory_values)
        accuracy_passed = base_error <= criteria.maximum_relative_l2
        robustness_passed = (
            degradation is not None and degradation <= criteria.maximum_shift_degradation
        )
        reproducibility_passed = (
            seed_count >= criteria.minimum_seeds
            and maximum_std <= criteria.maximum_seed_std
        )
        efficiency_passed = (
            maximum_inference <= criteria.maximum_inference_seconds
            and parameters <= criteria.maximum_parameter_count
        )
        if criteria.maximum_peak_memory_bytes is not None:
            efficiency_passed = (
                efficiency_passed
                and maximum_memory is not None
                and maximum_memory <= criteria.maximum_peak_memory_bytes
            )
        integrity_passed = audit_lookup[scenario].passed
        difficulty_audit = difficulty_lookup.get(scenario)
        baseline_hardness_passed = not criteria.require_baseline_hardness or (
            difficulty_audit is not None and difficulty_audit.passed
        )
        convergence_passed = not criteria.require_convergence or all(
            row.convergence_rate >= 1.0 for row in rows
        )
        parity = parity_lookup.get(family)
        current_checks = current_parity.get(family, ())
        parity_passed = not criteria.require_parity or (
            parity is not None
            and parity.verified
            and all(check.passed for check in current_checks)
        )
        comparison = comparison_lookup[key]
        if not comparison.comparable:
            comparison_passed = False
        elif comparison.mode == "capacity":
            comparison_passed = (
                comparison.capacity_ratio is not None
                and criteria.minimum_capacity_ratio
                <= comparison.capacity_ratio
                <= criteria.maximum_capacity_ratio
            )
        elif comparison.mode == "compute":
            comparison_passed = (
                comparison.compute_ratio is not None
                and criteria.minimum_compute_ratio
                <= comparison.compute_ratio
                <= criteria.maximum_compute_ratio
            )
        else:
            comparison_passed = False
        external_audit = external_audit_lookup.get(architecture)
        external_audit_passed = (
            None
            if scope != "external"
            else (
                external_audit is not None
                and external_audit.eligible
                and external_audit.artifact_verified
            )
        )
        reasons = []
        if scope == "reference":
            reasons.append("reference baselines are not promotion candidates")
        if not production_run:
            reasons.append("quick smoke profiles are not promotion-eligible")
        if not provenance_pinned:
            reasons.append("immutable benchmark commit identity is required")
        if not accuracy_passed:
            reasons.append("accuracy gate failed")
        if not robustness_passed:
            reasons.append("robustness gate failed or no shifted evaluation exists")
        if not reproducibility_passed:
            reasons.append("seed-count or seed-variance gate failed")
        if not efficiency_passed:
            reasons.append("runtime, memory, or parameter-count gate failed")
        if not integrity_passed:
            reasons.append("scenario integrity gate failed")
        if not baseline_hardness_passed:
            reasons.append("persistence, leakage, or POD-rank difficulty gate failed")
        if not convergence_passed:
            reasons.append("one or more selected learning curves did not converge")
        if not parity_passed:
            reasons.append("verified upstream family parity evidence is absent")
        if not comparison_passed:
            if comparison.mode == "pareto":
                reasons.append("Pareto reporting is not a promotion matching gate")
            else:
                reasons.append("capacity or compute matching tolerance failed")
        if external_audit_passed is False:
            reasons.append("external provenance and license audit failed or is absent")
        promoted = scope != "reference" and not reasons
        if promoted and scope == "external":
            tier: PromotionTier = "external"
        elif promoted and scope == "specialized":
            tier = "specialized"
        elif promoted:
            tier = "validated"
        else:
            tier = "experimental"
        reports.append(
            ArchitecturePromotionReport(
                scenario=scenario,
                architecture=architecture,
                size_scale=size_scale,
                family=family,
                scope=scope,
                tier=tier,
                accuracy_passed=accuracy_passed,
                robustness_passed=robustness_passed,
                reproducibility_passed=reproducibility_passed,
                efficiency_passed=efficiency_passed,
                integrity_passed=integrity_passed,
                parity_passed=parity_passed,
                comparison_passed=comparison_passed,
                baseline_hardness_passed=baseline_hardness_passed,
                convergence_passed=convergence_passed,
                external_audit_passed=external_audit_passed,
                in_distribution_relative_l2=base_error,
                worst_shift_degradation=degradation,
                maximum_seed_std=maximum_std,
                maximum_inference_seconds=maximum_inference,
                parameter_count_mean=parameters,
                peak_memory_bytes_mean=maximum_memory,
                seed_count=seed_count,
                promoted=promoted,
                reasons=tuple(reasons),
            )
        )
    return tuple(reports)


def _portfolio_promotions(
    reports: tuple[ArchitecturePromotionReport, ...],
    criteria: PromotionCriteria,
    /,
) -> tuple[ArchitecturePortfolioPromotion, ...]:
    grouped = {}
    for report in reports:
        grouped.setdefault((report.architecture, report.size_scale), []).append(report)
    portfolio = []
    for architecture, size_scale in sorted(grouped):
        rows = grouped[(architecture, size_scale)]
        family = rows[0].family
        scope = rows[0].scope
        if any(row.family != family or row.scope != scope for row in rows):
            raise ValueError(
                f"Architecture {architecture!r} has inconsistent family or scope."
            )
        scenario_count = len(rows)
        passed_scenarios = sum(row.promoted for row in rows)
        if scope == "external":
            required_scenarios = criteria.minimum_external_scenarios
        elif scope == "general":
            required_scenarios = criteria.minimum_general_scenarios
        else:
            required_scenarios = 1
        reasons = []
        if scope == "reference":
            reasons.append("reference baselines are not promotion candidates")
        if passed_scenarios != scenario_count:
            reasons.append("one or more scenario-level promotion gates failed")
        if scenario_count < required_scenarios:
            reasons.append(f"requires at least {required_scenarios} compatible scenarios")
        promoted = scope != "reference" and not reasons
        if promoted and scope == "external":
            tier: PromotionTier = "external"
        elif promoted and scope == "specialized":
            tier = "specialized"
        elif promoted:
            tier = "validated"
        else:
            tier = "experimental"
        portfolio.append(
            ArchitecturePortfolioPromotion(
                architecture=architecture,
                size_scale=size_scale,
                family=family,
                scope=scope,
                tier=tier,
                scenario_count=scenario_count,
                passed_scenarios=passed_scenarios,
                promoted=promoted,
                reasons=tuple(reasons),
            )
        )
    return tuple(portfolio)


def _trial_checkpoint_path(
    protocol: OperatorBenchmarkProtocol,
    scenario: OperatorBenchmarkScenario,
    architecture: OperatorArchitecture,
    /,
    *,
    seed: int,
    learning_rate: float,
    size_scale: float,
    normalization: str,
) -> tuple[Path | None, dict[str, object]]:
    identity = {
        "scenario_checksum": scenario_checksum(scenario),
        "architecture": architecture.name,
        "family": architecture.family,
        "architecture_configuration": dict(
            architecture.configuration(scenario, size_scale=size_scale)
        ),
        "seed": int(seed),
        "learning_rate": float(learning_rate),
        "size_scale": float(size_scale),
        "normalization": normalization,
        "comparison": protocol.comparison,
        "commit_identity": protocol.commit_identity,
    }
    if protocol.checkpoint_directory is None or not architecture.trainable:
        return None, identity
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    digest = hashlib.sha256(encoded).hexdigest()[:20]
    scenario_name = "".join(
        character if character.isalnum() or character in "-_" else "-"
        for character in scenario.name
    )
    architecture_name = "".join(
        character if character.isalnum() or character in "-_" else "-"
        for character in architecture.name
    )
    path = (
        Path(protocol.checkpoint_directory) / scenario_name / architecture_name / digest
    )
    return path, identity


def _training_subset_scenario(
    scenario: OperatorBenchmarkScenario,
    count: int,
    /,
) -> OperatorBenchmarkScenario:
    total = int(scenario.train_batch.case_shape[0])
    requested = int(count)
    if requested <= 0 or requested > total:
        raise ValueError("Training subset count must lie in [1, train_cases].")
    indices = jnp.arange(requested)
    target = (
        scenario.train_target.take(indices)
        if isinstance(scenario.train_target, phx.nn.operator.OperatorTargetBatch)
        else jnp.take(scenario.train_target, indices, axis=0)
    )
    return replace(
        scenario,
        train_batch=scenario.train_batch.take(indices),
        train_target=target,
        case_ids=scenario.case_ids[:requested],
        metadata=scenario.metadata + (("sample_efficiency_train_cases", str(requested)),),
    )


def _sample_efficiency_curve(
    architecture: OperatorArchitecture,
    scenario: OperatorBenchmarkScenario,
    protocol: OperatorBenchmarkProtocol,
    /,
    *,
    seed: int,
    size_scale: float,
    learning_rate: float,
    planned_steps: int,
    full_result: OperatorBenchmarkResult,
    evaluation: OperatorBenchmarkEvaluation,
    full_evaluation,
) -> SampleEfficiencyCurve:
    total = int(scenario.train_batch.case_shape[0])
    counts = tuple(
        dict.fromkeys(
            max(1, min(total, round(total * fraction)))
            for fraction in protocol.sample_fractions
        )
    )
    fractions = []
    relative_l2 = []
    maximum_absolute_error = []
    training_seconds = []
    parameter_counts = []
    for count in counts:
        fractions.append(float(count / total))
        if count == total:
            current_evaluation = full_evaluation
            current_result = full_result
        else:
            subset = _training_subset_scenario(scenario, count)
            training_subset = architecture.training_scenario(subset)
            model = architecture.build(
                subset,
                seed,
                size_scale=size_scale,
            )
            model, _ = _normalized_model(
                model,
                training_subset,
                architecture,
                protocol,
            )
            model, current_result = run_operator_benchmark(
                model,
                training_subset,
                steps=planned_steps,
                learning_rate=learning_rate,
                repeats=protocol.repeats,
                size_scale=size_scale,
                architecture=architecture.name,
                family=architecture.family,
                architecture_configuration=architecture.configuration(
                    subset,
                    size_scale=size_scale,
                ),
                seed=seed,
                trainable=architecture.trainable,
                validation_interval=protocol.validation_interval,
                patience=protocol.patience,
                minimum_delta=protocol.minimum_delta,
                relative_minimum_delta=protocol.relative_minimum_delta,
                checkpoint_key=jr.key(seed),
                run_evaluations=False,
            )
            current_evaluation = evaluate_operator(
                model,
                evaluation,
                repeats=protocol.repeats,
            )
        relative_l2.append(float(current_evaluation.relative_l2))
        maximum_absolute_error.append(float(current_evaluation.maximum_absolute_error))
        training_seconds.append(float(current_result.training_seconds))
        parameter_counts.append(int(current_result.parameter_count))

    x_values = np.asarray(fractions, dtype=float)
    y_values = np.asarray(relative_l2, dtype=float)
    if len(fractions) == 1:
        area = float(y_values[0])
    else:
        area = float(
            np.trapezoid(y_values, x_values)
            / max(float(x_values[-1] - x_values[0]), 1e-12)
        )
    return SampleEfficiencyCurve(
        scenario=scenario.name,
        architecture=architecture.name,
        family=architecture.family,
        seed=int(seed),
        evaluation=evaluation.name,
        learning_rate=float(learning_rate),
        size_scale=float(size_scale),
        sample_fractions=tuple(fractions),
        train_cases=counts,
        relative_l2=tuple(relative_l2),
        maximum_absolute_error=tuple(maximum_absolute_error),
        training_seconds=tuple(training_seconds),
        parameter_counts=tuple(parameter_counts),
        area_under_sample_error_curve=area,
    )


def run_operator_benchmark_v2(
    ladders: tuple[OperatorBenchmarkLadder, ...],
    /,
    *,
    protocol: OperatorBenchmarkProtocol = OperatorBenchmarkProtocol(),
    architecture_names: tuple[str, ...] | None = None,
    architecture_extensions: Callable[
        [OperatorBenchmarkScenario], tuple[OperatorArchitecture, ...]
    ]
    | None = None,
    family_parity: tuple[FamilyParityEvidence, ...] = (),
    external_audits: tuple[ExternalCandidateAudit, ...] = (),
    promotion_criteria: PromotionCriteria = PromotionCriteria(),
    difficulty: Literal["easy", "hard"] | None = None,
) -> OperatorBenchmarkV2Result:
    """Run audited, normalized, searched, and promotion-gated benchmark ladders."""
    unsplit = flatten_operator_benchmark_ladders(ladders, difficulty=difficulty)
    scenarios = tuple(
        split_operator_scenario(
            scenario,
            seed=protocol.split_seed,
            train_fraction=protocol.train_fraction,
            validation_fraction=protocol.validation_fraction,
        )
        if scenario.validation is None
        else scenario
        for scenario in unsplit
    )
    difficulty_scenarios = scenarios
    if protocol.sensor_training_dropout > 0.0:
        scenarios = tuple(
            add_training_sensor_dropout(
                scenario,
                drop_fraction=protocol.sensor_training_dropout,
                seed=protocol.split_seed + index,
            )
            for index, scenario in enumerate(scenarios)
        )
    audits = tuple(
        audit_operator_scenario(
            scenario,
            near_identity_threshold=protocol.near_identity_threshold,
            quick=protocol.quick,
        )
        for scenario in scenarios
    )
    failed = tuple(audit for audit in audits if not audit.passed)
    if failed:
        messages = "; ".join(
            f"{audit.scenario}: {', '.join(audit.reasons)}" for audit in failed
        )
        raise ValueError(f"Benchmark scenario integrity audit failed: {messages}")
    symmetry_audits = tuple(
        audit_symmetry_scenario(scenario)
        for scenario in scenarios
        if scenario.symmetry is not None
    )
    failed_symmetry = tuple(audit for audit in symmetry_audits if not audit.passed)
    if failed_symmetry:
        messages = "; ".join(
            f"{audit.scenario}: {', '.join(audit.reasons)}" for audit in failed_symmetry
        )
        raise ValueError(f"Benchmark symmetry audit failed: {messages}")
    difficulty_audits = tuple(
        audit_scenario_difficulty(scenario, promotion_criteria)
        for scenario in difficulty_scenarios
    )
    selected_names = None if architecture_names is None else set(architecture_names)
    architecture_sets = []
    for scenario in scenarios:
        architectures = compatible_architectures(scenario, quick=protocol.quick)
        if architecture_extensions is not None:
            architectures += tuple(architecture_extensions(scenario))
        names = tuple(architecture.name for architecture in architectures)
        if len(set(names)) != len(names):
            raise ValueError(
                f"Duplicate architecture name for scenario {scenario.name!r}."
            )
        architecture_sets.append((scenario, architectures))
    if selected_names is not None:
        available_names = {
            architecture.name
            for _, architectures in architecture_sets
            for architecture in architectures
        }
        unavailable_names = selected_names - available_names
        if unavailable_names:
            unavailable = ", ".join(sorted(unavailable_names))
            raise ValueError(
                f"Requested architectures are incompatible with every selected "
                f"scenario: {unavailable}."
            )

    comparisons = []
    trials = []
    selected_results = []
    sample_efficiency = []
    symmetry_results = []
    scopes: dict[tuple[str, str] | tuple[str, str, float], str] = {}
    for scenario, architectures in architecture_sets:
        if selected_names is not None:
            architectures = tuple(
                architecture
                for architecture in architectures
                if architecture.name in selected_names
            )
        if not architectures:
            raise ValueError(f"No compatible architectures for {scenario.name!r}.")
        target_parameters = _target_parameters(architectures, scenario, protocol)
        profiles = []
        for architecture in architectures:
            training_scenario = architecture.training_scenario(scenario)
            parameter_counts = _architecture_parameter_counts(
                architecture,
                scenario,
                protocol,
            )
            minimum_parameters = min(parameter_counts)
            maximum_parameters = max(parameter_counts)
            if protocol.comparison == "pareto" and architecture.trainable:
                choices = tuple(zip(protocol.size_scales, parameter_counts))
            else:
                choices = (
                    _select_size_scale(
                        architecture,
                        scenario,
                        protocol,
                        target_parameters,
                    ),
                )
            for size_scale, count in choices:
                step_flops = 0
                step_bytes = 0
                if (
                    protocol.comparison in ("compute", "pareto")
                    and architecture.trainable
                    and protocol.steps > 0
                ):
                    probe = architecture.build(
                        scenario,
                        0,
                        size_scale=size_scale,
                    )
                    probe, _ = _normalized_model(
                        probe,
                        training_scenario,
                        architecture,
                        protocol,
                    )
                    step_flops, step_bytes = training_step_cost(
                        probe,
                        training_scenario,
                    )
                profiles.append(
                    (
                        architecture,
                        float(size_scale),
                        int(count),
                        int(minimum_parameters),
                        int(maximum_parameters),
                        step_flops,
                        step_bytes,
                    )
                )
        if protocol.comparison == "compute":
            measured_flops = [
                step_flops
                for architecture, _, _, _, _, step_flops, _ in profiles
                if (
                    architecture.trainable
                    and architecture.promotion_scope != "reference"
                    and step_flops > 0
                )
            ]
            if protocol.compute_budget is not None:
                target_compute = int(protocol.compute_budget)
            elif measured_flops:
                target_compute = max(
                    1,
                    round(float(np.median(np.asarray(measured_flops)))) * protocol.steps,
                )
            else:
                target_compute = 1
        else:
            target_compute = None
        for (
            architecture,
            size_scale,
            count,
            minimum_parameters,
            maximum_parameters,
            step_flops,
            step_bytes,
        ) in profiles:
            training_scenario = architecture.training_scenario(scenario)
            comparison = _comparison_record(
                architecture,
                scenario,
                protocol,
                target_parameters,
                size_scale,
                count,
                minimum_parameters,
                maximum_parameters,
                target_compute,
                step_flops,
                step_bytes,
            )
            comparisons.append(comparison)
            scopes[(scenario.name, architecture.name, float(size_scale))] = (
                architecture.promotion_scope
            )
            seeds = protocol.seeds if architecture.trainable else (protocol.seeds[0],)
            for seed in seeds:
                rates = (
                    protocol.learning_rates
                    if architecture.trainable
                    else (protocol.learning_rates[0],)
                )
                trial_results = []
                for learning_rate in rates:
                    model = architecture.build(
                        scenario,
                        seed,
                        size_scale=size_scale,
                    )
                    model, normalization = _normalized_model(
                        model,
                        training_scenario,
                        architecture,
                        protocol,
                    )
                    checkpoint_path, checkpoint_metadata = _trial_checkpoint_path(
                        protocol,
                        scenario,
                        architecture,
                        seed=seed,
                        learning_rate=learning_rate,
                        size_scale=size_scale,
                        normalization=normalization,
                    )
                    trained, result = run_operator_benchmark(
                        model,
                        training_scenario,
                        steps=comparison.planned_steps,
                        learning_rate=learning_rate,
                        repeats=protocol.repeats,
                        size_scale=size_scale,
                        architecture=architecture.name,
                        family=architecture.family,
                        architecture_configuration=architecture.configuration(
                            scenario,
                            size_scale=size_scale,
                        ),
                        seed=seed,
                        trainable=architecture.trainable,
                        validation_interval=protocol.validation_interval,
                        patience=protocol.patience,
                        minimum_delta=protocol.minimum_delta,
                        relative_minimum_delta=protocol.relative_minimum_delta,
                        checkpoint_path=checkpoint_path,
                        resume=protocol.resume,
                        checkpoint_metadata=checkpoint_metadata,
                        checkpoint_key=jr.key(seed),
                        run_evaluations=False,
                    )
                    trial_results.append((trained, result, learning_rate, normalization))
                selected_index = _selected_trial_index(
                    [result for _, result, _, _ in trial_results]
                )
                (
                    selected_model,
                    selected_result,
                    selected_learning_rate,
                    _,
                ) = trial_results[selected_index]
                selected_evaluations = tuple(
                    evaluate_operator(
                        selected_model,
                        evaluation,
                        repeats=protocol.repeats,
                    )
                    for evaluation in scenario.evaluations
                )
                selected_results.append(
                    replace(selected_result, evaluations=selected_evaluations)
                )
                sample_evaluation, sample_evaluation_result = next(
                    (
                        (evaluation, evaluation_result)
                        for evaluation, evaluation_result in zip(
                            scenario.evaluations,
                            selected_evaluations,
                            strict=True,
                        )
                        if evaluation.shift == "in_distribution"
                    ),
                    (scenario.evaluations[0], selected_evaluations[0]),
                )
                sample_efficiency.append(
                    _sample_efficiency_curve(
                        architecture,
                        scenario,
                        protocol,
                        seed=seed,
                        size_scale=size_scale,
                        learning_rate=selected_learning_rate,
                        planned_steps=comparison.planned_steps,
                        full_result=selected_result,
                        evaluation=sample_evaluation,
                        full_evaluation=sample_evaluation_result,
                    )
                )
                if scenario.symmetry is not None:
                    symmetry_evaluation = sample_evaluation
                    symmetry_result = evaluate_operator_symmetry(
                        selected_model,
                        symmetry_evaluation,
                        scenario.symmetry,
                    )
                    symmetry_results.append(
                        SymmetryBenchmarkRecord(
                            scenario=scenario.name,
                            architecture=architecture.name,
                            family=architecture.family,
                            seed=int(seed),
                            size_scale=float(size_scale),
                            evaluation=symmetry_result.name,
                            declared_group=symmetry_result.declared_group,
                            audit_group=symmetry_result.audit_group,
                            element_relative_l2=(symmetry_result.element_relative_l2),
                            element_maximum_absolute_error=(
                                symmetry_result.element_maximum_absolute_error
                            ),
                            mean_equivariance_defect=(
                                symmetry_result.mean_equivariance_defect
                            ),
                            worst_equivariance_defect=(
                                symmetry_result.worst_equivariance_defect
                            ),
                            maximum_absolute_equivariance_error=(
                                symmetry_result.maximum_absolute_equivariance_error
                            ),
                            mean_rotated_pair_difference=(
                                symmetry_result.mean_rotated_pair_difference
                            ),
                            mean_reflected_pair_difference=(
                                symmetry_result.mean_reflected_pair_difference
                            ),
                            reference_worst_defect=max(
                                defect
                                for _, defect in scenario.symmetry.reference_defects
                            ),
                        )
                    )
                for index, (_, result, learning_rate, normalization) in enumerate(
                    trial_results
                ):
                    trials.append(
                        HyperparameterTrial(
                            scenario=scenario.name,
                            architecture=architecture.name,
                            family=architecture.family,
                            seed=int(seed),
                            learning_rate=float(learning_rate),
                            size_scale=float(size_scale),
                            normalization=normalization,
                            architecture_configuration=json.dumps(
                                dict(
                                    architecture.configuration(
                                        scenario,
                                        size_scale=size_scale,
                                    )
                                ),
                                sort_keys=True,
                            ),
                            parameter_count=result.parameter_count,
                            training_steps=result.training_steps,
                            training_seconds=result.training_seconds,
                            initial_loss=result.initial_loss,
                            final_loss=result.final_loss,
                            validation_loss=result.validation_loss,
                            learning_curve=result.losses,
                            selected=index == selected_index,
                            validation_steps=result.validation_steps,
                            validation_curve=result.validation_losses,
                            stopped_early=result.stopped_early,
                            converged=result.converged,
                            resumed_from_step=result.resumed_from_step,
                        )
                    )
    result_tuple = tuple(selected_results)
    aggregate_tuple = aggregate_benchmark_results(result_tuple)
    comparisons_tuple = tuple(comparisons)
    trials_tuple = tuple(trials)
    pareto_fronts = _pareto_fronts(
        aggregate_tuple,
        trials_tuple,
        comparisons_tuple,
    )
    promotions = _promotion_reports(
        aggregate_tuple,
        audits,
        comparisons_tuple,
        scopes,
        family_parity,
        promotion_criteria,
        production_run=protocol.profile == "decision" and not protocol.quick,
        provenance_pinned=protocol.commit_identity.strip() not in ("", "working-tree"),
        external_audits=external_audits,
        difficulty_audits=difficulty_audits,
    )
    portfolio_promotions = _portfolio_promotions(promotions, promotion_criteria)
    return OperatorBenchmarkV2Result(
        metadata=benchmark_metadata(
            scenarios,
            commit_identity=protocol.commit_identity,
        ),
        protocol=protocol,
        ladders=ladders,
        audits=audits,
        symmetry_audits=symmetry_audits,
        kernel_parity=kernel_parity_checks(scenarios),
        family_parity=family_parity,
        external_audits=external_audits,
        comparisons=comparisons_tuple,
        trials=trials_tuple,
        sample_efficiency=tuple(sample_efficiency),
        results=result_tuple,
        symmetry_results=tuple(symmetry_results),
        aggregates=aggregate_tuple,
        difficulty_audits=difficulty_audits,
        pareto_fronts=pareto_fronts,
        promotions=promotions,
        portfolio_promotions=portfolio_promotions,
    )


def _rows(values):
    return [asdict(value) for value in values]


def save_benchmark_v2_artifacts(
    directory: str | Path,
    result: OperatorBenchmarkV2Result,
    /,
) -> tuple[Path, Path, Path, Path, Path, Path, Path, Path, Path]:
    """Write metric, sample, symmetry, audit, Pareto, and promotion artifacts."""
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    json_path = root / "operator_benchmarks_v2.json"
    aggregate_path = root / "operator_benchmarks_v2.parquet"
    trial_path = root / "operator_benchmark_trials_v2.parquet"
    sample_efficiency_path = root / "operator_sample_efficiency_v2.parquet"
    promotion_path = root / "operator_promotions_v2.parquet"
    symmetry_path = root / "operator_symmetry_v2.parquet"
    portfolio_path = root / "operator_portfolio_promotions_v2.parquet"
    difficulty_path = root / "operator_scenario_difficulty_v2.parquet"
    pareto_path = root / "operator_pareto_fronts_v2.parquet"
    json_path.write_text(
        json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    pl.DataFrame(_rows(result.aggregates)).write_parquet(aggregate_path)
    pl.DataFrame(_rows(result.trials)).write_parquet(trial_path)
    pl.DataFrame(_rows(result.sample_efficiency)).write_parquet(sample_efficiency_path)
    symmetry_rows = _rows(result.symmetry_results)
    symmetry_frame = (
        pl.DataFrame(symmetry_rows)
        if symmetry_rows
        else pl.DataFrame(
            schema={
                "scenario": pl.String,
                "architecture": pl.String,
                "family": pl.String,
                "seed": pl.Int64,
                "size_scale": pl.Float64,
                "evaluation": pl.String,
                "declared_group": pl.String,
                "audit_group": pl.String,
                "element_relative_l2": pl.List(pl.Float64),
                "element_maximum_absolute_error": pl.List(pl.Float64),
                "mean_equivariance_defect": pl.Float64,
                "worst_equivariance_defect": pl.Float64,
                "maximum_absolute_equivariance_error": pl.Float64,
                "mean_rotated_pair_difference": pl.Float64,
                "mean_reflected_pair_difference": pl.Float64,
                "reference_worst_defect": pl.Float64,
            }
        )
    )
    symmetry_frame.write_parquet(symmetry_path)
    pl.DataFrame(_rows(result.difficulty_audits)).write_parquet(difficulty_path)
    pareto_rows = [
        asdict(point) for front in result.pareto_fronts for point in front.points
    ]
    pl.DataFrame(pareto_rows).write_parquet(pareto_path)
    pl.DataFrame(_rows(result.promotions)).write_parquet(promotion_path)
    pl.DataFrame(_rows(result.portfolio_promotions)).write_parquet(portfolio_path)
    return (
        json_path,
        aggregate_path,
        trial_path,
        sample_efficiency_path,
        symmetry_path,
        promotion_path,
        portfolio_path,
        difficulty_path,
        pareto_path,
    )


__all__ = [
    "ArchitecturePromotionReport",
    "ArchitecturePortfolioPromotion",
    "BenchmarkProfile",
    "BenchmarkComparisonRecord",
    "BenchmarkParetoFront",
    "BenchmarkParetoPoint",
    "FamilyParityEvidence",
    "GeometryScenarioIntegrityAudit",
    "HyperparameterTrial",
    "KernelParityCheck",
    "NearIdentityDiagnostic",
    "OperatorBenchmarkLadder",
    "native_kernel_parity_checks",
    "OperatorBenchmarkProtocol",
    "OperatorBenchmarkV2Result",
    "PromotionCriteria",
    "SampleEfficiencyCurve",
    "ScenarioDifficultyAudit",
    "ScenarioIntegrityAudit",
    "audit_geometry_scenario",
    "audit_operator_scenario",
    "audit_scenario_difficulty",
    "flatten_operator_benchmark_ladders",
    "kernel_parity_checks",
    "load_family_parity_evidence",
    "run_operator_benchmark_v2",
    "save_benchmark_v2_artifacts",
    "standard_operator_benchmark_ladders",
]
