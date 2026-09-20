#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Scalable Pfaffians, phase reweighting, multishift solves, and limit campaigns."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike, Key

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._limit_study import (
    run_scientific_limit_study,
    ScientificLimitAxis,
    ScientificLimitDatum,
    ScientificLimitStudyPlan,
    ScientificLimitStudyResult,
    ScientificLimitVariation,
)
from ..._strict import StrictModule
from ...linalg import (
    AbstractLinearOperator,
    evaluate_pfaffian,
    PfaffianPolicy,
    ShiftedLinearSystemFamily,
    ShiftedSolvePlan,
    ShiftedSolvePolicy,
    ShiftedSolveResult,
    solve_shifted,
)
from ._rhmc import (
    PreparedTwistedN2RHMC,
    sample_twisted_n2_rhmc,
    TwistedN2RHMCRun,
)


SupersymmetricLimitKind: TypeAlias = Literal[
    "regulator", "continuum", "thermodynamic", "large-rank"
]


class ScalablePfaffianPlan(StrictModule):
    """Guarded cubic skew-elimination Pfaffian plan."""

    maximum_dimension: int = eqx.field(static=True)
    antisymmetry_tolerance: float = eqx.field(static=True)
    pivot_tolerance: float = eqx.field(static=True)
    determinant_tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        /,
        *,
        maximum_dimension: int = 8192,
        antisymmetry_tolerance: float = 1e-10,
        pivot_tolerance: float = 1e-14,
        determinant_tolerance: float = 1e-8,
    ):
        maximum = int(maximum_dimension)
        antisymmetry = float(antisymmetry_tolerance)
        pivot = float(pivot_tolerance)
        determinant = float(determinant_tolerance)
        if maximum < 2 or maximum % 2:
            raise ValueError("maximum_dimension must be positive and even.")
        if any(
            not math.isfinite(value) or value < 0.0
            for value in (antisymmetry, pivot, determinant)
        ):
            raise ValueError("Pfaffian tolerances must be finite and non-negative.")
        content = {
            "kind": "scalable-pfaffian-plan",
            "maximum_dimension": maximum,
            "antisymmetry_tolerance": antisymmetry,
            "pivot_tolerance": pivot,
            "determinant_tolerance": determinant,
            "algorithm": "native-pivoted-skew-ldlt",
        }
        self.maximum_dimension = maximum
        self.antisymmetry_tolerance = antisymmetry
        self.pivot_tolerance = pivot
        self.determinant_tolerance = determinant
        self.plan_id = canonical_fingerprint(content)


class ScalablePfaffianEvidence(StrictModule):
    pfaffian: Array
    phase: Array
    log_magnitude: Array
    antisymmetry_residual: Array
    determinant_identity_residual: Array
    pivot_magnitudes: Array
    singular: Array
    value_finite: Array
    native_status: Array
    accepted: Array
    plan_id: str = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)


def scalable_pfaffian(
    matrix: ArrayLike,
    plan: ScalablePfaffianPlan,
    /,
) -> ScalablePfaffianEvidence:
    """Compute a complex Pfaffian using the native reusable skew factorization."""

    if not isinstance(plan, ScalablePfaffianPlan):
        raise TypeError("plan must be ScalablePfaffianPlan.")
    value = np.asarray(matrix, dtype=np.complex128)
    if value.ndim != 2 or value.shape[0] != value.shape[1]:
        raise ValueError("Pfaffian input must be one square matrix.")
    dimension = value.shape[0]
    if dimension % 2 or dimension > plan.maximum_dimension:
        raise ValueError("Pfaffian dimension is odd or exceeds maximum_dimension.")
    capacity = plan.maximum_dimension
    itemsize = np.dtype(np.complex128).itemsize
    storage_limit = (
        (2 * capacity * capacity + capacity // 2) * itemsize
        + capacity * np.dtype(np.int32).itemsize
        + 4 * np.dtype(np.float64).itemsize
        + itemsize
    )
    workspace_limit = itemsize * (5 * capacity * capacity + 3 * capacity + 1)
    result = evaluate_pfaffian(
        value,
        PfaffianPolicy(
            skew_mode="require",
            antisymmetry_tolerance=plan.antisymmetry_tolerance,
            pivot_tolerance=plan.pivot_tolerance,
            verify_determinant=True,
            max_dimension=plan.maximum_dimension,
            max_storage_bytes=storage_limit,
            max_workspace_bytes=workspace_limit,
        ),
    )
    antisymmetric = bool(np.asarray(result.antisymmetric))
    if not antisymmetric:
        raise ValueError("Pfaffian input violates the antisymmetry tolerance.")
    pfaffian = complex(np.asarray(result.value))
    signed_phase = complex(np.asarray(result.sign))
    phase = float(np.angle(signed_phase)) if signed_phase else 0.0
    log_magnitude = float(np.asarray(result.log_abs))
    antisymmetry = float(np.asarray(result.antisymmetry_residual))
    determinant_residual = float(np.asarray(result.determinant_identity_residual))
    pivot_table = np.asarray(result.pivot_magnitudes)
    singular = bool(np.asarray(result.singular))
    value_finite = bool(np.asarray(result.value_finite))
    native_status = int(np.asarray(result.status))
    accepted = (
        bool(np.asarray(result.successful))
        and not singular
        and determinant_residual <= plan.determinant_tolerance
        and math.isfinite(phase)
        and math.isfinite(log_magnitude)
    )
    evidence_id = canonical_fingerprint(
        {
            "kind": "scalable-pfaffian-evidence",
            "plan": plan.plan_id,
            "matrix": array_tree_fingerprint(value),
            "sign": (signed_phase.real, signed_phase.imag),
            "log_magnitude": log_magnitude,
            "antisymmetry_residual": antisymmetry,
            "determinant_identity_residual": determinant_residual,
            "pivots": array_tree_fingerprint(pivot_table),
            "singular": singular,
            "value_finite": value_finite,
            "native_status": native_status,
        }
    )
    return ScalablePfaffianEvidence(
        pfaffian=jnp.asarray(pfaffian),
        phase=jnp.asarray(phase),
        log_magnitude=jnp.asarray(log_magnitude),
        antisymmetry_residual=jnp.asarray(antisymmetry),
        determinant_identity_residual=jnp.asarray(determinant_residual),
        pivot_magnitudes=jnp.asarray(pivot_table),
        singular=jnp.asarray(singular),
        value_finite=jnp.asarray(value_finite),
        native_status=jnp.asarray(native_status, dtype=jnp.int32),
        accepted=jnp.asarray(accepted),
        plan_id=plan.plan_id,
        evidence_id=evidence_id,
    )


class PfaffianPhaseChainEvidence(StrictModule):
    phases: Array
    unwrapped_phases: Array
    phase_factors: Array
    mean_phase_factor: Array
    phase_effective_sample_size: Array
    accepted: Array
    evidence_id: str = eqx.field(static=True)


def assess_pfaffian_phase_chain(
    pfaffians: ArrayLike,
    /,
    *,
    minimum_effective_samples: float = 4.0,
) -> PfaffianPhaseChainEvidence:
    values = np.asarray(pfaffians, dtype=np.complex128)
    if values.ndim != 1 or values.size < 2 or not np.all(np.isfinite(values)):
        raise ValueError(
            "Pfaffian chain must be a finite vector with at least two values."
        )
    if np.any(np.abs(values) == 0.0):
        raise ValueError("Pfaffian phase chains cannot contain exact zeros.")
    phases = np.angle(values)
    unwrapped = np.unwrap(phases)
    factors = values / np.abs(values)
    mean = np.mean(factors)
    effective = float(abs(np.sum(factors)) ** 2 / values.size)
    accepted = effective >= float(minimum_effective_samples)
    evidence_id = canonical_fingerprint(
        {
            "kind": "pfaffian-phase-chain-evidence",
            "values": array_tree_fingerprint(values),
            "minimum_effective_samples": float(minimum_effective_samples),
        }
    )
    return PfaffianPhaseChainEvidence(
        phases=jnp.asarray(phases),
        unwrapped_phases=jnp.asarray(unwrapped),
        phase_factors=jnp.asarray(factors),
        mean_phase_factor=jnp.asarray(mean),
        phase_effective_sample_size=jnp.asarray(effective),
        accepted=jnp.asarray(accepted),
        evidence_id=evidence_id,
    )


class PhaseReweightingEvidence(StrictModule):
    reweighted_mean: Array
    standard_error: Array
    phase_mean: Array
    phase_effective_sample_size: Array
    overlap_accepted: Array
    finite: Array
    evidence_id: str = eqx.field(static=True)


def phase_reweight_observable(
    observables: ArrayLike,
    pfaffians: ArrayLike,
    /,
    *,
    minimum_effective_samples: float = 4.0,
) -> PhaseReweightingEvidence:
    """Reweight a phase-quenched chain with covariance-aware jackknife error."""

    observable = np.asarray(observables)
    values = np.asarray(pfaffians, dtype=np.complex128)
    if observable.shape[0] != values.size or values.ndim != 1 or values.size < 3:
        raise ValueError(
            "Observable and Pfaffian sample axes must align and contain three samples."
        )
    if not np.all(np.isfinite(observable)) or not np.all(np.isfinite(values)):
        raise ValueError("Phase reweighting inputs must be finite.")
    if np.any(np.abs(values) == 0.0):
        raise ValueError("Phase reweighting cannot use zero Pfaffians.")
    phase = values / np.abs(values)
    denominator = np.sum(phase)
    if abs(denominator) == 0.0:
        raise ValueError("Phase reweighting denominator vanishes exactly.")
    expanded = phase.reshape((values.size,) + (1,) * (observable.ndim - 1))
    mean = np.sum(expanded * observable, axis=0) / denominator
    jackknife = []
    for index in range(values.size):
        mask = np.arange(values.size) != index
        denominator_i = np.sum(phase[mask])
        jackknife.append(
            np.sum(expanded[mask] * observable[mask], axis=0) / denominator_i
        )
    jackknife_values = np.stack(jackknife, axis=0)
    centered = jackknife_values - np.mean(jackknife_values, axis=0)
    error = np.sqrt(
        (values.size - 1) / values.size * np.sum(np.abs(centered) ** 2, axis=0)
    )
    effective = float(abs(denominator) ** 2 / values.size)
    accepted = effective >= float(minimum_effective_samples)
    finite = bool(np.all(np.isfinite(mean)) and np.all(np.isfinite(error)))
    evidence_id = canonical_fingerprint(
        {
            "kind": "phase-reweighting-evidence",
            "observables": array_tree_fingerprint(observable),
            "pfaffians": array_tree_fingerprint(values),
            "minimum_effective_samples": float(minimum_effective_samples),
        }
    )
    return PhaseReweightingEvidence(
        reweighted_mean=jnp.asarray(mean),
        standard_error=jnp.asarray(error),
        phase_mean=jnp.asarray(np.mean(phase)),
        phase_effective_sample_size=jnp.asarray(effective),
        overlap_accepted=jnp.asarray(accepted),
        finite=jnp.asarray(finite),
        evidence_id=evidence_id,
    )


def _integrated_autocorrelation(values: np.ndarray, /) -> float:
    centered = values - np.mean(values)
    variance = float(np.mean(np.abs(centered) ** 2))
    if variance == 0.0:
        return 0.5
    integrated = 0.5
    for lag in range(1, values.size):
        correlation = float(
            np.real(np.mean(np.conj(centered[:-lag]) * centered[lag:])) / variance
        )
        if correlation <= 0.0:
            break
        integrated += correlation
    return integrated


class SupersymmetricObservableEvidence(StrictModule):
    bosonic_action_mean: Array
    bosonic_action_standard_error: Array
    ward_means: Array
    ward_standard_errors: Array
    polyakov_mean: Array
    scalar_eigenvalue_mean: Array
    integrated_autocorrelation_times: Array
    finite: Array
    evidence_id: str = eqx.field(static=True)


def assess_supersymmetric_observables(
    bosonic_actions: ArrayLike,
    ward_observables: ArrayLike,
    polyakov_loops: ArrayLike,
    scalar_eigenvalues: ArrayLike,
    /,
) -> SupersymmetricObservableEvidence:
    """Summarize gauge-invariant chain observables without inferring a continuum claim."""

    bosonic = np.asarray(bosonic_actions, dtype=np.float64)
    wards = np.asarray(ward_observables)
    polyakov = np.asarray(polyakov_loops)
    scalars = np.asarray(scalar_eigenvalues, dtype=np.float64)
    count = bosonic.size
    if bosonic.ndim != 1 or count < 2:
        raise ValueError("Bosonic actions must provide at least two chain samples.")
    if any(value.shape[0] != count for value in (wards, polyakov, scalars)):
        raise ValueError("Every supersymmetric observable must share one sample axis.")
    if not all(
        np.all(np.isfinite(value)) for value in (bosonic, wards, polyakov, scalars)
    ):
        raise ValueError("Supersymmetric observable samples must be finite.")
    ward_flat = wards.reshape((count, -1))
    scalar_flat = scalars.reshape((count, -1))
    series = [bosonic]
    series.extend(ward_flat[:, index] for index in range(ward_flat.shape[1]))
    autocorrelations = np.asarray(
        [_integrated_autocorrelation(np.asarray(value)) for value in series]
    )
    ward_errors = np.std(ward_flat, axis=0, ddof=1) / math.sqrt(count)
    evidence_id = canonical_fingerprint(
        {
            "kind": "supersymmetric-observable-evidence",
            "bosonic": array_tree_fingerprint(bosonic),
            "wards": array_tree_fingerprint(wards),
            "polyakov": array_tree_fingerprint(polyakov),
            "scalars": array_tree_fingerprint(scalars),
        }
    )
    return SupersymmetricObservableEvidence(
        bosonic_action_mean=jnp.asarray(np.mean(bosonic)),
        bosonic_action_standard_error=jnp.asarray(
            np.std(bosonic, ddof=1) / math.sqrt(count)
        ),
        ward_means=jnp.asarray(np.mean(ward_flat, axis=0)),
        ward_standard_errors=jnp.asarray(ward_errors),
        polyakov_mean=jnp.asarray(np.mean(polyakov, axis=0)),
        scalar_eigenvalue_mean=jnp.asarray(np.mean(scalar_flat, axis=0)),
        integrated_autocorrelation_times=jnp.asarray(autocorrelations),
        finite=jnp.asarray(True),
        evidence_id=evidence_id,
    )


class RationalSpectralEvidence(StrictModule):
    spectral_lower: Array
    spectral_upper: Array
    action_maximum_relative_error: Array
    refresh_maximum_relative_error: Array
    action_certified: Array
    refresh_certified: Array
    evidence_id: str = eqx.field(static=True)


def assess_rhmc_rational_spectrum(
    prepared: PreparedTwistedN2RHMC,
    /,
) -> RationalSpectralEvidence:
    if not isinstance(prepared, PreparedTwistedN2RHMC):
        raise TypeError("prepared must be PreparedTwistedN2RHMC.")
    action = prepared.pseudofermion.action_approximation
    refresh = prepared.pseudofermion.refresh_approximation
    lower = prepared.spectral_interval.lower
    upper = prepared.spectral_interval.upper
    evidence_id = canonical_fingerprint(
        {
            "kind": "rhmc-rational-spectral-evidence",
            "prepared": prepared.prepared_id,
            "interval": prepared.spectral_interval.certificate_id,
            "action": action.certificate_id,
            "refresh": refresh.certificate_id,
        }
    )
    return RationalSpectralEvidence(
        spectral_lower=lower,
        spectral_upper=upper,
        action_maximum_relative_error=action.maximum_relative_error,
        refresh_maximum_relative_error=refresh.maximum_relative_error,
        action_certified=action.successful,
        refresh_certified=refresh.successful,
        evidence_id=evidence_id,
    )


def solve_pseudofermion_multishift(
    operator: AbstractLinearOperator,
    right_hand_side: ArrayLike,
    shifts: ArrayLike,
    /,
    *,
    policy: ShiftedSolvePolicy | ShiftedSolvePlan | None = None,
) -> ShiftedSolveResult:
    """Execute the native shared-Krylov multishift route used by pseudofermions."""

    if not isinstance(operator, AbstractLinearOperator):
        raise TypeError("operator must be AbstractLinearOperator.")
    family = ShiftedLinearSystemFamily(operator, shifts)
    return solve_shifted(family, right_hand_side, policy=policy)


def run_supersymmetric_limit_campaign(
    kind: SupersymmetricLimitKind,
    coordinates: Sequence[float],
    values: Sequence[float],
    standard_errors: Sequence[float],
    /,
) -> ScientificLimitStudyResult:
    """Run a prespecified linear/quadratic regulator, continuum, volume, or rank limit."""

    if kind not in ("regulator", "continuum", "thermodynamic", "large-rank"):
        raise ValueError("Unknown supersymmetric limit kind.")
    points = tuple(float(value) for value in coordinates)
    observations = tuple(float(value) for value in values)
    errors = tuple(float(value) for value in standard_errors)
    if len(points) < 4 or len(points) != len(observations) or len(points) != len(errors):
        raise ValueError("Supersymmetric limit campaigns require four aligned points.")
    axis = ScientificLimitAxis(kind, 0.0, minimum_span=max(points) - min(points))
    plan = ScientificLimitStudyPlan(
        (axis,),
        (
            ScientificLimitVariation("linear", {kind: 1}, minimum_points=3),
            ScientificLimitVariation("quadratic", {kind: 2}, minimum_points=4),
        ),
    )
    data = tuple(
        ScientificLimitDatum(
            f"{kind}-{index}",
            {kind: coordinate},
            observation,
            error,
        )
        for index, (coordinate, observation, error) in enumerate(
            zip(points, observations, errors, strict=True)
        )
    )
    return run_scientific_limit_study(plan, data)


@dataclass(frozen=True, slots=True)
class DistributedTwistedN2Run:
    runs: tuple[TwistedN2RHMCRun, ...]
    device_ids: tuple[int, ...]
    total_draws: int
    run_id: str


def sample_distributed_twisted_n2_rhmc(
    prepared: PreparedTwistedN2RHMC,
    initial_configurations: Sequence[ArrayLike],
    keys: Sequence[Key[Array, ""]],
    /,
    *,
    num_draws: int,
) -> DistributedTwistedN2Run:
    """Run independently addressed chains on explicitly assigned local devices."""

    if not isinstance(prepared, PreparedTwistedN2RHMC):
        raise TypeError("prepared must be PreparedTwistedN2RHMC.")
    configurations = tuple(initial_configurations)
    keys_ = tuple(keys)
    devices = tuple(jax.local_devices())
    if not configurations or len(configurations) != len(keys_):
        raise ValueError("Distributed RHMC configurations and keys must align.")
    if len(configurations) > len(devices):
        raise ValueError("One local device is required per distributed RHMC chain.")
    runs: list[TwistedN2RHMCRun] = []
    device_ids: list[int] = []
    for index, (configuration, key) in enumerate(zip(configurations, keys_, strict=True)):
        device = devices[index]
        with jax.default_device(device):
            run = sample_twisted_n2_rhmc(
                prepared,
                jax.device_put(configuration, device),
                jax.device_put(key, device),
                num_draws=num_draws,
            )
        runs.append(run)
        device_ids.append(index)
    run_id = canonical_fingerprint(
        {
            "kind": "distributed-twisted-n2-rhmc-run",
            "prepared": prepared.prepared_id,
            "chains": len(runs),
            "device_ids": device_ids,
            "num_draws": int(num_draws),
        }
    )
    return DistributedTwistedN2Run(
        tuple(runs),
        tuple(device_ids),
        len(runs) * int(num_draws),
        run_id,
    )


__all__ = [
    "DistributedTwistedN2Run",
    "PfaffianPhaseChainEvidence",
    "PhaseReweightingEvidence",
    "RationalSpectralEvidence",
    "ScalablePfaffianEvidence",
    "ScalablePfaffianPlan",
    "SupersymmetricLimitKind",
    "SupersymmetricObservableEvidence",
    "assess_pfaffian_phase_chain",
    "assess_rhmc_rational_spectrum",
    "assess_supersymmetric_observables",
    "phase_reweight_observable",
    "run_supersymmetric_limit_campaign",
    "sample_distributed_twisted_n2_rhmc",
    "scalable_pfaffian",
    "solve_pseudofermion_multishift",
]
