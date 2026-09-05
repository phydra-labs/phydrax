#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Actual two-cause probability calibration and staged external qualification.

Fixed candidate/topology support is differentiated; imported transport/chemistry,
threshold decisions and initial-lesion realizations are not. A proper caller-
declared Gaussian prior on logits defines a Laplace posterior approximation.
Transport, chemical G values and target-reaction comparisons remain independent
stage evidence; fitting lesion yields cannot validate those upstream engines.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import jax
import jax.numpy as jnp
import numpy as np

from ..._fingerprint import canonical_fingerprint
from ...linalg import DenseLinearOperator
from ...linalg.svd import svd, SVDProblem, SVDSolvePolicy
from ...optim import (
    least_squares,
    LeastSquaresResult,
    LevenbergMarquardt,
    OptimizationTermination,
)
from ...qualification import ReferenceArtifactManifest
from ...units import conversion_factor, ONE, UnitDefinition
from ...uq import (
    DenseCovariance,
    fit_laplace,
    LaplaceResult,
    ParameterSpace,
    PosteriorProblem,
    propagate_linearized,
)
from ._interactions import _nonnegative, _text
from ._lesions import LesionCandidates
from ._quantities import PER_GRAY, PER_JOULE


@dataclass(frozen=True, slots=True)
class LesionExpectationSupport:
    """Per-site multiplicity and canonical (Gy, kg, count) yield denominator.

    The denominator is the value returned by ``radiation_yield`` for the chosen
    convention; measured-unit conversion belongs to the calibration dataset.
    """

    direct_multiplicity: tuple[int, ...]
    indirect_multiplicity: tuple[int, ...]
    denominator: float
    candidate_artifact_id: str

    def __post_init__(self):
        if (
            not isinstance(self.direct_multiplicity, tuple)
            or not isinstance(self.indirect_multiplicity, tuple)
            or len(self.direct_multiplicity) != len(self.indirect_multiplicity)
        ):
            raise ValueError(
                "Direct/indirect site multiplicities must be aligned immutable tuples."
            )
        if any(
            type(n) is not int or n < 0
            for n in (*self.direct_multiplicity, *self.indirect_multiplicity)
        ):
            raise ValueError("Candidate multiplicities must be nonnegative integers.")
        if not math.isfinite(self.denominator) or self.denominator <= 0:
            raise ValueError(
                "Expected lesion yields need a positive physical denominator."
            )
        _text(self.candidate_artifact_id, "candidate artifact")


def prepare_lesion_expectation(
    candidates: LesionCandidates, *, denominator: float
) -> LesionExpectationSupport:
    """Compile the R1 candidate ledger for fitting one direct/indirect probability.

    Calibration requires unthinned unit-probability candidates. A fractional
    geometry route or preselected lesion realization cannot masquerade as such a
    trial. Repeated same-site reactions use a union probability, not summed yields.
    """
    grouped = {}
    for candidate in candidates.candidates:
        if candidate.probability != 1.0:
            raise ValueError(
                "Probability calibration requires unthinned unit-probability candidates."
            )
        if candidate.cause not in ("direct", "indirect"):
            raise ValueError("Unknown candidate cause.")
        counts = grouped.setdefault((candidate.history, candidate.target_id), [0, 0])
        counts[int(candidate.cause == "indirect")] += 1
    counts = [grouped[key] for key in sorted(grouped)]
    identity = canonical_fingerprint(
        {
            "mapping": candidates.mapping_id,
            "policy": candidates.policy_id,
            "candidates": [asdict(item) for item in candidates.candidates],
        }
    )
    return LesionExpectationSupport(
        tuple(item[0] for item in counts),
        tuple(item[1] for item in counts),
        denominator,
        identity,
    )


def expected_initial_lesion_yield(
    logits, direct_multiplicity, indirect_multiplicity, active_mask, denominators
):
    """JIT/grad-safe exact independent-candidate union probability on fixed support.

    Shape: logits (2,), multiplicities/mask (condition, site), denominators
    (condition,). Padding is inactive; it never becomes an undamaged physical site.
    """
    log_survival = jnp.asarray(direct_multiplicity) * jax.nn.log_sigmoid(
        -logits[0]
    ) + jnp.asarray(indirect_multiplicity) * jax.nn.log_sigmoid(-logits[1])
    probabilities = jnp.where(active_mask, -jnp.expm1(log_survival), 0.0)
    return jnp.sum(probabilities, axis=-1) / denominators


@dataclass(frozen=True, slots=True)
class RadiationCondition:
    condition_id: str
    dose_gy: float
    oxygen_mol_per_m3: float
    scavenger_mol_per_m3: float
    chemistry_endpoint_s: float

    def __post_init__(self):
        _text(self.condition_id, "condition")
        for value in (
            self.dose_gy,
            self.oxygen_mol_per_m3,
            self.scavenger_mol_per_m3,
            self.chemistry_endpoint_s,
        ):
            _nonnegative(value, "physical condition")

    def physical_key(self) -> tuple[float, ...]:
        return (
            self.dose_gy,
            self.oxygen_mol_per_m3,
            self.scavenger_mol_per_m3,
            self.chemistry_endpoint_s,
        )


@dataclass(frozen=True, slots=True)
class RadiationCalibrationData:
    observation_ids: tuple[str, ...]
    conditions: tuple[RadiationCondition, ...]
    supports: tuple[LesionExpectationSupport, ...]
    observed_yields: tuple[float, ...]
    standard_errors: tuple[float, ...]
    yield_unit: UnitDefinition
    normalization_convention: str
    reference: ReferenceArtifactManifest
    source_kind: str

    def __post_init__(self):
        n = len(self.observation_ids)
        arrays = (
            self.observation_ids,
            self.conditions,
            self.supports,
            self.observed_yields,
            self.standard_errors,
        )
        if not n or any(
            not isinstance(values, tuple) or len(values) != n for values in arrays
        ):
            raise ValueError(
                "Calibration observations require nonempty aligned immutable rows."
            )
        if (
            len(set(self.observation_ids)) != n
            or len({c.condition_id for c in self.conditions}) != n
        ):
            raise ValueError(
                "Observation and condition IDs must be unique within a split."
            )
        for value in self.observed_yields:
            _nonnegative(value, "observed initial lesion yield")
        if any(not math.isfinite(value) or value <= 0 for value in self.standard_errors):
            raise ValueError(
                "Measured yield standard errors must be finite and positive."
            )
        if self.source_kind not in ("synthetic", "experimental", "external-reference"):
            raise ValueError("Calibration source kind must be explicit.")
        if self.normalization_convention not in (
            "per-primary",
            "per-Gy",
            "per-Gy-per-Mbp",
            "per-Gy-per-molecule",
            "per-Gy-per-kg",
        ):
            raise ValueError("Calibration yield normalization must be explicit.")
        expected_unit = (
            ONE
            if self.normalization_convention == "per-primary"
            else PER_JOULE
            if self.normalization_convention == "per-Gy-per-kg"
            else PER_GRAY
        )
        conversion_factor(self.yield_unit, expected_unit)
        self.reference.require_uncertainty()

    def fingerprint(self) -> str:
        return canonical_fingerprint(
            {
                "observations": self.observation_ids,
                "conditions": [asdict(item) for item in self.conditions],
                "supports": [asdict(item) for item in self.supports],
                "yields": self.observed_yields,
                "errors": self.standard_errors,
                "unit": self.yield_unit.unit_id,
                "normalization": self.normalization_convention,
                "reference": self.reference.manifest_id,
                "kind": self.source_kind,
            }
        )

    def prepared_arrays(self):
        capacity = max(1, max(len(item.direct_multiplicity) for item in self.supports))
        direct = np.zeros((len(self.supports), capacity), dtype=np.int64)
        indirect = np.zeros_like(direct)
        mask = np.zeros(direct.shape, dtype=bool)
        for row, support in enumerate(self.supports):
            n = len(support.direct_multiplicity)
            direct[row, :n] = support.direct_multiplicity
            indirect[row, :n] = support.indirect_multiplicity
            mask[row, :n] = True
        canonical_unit = (
            ONE
            if self.normalization_convention == "per-primary"
            else PER_JOULE
            if self.normalization_convention == "per-Gy-per-kg"
            else PER_GRAY
        )
        denominator_scale = float(conversion_factor(self.yield_unit, canonical_unit))
        return (
            jnp.asarray(direct),
            jnp.asarray(indirect),
            jnp.asarray(mask),
            jnp.asarray([item.denominator * denominator_scale for item in self.supports]),
        )


@dataclass(frozen=True, slots=True)
class RadiationStageEvidence:
    """Independent source-stage comparison, with its own known uncertainty.

    chemical-G rows may span time; condition IDs must identify those time points.
    Complete radiolysis coverage must be established externally, not inferred from
    the selected damage-reaction ntuple adapter.
    """

    stage: str
    condition_ids: tuple[str, ...]
    predicted: tuple[float, ...]
    observed: tuple[float, ...]
    standard_errors: tuple[float, ...]
    unit: UnitDefinition
    reference: ReferenceArtifactManifest
    source_kind: str
    maximum_standardized_rms: float
    upstream_artifact_ids: tuple[str, ...]

    def __post_init__(self):
        if self.stage not in (
            "transport",
            "chemical-G",
            "target-reactions",
            "lesion-yields",
        ):
            raise ValueError("Unknown radiation qualification stage.")
        n = len(self.condition_ids)
        if (
            not n
            or any(
                not isinstance(values, tuple) or len(values) != n
                for values in (
                    self.condition_ids,
                    self.predicted,
                    self.observed,
                    self.standard_errors,
                )
            )
            or len(set(self.condition_ids)) != n
        ):
            raise ValueError("Stage evidence must contain unique aligned condition rows.")
        if not all(math.isfinite(value) for value in (*self.predicted, *self.observed)):
            raise ValueError("Stage values must be finite.")
        if any(not math.isfinite(value) or value <= 0 for value in self.standard_errors):
            raise ValueError(
                "Stage comparison requires measured positive standard errors."
            )
        if (
            not math.isfinite(self.maximum_standardized_rms)
            or self.maximum_standardized_rms <= 0
        ):
            raise ValueError("Stage acceptance threshold must be declared positive.")
        if self.source_kind not in ("synthetic", "experimental", "external-reference"):
            raise ValueError("Stage source kind must be explicit.")
        if (
            not isinstance(self.upstream_artifact_ids, tuple)
            or not self.upstream_artifact_ids
        ):
            raise ValueError(
                "Stage comparisons require external upstream artifact lineage."
            )
        self.reference.require_uncertainty()

    @property
    def standardized_rms(self) -> float:
        return math.sqrt(
            math.fsum(
                ((p - y) / s) ** 2
                for p, y, s in zip(
                    self.predicted, self.observed, self.standard_errors, strict=True
                )
            )
            / len(self.observed)
        )

    @property
    def accepted(self) -> bool:
        return self.standardized_rms <= self.maximum_standardized_rms


@dataclass(frozen=True, slots=True)
class RadiationCalibrationResult:
    fit: LeastSquaresResult
    posterior: LaplaceResult
    probabilities: jax.Array
    heldout_predictions: jax.Array
    heldout_parameter_variance: jax.Array
    heldout_standardized_residuals: jax.Array
    likelihood_rank: int
    likelihood_singular_values: jax.Array
    stage_evidence: tuple[RadiationStageEvidence, ...]
    gates: tuple[str, ...]
    train_id: str
    heldout_id: str
    prediction_unit: UnitDefinition
    normalization_convention: str

    @property
    def scientifically_qualified(self) -> bool:
        return not self.gates


def calibrate_radiation_lesions(
    training: RadiationCalibrationData,
    heldout: RadiationCalibrationData,
    *,
    initial_logits,
    prior_mean,
    prior_standard_deviation,
    stage_evidence: tuple[RadiationStageEvidence, ...] = (),
    maximum_heldout_standardized_rms: float = 2.0,
    termination: OptimizationTermination | None = None,
    commercial_use=False,
) -> RadiationCalibrationResult:
    """Fit two independent-candidate probabilities, then predict withheld conditions.

    Likelihood is Gaussian with calibrated independent standard errors; correlated
    observation/transport uncertainties require a different explicit likelihood.
    The proper Gaussian logit prior is NOT data identifiability. Native likelihood
    SVD reports rank separately, even when prior curvature yields finite posterior
    intervals. Native optim and UQ execute the fit and uncertainty propagation.
    """
    training.reference.require_rights(training_use=True, commercial_use=commercial_use)
    heldout.reference.require_rights(commercial_use=commercial_use)
    if (
        set(training.observation_ids) & set(heldout.observation_ids)
        or {c.condition_id for c in training.conditions}
        & {c.condition_id for c in heldout.conditions}
        or {c.physical_key() for c in training.conditions}
        & {c.physical_key() for c in heldout.conditions}
    ):
        raise ValueError(
            "Calibration and validation require independent withheld physical conditions/observations."
        )
    if training.normalization_convention != heldout.normalization_convention:
        raise ValueError(
            "Calibration/heldout yield denominators use different conventions."
        )
    scale = float(conversion_factor(heldout.yield_unit, training.yield_unit))
    if (
        not math.isfinite(maximum_heldout_standardized_rms)
        or maximum_heldout_standardized_rms <= 0
    ):
        raise ValueError("Held-out acceptance threshold must be positive.")
    initial = jnp.asarray(initial_logits, dtype=float)
    mean = jnp.asarray(prior_mean, dtype=float)
    prior_sd = jnp.asarray(prior_standard_deviation, dtype=float)
    if (
        initial.shape != (2,)
        or mean.shape != (2,)
        or prior_sd.shape != (2,)
        or not bool(jnp.all(jnp.isfinite(initial)))
        or not bool(jnp.all(jnp.isfinite(mean)))
        or not bool(jnp.all(jnp.isfinite(prior_sd) & (prior_sd > 0)))
    ):
        raise ValueError(
            "Two finite logit parameters and a proper positive Gaussian prior are required."
        )
    train_arrays = training.prepared_arrays()
    heldout_arrays = heldout.prepared_arrays()
    observed = jnp.asarray(training.observed_yields)
    sigma = jnp.asarray(training.standard_errors)

    def likelihood_residual(logits):
        return (expected_initial_lesion_yield(logits, *train_arrays) - observed) / sigma

    def residual(logits, args):
        del args
        return jnp.concatenate((likelihood_residual(logits), (logits - mean) / prior_sd))

    fit = least_squares(
        residual, initial, method=LevenbergMarquardt(), termination=termination
    )
    if not bool(fit.successful):
        raise RuntimeError(
            f"Radiation calibration did not converge (native status {int(fit.status)})."
        )
    space = ParameterSpace(
        fit.parameters,
        log_prior=lambda logits: -0.5 * jnp.sum(((logits - mean) / prior_sd) ** 2),
    )
    posterior_problem = PosteriorProblem(
        space, lambda logits: -0.5 * jnp.sum(likelihood_residual(logits) ** 2)
    )
    posterior = fit_laplace(posterior_problem, fit.parameters)
    propagation = propagate_linearized(
        lambda logits: expected_initial_lesion_yield(logits, *heldout_arrays) * scale,
        fit.parameters,
        DenseCovariance(posterior.covariance),
        source="epistemic",
    )
    parameter_variance = propagation.exact_variance()
    heldout_sigma = jnp.asarray(heldout.standard_errors) * scale
    heldout_observed = jnp.asarray(heldout.observed_yields) * scale
    standardized = (propagation.mean - heldout_observed) / jnp.sqrt(
        heldout_sigma**2 + parameter_variance
    )
    jacobian = jax.jacrev(likelihood_residual)(fit.parameters)
    rank_evidence = svd(
        SVDProblem(DenseLinearOperator(jacobian)),
        policy=SVDSolvePolicy(count=min(jacobian.shape)),
    )
    if not bool(rank_evidence.successful):
        raise RuntimeError("Native likelihood identifiability decomposition failed.")
    rank = int(rank_evidence.numerical_rank)
    gates = []
    if rank < 2:
        gates.append(
            "Likelihood does not identify both cause probabilities; posterior is prior-constrained."
        )
    if training.source_kind != "experimental" or heldout.source_kind != "experimental":
        gates.append(
            "Rights-cleared experimental training and held-out lesion-yield observations are required."
        )
    if float(jnp.sqrt(jnp.mean(standardized**2))) > maximum_heldout_standardized_rms:
        gates.append(
            "Held-out initial-lesion predictions exceed the declared standardized-RMS criterion."
        )
    for evidence in stage_evidence:
        evidence.reference.require_rights(commercial_use=commercial_use)
        if not evidence.accepted:
            gates.append(
                f"Stage {evidence.stage} exceeds its declared uncertainty criterion."
            )
    for stage in ("transport", "chemical-G", "target-reactions", "lesion-yields"):
        if not any(
            item.stage == stage and item.source_kind != "synthetic" and item.accepted
            for item in stage_evidence
        ):
            gates.append(
                f"Independent rights-cleared {stage} stage evidence is missing or unqualified."
            )
    return RadiationCalibrationResult(
        fit,
        posterior,
        jax.nn.sigmoid(fit.parameters),
        propagation.mean,
        parameter_variance,
        standardized,
        rank,
        rank_evidence.singular_values,
        stage_evidence,
        tuple(gates),
        training.fingerprint(),
        heldout.fingerprint(),
        training.yield_unit,
        training.normalization_convention,
    )
