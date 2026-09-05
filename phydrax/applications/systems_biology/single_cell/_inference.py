#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Measured-count moment inference and qualified count-derived drift estimates.

Stationary snapshots identify rate ratios, never an absolute clock. A supplied
independent rate calibration can fix that gauge. First-order fit covariance is
not a posterior and does not include unknown assay/model discrepancy.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from phydrax._fingerprint import canonical_fingerprint
from phydrax.optim import (
    least_squares,
    LeastSquaresResult,
    LevenbergMarquardt,
    OptimizationTermination,
)
from phydrax.qualification import ReferenceArtifactManifest
from phydrax.units import conversion_factor, SECOND, UnitDefinition
from phydrax.uq import DenseCovariance, LinearizedPropagationResult, propagate_linearized

from .._gene_expression import (
    PreparedTelegraphGeneExpression,
    TelegraphFitTarget,
    TelegraphGeneExpressionPlan,
)
from ._assay import TranscriptCountAssay, TranscriptCounts
from ._scenario import _label


@dataclass(frozen=True, slots=True)
class StationaryCountTarget:
    observations: TranscriptCounts
    target: TelegraphFitTarget
    equilibrium_evidence_id: str

    @classmethod
    def from_counts(
        cls,
        observations: TranscriptCounts,
        standard_errors: ArrayLike,
        /,
        *,
        equilibrium_evidence_id: str,
    ) -> StationaryCountTarget:
        """Use independent complete U/S snapshot pairs and declared moment errors.

        Moment order is mean(U), mean(S), var(U), var(S), cov(U,S). Equilibrium is
        an explicit experimental assumption, not inferred from a saved time label.
        Variances/covariance use the unbiased n−1 denominator.
        """
        _label(equilibrium_evidence_id, "equilibrium_evidence_id")
        selected = np.asarray(observations.valid).all(axis=-1)
        values = np.asarray(observations.counts)[selected]
        if values.shape[0] < 3:
            raise ValueError(
                "Stationary moment inference needs at least three complete independent snapshots."
            )
        means = jnp.mean(jnp.asarray(values), axis=0)
        centered = jnp.asarray(values) - means
        covariance = (centered.T @ centered) / (values.shape[0] - 1)
        moments = jnp.stack(
            (means[0], means[1], covariance[0, 0], covariance[1, 1], covariance[0, 1])
        )
        return cls(
            observations,
            TelegraphFitTarget(moments, standard_errors),
            equilibrium_evidence_id,
        )


def predicted_count_moments(
    model: PreparedTelegraphGeneExpression,
    rates: ArrayLike,
    assay: TranscriptCountAssay,
    /,
) -> Array:
    """Differentiable exact stationary observable moments under calibrated capture."""
    latent = model.stationary_moments(rates)
    mean_u, var_u = assay.unspliced.observed_moments(
        latent.nascent_mean, latent.nascent_variance
    )
    mean_s, var_s = assay.spliced.observed_moments(
        latent.mature_mean, latent.mature_variance
    )
    covariance = (
        assay.unspliced.plan.capture_probability
        * assay.spliced.plan.capture_probability
        * latent.nascent_mature_covariance
    )
    return jnp.where(
        latent.valid, jnp.stack((mean_u, mean_s, var_u, var_s, covariance)), jnp.nan
    )


@dataclass(frozen=True, slots=True)
class TranscriptIdentifiability:
    sensitivity: Array
    singular_values: Array
    rank: int
    free_parameter_indices: tuple[int, ...]
    absolute_time_calibrated: bool

    @property
    def locally_identifiable(self) -> bool:
        return self.rank == len(self.free_parameter_indices)


def _sensitivity_evidence(
    sensitivity: Array, free: tuple[int, ...], calibrated: bool
) -> TranscriptIdentifiability:
    host = np.asarray(sensitivity)
    singular = np.linalg.svd(host, compute_uv=False)
    threshold = (
        10
        * max(host.shape)
        * np.finfo(host.dtype).eps
        * max(float(np.max(singular, initial=0)), 1.0)
    )
    rank = int(np.sum(singular > threshold))
    return TranscriptIdentifiability(
        sensitivity, jnp.asarray(singular), rank, free, calibrated
    )


@dataclass(frozen=True, slots=True)
class TranscriptFit:
    target: StationaryCountTarget
    assay: TranscriptCountAssay
    model: PreparedTelegraphGeneExpression
    result: LeastSquaresResult
    rates: Array
    rate_time_unit: UnitDefinition
    identifiability: TranscriptIdentifiability
    free_log_rate_covariance: Array | None
    count_prediction_uq: LinearizedPropagationResult | None
    rate_calibration: ReferenceArtifactManifest | None
    fit_id: str

    def predict_count_moments(
        self, assay: TranscriptCountAssay | None = None, /
    ) -> Array:
        return predicted_count_moments(
            self.model, self.rates, self.assay if assay is None else assay
        )

    def held_out_residuals(self, target: StationaryCountTarget, /) -> Array:
        if set(target.observations.cell_ids) & set(self.target.observations.cell_ids):
            raise ValueError("Held-out prediction requires disjoint cell identities.")
        if (
            target.observations.gene != self.target.observations.gene
            or target.observations.assay_id != self.assay.assay_id
        ):
            raise ValueError(
                "Held-out target must bind the same gene and calibrated assay."
            )
        return (
            self.predict_count_moments() - target.target.moments
        ) / target.target.standard_errors


def fit_stationary_counts(
    target: StationaryCountTarget,
    assay: TranscriptCountAssay,
    initial_rates: ArrayLike,
    /,
    *,
    fixed_rates: Mapping[int, float] | None = None,
    rate_time_unit: UnitDefinition = SECOND,
    rate_calibration: ReferenceArtifactManifest | None = None,
    maximum_steps: int = 128,
) -> TranscriptFit:
    """Native positive-log-rate moment fitting, with honest rank and clock evidence.

    ``fixed_rates`` maps rate indices to independently calibrated values in inverse
    ``rate_time_unit``. No fixed clock means all-rate scaling is unidentifiable;
    covariance and physical-velocity claims are refused in that case. The objective
    assumes caller-declared independent moment errors, not an exact count likelihood.
    """
    conversion_factor(rate_time_unit, SECOND)
    if target.observations.assay_id != assay.assay_id:
        raise ValueError("Count target is bound to another assay calibration.")
    initial = np.asarray(initial_rates, dtype=float)
    if initial.shape != (5,) or np.any(~np.isfinite(initial)) or np.any(initial <= 0):
        raise ValueError("Initial rates must be a finite positive vector of length five.")
    fixed = {} if fixed_rates is None else dict(fixed_rates)
    if any(
        isinstance(i, bool) or not isinstance(i, int) or i not in range(5) for i in fixed
    ):
        raise ValueError("Fixed-rate keys must be rate indices 0 through 4.")
    if any(not np.isfinite(value) or value <= 0 for value in fixed.values()):
        raise ValueError("Fixed rates must be finite and positive.")
    if fixed:
        if rate_calibration is None:
            raise ValueError(
                "Fixing physical rate values requires an independent calibration manifest."
            )
        rate_calibration.require_rights()
        rate_calibration.require_uncertainty()
    elif rate_calibration is not None:
        raise ValueError(
            "A rate calibration without any fixed rates does not identify a clock."
        )
    free = tuple(i for i in range(5) if i not in fixed)
    if not free:
        raise ValueError("At least one rate must remain free to fit measured counts.")
    base = initial.copy()
    for index, value in fixed.items():
        base[index] = value
    model = TelegraphGeneExpressionPlan(
        *tuple(base), name="single-cell-stationary-count-fit"
    ).prepare()
    indices = jnp.asarray(free, dtype=jnp.int32)
    logs = jnp.log(jnp.asarray(base))

    def realize(free_logs):
        return jnp.exp(logs.at[indices].set(free_logs))

    def forward(free_logs):
        return predicted_count_moments(model, realize(free_logs), assay)

    def residual(free_logs, args):
        del args
        return (
            forward(free_logs) - target.target.moments
        ) / target.target.standard_errors

    result = least_squares(
        residual,
        logs[indices],
        method=LevenbergMarquardt(),
        termination=OptimizationTermination(maximum_steps=maximum_steps),
    )
    rates = realize(result.parameters)
    sensitivity = jax.jacrev(lambda value: residual(value, None))(result.parameters)
    evidence = _sensitivity_evidence(sensitivity, free, bool(fixed))
    covariance, prediction_uq = None, None
    if bool(result.successful) and evidence.locally_identifiable:
        # Host SVD supplies local rank and a covariance factor, never a regularized
        # inverse of an unidentifiable sensitivity. Declared errors are absolute.
        _, singular, right = np.linalg.svd(np.asarray(sensitivity), full_matrices=False)
        factor = right.T / singular[None, :]
        covariance = jnp.asarray(factor @ factor.T)
        prediction_uq = propagate_linearized(
            forward, result.parameters, DenseCovariance(covariance), source="input"
        )
    identity = canonical_fingerprint(
        {
            "kind": "transcript-count-fit",
            "target": target.target.target_id,
            "observations": target.observations.observation_id,
            "equilibrium": target.equilibrium_evidence_id,
            "assay": assay.assay_id,
            "rates": np.asarray(rates).tolist(),
            "fixed": sorted(fixed.items()),
            "rate_time_unit": rate_time_unit.unit_id,
            "calibration": None
            if rate_calibration is None
            else rate_calibration.manifest_id,
        }
    )
    return TranscriptFit(
        target,
        assay,
        model,
        result,
        rates,
        rate_time_unit,
        evidence,
        covariance,
        prediction_uq,
        rate_calibration,
        identity,
    )


@dataclass(frozen=True, slots=True)
class TranscriptVelocityEvidence:
    """Capture-corrected drift estimator, not a sampled-path derivative or lineage.

    Negative deconvolved estimates are retained: clipping would introduce bias.
    This is not a posterior latent velocity, and cannot imply an energy landscape.
    """

    estimates: Array
    valid: Array
    observations_id: str
    fit_id: str
    inverse_time_unit: UnitDefinition
    estimator: str = "calibrated-count-linear-drift"
    preprocessing: str = "subtract-background-divide-capture-no-clipping"
    uncertainty: str = "conditional-on-fitted-rates-and-assay;not-a-posterior"


def predict_transcript_velocity(
    fit: TranscriptFit, observations: TranscriptCounts, /
) -> TranscriptVelocityEvidence:
    """Estimate βU−γS solely from held/measured U/S counts, never stored latent truth."""
    if (
        not bool(fit.result.successful)
        or not fit.identifiability.locally_identifiable
        or not fit.identifiability.absolute_time_calibrated
    ):
        raise ValueError(
            "Physical velocity requires a successful identifiable fit with an independent rate clock."
        )
    if (
        observations.assay_id != fit.assay.assay_id
        or observations.gene != fit.target.observations.gene
    ):
        raise ValueError("Velocity observations must bind the fitted gene and assay.")
    captures = jnp.stack(
        (
            fit.assay.unspliced.plan.capture_probability,
            fit.assay.spliced.plan.capture_probability,
        )
    )
    if bool(jnp.any(captures <= 0)):
        raise ValueError(
            "Count-derived drift requires positive capture in both channels."
        )
    backgrounds = jnp.stack(
        (fit.assay.unspliced.plan.background_rate, fit.assay.spliced.plan.background_rate)
    )
    corrected = (observations.counts - backgrounds) / captures
    estimates = fit.rates[3] * corrected[:, 0] - fit.rates[4] * corrected[:, 1]
    valid = jnp.all(observations.valid, axis=-1)
    from phydrax.units import derived_unit

    unit = derived_unit("transcript-drift-rate", ((fit.rate_time_unit, -1),))
    return TranscriptVelocityEvidence(
        jnp.where(valid, estimates, jnp.nan),
        valid,
        observations.observation_id,
        fit.fit_id,
        unit,
    )
