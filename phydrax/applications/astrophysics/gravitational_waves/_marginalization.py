#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import itertools
from collections.abc import Mapping, Sequence
from typing import Any, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.scipy as jsp
import numpy as np
from jaxtyping import Array, ArrayLike, PyTree

from ...._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ...._probability import AbstractProbabilityLaw
from ...._sampling import derive_key, SampleAddress
from ...._strict import StrictModule
from ...._trainable import NonTrainableState
from ....integration import GaussLegendreRule, interval_rule_data
from ....special import ive
from ....uq import AbstractPosteriorTerm
from .._photometry import ObservationDataProvenance
from ._likelihood import GravitationalWaveLikelihoodPlan
from ._status import GravitationalWaveStatus


_RECONSTRUCTION_ADDRESS = SampleAddress(
    "gravitational-wave",
    "marginalized-reconstruction",
    target="joint-nuisance",
    role="posterior",
)
CalibrationCorrectionConvention = Literal["data", "template"]


def _identifier(value: str, role: str, /) -> str:
    identifier = str(value).strip()
    if not identifier:
        raise ValueError(f"{role} must be non-empty.")
    return identifier


def _replace(parameters: PyTree[Any], updates: Mapping[str, Any], /) -> dict[str, Any]:
    if not isinstance(parameters, Mapping):
        raise TypeError("Marginalized gravitational-wave parameters must be a mapping.")
    result = dict(parameters)
    result.update(updates)
    return result


def _log_i0(value: ArrayLike, /) -> Array:
    magnitude = jnp.abs(jnp.asarray(value))
    return magnitude + jnp.log(ive(0.0, magnitude))


class PhaseMarginalizationPlan(StrictModule, NonTrainableState):
    parameter: str = eqx.field(static=True)
    low: float = eqx.field(static=True)
    high: float = eqx.field(static=True)
    reconstruction_nodes: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        parameter: str = "phase",
        /,
        *,
        low: float = 0.0,
        high: float = 2.0 * np.pi,
        reconstruction_nodes: int = 256,
    ):
        name = _identifier(parameter, "phase parameter")
        lower, upper = float(low), float(high)
        count = int(reconstruction_nodes)
        if (
            not np.isfinite(lower)
            or not np.isfinite(upper)
            or lower >= upper
            or count < 16
        ):
            raise ValueError("Phase support and reconstruction capacity are invalid.")
        self.parameter = name
        self.low = lower
        self.high = upper
        self.reconstruction_nodes = count
        self.plan_id = canonical_fingerprint(
            {
                "kind": "gravitational-wave-phase-marginalization",
                "parameter": name,
                "support": [lower, upper],
                "reconstruction_nodes": count,
            }
        )

    @property
    def nodes(self) -> Array:
        spacing = (self.high - self.low) / self.reconstruction_nodes
        return self.low + spacing * (0.5 + jnp.arange(self.reconstruction_nodes))


class TimeMarginalizationPlan(StrictModule, NonTrainableState):
    nodes: Array
    log_weights: Array
    parameter: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        low: float,
        high: float,
        /,
        *,
        num_nodes: int,
        parameter: str = "geocent_time",
    ):
        lower, upper = float(low), float(high)
        count = int(num_nodes)
        name = _identifier(parameter, "time parameter")
        if (
            not np.isfinite(lower)
            or not np.isfinite(upper)
            or lower >= upper
            or count < 2
        ):
            raise ValueError("Time marginalization support and node count are invalid.")
        spacing = (upper - lower) / count
        self.nodes = lower + spacing * (0.5 + jnp.arange(count, dtype=float))
        self.log_weights = jnp.full((count,), -jnp.log(float(count)))
        self.parameter = name
        self.plan_id = canonical_fingerprint(
            {
                "kind": "gravitational-wave-time-marginalization",
                "parameter": name,
                "support": [lower, upper],
                "nodes": count,
                "rule": "uniform-midpoint",
            }
        )


class DistanceMarginalizationPlan(StrictModule, NonTrainableState):
    nodes: Array
    log_weights: Array
    prior_mass_error: Array
    prior: AbstractProbabilityLaw
    parameter: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        low: float,
        high: float,
        prior: AbstractProbabilityLaw,
        /,
        *,
        order: int = 64,
        parameter: str = "luminosity_distance",
    ):
        lower, upper = float(low), float(high)
        count = int(order)
        name = _identifier(parameter, "distance parameter")
        if (
            not np.isfinite(lower)
            or not np.isfinite(upper)
            or lower <= 0.0
            or lower >= upper
            or count < 4
            or not isinstance(prior, AbstractProbabilityLaw)
        ):
            raise ValueError(
                "Distance marginalization support, prior, or order is invalid."
            )
        rule = interval_rule_data(GaussLegendreRule(count))
        reference_nodes = jnp.asarray(rule.nodes)
        reference_weights = jnp.asarray(rule.weights)
        nodes = 0.5 * ((upper - lower) * reference_nodes + upper + lower)
        weights = (
            0.5 * (upper - lower) * reference_weights * jnp.exp(prior.log_prob(nodes))
        )
        mass = jnp.sum(weights)
        if not bool(jnp.isfinite(mass) & (mass > 0.0)):
            raise ValueError("Distance quadrature has no finite prior mass.")
        self.nodes = nodes
        self.log_weights = jnp.log(weights) - jnp.log(mass)
        self.prior_mass_error = jnp.abs(mass - 1.0)
        self.prior = prior
        self.parameter = name
        self.plan_id = canonical_fingerprint(
            {
                "kind": "gravitational-wave-distance-marginalization",
                "parameter": name,
                "support": [lower, upper],
                "order": count,
                "prior": {
                    "type": type(prior).__qualname__,
                    "values": array_tree_fingerprint(prior),
                },
            }
        )


class CalibrationResponseEnsemble(StrictModule, NonTrainableState):
    frequency: Array
    template_multipliers: Array
    log_weights: Array
    provenance: ObservationDataProvenance
    detector_ids: tuple[str, ...] = eqx.field(static=True)
    convention: CalibrationCorrectionConvention = eqx.field(static=True)
    ensemble_id: str = eqx.field(static=True)

    def __init__(
        self,
        frequency: ArrayLike,
        responses: ArrayLike,
        detector_ids: Sequence[str],
        provenance: ObservationDataProvenance,
        /,
        *,
        convention: CalibrationCorrectionConvention,
        log_weights: ArrayLike | None = None,
        ensemble_id: str = "calibration-response-ensemble",
    ):
        frequencies = np.asarray(frequency, dtype=float)
        curves = np.asarray(responses)
        identifiers = tuple(str(value).strip() for value in detector_ids)
        if (
            frequencies.ndim != 1
            or curves.ndim != 3
            or curves.shape[0] == 0
            or curves.shape[1:] != (len(identifiers), frequencies.size)
            or not np.issubdtype(curves.dtype, np.complexfloating)
            or np.any(~np.isfinite(frequencies))
            or np.any(~np.isfinite(curves))
            or np.any(np.diff(frequencies) <= 0.0)
            or not identifiers
            or any(not value for value in identifiers)
            or len(set(identifiers)) != len(identifiers)
            or convention not in ("data", "template")
        ):
            raise ValueError(
                "Calibration frequency, response, detector, or convention is invalid."
            )
        if convention == "data":
            if np.any(curves == 0.0):
                raise ValueError("Data-convention calibration responses must be nonzero.")
            multipliers = 1.0 / curves
        else:
            multipliers = curves
        if log_weights is None:
            logs = np.full((curves.shape[0],), -np.log(float(curves.shape[0])))
        else:
            logs = np.asarray(log_weights, dtype=float)
            if logs.shape != (curves.shape[0],) or np.any(np.isnan(logs)):
                raise ValueError("Calibration log weights must match the curve axis.")
            normalizer = float(jsp.special.logsumexp(jnp.asarray(logs)))
            if not np.isfinite(normalizer):
                raise ValueError("Calibration log weights have no finite mass.")
            logs = logs - normalizer
        if not isinstance(provenance, ObservationDataProvenance):
            raise TypeError("provenance must be ObservationDataProvenance.")
        label = _identifier(ensemble_id, "calibration ensemble ID")
        self.frequency = jnp.asarray(frequencies)
        self.template_multipliers = jnp.asarray(multipliers)
        self.log_weights = jnp.asarray(logs)
        self.provenance = provenance
        self.detector_ids = identifiers
        self.convention = convention
        self.ensemble_id = canonical_fingerprint(
            {
                "kind": "gravitational-wave-calibration-ensemble",
                "label": label,
                "detectors": list(identifiers),
                "convention": convention,
                "content": array_tree_fingerprint(
                    {"frequency": frequencies, "responses": curves, "log_weights": logs}
                )["sha256"],
                "provenance": provenance.provenance_id,
            }
        )

    @property
    def count(self) -> int:
        return int(self.log_weights.shape[0])

    def response(self, index: ArrayLike, frequency: ArrayLike, /) -> Array:
        frequencies = jnp.asarray(frequency)
        if frequencies.shape != self.frequency.shape:
            raise ValueError("Calibration and likelihood frequency shapes disagree.")
        return jnp.take(
            self.template_multipliers, jnp.asarray(index, dtype=jnp.int32), axis=0
        )


class CalibrationMarginalizationPlan(StrictModule, NonTrainableState):
    ensemble: CalibrationResponseEnsemble
    parameter: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        ensemble: CalibrationResponseEnsemble,
        /,
        *,
        parameter: str = "calibration_index",
    ):
        if not isinstance(ensemble, CalibrationResponseEnsemble):
            raise TypeError("ensemble must be CalibrationResponseEnsemble.")
        self.ensemble = ensemble
        self.parameter = _identifier(parameter, "calibration parameter")
        self.plan_id = canonical_fingerprint(
            {
                "kind": "gravitational-wave-calibration-marginalization",
                "ensemble": ensemble.ensemble_id,
                "parameter": self.parameter,
            }
        )


class GravitationalWaveMarginalizationEvaluation(StrictModule):
    log_probability: Array
    evaluated_points: Array
    distance_prior_mass_error: Array
    valid: Array
    status: Array
    plan_id: str = eqx.field(static=True)


class GravitationalWaveMarginalizationPlan(StrictModule):
    """Fixed-grid nuisance integration preserving full normalized likelihoods."""

    likelihood: GravitationalWaveLikelihoodPlan
    phase: PhaseMarginalizationPlan | None
    time: TimeMarginalizationPlan | None
    distance: DistanceMarginalizationPlan | None
    calibration: CalibrationMarginalizationPlan | None
    maximum_grid_points: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        likelihood: GravitationalWaveLikelihoodPlan,
        /,
        *,
        phase: PhaseMarginalizationPlan | None = None,
        time: TimeMarginalizationPlan | None = None,
        distance: DistanceMarginalizationPlan | None = None,
        calibration: CalibrationMarginalizationPlan | None = None,
        maximum_grid_points: int = 1_000_000,
        phase_separable_calibration: bool = False,
    ):
        if not isinstance(likelihood, GravitationalWaveLikelihoodPlan):
            raise TypeError("likelihood must be GravitationalWaveLikelihoodPlan.")
        values = (phase, time, distance, calibration)
        expected = (
            PhaseMarginalizationPlan,
            TimeMarginalizationPlan,
            DistanceMarginalizationPlan,
            CalibrationMarginalizationPlan,
        )
        if all(value is None for value in values):
            raise ValueError("At least one nuisance marginalization plan is required.")
        if any(
            value is not None and not isinstance(value, kind)
            for value, kind in zip(values, expected, strict=True)
        ):
            raise TypeError("Marginalization components have incompatible types.")
        capabilities = likelihood.waveform.capabilities
        if phase is not None:
            if (
                capabilities.phase_parameter != phase.parameter
                or capabilities.phase_harmonic is None
            ):
                raise ValueError(
                    "Waveform does not declare the requested global phase action."
                )
            periods = (
                abs(capabilities.phase_harmonic)
                * (phase.high - phase.low)
                / (2.0 * np.pi)
            )
            if not np.isclose(periods, round(periods), rtol=0.0, atol=1.0e-12):
                raise ValueError(
                    "Phase support must contain an integer number of waveform periods."
                )
            if calibration is not None and not phase_separable_calibration:
                raise ValueError(
                    "Phase/calibration composition requires explicit separability."
                )
        if distance is not None and capabilities.distance_parameter != distance.parameter:
            raise ValueError("Waveform does not declare the requested distance scaling.")
        if calibration is not None:
            if likelihood.calibration_fn is None:
                raise ValueError(
                    "Calibration marginalization requires a calibration response callback."
                )
            if calibration.ensemble.detector_ids != likelihood.network.detector_ids:
                raise ValueError("Calibration detector IDs do not match likelihood data.")
            if not bool(
                jnp.array_equal(
                    calibration.ensemble.frequency, likelihood.network.frequency
                )
            ):
                raise ValueError(
                    "Calibration frequency grid does not match likelihood data."
                )
        count = (
            (1 if time is None else int(time.nodes.size))
            * (1 if distance is None else int(distance.nodes.size))
            * (1 if calibration is None else calibration.ensemble.count)
        )
        limit = int(maximum_grid_points)
        if limit <= 0 or count > limit:
            raise ValueError("Marginalization grid exceeds maximum_grid_points.")
        self.likelihood = likelihood
        self.phase = phase
        self.time = time
        self.distance = distance
        self.calibration = calibration
        self.maximum_grid_points = limit
        self.plan_id = canonical_fingerprint(
            {
                "kind": "gravitational-wave-marginalization",
                "likelihood": likelihood.likelihood_id,
                "phase": None if phase is None else phase.plan_id,
                "time": None if time is None else time.plan_id,
                "distance": None if distance is None else distance.plan_id,
                "calibration": None if calibration is None else calibration.plan_id,
                "maximum_grid_points": limit,
                "phase_separable_calibration": bool(phase_separable_calibration),
            }
        )

    def _nonphase_choices(self) -> tuple[tuple[dict[str, Any], Array], ...]:
        calibration_choices = (
            (({}, jnp.asarray(0.0)),)
            if self.calibration is None
            else tuple(
                (
                    {self.calibration.parameter: jnp.asarray(index, dtype=jnp.int32)},
                    self.calibration.ensemble.log_weights[index],
                )
                for index in range(self.calibration.ensemble.count)
            )
        )
        distance_choices = (
            (({}, jnp.asarray(0.0)),)
            if self.distance is None
            else tuple(
                (
                    {self.distance.parameter: self.distance.nodes[index]},
                    self.distance.log_weights[index],
                )
                for index in range(int(self.distance.nodes.size))
            )
        )
        time_choices = (
            (({}, jnp.asarray(0.0)),)
            if self.time is None
            else tuple(
                (
                    {self.time.parameter: self.time.nodes[index]},
                    self.time.log_weights[index],
                )
                for index in range(int(self.time.nodes.size))
            )
        )
        choices = []
        for calibration, distance, time in itertools.product(
            calibration_choices, distance_choices, time_choices
        ):
            updates = {**calibration[0], **distance[0], **time[0]}
            choices.append((updates, calibration[1] + distance[1] + time[1]))
        return tuple(choices)

    def _phase_marginalized_value(self, parameters: PyTree[Any], /) -> Array:
        if self.phase is None:
            return self.likelihood.log_probability(parameters)
        reference = _replace(parameters, {self.phase.parameter: jnp.asarray(0.0)})
        evaluation = self.likelihood.evaluate(reference)
        complex_inner = jnp.sum(evaluation.complex_data_signal_inner_product)
        signal_norm = jnp.sum(evaluation.signal_norm)
        value = (
            evaluation.noise_log_probability - 0.5 * signal_norm + _log_i0(complex_inner)
        )
        return jnp.where(evaluation.valid, value, -jnp.inf)

    def evaluate(
        self, parameters: PyTree[Any], /
    ) -> GravitationalWaveMarginalizationEvaluation:
        values = []
        for updates, log_weight in self._nonphase_choices():
            values.append(
                self._phase_marginalized_value(_replace(parameters, updates)) + log_weight
            )
        stacked = jnp.stack(tuple(values))
        log_probability = jsp.special.logsumexp(stacked)
        valid = jnp.isfinite(log_probability) & jnp.all(~jnp.isnan(stacked))
        status = jnp.where(
            valid,
            int(GravitationalWaveStatus.SUCCESS),
            int(GravitationalWaveStatus.MARGINALIZATION_FAILURE),
        ).astype(jnp.int32)
        mass_error = (
            jnp.asarray(0.0) if self.distance is None else self.distance.prior_mass_error
        )
        return GravitationalWaveMarginalizationEvaluation(
            log_probability,
            jnp.asarray(len(values), dtype=jnp.int32),
            mass_error,
            valid,
            status,
            self.plan_id,
        )

    def log_probability(self, parameters: PyTree[Any], /) -> Array:
        result = self.evaluate(parameters)
        return jnp.where(result.valid, result.log_probability, -jnp.inf)

    def _joint_choices(self) -> tuple[tuple[dict[str, Any], Array], ...]:
        phase_choices = (
            (({}, jnp.asarray(0.0)),)
            if self.phase is None
            else tuple(
                (
                    {self.phase.parameter: self.phase.nodes[index]},
                    -jnp.log(float(self.phase.reconstruction_nodes)),
                )
                for index in range(self.phase.reconstruction_nodes)
            )
        )
        choices = []
        for nonphase, phase in itertools.product(self._nonphase_choices(), phase_choices):
            choices.append(({**nonphase[0], **phase[0]}, nonphase[1] + phase[1]))
        if len(choices) > self.maximum_grid_points:
            raise ValueError("Joint reconstruction grid exceeds maximum_grid_points.")
        return tuple(choices)


class GravitationalWaveMarginalizedPosteriorTerm(AbstractPosteriorTerm):
    marginalization: GravitationalWaveMarginalizationPlan

    def __init__(
        self,
        marginalization: GravitationalWaveMarginalizationPlan,
        /,
        *,
        label: str = "gravitational_wave_marginalized_network",
    ):
        if not isinstance(marginalization, GravitationalWaveMarginalizationPlan):
            raise TypeError(
                "marginalization must be GravitationalWaveMarginalizationPlan."
            )
        self.marginalization = marginalization
        self.label = _identifier(label, "posterior-term label")

    def per_case_log_prob(self, parameters: PyTree[Any], /) -> Array:
        return self.marginalization.log_probability(parameters).reshape((1,))


class MarginalizedParameterDraws(StrictModule):
    samples: PyTree[Array]
    selected_log_probability: Array
    valid: Array
    root_key: Array
    sample_shape: tuple[int, ...] = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)


def reconstruct_marginalized_parameters(
    plan: GravitationalWaveMarginalizationPlan,
    key: Array,
    samples: Mapping[str, ArrayLike],
    /,
    *,
    sample_ndim: int = 1,
) -> MarginalizedParameterDraws:
    if not isinstance(plan, GravitationalWaveMarginalizationPlan):
        raise TypeError("plan must be GravitationalWaveMarginalizationPlan.")
    if not isinstance(samples, Mapping) or not samples:
        raise TypeError("samples must be a non-empty parameter mapping.")
    rank = int(sample_ndim)
    leaves = tuple(jnp.asarray(value) for value in samples.values())
    if rank <= 0 or any(value.ndim < rank for value in leaves):
        raise ValueError("sample_ndim must select nonempty leading sample axes.")
    sample_shape = tuple(int(size) for size in leaves[0].shape[:rank])
    if any(tuple(value.shape[:rank]) != sample_shape for value in leaves[1:]):
        raise ValueError("Every sample leaf must share the leading sample shape.")
    count = int(np.prod(sample_shape))
    if count <= 0:
        raise ValueError("Marginalized reconstruction requires at least one sample.")
    flattened = {
        name: jnp.asarray(value).reshape((count, *jnp.asarray(value).shape[rank:]))
        for name, value in samples.items()
    }
    choices = plan._joint_choices()
    selected_samples = []
    selected_logs = []
    for sample_index in range(count):
        parameters = {name: value[sample_index] for name, value in flattened.items()}
        candidates = []
        for updates, log_weight in choices:
            candidate = _replace(parameters, updates)
            candidates.append(plan.likelihood.log_probability(candidate) + log_weight)
        candidate_logs = jnp.stack(tuple(candidates))
        normalized = candidate_logs - jsp.special.logsumexp(candidate_logs)
        selected = jr.categorical(
            derive_key(key, _RECONSTRUCTION_ADDRESS, sample_index), normalized
        )
        selected_index = int(selected)
        selected_samples.append(_replace(parameters, choices[selected_index][0]))
        selected_logs.append(normalized[selected_index])
    stacked = jax.tree_util.tree_map(lambda *values: jnp.stack(values), *selected_samples)
    restored = jax.tree_util.tree_map(
        lambda value: value.reshape((*sample_shape, *value.shape[1:])), stacked
    )
    logs = jnp.stack(tuple(selected_logs)).reshape(sample_shape)
    valid = jnp.isfinite(logs)
    return MarginalizedParameterDraws(
        restored,
        logs,
        valid,
        jnp.asarray(key),
        sample_shape,
        plan.plan_id,
    )


__all__ = [
    "CalibrationCorrectionConvention",
    "CalibrationMarginalizationPlan",
    "CalibrationResponseEnsemble",
    "DistanceMarginalizationPlan",
    "GravitationalWaveMarginalizationEvaluation",
    "GravitationalWaveMarginalizationPlan",
    "GravitationalWaveMarginalizedPosteriorTerm",
    "MarginalizedParameterDraws",
    "PhaseMarginalizationPlan",
    "TimeMarginalizationPlan",
    "reconstruct_marginalized_parameters",
]
