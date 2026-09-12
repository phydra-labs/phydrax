#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike, PyTree

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..integration import WeightedSampleTarget
from ._particle import effective_sample_size, normalize_log_weights
from ._posterior_reweighting import _flatten_target
from ._posterior_terms import AbstractPosteriorTerm


EvidenceKind = Literal["absolute", "noise-relative", "omitted-constant"]


def _identifier(value: str, role: str, /) -> str:
    identifier = str(value).strip()
    if not identifier:
        raise ValueError(f"{role} must be non-empty.")
    return identifier


class EventPosterior(StrictModule, NonTrainableState):
    """One event posterior with source-prior and dependence evidence."""

    posterior: WeightedSampleTarget
    sampling_log_prior: Array
    log_evidence: Array
    source_effective_sample_size: Array
    event_id: str = eqx.field(static=True)
    parameterization_id: str = eqx.field(static=True)
    likelihood_id: str = eqx.field(static=True)
    provider_id: str = eqx.field(static=True)
    inference_method: str = eqx.field(static=True)
    approximation: str = eqx.field(static=True)
    evidence_kind: EvidenceKind = eqx.field(static=True)
    has_evidence: bool = eqx.field(static=True)
    event_posterior_id: str = eqx.field(static=True)

    def __init__(
        self,
        posterior: WeightedSampleTarget,
        sampling_log_prior: ArrayLike,
        /,
        *,
        event_id: str,
        parameterization_id: str,
        likelihood_id: str,
        provider_id: str,
        inference_method: str,
        approximation: str,
        source_effective_sample_size: ArrayLike,
        log_evidence: ArrayLike | None = None,
        evidence_kind: EvidenceKind = "omitted-constant",
    ):
        samples, weights, active, shape = _flatten_target(posterior)
        del samples
        if posterior.support_valid is not None and not bool(
            jnp.all(posterior.support_valid)
        ):
            raise ValueError("Event posterior reports invalid proposal support.")
        prior = jnp.asarray(sampling_log_prior)
        if prior.shape != shape or bool(
            jnp.any(active.reshape(shape) & ~jnp.isfinite(prior))
        ):
            raise ValueError(
                "Sampling-prior values must be finite on every active posterior sample."
            )
        source_ess = jnp.asarray(source_effective_sample_size, dtype=float).reshape(())
        active_count = jnp.sum(active)
        if not bool(
            jnp.isfinite(source_ess)
            & (source_ess > 0.0)
            & (source_ess <= active_count + 1.0e-8)
        ):
            raise ValueError(
                "Source effective sample size must lie within the active sample count."
            )
        if evidence_kind not in ("absolute", "noise-relative", "omitted-constant"):
            raise ValueError("Unknown event evidence kind.")
        has_evidence = log_evidence is not None
        evidence = jnp.asarray(0.0 if log_evidence is None else log_evidence).reshape(())
        if has_evidence and not bool(jnp.isfinite(evidence)):
            raise ValueError("Event evidence must be finite when supplied.")
        if has_evidence == (evidence_kind == "omitted-constant"):
            raise ValueError("Evidence kind must agree with evidence availability.")
        identifiers = tuple(
            _identifier(value, role)
            for value, role in (
                (event_id, "event ID"),
                (parameterization_id, "parameterization ID"),
                (likelihood_id, "likelihood ID"),
                (provider_id, "provider ID"),
                (inference_method, "inference method"),
                (approximation, "approximation"),
            )
        )
        self.posterior = posterior
        self.sampling_log_prior = prior
        self.log_evidence = evidence
        self.source_effective_sample_size = source_ess
        (
            self.event_id,
            self.parameterization_id,
            self.likelihood_id,
            self.provider_id,
            self.inference_method,
            self.approximation,
        ) = identifiers
        self.evidence_kind = evidence_kind
        self.has_evidence = has_evidence
        self.event_posterior_id = canonical_fingerprint(
            {
                "kind": "event-posterior",
                "event": self.event_id,
                "parameterization": self.parameterization_id,
                "likelihood": self.likelihood_id,
                "provider": self.provider_id,
                "method": self.inference_method,
                "approximation": self.approximation,
                "evidence_kind": evidence_kind,
                "sample_shape": list(shape),
                "content": array_tree_fingerprint(
                    {
                        "samples": posterior.samples,
                        "weights": weights,
                        "sampling_log_prior": prior,
                        "log_evidence": evidence,
                        "source_effective_sample_size": source_ess,
                    }
                )["sha256"],
            }
        )


class PopulationSampleBatch(StrictModule, NonTrainableState):
    """Fixed-capacity events in one common physical population parameterization."""

    parameters: PyTree[Array]
    log_weights: Array
    sampling_log_prior: Array
    mask: Array
    log_evidence: Array
    has_evidence: Array
    source_effective_sample_size: Array
    evidence_kinds: tuple[EvidenceKind, ...] = eqx.field(static=True)
    event_ids: tuple[str, ...] = eqx.field(static=True)
    event_posterior_ids: tuple[str, ...] = eqx.field(static=True)
    parameterization_id: str = eqx.field(static=True)
    capacity: int = eqx.field(static=True)
    batch_id: str = eqx.field(static=True)

    @property
    def num_events(self) -> int:
        return len(self.event_ids)


def prepare_population_sample_batch(
    events: Sequence[EventPosterior],
    /,
    *,
    capacity: int | None = None,
) -> PopulationSampleBatch:
    items = tuple(events)
    if not items or any(not isinstance(item, EventPosterior) for item in items):
        raise TypeError("events must contain EventPosterior values.")
    parameterization = items[0].parameterization_id
    if any(item.parameterization_id != parameterization for item in items):
        raise ValueError("Every event must use the same population parameterization.")
    event_ids = tuple(item.event_id for item in items)
    if len(set(event_ids)) != len(event_ids):
        raise ValueError("Population event IDs must be unique.")
    flattened = []
    counts = []
    for item in items:
        samples, weights, active, shape = _flatten_target(item.posterior)
        normalized, _, valid = normalize_log_weights(jnp.where(active, weights, -jnp.inf))
        if not bool(valid):
            raise ValueError(f"Event {item.event_id!r} has invalid posterior weights.")
        count = int(np.prod(shape))
        flattened.append(
            (samples, normalized, item.sampling_log_prior.reshape((count,)), active)
        )
        counts.append(count)
    width = max(counts) if capacity is None else int(capacity)
    if width <= 0 or any(count > width for count in counts):
        raise ValueError("Population sample capacity must cover every event.")
    structure = jax.tree_util.tree_structure(flattened[0][0])
    if any(
        jax.tree_util.tree_structure(value[0]) != structure for value in flattened[1:]
    ):
        raise ValueError(
            "Event posterior parameter PyTrees must have identical structure."
        )
    reference_shapes = tuple(
        value.shape[1:] for value in jax.tree_util.tree_leaves(flattened[0][0])
    )
    if any(
        tuple(value.shape[1:] for value in jax.tree_util.tree_leaves(samples))
        != reference_shapes
        for samples, _, _, _ in flattened[1:]
    ):
        raise ValueError("Event posterior parameter event shapes must agree.")

    padded_trees = []
    weight_rows = []
    prior_rows = []
    mask_rows = []
    for count, (samples, weights, prior, active) in zip(counts, flattened, strict=True):
        padding = width - count
        padded_trees.append(
            jax.tree_util.tree_map(
                lambda value, padding=padding: jnp.concatenate(
                    (value, jnp.zeros((padding, *value.shape[1:]), dtype=value.dtype)),
                    axis=0,
                ),
                samples,
            )
        )
        weight_rows.append(jnp.pad(weights, (0, padding), constant_values=-jnp.inf))
        prior_rows.append(jnp.pad(prior, (0, padding), constant_values=0.0))
        mask_rows.append(jnp.pad(active, (0, padding), constant_values=False))
    parameters = jax.tree_util.tree_map(lambda *values: jnp.stack(values), *padded_trees)
    log_weights = jnp.stack(tuple(weight_rows))
    sampling_log_prior = jnp.stack(tuple(prior_rows))
    mask = jnp.stack(tuple(mask_rows))
    posterior_ids = tuple(item.event_posterior_id for item in items)
    evidence_kinds = tuple(item.evidence_kind for item in items)
    batch_id = canonical_fingerprint(
        {
            "kind": "population-sample-batch",
            "events": list(posterior_ids),
            "parameterization": parameterization,
            "evidence_kinds": list(evidence_kinds),
            "capacity": width,
            "content": array_tree_fingerprint(
                {
                    "parameters": parameters,
                    "log_weights": log_weights,
                    "sampling_log_prior": sampling_log_prior,
                    "mask": mask,
                }
            )["sha256"],
        }
    )
    return PopulationSampleBatch(
        parameters,
        log_weights,
        sampling_log_prior,
        mask,
        jnp.stack(tuple(item.log_evidence for item in items)),
        jnp.asarray(tuple(item.has_evidence for item in items), dtype=bool),
        jnp.stack(tuple(item.source_effective_sample_size for item in items)),
        evidence_kinds,
        event_ids,
        posterior_ids,
        parameterization,
        width,
        batch_id,
    )


class SelectionInjectionSet(StrictModule, NonTrainableState):
    parameters: PyTree[Array]
    proposal_log_prob: Array
    detection_probability: Array
    mask: Array
    campaign_id: str = eqx.field(static=True)
    parameterization_id: str = eqx.field(static=True)
    selection_id: str = eqx.field(static=True)

    def __init__(
        self,
        parameters: PyTree[ArrayLike],
        proposal_log_prob: ArrayLike,
        detection_probability: ArrayLike,
        /,
        *,
        campaign_id: str,
        parameterization_id: str,
        mask: ArrayLike | None = None,
    ):
        leaves = tuple(
            jnp.asarray(value) for value in jax.tree_util.tree_leaves(parameters)
        )
        if not leaves or leaves[0].ndim < 1:
            raise ValueError("Selection injections require a leading draw axis.")
        count = int(leaves[0].shape[0])
        if any(value.ndim < 1 or int(value.shape[0]) != count for value in leaves):
            raise ValueError("Every selection parameter leaf must share the draw axis.")
        proposal = jnp.asarray(proposal_log_prob, dtype=float)
        detection = jnp.asarray(detection_probability, dtype=float)
        active = (
            jnp.ones((count,), dtype=bool)
            if mask is None
            else jnp.asarray(mask, dtype=bool)
        )
        if (
            proposal.shape != (count,)
            or detection.shape != (count,)
            or active.shape != (count,)
            or not bool(jnp.any(active))
            or bool(
                jnp.any(active & (~jnp.isfinite(proposal) | ~jnp.isfinite(detection)))
            )
            or bool(jnp.any(active & ((detection < 0.0) | (detection > 1.0))))
        ):
            raise ValueError(
                "Selection proposal, detection probabilities, or mask are invalid."
            )
        if any(bool(jnp.any(~jnp.isfinite(value[active]))) for value in leaves):
            raise ValueError("Active selection-injection parameters must be finite.")
        campaign = _identifier(campaign_id, "selection campaign ID")
        parameterization = _identifier(
            parameterization_id, "selection parameterization ID"
        )
        self.parameters = jax.tree_util.tree_map(jnp.asarray, parameters)
        self.proposal_log_prob = proposal
        self.detection_probability = detection
        self.mask = active
        self.campaign_id = campaign
        self.parameterization_id = parameterization
        self.selection_id = canonical_fingerprint(
            {
                "kind": "selection-injection-set",
                "campaign": campaign,
                "parameterization": parameterization,
                "content": array_tree_fingerprint(
                    {
                        "parameters": self.parameters,
                        "proposal_log_prob": proposal,
                        "detection_probability": detection,
                        "mask": active,
                    }
                )["sha256"],
            }
        )

    @property
    def draw_count(self) -> int:
        return int(self.mask.shape[0])


class SelectionEfficiencyEstimate(StrictModule):
    efficiency: Array
    log_efficiency: Array
    importance_effective_sample_size: Array
    standard_error: Array
    active_draws: Array
    valid: Array
    selection_id: str = eqx.field(static=True)


def estimate_selection_efficiency(
    injections: SelectionInjectionSet,
    hyperparameters: PyTree[Any],
    population_log_prob: Callable[[PyTree[Any], PyTree[Any]], Array],
    /,
) -> SelectionEfficiencyEstimate:
    if not isinstance(injections, SelectionInjectionSet) or not callable(
        population_log_prob
    ):
        raise TypeError(
            "Selection estimate requires injections and a population log density."
        )
    population = jax.vmap(lambda sample: population_log_prob(hyperparameters, sample))(
        injections.parameters
    )
    if population.shape != injections.proposal_log_prob.shape:
        raise ValueError("Population log density must return one scalar per injection.")
    detected_log = jnp.where(
        injections.detection_probability > 0.0,
        jnp.log(injections.detection_probability),
        -jnp.inf,
    )
    contributions = jnp.where(
        injections.mask,
        population - injections.proposal_log_prob + detected_log,
        -jnp.inf,
    )
    active_count = jnp.sum(injections.mask)
    log_efficiency = jax.scipy.special.logsumexp(contributions) - jnp.log(active_count)
    normalized, _, weights_valid = normalize_log_weights(contributions)
    ess = effective_sample_size(normalized)
    efficiency = jnp.exp(log_efficiency)
    anchor = jnp.max(contributions)
    scaled = jnp.where(injections.mask, jnp.exp(contributions - anchor), 0.0)
    scaled_mean = jnp.sum(scaled) / active_count
    scaled_variance = jnp.sum(
        jnp.where(injections.mask, (scaled - scaled_mean) ** 2, 0.0)
    ) / jnp.maximum(active_count - 1, 1)
    standard_error = jnp.exp(anchor) * jnp.sqrt(scaled_variance / active_count)
    valid = (
        weights_valid
        & jnp.isfinite(log_efficiency)
        & jnp.isfinite(standard_error)
        & (efficiency > 0.0)
        & (efficiency <= 1.0 + 1.0e-8)
    )
    return SelectionEfficiencyEstimate(
        efficiency,
        log_efficiency,
        ess,
        standard_error,
        active_count,
        valid,
        injections.selection_id,
    )


class PopulationLikelihoodDiagnostics(StrictModule):
    event_log_factors: Array
    event_importance_effective_sample_size: Array
    selection: SelectionEfficiencyEstimate | None
    valid: Array
    batch_id: str = eqx.field(static=True)


class PopulationPosteriorTerm(AbstractPosteriorTerm):
    """Posterior-recycling likelihood conditional on the observed catalog size."""

    batch: PopulationSampleBatch
    population_log_prob: Callable[[PyTree[Any], PyTree[Any]], Array] = eqx.field(
        static=True
    )
    selection: SelectionInjectionSet | None
    minimum_event_effective_sample_size: float = eqx.field(static=True)
    include_event_evidence: bool = eqx.field(static=True)

    def __init__(
        self,
        batch: PopulationSampleBatch,
        population_log_prob: Callable[[PyTree[Any], PyTree[Any]], Array],
        /,
        *,
        selection: SelectionInjectionSet | None = None,
        minimum_event_effective_sample_size: float = 20.0,
        include_event_evidence: bool = False,
        label: str = "population_posterior_recycling",
    ):
        if not isinstance(batch, PopulationSampleBatch) or not callable(
            population_log_prob
        ):
            raise TypeError(
                "Population term requires a sample batch and population density."
            )
        if selection is not None and not isinstance(selection, SelectionInjectionSet):
            raise TypeError("selection must be SelectionInjectionSet or None.")
        if (
            selection is not None
            and selection.parameterization_id != batch.parameterization_id
        ):
            raise ValueError("Selection and event parameterizations must agree.")
        minimum = float(minimum_event_effective_sample_size)
        if not np.isfinite(minimum) or minimum <= 0.0:
            raise ValueError("Minimum event effective sample size must be positive.")
        if include_event_evidence and (
            not bool(jnp.all(batch.has_evidence))
            or any(kind != "absolute" for kind in batch.evidence_kinds)
        ):
            raise ValueError(
                "Absolute population likelihood requires absolute evidence for every event."
            )
        self.batch = batch
        self.population_log_prob = population_log_prob
        self.selection = selection
        self.minimum_event_effective_sample_size = minimum
        self.include_event_evidence = bool(include_event_evidence)
        self.label = _identifier(label, "population posterior label")

    def _event_components(self, hyperparameters: PyTree[Any], /) -> tuple[Array, Array]:
        factors = []
        importance_ess = []
        for event_index in range(self.batch.num_events):
            samples = jax.tree_util.tree_map(
                lambda value, event_index=event_index: value[event_index],
                self.batch.parameters,
            )
            population = jax.vmap(
                lambda sample: self.population_log_prob(hyperparameters, sample)
            )(samples)
            if population.shape != (self.batch.capacity,):
                raise ValueError(
                    "Population log density must return one scalar per event sample."
                )
            log_importance = population - self.batch.sampling_log_prior[event_index]
            candidates = jnp.where(
                self.batch.mask[event_index],
                self.batch.log_weights[event_index] + log_importance,
                -jnp.inf,
            )
            normalized, log_factor, valid = normalize_log_weights(candidates)
            factors.append(jnp.where(valid, log_factor, -jnp.inf))
            importance_ess.append(effective_sample_size(normalized))
        return jnp.stack(tuple(factors)), jnp.stack(tuple(importance_ess))

    def diagnostics(
        self, hyperparameters: PyTree[Any], /
    ) -> PopulationLikelihoodDiagnostics:
        factors, importance_ess = self._event_components(hyperparameters)
        selection = (
            None
            if self.selection is None
            else estimate_selection_efficiency(
                self.selection, hyperparameters, self.population_log_prob
            )
        )
        source_valid = jnp.all(
            self.batch.source_effective_sample_size
            >= self.minimum_event_effective_sample_size
        )
        importance_valid = jnp.all(
            importance_ess >= self.minimum_event_effective_sample_size
        )
        selection_valid = jnp.asarray(True) if selection is None else selection.valid
        valid = (
            source_valid
            & importance_valid
            & selection_valid
            & jnp.all(jnp.isfinite(factors))
        )
        return PopulationLikelihoodDiagnostics(
            factors,
            importance_ess,
            selection,
            valid,
            self.batch.batch_id,
        )

    def per_case_log_prob(self, hyperparameters: PyTree[Any], /) -> Array:
        diagnostics = self.diagnostics(hyperparameters)
        values = diagnostics.event_log_factors
        if self.include_event_evidence:
            values = values + self.batch.log_evidence
        if diagnostics.selection is not None:
            values = values - diagnostics.selection.log_efficiency
        return jnp.where(diagnostics.valid, values, -jnp.inf)


class PoissonPopulationPosteriorTerm(AbstractPosteriorTerm):
    """Rate-aware detected Poisson-process population likelihood."""

    conditional: PopulationPosteriorTerm
    rate: Callable[[PyTree[Any]], Array] = eqx.field(static=True)
    selection: SelectionInjectionSet

    def __init__(
        self,
        batch: PopulationSampleBatch,
        population_log_prob: Callable[[PyTree[Any], PyTree[Any]], Array],
        rate: Callable[[PyTree[Any]], Array],
        selection: SelectionInjectionSet,
        /,
        *,
        minimum_event_effective_sample_size: float = 20.0,
        include_event_evidence: bool = False,
        label: str = "poisson_population_process",
    ):
        if not callable(rate) or not isinstance(selection, SelectionInjectionSet):
            raise TypeError(
                "Poisson population term requires rate and selection contracts."
            )
        if selection.parameterization_id != batch.parameterization_id:
            raise ValueError("Selection and event parameterizations must agree.")
        self.conditional = PopulationPosteriorTerm(
            batch,
            population_log_prob,
            selection=None,
            minimum_event_effective_sample_size=minimum_event_effective_sample_size,
            include_event_evidence=include_event_evidence,
        )
        self.rate = rate
        self.selection = selection
        self.label = _identifier(label, "Poisson population label")

    def per_case_log_prob(self, hyperparameters: PyTree[Any], /) -> Array:
        values = self.conditional.per_case_log_prob(hyperparameters)
        rate = jnp.asarray(self.rate(hyperparameters), dtype=float).reshape(())
        selection = estimate_selection_efficiency(
            self.selection,
            hyperparameters,
            self.conditional.population_log_prob,
        )
        valid = jnp.isfinite(rate) & (rate > 0.0) & selection.valid
        expected = rate * selection.efficiency
        value = (
            values
            + jnp.log(jnp.where(valid, rate, 1.0))
            - expected / (self.conditional.batch.num_events)
        )
        return jnp.where(valid, value, -jnp.inf)


__all__ = [
    "EventPosterior",
    "EvidenceKind",
    "PoissonPopulationPosteriorTerm",
    "PopulationLikelihoodDiagnostics",
    "PopulationPosteriorTerm",
    "PopulationSampleBatch",
    "SelectionEfficiencyEstimate",
    "SelectionInjectionSet",
    "estimate_selection_efficiency",
    "prepare_population_sample_batch",
]
