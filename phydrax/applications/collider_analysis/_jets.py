#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from enum import StrEnum

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from phydrax import ein

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...particle_physics import HEPProviderBinding


class JetAlgorithm(StrEnum):
    KT = "kt"
    CAMBRIDGE_AACHEN = "cambridge-aachen"
    ANTI_KT = "anti-kt"

    @property
    def distance_power(self) -> float:
        if self is JetAlgorithm.KT:
            return 1.0
        if self is JetAlgorithm.CAMBRIDGE_AACHEN:
            return 0.0
        return -1.0


class JetRecombinationScheme(StrEnum):
    E_SCHEME = "E-scheme"


class JetDefinition(StrictModule, NonTrainableState):
    algorithm: JetAlgorithm = eqx.field(static=True)
    radius: float = eqx.field(static=True)
    recombination_scheme: JetRecombinationScheme = eqx.field(static=True)
    minimum_transverse_momentum: float = eqx.field(static=True)
    definition_id: str = eqx.field(static=True)

    def __init__(
        self,
        algorithm: JetAlgorithm,
        radius: float,
        /,
        *,
        recombination_scheme: JetRecombinationScheme = JetRecombinationScheme.E_SCHEME,
        minimum_transverse_momentum: float = 0.0,
    ):
        radius_ = float(radius)
        minimum = float(minimum_transverse_momentum)
        if not isinstance(algorithm, JetAlgorithm) or not isinstance(
            recombination_scheme, JetRecombinationScheme
        ):
            raise TypeError(
                "Jet algorithm and recombination scheme must be explicit enums."
            )
        if (
            not math.isfinite(radius_)
            or radius_ <= 0.0
            or not math.isfinite(minimum)
            or minimum < 0.0
        ):
            raise ValueError("Jet radius and transverse-momentum threshold are invalid.")
        self.algorithm = algorithm
        self.radius = radius_
        self.recombination_scheme = recombination_scheme
        self.minimum_transverse_momentum = minimum
        self.definition_id = canonical_fingerprint(
            {
                "kind": "jet-definition",
                "algorithm": algorithm.value,
                "radius": radius_,
                "recombination": recombination_scheme.value,
                "minimum_pt": minimum,
            }
        )


class JetInputBatch(StrictModule, NonTrainableState):
    event_ids: Array
    momenta: Array
    active: Array
    valid: Array
    source_collection_id: str = eqx.field(static=True)
    momentum_unit_id: str = eqx.field(static=True)
    constituent_capacity: int = eqx.field(static=True)

    def __init__(
        self,
        event_ids: ArrayLike,
        momenta: ArrayLike,
        active: ArrayLike,
        /,
        *,
        source_collection_id: str,
        momentum_unit_id: str,
    ):
        event_ids_ = jnp.asarray(event_ids)
        momenta_ = jnp.asarray(momenta)
        active_ = jnp.asarray(active, dtype=jnp.bool_)
        if (
            event_ids_.ndim != 1
            or momenta_.ndim != 3
            or momenta_.shape[-1] != 4
            or active_.shape != momenta_.shape[:2]
            or event_ids_.shape != (momenta_.shape[0],)
        ):
            raise ValueError(
                "Jet inputs require event IDs and (event, constituent, E/px/py/pz) arrays."
            )
        source = str(source_collection_id).strip()
        unit = str(momentum_unit_id).strip()
        if not source or not unit:
            raise ValueError("Jet source collection and momentum unit are required.")
        spatial_norm = jnp.linalg.norm(momenta_[..., 1:], axis=-1)
        valid = jnp.all(jnp.isfinite(momenta_), axis=-1) & (
            momenta_[..., 0] >= spatial_norm
        )
        self.event_ids = event_ids_
        self.momenta = momenta_
        self.active = active_
        self.valid = jnp.where(active_, valid, True)
        self.source_collection_id = source
        self.momentum_unit_id = unit
        self.constituent_capacity = momenta_.shape[1]


class JetCollection(StrictModule, NonTrainableState):
    event_ids: Array
    momenta: Array
    constituent_weights: Array
    active: Array
    valid: Array
    derivative_valid: Array
    definition_id: str = eqx.field(static=True)
    source_collection_id: str = eqx.field(static=True)
    provider_id: str = eqx.field(static=True)


class JetProviderPlan(StrictModule, NonTrainableState):
    definition: JetDefinition
    provider: HEPProviderBinding
    plan_id: str = eqx.field(static=True)

    def __init__(self, definition: JetDefinition, provider: HEPProviderBinding, /):
        if not isinstance(definition, JetDefinition) or not isinstance(
            provider, HEPProviderBinding
        ):
            raise TypeError(
                "definition and provider must use jet/HEP provider contracts."
            )
        if not provider.supports("hep.jet-clustering"):
            raise ValueError("Provider profile does not support hep.jet-clustering.")
        self.definition = definition
        self.provider = provider
        self.plan_id = canonical_fingerprint(
            {
                "kind": "jet-provider-plan",
                "definition": definition.definition_id,
                "provider": provider.binding_id,
            }
        )


def _rapidity_phi(momentum):
    energy = momentum[..., 0]
    px = momentum[..., 1]
    py = momentum[..., 2]
    pz = momentum[..., 3]
    tiny = jnp.finfo(momentum.dtype).tiny
    rapidity = 0.5 * jnp.log(
        jnp.maximum(energy + pz, tiny) / jnp.maximum(energy - pz, tiny)
    )
    phi = jnp.arctan2(py, px)
    transverse_momentum = jnp.sqrt(px * px + py * py)
    return rapidity, phi, transverse_momentum


def _delta_phi(first, second):
    return jnp.arctan2(jnp.sin(first - second), jnp.cos(first - second))


def _cluster_one(definition: JetDefinition, momenta, active):
    capacity = momenta.shape[0]
    constituents = jnp.eye(capacity, dtype=momenta.dtype)
    jets = jnp.zeros_like(momenta)
    jet_constituents = jnp.zeros((capacity, capacity), dtype=momenta.dtype)

    def iteration(state, _):
        (
            work,
            work_constituents,
            work_active,
            output,
            output_constituents,
            output_count,
        ) = state
        rapidity, phi, transverse_momentum = _rapidity_phi(work)
        safe_pt = jnp.maximum(transverse_momentum, jnp.finfo(work.dtype).tiny)
        scale = safe_pt ** (2.0 * definition.algorithm.distance_power)
        delta_y = rapidity[:, None] - rapidity[None, :]
        delta_phi = _delta_phi(phi[:, None], phi[None, :])
        pair_distance = (
            jnp.minimum(scale[:, None], scale[None, :])
            * (delta_y * delta_y + delta_phi * delta_phi)
            / definition.radius**2
        )
        indices = jnp.arange(capacity)
        upper = indices[:, None] < indices[None, :]
        pair_distance = jnp.where(
            upper & work_active[:, None] & work_active[None, :], pair_distance, jnp.inf
        )
        flat_pair = jnp.argmin(pair_distance)
        first = flat_pair // capacity
        second = flat_pair % capacity
        minimum_pair = pair_distance[first, second]
        beam_distance = jnp.where(work_active, scale, jnp.inf)
        beam_index = jnp.argmin(beam_distance)
        minimum_beam = beam_distance[beam_index]
        merge = minimum_pair < minimum_beam
        has_active = jnp.any(work_active)
        merged_momentum = work[first] + work[second]
        merged_constituents = work_constituents[first] + work_constituents[second]
        work_after_merge = work.at[first].set(merged_momentum)
        constituents_after_merge = work_constituents.at[first].set(merged_constituents)
        active_after_merge = work_active.at[second].set(False)
        output_after_beam = output.at[output_count].set(work[beam_index])
        output_constituents_after_beam = output_constituents.at[output_count].set(
            work_constituents[beam_index]
        )
        active_after_beam = work_active.at[beam_index].set(False)
        next_work = jnp.where(merge & has_active, work_after_merge, work)
        next_constituents = jnp.where(
            merge & has_active, constituents_after_merge, work_constituents
        )
        next_active = jnp.where(merge & has_active, active_after_merge, active_after_beam)
        next_output = jnp.where((~merge & has_active), output_after_beam, output)
        next_output_constituents = jnp.where(
            (~merge & has_active),
            output_constituents_after_beam,
            output_constituents,
        )
        next_count = output_count + ((~merge) & has_active).astype(jnp.int32)
        return (
            next_work,
            next_constituents,
            next_active,
            next_output,
            next_output_constituents,
            next_count,
        ), None

    initial = (
        momenta,
        constituents,
        active,
        jets,
        jet_constituents,
        jnp.asarray(0, dtype=jnp.int32),
    )
    final, _ = jax.lax.scan(iteration, initial, xs=None, length=capacity)
    _, _, _, jets, jet_constituents, count = final
    _, _, transverse_momentum = _rapidity_phi(jets)
    order = jnp.argsort(transverse_momentum)[::-1]
    jets = jets[order]
    jet_constituents = jet_constituents[order]
    sorted_pt = transverse_momentum[order]
    active_jets = (jnp.arange(capacity) < count) & (
        sorted_pt >= definition.minimum_transverse_momentum
    )
    return jets, jet_constituents, active_jets


def cluster_sequential_jets(
    definition: JetDefinition,
    inputs: JetInputBatch,
    /,
) -> JetCollection:
    """Bounded native reference for inclusive longitudinally invariant clustering."""
    if not isinstance(definition, JetDefinition) or not isinstance(inputs, JetInputBatch):
        raise TypeError("definition and inputs must use jet contracts.")
    jets, constituent_weights, active = jax.vmap(
        lambda momenta, event_active: _cluster_one(definition, momenta, event_active)
    )(inputs.momenta, inputs.active & inputs.valid)
    spatial_norm = jnp.linalg.norm(jets[..., 1:], axis=-1)
    valid = (
        jnp.all(jnp.isfinite(jets), axis=-1)
        & (jets[..., 0] >= spatial_norm)
        & jnp.isclose(
            jnp.sum(constituent_weights, axis=1), inputs.active.astype(jets.dtype)
        ).all(axis=-1)[:, None]
    )
    return JetCollection(
        inputs.event_ids,
        jets,
        constituent_weights,
        active,
        jnp.where(active, valid, True),
        jnp.zeros_like(active),
        definition.definition_id,
        inputs.source_collection_id,
        "phydrax-native-sequential-reference",
    )


class FuzzyJetPlan(StrictModule, NonTrainableState):
    component_count: int = eqx.field(static=True)
    radius_scale: float = eqx.field(static=True)
    iteration_count: int = eqx.field(static=True)
    pileup_density: float = eqx.field(static=True)
    minimum_component_weight: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        component_count: int,
        radius_scale: float,
        /,
        *,
        iteration_count: int = 32,
        pileup_density: float = 1.0e-6,
        minimum_component_weight: float = 1.0e-6,
    ):
        count = int(component_count)
        radius = float(radius_scale)
        iterations = int(iteration_count)
        pileup = float(pileup_density)
        minimum = float(minimum_component_weight)
        if (
            count < 1
            or not math.isfinite(radius)
            or radius <= 0.0
            or iterations < 1
            or not math.isfinite(pileup)
            or pileup <= 0.0
            or not math.isfinite(minimum)
            or minimum <= 0.0
        ):
            raise ValueError(
                "Fuzzy-jet component, scale, iteration, pileup, and weight policy are invalid."
            )
        self.component_count = count
        self.radius_scale = radius
        self.iteration_count = iterations
        self.pileup_density = pileup
        self.minimum_component_weight = minimum
        self.plan_id = canonical_fingerprint(
            {
                "kind": "fuzzy-jet-plan",
                "components": count,
                "radius_scale": radius,
                "iterations": iterations,
                "pileup_density": pileup,
                "minimum_weight": minimum,
            }
        )


class FuzzyJetResult(StrictModule, NonTrainableState):
    jets: JetCollection
    responsibilities: Array
    component_means: Array
    component_weights: Array
    pileup_responsibility: Array
    log_likelihood: Array
    convergence_residual: Array
    local_optimum_valid: Array
    plan_id: str = eqx.field(static=True)


def _fuzzy_one(plan: FuzzyJetPlan, momenta, active):
    rapidity, phi, transverse_momentum = _rapidity_phi(momenta)
    capacity = momenta.shape[0]
    order = jnp.argsort(jnp.where(active, transverse_momentum, -jnp.inf))[::-1]
    seeds = order[: plan.component_count]
    means = jnp.stack((rapidity[seeds], phi[seeds]), axis=-1)
    weights = jnp.full(
        (plan.component_count,), 1.0 / plan.component_count, dtype=momenta.dtype
    )
    responsibilities = jnp.zeros((capacity, plan.component_count), dtype=momenta.dtype)
    likelihood = jnp.asarray(-jnp.inf, dtype=momenta.dtype)

    def iteration(state, _):
        means_, weights_, _previous_responsibilities, _previous_likelihood = state
        delta_y = rapidity[:, None] - means_[None, :, 0]
        delta_phi = _delta_phi(phi[:, None], means_[None, :, 1])
        log_components = (
            jnp.log(jnp.maximum(weights_, jnp.finfo(momenta.dtype).tiny))[None, :]
            - 0.5 * (delta_y * delta_y + delta_phi * delta_phi) / plan.radius_scale**2
        )
        pileup_log = jnp.full(
            (capacity, 1), jnp.log(plan.pileup_density), dtype=momenta.dtype
        )
        log_all = jnp.concatenate((log_components, pileup_log), axis=1)
        normalizer = jax.scipy.special.logsumexp(log_all, axis=1, keepdims=True)
        all_responsibilities = jnp.exp(log_all - normalizer) * active[:, None]
        responsibilities_ = all_responsibilities[:, : plan.component_count]
        weighted = transverse_momentum[:, None] * responsibilities_
        totals = jnp.sum(weighted, axis=0)
        safe_totals = jnp.maximum(totals, jnp.finfo(momenta.dtype).tiny)
        mean_y = jnp.sum(weighted * rapidity[:, None], axis=0) / safe_totals
        mean_phi = jnp.arctan2(
            jnp.sum(weighted * jnp.sin(phi)[:, None], axis=0),
            jnp.sum(weighted * jnp.cos(phi)[:, None], axis=0),
        )
        means_next = jnp.stack((mean_y, mean_phi), axis=-1)
        weights_next = safe_totals / jnp.maximum(
            jnp.sum(safe_totals), jnp.finfo(momenta.dtype).tiny
        )
        likelihood_next = jnp.sum(
            jnp.where(active, transverse_momentum * normalizer[:, 0], 0.0)
        )
        return (means_next, weights_next, responsibilities_, likelihood_next), None

    (means, weights, responsibilities, likelihood), _ = jax.lax.scan(
        iteration,
        (means, weights, responsibilities, likelihood),
        xs=None,
        length=plan.iteration_count,
    )
    delta_y = rapidity[:, None] - means[None, :, 0]
    delta_phi = _delta_phi(phi[:, None], means[None, :, 1])
    log_components = (
        jnp.log(jnp.maximum(weights, jnp.finfo(momenta.dtype).tiny))[None, :]
        - 0.5 * (delta_y * delta_y + delta_phi * delta_phi) / plan.radius_scale**2
    )
    pileup_log = jnp.full(
        (capacity, 1), jnp.log(plan.pileup_density), dtype=momenta.dtype
    )
    log_all = jnp.concatenate((log_components, pileup_log), axis=1)
    all_responsibilities = (
        jnp.exp(log_all - jax.scipy.special.logsumexp(log_all, axis=1, keepdims=True))
        * active[:, None]
    )
    final_responsibilities = all_responsibilities[:, : plan.component_count]
    pileup = all_responsibilities[:, -1]
    jet_momenta = ein.contract("nk,nq->kq", final_responsibilities, momenta)
    component_active = weights >= plan.minimum_component_weight
    residual = jnp.max(jnp.abs(final_responsibilities - responsibilities))
    seed_pt = transverse_momentum[seeds]
    distinct_seed_margin = jnp.min(
        jnp.where(
            jnp.eye(plan.component_count, dtype=jnp.bool_),
            jnp.inf,
            jnp.abs(seed_pt[:, None] - seed_pt[None, :]),
        )
    )
    valid = jnp.all(jnp.isfinite(jet_momenta)) & jnp.isfinite(likelihood)
    local_optimum_valid = valid & (distinct_seed_margin > 0.0) & jnp.all(component_active)
    return (
        jet_momenta,
        final_responsibilities.T,
        component_active,
        means,
        weights,
        pileup,
        likelihood,
        residual,
        local_optimum_valid,
    )


def cluster_fuzzy_jets(plan: FuzzyJetPlan, inputs: JetInputBatch, /) -> FuzzyJetResult:
    """Soft probabilistic jet reference with an explicit uniform pileup component."""
    if not isinstance(plan, FuzzyJetPlan) or not isinstance(inputs, JetInputBatch):
        raise TypeError("plan and inputs must use fuzzy-jet contracts.")
    if plan.component_count > inputs.constituent_capacity:
        raise ValueError("Fuzzy component count exceeds constituent capacity.")
    results = jax.vmap(lambda momenta, active: _fuzzy_one(plan, momenta, active))(
        inputs.momenta, inputs.active & inputs.valid
    )
    (
        jet_momenta,
        responsibilities,
        component_active,
        means,
        weights,
        pileup,
        likelihood,
        residual,
        local_valid,
    ) = results
    spatial_norm = jnp.linalg.norm(jet_momenta[..., 1:], axis=-1)
    jet_valid = jnp.all(jnp.isfinite(jet_momenta), axis=-1) & (
        jet_momenta[..., 0] >= spatial_norm
    )
    jets = JetCollection(
        inputs.event_ids,
        jet_momenta,
        responsibilities,
        component_active,
        jnp.where(component_active, jet_valid, True),
        jnp.broadcast_to(local_valid[:, None], component_active.shape) & component_active,
        plan.plan_id,
        inputs.source_collection_id,
        "phydrax-native-fuzzy-reference",
    )
    return FuzzyJetResult(
        jets,
        responsibilities,
        means,
        weights,
        pileup,
        likelihood,
        residual,
        local_valid,
        plan.plan_id,
    )


class JetObservables(StrictModule, NonTrainableState):
    transverse_momentum: Array
    rapidity: Array
    phi: Array
    mass: Array
    constituent_multiplicity: Array
    girth: Array
    valid: Array
    definition_id: str = eqx.field(static=True)


def jet_observables(jets: JetCollection, inputs: JetInputBatch, /) -> JetObservables:
    if not isinstance(jets, JetCollection) or not isinstance(inputs, JetInputBatch):
        raise TypeError("jets and inputs must use jet contracts.")
    rapidity, phi, transverse_momentum = _rapidity_phi(jets.momenta)
    mass_squared = jets.momenta[..., 0] ** 2 - jnp.sum(
        jets.momenta[..., 1:] ** 2, axis=-1
    )
    mass = jnp.sqrt(jnp.maximum(mass_squared, 0.0))
    input_rapidity, input_phi, input_pt = _rapidity_phi(inputs.momenta)
    delta_y = input_rapidity[:, None, :] - rapidity[:, :, None]
    delta_phi = _delta_phi(input_phi[:, None, :], phi[:, :, None])
    radius = jnp.sqrt(delta_y * delta_y + delta_phi * delta_phi)
    weighted_pt = jets.constituent_weights * input_pt[:, None, :]
    girth = jnp.sum(weighted_pt * radius, axis=-1) / jnp.maximum(
        jnp.sum(weighted_pt, axis=-1), jnp.finfo(jets.momenta.dtype).tiny
    )
    multiplicity = jnp.sum(jets.constituent_weights > 0.0, axis=-1, dtype=jnp.int32)
    valid = jets.valid & jnp.isfinite(girth)
    return JetObservables(
        transverse_momentum,
        rapidity,
        phi,
        mass,
        multiplicity,
        girth,
        valid,
        jets.definition_id,
    )


__all__ = [
    "FuzzyJetPlan",
    "FuzzyJetResult",
    "JetAlgorithm",
    "JetCollection",
    "JetDefinition",
    "JetInputBatch",
    "JetObservables",
    "JetProviderPlan",
    "JetRecombinationScheme",
    "cluster_fuzzy_jets",
    "cluster_sequential_jets",
    "jet_observables",
]
