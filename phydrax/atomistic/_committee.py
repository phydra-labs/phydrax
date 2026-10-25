#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Iterable, Sequence
from enum import StrEnum
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..typing import checked
from ._cell_dynamics import bind_atomistic_bundle_geometry
from ._frame import AtomisticFrame
from ._neighborhood_bundle import (
    AtomisticNeighborhoodBundleState,
    PreparedAtomisticNeighborhoodBundle,
)
from ._potential_program import PreparedAtomisticPotentialProgram


class OODPolicy(StrEnum):
    DIAGNOSE = "diagnose"
    REJECT = "reject"
    TERMINATE_SEGMENT = "terminate-segment"
    CONSERVATIVE_BLEND = "conservative-blend"
    NEXT_SEGMENT_FALLBACK = "next-segment-fallback"


class CommitteeReductionPolicy(StrictModule, NonTrainableState):
    energy_threshold: float = eqx.field(static=True)
    force_threshold: float = eqx.field(static=True)
    atom_threshold: float = eqx.field(static=True)
    policy: OODPolicy = eqx.field(static=True)
    transition_width: float = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        energy_threshold: float,
        force_threshold: float,
        atom_threshold: float,
        /,
        *,
        policy: OODPolicy = OODPolicy.DIAGNOSE,
        transition_width: float = 0.1,
    ) -> None:
        values = tuple(
            float(value)
            for value in (
                energy_threshold,
                force_threshold,
                atom_threshold,
                transition_width,
            )
        )
        if any(not np.isfinite(value) or value <= 0.0 for value in values):
            raise ValueError(
                "Committee thresholds and transition width must be positive."
            )
        (
            self.energy_threshold,
            self.force_threshold,
            self.atom_threshold,
            self.transition_width,
        ) = values
        self.policy = policy
        self.policy_id = canonical_fingerprint(
            {
                "kind": "committee-policy",
                "thresholds": list(values[:3]),
                "policy": policy.value,
                "transition_width": values[3],
            }
        )


class AtomisticUncertaintyEvidence(StrictModule):
    energy_standard_deviation: Array
    force_standard_deviation: Array
    atom_standard_deviation: Array
    maximum_force_standard_deviation: Array
    maximum_atom_standard_deviation: Array
    out_of_domain: Array
    successful: Array
    committee_id: str = eqx.field(static=True)


class CommitteeEvaluation(StrictModule):
    energy: Array
    forces: Array
    atom_energy: Array
    member_energies: Array
    member_forces: Array
    uncertainty: AtomisticUncertaintyEvidence
    successful: Array
    committee_id: str = eqx.field(static=True)


class CommitteeAtomisticPotential(StrictModule):
    """Committee of prepared potential programs over independently valid resources.

    Members of one prepared system may declare different scientific
    neighborhoods.  ``bundles`` are the members' distinct prepared resource
    bundles in first-appearance order and ``member_bundles`` maps each member
    to its bundle; members with the same prepared bundle share one accepted
    state.  ``evaluate`` consumes one bundle state per entry of ``bundles``,
    each certified at the evaluated geometry.
    """

    members: tuple[PreparedAtomisticPotentialProgram, ...]
    policy: CommitteeReductionPolicy
    bundles: tuple[PreparedAtomisticNeighborhoodBundle, ...]
    member_bundles: tuple[int, ...] = eqx.field(static=True)
    committee_id: str = eqx.field(static=True)

    def __init__(
        self,
        members: Iterable[PreparedAtomisticPotentialProgram],
        policy: CommitteeReductionPolicy,
        /,
    ) -> None:
        values = tuple(members)
        if len(values) < 2 or any(
            not isinstance(value, PreparedAtomisticPotentialProgram) for value in values
        ):
            raise TypeError(
                "Committee requires at least two prepared potential programs."
            )
        system_id = values[0].system.prepared_id
        if any(value.system.prepared_id != system_id for value in values[1:]):
            raise ValueError("Committee members belong to different atomistic systems.")
        bundles: list[PreparedAtomisticNeighborhoodBundle] = []
        identifiers: list[str] = []
        owners: list[int] = []
        for value in values:
            bundle = value.neighborhood_bundle
            if bundle.prepared_id not in identifiers:
                identifiers.append(bundle.prepared_id)
                bundles.append(bundle)
            owners.append(identifiers.index(bundle.prepared_id))
        self.members = values
        self.policy = policy
        self.bundles = tuple(bundles)
        self.member_bundles = tuple(owners)
        self.committee_id = canonical_fingerprint(
            {
                "kind": "atomistic-committee",
                "members": [value.prepared_id for value in values],
                "policy": policy.policy_id,
            }
        )

    def with_neighborhood_bundles(
        self, bundles: Sequence[PreparedAtomisticNeighborhoodBundle], /
    ) -> CommitteeAtomisticPotential:
        """Rebind grown resources; member numerical preparations are retained."""
        grown = tuple(bundles)
        if len(grown) != len(self.bundles) or any(
            new.plan.plan_id != old.plan.plan_id
            for new, old in zip(grown, self.bundles, strict=True)
        ):
            raise ValueError("Grown bundles must align with the committee bundles.")
        return CommitteeAtomisticPotential(
            tuple(
                member
                if grown[owner].prepared_id == member.neighborhood_bundle.prepared_id
                else member.with_neighborhood_bundle(grown[owner])
                for member, owner in zip(self.members, self.member_bundles, strict=True)
            ),
            self.policy,
        )

    def resources(
        self,
        positions: ArrayLike,
        previous: Sequence[AtomisticNeighborhoodBundleState] | None = None,
        /,
        *,
        cell_vectors: ArrayLike | None = None,
        image_counts: ArrayLike | None = None,
    ) -> CommitteeResources:
        """Host lifecycle: certified states of every bundle at one geometry.

        ``positions``, ``cell_vectors`` and ``image_counts`` are exactly the
        values later passed to ``evaluate``.  Each bundle refreshes its
        ``previous`` accepted state (or initializes) and replays capacity-only
        failures on its declared grown capacities; the returned committee is
        rebound to the grown resources.  An exhausted ladder is reported.
        """
        accepted = None if previous is None else tuple(previous)
        if accepted is not None and len(accepted) != len(self.bundles):
            raise ValueError("previous must hold one state per committee bundle.")
        system = self.members[0].system
        bundles = list(self.bundles)
        states: list[AtomisticNeighborhoodBundleState] = []
        growths = 0
        exhausted = False
        for index, bundle in enumerate(self.bundles):
            geometry = bind_atomistic_bundle_geometry(
                bundle,
                system,
                positions,
                cell_vectors=cell_vectors,
                image_counts=image_counts,
            )
            prior = None if accepted is None else accepted[index]

            def attempt(
                prepared: PreparedAtomisticNeighborhoodBundle,
                prior: AtomisticNeighborhoodBundleState | None = prior,
                geometry: Any = geometry,
            ) -> AtomisticNeighborhoodBundleState:
                if prior is None:
                    return prepared.initialize(geometry)
                return prepared.refresh(prepared.rebase(prior), geometry)

            state = attempt(bundle)
            while bool(np.asarray(state.capacity_only)[0]):
                try:
                    bundle = bundle.grow_capacities(state)
                except RuntimeError:
                    exhausted = True
                    break
                state = attempt(bundle)
                growths += 1
            bundles[index] = bundle
            states.append(state)
        committee = (
            self
            if all(
                new.prepared_id == old.prepared_id
                for new, old in zip(bundles, self.bundles, strict=True)
            )
            else self.with_neighborhood_bundles(bundles)
        )
        return CommitteeResources(committee, tuple(states), growths, exhausted)

    def _member_states(
        self, neighborhoods: Sequence[AtomisticNeighborhoodBundleState], /
    ) -> tuple[AtomisticNeighborhoodBundleState, ...]:
        states = tuple(neighborhoods)
        if len(states) != len(self.bundles):
            raise ValueError("neighborhoods must hold one state per committee bundle.")
        return tuple(states[owner] for owner in self.member_bundles)

    def evaluate(
        self,
        positions: ArrayLike,
        neighborhoods: Sequence[AtomisticNeighborhoodBundleState],
        /,
        **kwargs: Any,
    ) -> CommitteeEvaluation:
        evaluations = tuple(
            member.evaluate(positions, state, **kwargs)
            for member, state in zip(
                self.members, self._member_states(neighborhoods), strict=True
            )
        )
        energies = jnp.stack(tuple(value.energy for value in evaluations))
        forces = jnp.stack(tuple(value.forces for value in evaluations))
        atoms = jnp.stack(tuple(value.atom_energy for value in evaluations))
        mean_energy = jnp.mean(energies)
        mean_force = jnp.mean(forces, axis=0)
        mean_atom = jnp.mean(atoms, axis=0)
        energy_std = jnp.std(energies)
        force_std = jnp.std(forces, axis=0)
        atom_std = jnp.std(atoms, axis=0)
        maximum_force = jnp.max(jnp.sqrt(jnp.sum(force_std**2, axis=-1)))
        maximum_atom = jnp.max(atom_std)
        out = (
            (energy_std > self.policy.energy_threshold)
            | (maximum_force > self.policy.force_threshold)
            | (maximum_atom > self.policy.atom_threshold)
        )
        successful = jnp.all(
            jnp.stack(tuple(value.successful for value in evaluations))
        ) & jnp.all(jnp.isfinite(forces))
        uncertainty = AtomisticUncertaintyEvidence(
            energy_std,
            force_std,
            atom_std,
            maximum_force,
            maximum_atom,
            out,
            successful,
            self.committee_id,
        )
        accepted = successful & jnp.where(
            self.policy.policy is OODPolicy.REJECT, ~out, True
        )
        return CommitteeEvaluation(
            mean_energy,
            mean_force,
            mean_atom,
            energies,
            forces,
            uncertainty,
            accepted,
            self.committee_id,
        )


class CommitteeResources(StrictModule):
    """Committee rebound to its final resource levels and their certified states."""

    committee: CommitteeAtomisticPotential
    neighborhoods: tuple[AtomisticNeighborhoodBundleState, ...]
    capacity_growths: int = eqx.field(static=True)
    exhausted: bool = eqx.field(static=True)


class ConservativeUncertaintyBlend(StrictModule):
    committee: CommitteeAtomisticPotential
    baseline: PreparedAtomisticPotentialProgram
    blend_id: str = eqx.field(static=True)

    def __init__(
        self,
        committee: CommitteeAtomisticPotential,
        baseline: PreparedAtomisticPotentialProgram,
        /,
    ) -> None:
        if committee.members[0].system.prepared_id != baseline.system.prepared_id:
            raise ValueError("Committee and baseline belong to different systems.")
        self.committee = committee
        self.baseline = baseline
        self.blend_id = canonical_fingerprint(
            {
                "kind": "conservative-uncertainty-blend",
                "committee": committee.committee_id,
                "baseline": baseline.prepared_id,
            }
        )

    def evaluate(
        self,
        positions: ArrayLike,
        neighborhoods: Sequence[AtomisticNeighborhoodBundleState],
        baseline_neighborhoods: AtomisticNeighborhoodBundleState,
        /,
        **kwargs: Any,
    ) -> tuple[Array, Array, AtomisticUncertaintyEvidence, Array]:
        """One conservative blended energy and its force at a certified point.

        ``neighborhoods`` holds one state per committee bundle and
        ``baseline_neighborhoods`` the baseline program's own state, all
        certified at ``positions``.
        """
        position = jnp.asarray(positions)
        states = self.committee._member_states(neighborhoods)

        def energy(value: Array) -> tuple[Array, Array]:
            member_energies = jnp.stack(
                tuple(
                    member.energy(value, state, **kwargs)[0]
                    for member, state in zip(self.committee.members, states, strict=True)
                )
            )
            mean_energy = jnp.mean(member_energies)
            centered = member_energies - mean_energy
            epsilon = jnp.asarray(
                jnp.finfo(member_energies.dtype).eps, dtype=member_energies.dtype
            )
            energy_deviation = (
                jnp.sqrt(jnp.mean(centered * centered) + epsilon**2) - epsilon
            )
            baseline = self.baseline.energy(value, baseline_neighborhoods, **kwargs)[0]
            threshold = self.committee.policy.energy_threshold
            width = self.committee.policy.transition_width
            weight = jax.nn.sigmoid((threshold - energy_deviation) / width)
            return weight * mean_energy + (1.0 - weight) * baseline, weight

        (energy_value, weight), gradient = jax.value_and_grad(energy, has_aux=True)(
            position
        )
        uncertainty = self.committee.evaluate(
            position, neighborhoods, **kwargs
        ).uncertainty
        return energy_value, -gradient, uncertainty, weight


class SegmentFallbackDecision(StrictModule, NonTrainableState):
    use_fallback: bool = eqx.field(static=True)
    reason: str = eqx.field(static=True)
    committee_id: str = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)


class SegmentFallbackPolicy(StrictModule, NonTrainableState):
    fallback_provider_id: str = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(self, fallback_provider_id: str, /) -> None:
        identifier = str(fallback_provider_id).strip()
        if not identifier:
            raise ValueError("fallback_provider_id must be non-empty.")
        self.fallback_provider_id = identifier
        self.policy_id = canonical_fingerprint(
            {
                "kind": "atomistic-segment-fallback",
                "fallback_provider": identifier,
            }
        )

    def decide(
        self, uncertainty: AtomisticUncertaintyEvidence, /
    ) -> SegmentFallbackDecision:
        use_fallback = bool(uncertainty.out_of_domain | ~uncertainty.successful)
        reason = (
            "committee-evaluation-failed"
            if not bool(uncertainty.successful)
            else "out-of-domain"
            if bool(uncertainty.out_of_domain)
            else "primary"
        )
        return SegmentFallbackDecision(
            use_fallback,
            reason,
            uncertainty.committee_id,
            self.policy_id,
        )

    def choose_baseline(self, uncertainty: AtomisticUncertaintyEvidence, /) -> bool:
        return self.decide(uncertainty).use_fallback


class AcquisitionAggregation(StrEnum):
    MAXIMUM = "maximum"
    EUCLIDEAN = "euclidean"


class CommitteeAcquisitionScorePolicy(StrictModule, NonTrainableState):
    energy_scale: float = eqx.field(static=True)
    force_scale: float = eqx.field(static=True)
    atom_scale: float = eqx.field(static=True)
    aggregation: AcquisitionAggregation = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        energy_scale: float,
        force_scale: float,
        atom_scale: float,
        /,
        *,
        aggregation: AcquisitionAggregation = AcquisitionAggregation.MAXIMUM,
    ) -> None:
        scales = float(energy_scale), float(force_scale), float(atom_scale)
        if any(not np.isfinite(value) or value <= 0.0 for value in scales):
            raise ValueError(
                "Acquisition uncertainty scales must be finite and positive."
            )
        if not isinstance(aggregation, AcquisitionAggregation):
            raise TypeError("aggregation must be AcquisitionAggregation.")
        self.energy_scale, self.force_scale, self.atom_scale = scales
        self.aggregation = aggregation
        self.policy_id = canonical_fingerprint(
            {
                "kind": "committee-acquisition-score",
                "scales": [value.hex() for value in scales],
                "aggregation": aggregation.value,
            }
        )

    @checked
    def score(self, evidence: AtomisticUncertaintyEvidence, /) -> tuple[Array, Array]:
        components = jnp.asarray(
            [
                evidence.energy_standard_deviation / self.energy_scale,
                evidence.maximum_force_standard_deviation / self.force_scale,
                evidence.maximum_atom_standard_deviation / self.atom_scale,
            ]
        )
        score = (
            jnp.max(components)
            if self.aggregation is AcquisitionAggregation.MAXIMUM
            else jnp.sqrt(jnp.sum(components * components))
        )
        return score, components


class AcquisitionPlan(StrictModule, NonTrainableState):
    maximum_frames: int = eqx.field(static=True)
    minimum_score: float = eqx.field(static=True)
    scoring: CommitteeAcquisitionScorePolicy
    plan_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        maximum_frames: int,
        scoring: CommitteeAcquisitionScorePolicy,
        /,
        *,
        minimum_score: float = 0.0,
    ) -> None:
        if int(maximum_frames) <= 0 or float(minimum_score) < 0.0:
            raise ValueError("Acquisition capacity or minimum score is invalid.")
        self.maximum_frames = int(maximum_frames)
        self.minimum_score = float(minimum_score)
        self.scoring = scoring
        self.plan_id = canonical_fingerprint(
            {
                "kind": "atomistic-acquisition",
                "maximum_frames": self.maximum_frames,
                "minimum_score": self.minimum_score,
                "scoring": scoring.policy_id,
            }
        )

    def select(
        self,
        frames: Iterable[AtomisticFrame],
        uncertainty: Iterable[AtomisticUncertaintyEvidence],
        /,
        *,
        descriptors: ArrayLike | None = None,
    ) -> tuple[AcquisitionRecord, ...]:
        frame_values = tuple(frames)
        evidence = tuple(uncertainty)
        if len(frame_values) != len(evidence):
            raise ValueError("Acquisition frames and evidence must align.")
        scored = tuple(self.scoring.score(item) for item in evidence)
        scores = np.asarray([float(value[0]) for value in scored])
        component_scores = tuple(value[1] for value in scored)
        descriptor = (
            np.stack(
                tuple(
                    np.asarray(frame.positions).reshape((-1,)) for frame in frame_values
                )
            )
            if descriptors is None
            else np.asarray(descriptors, dtype=np.float64)
        )
        if descriptor.ndim != 2 or descriptor.shape[0] != len(frame_values):
            raise ValueError("Acquisition descriptors must have one vector per frame.")
        eligible = np.flatnonzero(scores >= self.minimum_score)
        if eligible.size == 0:
            return ()
        scale = np.std(descriptor[eligible], axis=0)
        normalized = descriptor / np.where(scale > 0.0, scale, 1.0)
        first = eligible[np.lexsort((eligible, -scores[eligible]))[0]]
        selected_indices = [int(first)]
        while len(selected_indices) < min(self.maximum_frames, eligible.size):
            remaining = np.asarray(
                [index for index in eligible if index not in selected_indices],
                dtype=np.int64,
            )
            displacement = (
                normalized[remaining, None, :]
                - normalized[np.asarray(selected_indices), :]
            )
            distance_squared = np.sum(displacement * displacement, axis=-1)
            minimum_distance = np.min(distance_squared, axis=1)
            choice = np.lexsort((remaining, -scores[remaining], -minimum_distance))[0]
            selected_indices.append(int(remaining[choice]))
        return tuple(
            AcquisitionRecord(
                frame_values[index],
                jnp.asarray(descriptor[index]),
                component_scores[index],
                index,
                float(scores[index]),
                "committee-uncertainty-diversity",
                evidence[index].committee_id,
                self.plan_id,
            )
            for index in selected_indices
        )


class AcquisitionRecord(StrictModule, NonTrainableState):
    frame: AtomisticFrame
    descriptor: Array
    component_scores: Array
    source_index: int = eqx.field(static=True)
    score: float = eqx.field(static=True)
    reason: str = eqx.field(static=True)
    model_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)


__all__ = [
    "AcquisitionAggregation",
    "AcquisitionPlan",
    "AcquisitionRecord",
    "CommitteeAcquisitionScorePolicy",
    "AtomisticUncertaintyEvidence",
    "CommitteeAtomisticPotential",
    "CommitteeEvaluation",
    "CommitteeReductionPolicy",
    "CommitteeResources",
    "ConservativeUncertaintyBlend",
    "OODPolicy",
    "SegmentFallbackPolicy",
    "SegmentFallbackDecision",
]
