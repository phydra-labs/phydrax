#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from enum import StrEnum

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization import ParticleNeighborhoodState
from ._frame import AtomisticFrame
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
    ):
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
    members: tuple[PreparedAtomisticPotentialProgram, ...]
    policy: CommitteeReductionPolicy
    committee_id: str = eqx.field(static=True)

    def __init__(self, members, policy: CommitteeReductionPolicy, /):
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
        self.members = values
        self.policy = policy
        self.committee_id = canonical_fingerprint(
            {
                "kind": "atomistic-committee",
                "members": [value.prepared_id for value in values],
                "policy": policy.policy_id,
            }
        )

    def evaluate(
        self, positions: ArrayLike, neighborhood: ParticleNeighborhoodState, /, **kwargs
    ) -> CommitteeEvaluation:
        evaluations = tuple(
            member.evaluate(positions, neighborhood, **kwargs) for member in self.members
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


class ConservativeUncertaintyBlend(StrictModule):
    committee: CommitteeAtomisticPotential
    baseline: PreparedAtomisticPotentialProgram
    blend_id: str = eqx.field(static=True)

    def __init__(
        self,
        committee: CommitteeAtomisticPotential,
        baseline: PreparedAtomisticPotentialProgram,
        /,
    ):
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
        self, positions: ArrayLike, neighborhood: ParticleNeighborhoodState, /, **kwargs
    ):
        position = jnp.asarray(positions)

        def energy(value):
            member_energies = jnp.stack(
                tuple(
                    member.energy(value, neighborhood, **kwargs)[0]
                    for member in self.committee.members
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
            baseline = self.baseline.energy(value, neighborhood, **kwargs)[0]
            threshold = self.committee.policy.energy_threshold
            width = self.committee.policy.transition_width
            weight = jax.nn.sigmoid((threshold - energy_deviation) / width)
            return weight * mean_energy + (1.0 - weight) * baseline, weight

        (energy_value, weight), gradient = jax.value_and_grad(energy, has_aux=True)(
            position
        )
        uncertainty = self.committee.evaluate(
            position, neighborhood, **kwargs
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

    def __init__(self, fallback_provider_id: str, /):
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


class AcquisitionPlan(StrictModule, NonTrainableState):
    maximum_frames: int = eqx.field(static=True)
    minimum_score: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(self, maximum_frames: int, /, *, minimum_score: float = 0.0):
        if int(maximum_frames) <= 0 or float(minimum_score) < 0.0:
            raise ValueError("Acquisition capacity or minimum score is invalid.")
        self.maximum_frames = int(maximum_frames)
        self.minimum_score = float(minimum_score)
        self.plan_id = canonical_fingerprint(
            {
                "kind": "atomistic-acquisition",
                "maximum_frames": self.maximum_frames,
                "minimum_score": self.minimum_score,
            }
        )

    def select(self, frames, uncertainty, /, *, descriptors: ArrayLike | None = None):
        frame_values = tuple(frames)
        evidence = tuple(uncertainty)
        if len(frame_values) != len(evidence):
            raise ValueError("Acquisition frames and evidence must align.")
        scores = np.asarray(
            [
                float(
                    item.energy_standard_deviation
                    + item.maximum_force_standard_deviation
                    + item.maximum_atom_standard_deviation
                )
                for item in evidence
            ]
        )
        descriptor = (
            np.stack(
                tuple(
                    np.asarray(frame.positions).reshape((-1,)) for frame in frame_values
                )
            )
            if descriptors is None
            else np.asarray(descriptors, dtype=float)
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
    source_index: int = eqx.field(static=True)
    score: float = eqx.field(static=True)
    reason: str = eqx.field(static=True)
    model_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)


__all__ = [
    "AcquisitionPlan",
    "AcquisitionRecord",
    "AtomisticUncertaintyEvidence",
    "CommitteeAtomisticPotential",
    "CommitteeEvaluation",
    "CommitteeReductionPolicy",
    "ConservativeUncertaintyBlend",
    "OODPolicy",
    "SegmentFallbackPolicy",
    "SegmentFallbackDecision",
]
