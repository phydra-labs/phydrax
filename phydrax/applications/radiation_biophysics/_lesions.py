#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Candidate formation and event-addressed realization of initial lesions.

All classification is host-side and discrete. Neither lesion decisions nor
clustering have pathwise derivatives, and none of these outputs implies repair.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

from ..._fingerprint import canonical_fingerprint
from ...units import conversion_factor, ELECTRONVOLT, SECOND, UnitDefinition
from ._interactions import (
    _nonnegative,
    _text,
    InteractionLedger,
    PrimaryHistoryKey,
    RadiationEventKey,
)
from ._reactions import ReactionLedger
from ._targets import RadiationTargetGeometry, TargetMapping


@dataclass(frozen=True, slots=True)
class IndirectLesionRule:
    channel: str
    species: str
    probability: float

    def __post_init__(self):
        _text(self.channel, "channel")
        _text(self.species, "species")
        if not math.isfinite(self.probability) or not 0 <= self.probability <= 1:
            raise ValueError("Lesion probability must lie in [0, 1].")


@dataclass(frozen=True, slots=True)
class LesionPolicy:
    policy_id: str
    direct_threshold: float
    energy_unit: UnitDefinition
    direct_probability: float
    indirect_rules: tuple[IndirectLesionRule, ...]
    chemistry_endpoint: float | None
    time_unit: UnitDefinition
    chemistry_model_id: str | None
    scavenging_model_id: str | None
    components: tuple[str, ...] = ("backbone",)

    def __post_init__(self):
        _text(self.policy_id, "lesion policy")
        if not math.isfinite(self.direct_threshold) or self.direct_threshold <= 0:
            raise ValueError("Direct deposited-energy threshold must be positive.")
        conversion_factor(self.energy_unit, ELECTRONVOLT)
        conversion_factor(self.time_unit, SECOND)
        if (
            not math.isfinite(self.direct_probability)
            or not 0 <= self.direct_probability <= 1
        ):
            raise ValueError("Direct lesion probability must lie in [0, 1].")
        if not isinstance(self.indirect_rules, tuple) or len(
            {(r.channel, r.species) for r in self.indirect_rules}
        ) != len(self.indirect_rules):
            raise ValueError(
                "Indirect channel/species rules must be immutable and unique."
            )
        if (
            not isinstance(self.components, tuple)
            or not self.components
            or len(set(self.components)) != len(self.components)
            or any(item not in ("base", "backbone") for item in self.components)
        ):
            raise ValueError(
                "Lesion policy requires explicit supported target components."
            )
        if self.chemistry_endpoint is not None:
            _nonnegative(self.chemistry_endpoint, "chemistry endpoint")
        if self.indirect_rules and (
            self.chemistry_endpoint is None
            or self.chemistry_model_id is None
            or self.scavenging_model_id is None
        ):
            raise ValueError(
                "Indirect lesions require endpoint, chemistry and scavenging models."
            )

    def fingerprint(self) -> str:
        return canonical_fingerprint(
            {
                "name": self.policy_id,
                "threshold": self.direct_threshold,
                "energy_unit": self.energy_unit.unit_id,
                "probability": self.direct_probability,
                "indirect": [asdict(item) for item in self.indirect_rules],
                "endpoint": self.chemistry_endpoint,
                "time_unit": self.time_unit.unit_id,
                "chemistry": self.chemistry_model_id,
                "scavenging": self.scavenging_model_id,
                "components": self.components,
            }
        )


@dataclass(frozen=True, slots=True)
class LesionCandidate:
    candidate_id: str
    history: PrimaryHistoryKey
    target_id: int
    cause: str
    probability: float
    event_keys: tuple[RadiationEventKey, ...]
    deposited_energy_ev: float | None


@dataclass(frozen=True, slots=True)
class LesionCandidates:
    candidates: tuple[LesionCandidate, ...]
    policy_id: str
    mapping_id: str
    geometry_id: str
    source_id: str


def candidate_radiation_lesions(
    physical: InteractionLedger,
    chemical: ReactionLedger,
    mapping: TargetMapping,
    geometry: RadiationTargetGeometry,
    policy: LesionPolicy,
    *,
    commercial_use=False,
) -> LesionCandidates:
    """Threshold cumulative deposition per target/primary, retaining all parent events.

    Threshold is inclusive. Kinetic-energy loss and carried energy are never used
    as deposited energy. Independent primaries and dose fractions are never fused.
    Chemical candidates are per reaction, not an energy surrogate.
    """
    physical.source.require_rights(commercial_use=commercial_use)
    if (
        mapping.ledger_ids != (physical.fingerprint(), chemical.fingerprint())
        or mapping.geometry_id != geometry.fingerprint()
    ):
        raise ValueError("Lesion input no longer matches its prepared mapping.")
    source = chemical.source
    if physical.source.fingerprint() != source.fingerprint():
        raise ValueError(
            "Physical and chemical ledgers must share a source configuration."
        )
    if policy.indirect_rules:
        if (
            source.chemistry_model_id != policy.chemistry_model_id
            or source.scavenging_model_id != policy.scavenging_model_id
        ):
            raise ValueError(
                "Lesion chemistry/scavenging policy mismatches external source."
            )
        requested_endpoint = policy.chemistry_endpoint * float(
            conversion_factor(policy.time_unit, source.time_unit)
        )
        if (
            source.chemistry_endpoint is None
            or requested_endpoint > source.chemistry_endpoint
        ):
            raise ValueError("External chemistry does not cover the requested endpoint.")
        if any(item.time is None for item in chemical.records) and not math.isclose(
            requested_endpoint, source.chemistry_endpoint, rel_tol=1e-12, abs_tol=0
        ):
            raise ValueError(
                "Untimed reactions cannot be re-filtered to another endpoint."
            )
    else:
        requested_endpoint = None
    physical_records = {item.key: item for item in physical.records}
    chemical_records = {item.key: item for item in chemical.records}
    sites = {item.target_id: item for item in geometry.sites}
    deposits: dict[
        tuple[PrimaryHistoryKey, int], list[tuple[RadiationEventKey, float]]
    ] = {}
    candidates = []
    energy_scale = float(conversion_factor(physical.source.energy_unit, ELECTRONVOLT))
    threshold = policy.direct_threshold * float(
        conversion_factor(policy.energy_unit, ELECTRONVOLT)
    )
    policy_id = policy.fingerprint()
    for hit in mapping.hits:
        if sites[hit.target_id].component not in policy.components:
            continue
        if hit.event_key.stage == "physical":
            record = physical_records[hit.event_key]
            deposits.setdefault((record.key.history, hit.target_id), []).append(
                (record.key, record.deposited_energy * energy_scale * hit.fraction)
            )
        elif policy.indirect_rules:
            record = chemical_records[hit.event_key]
            if record.time is not None and record.time > requested_endpoint:
                continue
            matched = [
                rule
                for rule in policy.indirect_rules
                if rule.channel == record.channel and rule.species in record.reactants
            ]
            if len(matched) > 1:
                raise ValueError("Reaction matches more than one indirect lesion rule.")
            if not matched:
                continue
            identity = canonical_fingerprint(
                {
                    "policy": policy_id,
                    "event": asdict(record.key),
                    "target": hit.target_id,
                    "cause": "indirect",
                }
            )
            candidates.append(
                LesionCandidate(
                    identity,
                    record.key.history,
                    hit.target_id,
                    "indirect",
                    matched[0].probability * hit.fraction,
                    (record.key,),
                    None,
                )
            )
    for (history, target_id), entries in sorted(deposits.items()):
        entries.sort(key=lambda item: item[0])
        energy = math.fsum(value for _, value in entries)
        if energy < threshold:
            continue
        event_keys = tuple(key for key, _ in entries)
        identity = canonical_fingerprint(
            {
                "policy": policy_id,
                "history": asdict(history),
                "events": [asdict(key) for key in event_keys],
                "target": target_id,
                "cause": "direct",
            }
        )
        candidates.append(
            LesionCandidate(
                identity,
                history,
                target_id,
                "direct",
                policy.direct_probability,
                event_keys,
                energy,
            )
        )
    return LesionCandidates(
        tuple(sorted(candidates, key=lambda item: item.candidate_id)),
        policy_id,
        mapping.report.report_id,
        mapping.geometry_id,
        physical.source.artifact.artifact_id,
    )


@dataclass(frozen=True, slots=True)
class InitialLesion:
    lesion_id: str
    history: PrimaryHistoryKey
    target_id: int
    causes: tuple[str, ...]
    candidate_ids: tuple[str, ...]
    event_keys: tuple[RadiationEventKey, ...]


@dataclass(frozen=True, slots=True)
class InitialLesionLedger:
    lesions: tuple[InitialLesion, ...]
    candidates: LesionCandidates
    accepted_candidate_ids: tuple[str, ...]
    realization_id: str


def realize_radiation_lesions(
    candidates: LesionCandidates,
    *,
    random_lineage: str,
    uniforms: tuple[tuple[str, float], ...] | None = None,
) -> InitialLesionLedger:
    """Realize independent Bernoulli candidates with stable event-addressed draws.

    Explicit uniforms are useful for coupled uncertainty studies. A draw u is
    accepted iff u < p; p=0 never and p=1 always. Duplicate candidates are refused;
    repeated physical damage at one site/history is one initial lesion retaining
    every accepted cause. Other primary histories remain independent lesions.
    """
    _text(random_lineage, "lesion random lineage")
    ids = {item.candidate_id for item in candidates.candidates}
    if len(ids) != len(candidates.candidates):
        raise ValueError("Duplicate candidate identity.")
    explicit = None if uniforms is None else dict(uniforms)
    if explicit is not None:
        if len(explicit) != len(uniforms) or set(explicit) != ids:
            raise ValueError("Explicit uniforms must cover every candidate exactly once.")
        if any(not math.isfinite(u) or not 0 <= u < 1 for u in explicit.values()):
            raise ValueError("Bernoulli uniforms must lie in [0, 1).")
    groups: dict[tuple[PrimaryHistoryKey, int], list[LesionCandidate]] = {}
    accepted = []
    for candidate in candidates.candidates:
        if (
            not math.isfinite(candidate.probability)
            or not 0 <= candidate.probability <= 1
        ):
            raise ValueError("Candidate probability must lie in [0, 1].")
        draw = (
            int(canonical_fingerprint([random_lineage, candidate.candidate_id])[:13], 16)
            / 2**52
            if explicit is None
            else explicit[candidate.candidate_id]
        )
        if draw < candidate.probability:
            accepted.append(candidate.candidate_id)
            groups.setdefault((candidate.history, candidate.target_id), []).append(
                candidate
            )
    lesions = []
    for (history, target_id), parents in sorted(groups.items()):
        parent_ids = tuple(sorted(item.candidate_id for item in parents))
        identity = canonical_fingerprint(
            {"history": asdict(history), "target": target_id, "parents": parent_ids}
        )
        lesions.append(
            InitialLesion(
                identity,
                history,
                target_id,
                tuple(sorted({item.cause for item in parents})),
                parent_ids,
                tuple(sorted({key for item in parents for key in item.event_keys})),
            )
        )
    realization_id = canonical_fingerprint(
        {
            "lineage": random_lineage,
            "policy": candidates.policy_id,
            "mapping": candidates.mapping_id,
            "uniforms": None if explicit is None else sorted(explicit.items()),
            "accepted": sorted(accepted),
        }
    )
    return InitialLesionLedger(
        tuple(lesions), candidates, tuple(sorted(accepted)), realization_id
    )
