#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Persistent, certified crystal deduplication with explicit policies.

A `DedupBank` is a directory of exclusively published canonical JSON commits
``commits/<sequence>.json``; each commit carries a `phydrax.NumericRevision`
and `RevisionLineage` naming its parent, the decisions it adds and the exact
structures that later comparisons need.  Reopening verifies every commit,
its canonical bytes, identities and lineage before any decision is used, so a
restarted campaign continues with the same representatives and refuses a
conflicting writer.

Matches are certified by `certify_equivalence`; an ``UNDETERMINED`` outcome is
recorded and never merges.  Two policies are distinct:

* ``greedy``: candidates are processed in the declared priority order and a
  candidate duplicates the first stored representative it is certified
  equivalent to; representatives are never replaced.
* ``components``: a candidate joins every component containing a member it is
  certified equivalent to, merging those components.  The tolerance relation
  need not be transitive, so a component certifies only its recorded edges,
  not every member pair; its representative is the member of best priority.

Priority is declared: ``(priority, ordinal, structure_id)`` ascending, or
descending in ``priority``.  It orders each submitted batch; a later batch
never displaces earlier decisions.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Sequence
from math import isfinite
from pathlib import Path
from typing import Any, final, Literal, NoReturn, TypeAlias

import equinox as eqx
import numpy as np

from ..._external_resource import read_bounded_resource, ResourceLimits, ResourceReadError
from ..._fingerprint import canonical_fingerprint, canonical_json
from ..._host_io import open_directory_beneath, remote_location
from ..._identity import NumericRevision, SemanticProvenance
from ..._publication import publish_bytes
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...lifecycle import RevisionLineage
from ...typing import parse
from ._equivalence import (
    certify_equivalence,
    CrystalStructure,
    EquivalencePolicy,
    EquivalenceResult,
    EquivalenceStatus,
)
from ._input import MolecularComponent


DedupMode: TypeAlias = Literal["greedy", "components"]
PriorityOrder: TypeAlias = Literal["ascending", "descending"]
DedupRole: TypeAlias = Literal["representative", "duplicate", "member"]
DedupOperation: TypeAlias = Literal["create", "submit"]
DedupFailure: TypeAlias = Literal["malformed", "limit", "conflict", "identity"]

_COMMIT_NAME = re.compile(r"([0-9]{20})\.json")
_COMMIT_TEMPORARY = re.compile(r"\.[0-9]{20}\.json\.[0-9a-f]{32}\.tmp")
_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")


class DedupBankError(ValueError):
    """Fail-closed dedup-bank error with a stable failure category."""

    reason: DedupFailure

    def __init__(self, reason: DedupFailure, message: str, /) -> None:
        self.reason = reason
        super().__init__(message)


def _positive(value: int, name: str, /) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")
    if value < 1:
        raise ValueError(f"{name} must be positive.")
    return value


@final
class DedupPolicy(StrictModule):
    """Dedup mode, declared priority direction and the certifying equivalence policy."""

    mode: DedupMode = eqx.field(static=True)
    priority_order: PriorityOrder = eqx.field(static=True)
    equivalence: EquivalencePolicy
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        mode: DedupMode,
        priority_order: PriorityOrder,
        equivalence: EquivalencePolicy,
    ) -> None:
        mode_ = parse(mode, DedupMode, "mode")
        order = parse(priority_order, PriorityOrder, "priority_order")
        if not isinstance(equivalence, EquivalencePolicy):
            raise TypeError("equivalence must be an EquivalencePolicy.")
        self.mode = mode_
        self.priority_order = order
        self.equivalence = equivalence
        self.policy_id = canonical_fingerprint(
            {
                "kind": "crystal-dedup-policy",
                "mode": mode_,
                "priority_order": order,
                "equivalence": equivalence.policy_id,
            }
        )


@final
class DedupBankLimits(StrictModule):
    """Finite bounds on commit bytes, commits, decisions and stored atoms."""

    max_commit_bytes: int = eqx.field(static=True)
    max_commits: int = eqx.field(static=True)
    max_entries: int = eqx.field(static=True)
    max_stored_atoms: int = eqx.field(static=True)

    def __init__(
        self,
        *,
        max_commit_bytes: int,
        max_commits: int,
        max_entries: int,
        max_stored_atoms: int,
    ) -> None:
        self.max_commit_bytes = _positive(max_commit_bytes, "max_commit_bytes")
        self.max_commits = _positive(max_commits, "max_commits")
        self.max_entries = _positive(max_entries, "max_entries")
        self.max_stored_atoms = _positive(max_stored_atoms, "max_stored_atoms")


@final
class DedupCandidate(StrictModule, NonTrainableState):
    """One structure with its declared priority (e.g. energy) and stable ordinal."""

    structure: CrystalStructure
    priority: float = eqx.field(static=True)
    ordinal: int = eqx.field(static=True)

    def __init__(
        self, structure: CrystalStructure, /, *, priority: float, ordinal: int
    ) -> None:
        if not isinstance(structure, CrystalStructure):
            raise TypeError("structure must be a CrystalStructure.")
        if isinstance(priority, bool) or not isinstance(priority, (int, float)):
            raise TypeError("priority must be a real number.")
        if not isfinite(float(priority)):
            raise ValueError("priority must be finite.")
        if isinstance(ordinal, bool) or not isinstance(ordinal, int) or ordinal < 0:
            raise ValueError("ordinal must be a nonnegative integer.")
        self.structure = structure
        self.priority = float(priority)
        self.ordinal = ordinal


@final
class DedupComparison(StrictModule):
    """One recorded certified comparison against a stored structure."""

    other_id: str = eqx.field(static=True)
    status: EquivalenceStatus = eqx.field(static=True)
    result_id: str = eqx.field(static=True)
    certificate_id: str | None = eqx.field(static=True)
    enantiomorphic: bool | None = eqx.field(static=True)


@final
class DedupDecision(StrictModule):
    """Durable decision for one structure identity.

    ``representative_id`` is, for ``greedy``, the representative this
    structure duplicates (itself for a representative).  ``component_id`` is
    the founding member of the structure's component at decision time;
    ``merged_components`` lists components merged into it by this decision.
    """

    structure_id: str = eqx.field(static=True)
    content_id: str = eqx.field(static=True)
    priority: float = eqx.field(static=True)
    ordinal: int = eqx.field(static=True)
    sequence: int = eqx.field(static=True)
    role: DedupRole = eqx.field(static=True)
    representative_id: str = eqx.field(static=True)
    component_id: str = eqx.field(static=True)
    merged_components: tuple[str, ...] = eqx.field(static=True)
    comparisons: tuple[DedupComparison, ...]
    decision_id: str = eqx.field(static=True)

    def __init__(
        self,
        structure_id: str,
        content_id: str,
        /,
        *,
        priority: float,
        ordinal: int,
        sequence: int,
        role: DedupRole,
        representative_id: str,
        component_id: str,
        merged_components: Sequence[str],
        comparisons: Sequence[DedupComparison],
    ) -> None:
        self.structure_id = structure_id
        self.content_id = content_id
        self.priority = float(priority)
        self.ordinal = ordinal
        self.sequence = sequence
        self.role = parse(role, DedupRole, "role")
        self.representative_id = representative_id
        self.component_id = component_id
        self.merged_components = tuple(merged_components)
        self.comparisons = tuple(comparisons)
        self.decision_id = canonical_fingerprint(_decision_payload(self))


def _comparison_payload(comparison: DedupComparison, /) -> dict[str, Any]:
    return {
        "other_id": comparison.other_id,
        "status": comparison.status.value,
        "result_id": comparison.result_id,
        "certificate_id": comparison.certificate_id,
        "enantiomorphic": comparison.enantiomorphic,
    }


def _decision_payload(decision: DedupDecision, /) -> dict[str, Any]:
    return {
        "structure_id": decision.structure_id,
        "content_id": decision.content_id,
        "priority": float(decision.priority).hex(),
        "ordinal": decision.ordinal,
        "sequence": decision.sequence,
        "role": decision.role,
        "representative_id": decision.representative_id,
        "component_id": decision.component_id,
        "merged_components": list(decision.merged_components),
        "comparisons": [_comparison_payload(item) for item in decision.comparisons],
    }


def _structure_payload(structure: CrystalStructure, /) -> dict[str, Any]:
    molecules: dict[int, list[int]] = {}
    for atom, (m, local) in enumerate(
        zip(
            structure.molecule_index.tolist(),
            structure.molecule_local.tolist(),
            strict=True,
        )
    ):
        if m >= 0:
            molecules.setdefault(m, [0] * structure.molecule_components[m].atom_count)[
                local
            ] = atom
    return {
        "structure_id": structure.structure_id,
        "content_id": structure.content_id,
        "vectors": [float(v).hex() for v in structure.vectors.reshape(-1)],
        "fractional": [float(v).hex() for v in structure.fractional.reshape(-1)],
        "species": [int(v) for v in structure.species],
        "molecules": [
            [structure.molecule_components[m].component_id, molecules[m]]
            for m in range(len(structure.molecule_components))
        ],
    }


def _policy_payload(policy: DedupPolicy, /) -> dict[str, Any]:
    equivalence = policy.equivalence
    return {
        "mode": policy.mode,
        "priority_order": policy.priority_order,
        "equivalence": {
            "position_tolerance": float(equivalence.position_tolerance).hex(),
            "lattice_tolerance": float(equivalence.lattice_tolerance).hex(),
            "supercell": equivalence.supercell,
            "enantiomorph": equivalence.enantiomorph,
            "maximum_atoms": equivalence.maximum_atoms,
            "maximum_translation_order": equivalence.maximum_translation_order,
            "maximum_lattice_vectors": equivalence.maximum_lattice_vectors,
            "maximum_basis_checks": equivalence.maximum_basis_checks,
            "maximum_bases": equivalence.maximum_bases,
            "maximum_assignments": equivalence.maximum_assignments,
            "maximum_mappings": equivalence.maximum_mappings,
            "maximum_nodes": equivalence.maximum_nodes,
        },
        "policy_id": policy.policy_id,
    }


@final
class DedupBankCommit(StrictModule, NonTrainableState):
    """One immutable bank revision with lifecycle identity and lineage."""

    policy: DedupPolicy
    revision: NumericRevision
    lineage: RevisionLineage
    bank_name: str = eqx.field(static=True)
    bank_id: str = eqx.field(static=True)
    sequence: int = eqx.field(static=True)
    operation: DedupOperation = eqx.field(static=True)
    decisions: tuple[DedupDecision, ...]
    structures: tuple[CrystalStructure, ...]
    parent_commit_id: str | None = eqx.field(static=True)
    commit_id: str = eqx.field(static=True)

    def __init__(
        self,
        policy: DedupPolicy,
        /,
        *,
        bank_name: str,
        sequence: int,
        operation: DedupOperation,
        decisions: Sequence[DedupDecision],
        structures: Sequence[CrystalStructure],
        parent_commit_id: str | None = None,
        parent_revision_id: str | None = None,
        parent_lineage_id: str | None = None,
    ) -> None:
        if not isinstance(policy, DedupPolicy):
            raise TypeError("policy must be a DedupPolicy.")
        if not isinstance(bank_name, str) or _NAME.fullmatch(bank_name) is None:
            raise ValueError("bank_name must be a short portable name.")
        operation_ = parse(operation, DedupOperation, "operation")
        if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence < 0:
            raise ValueError("sequence must be a nonnegative integer.")
        parents = (parent_commit_id, parent_revision_id, parent_lineage_id)
        if (sequence == 0) != (operation_ == "create"):
            raise ValueError("Exactly sequence zero creates a bank.")
        if operation_ == "create" and (any(p is not None for p in parents) or decisions):
            raise ValueError("Bank creation has no parent and no decisions.")
        if operation_ == "submit" and (any(p is None for p in parents) or not decisions):
            raise ValueError("A submission names its complete parent and adds decisions.")
        decisions_ = tuple(decisions)
        structures_ = tuple(structures)
        if any(decision.sequence != sequence for decision in decisions_):
            raise ValueError("Every decision of a commit carries the commit sequence.")
        bank_id = canonical_fingerprint(
            {
                "kind": "crystal-dedup-bank",
                "name": bank_name,
                "policy_id": policy.policy_id,
            }
        )
        revision = NumericRevision(
            SemanticProvenance(
                {
                    "kind": "crystal-dedup-bank",
                    "bank_id": bank_id,
                    "policy_id": policy.policy_id,
                }
            ),
            {
                "sequence": sequence,
                "operation": operation_,
                "decisions": [decision.decision_id for decision in decisions_],
                "structures": [structure.content_id for structure in structures_],
            },
        )
        self.policy = policy
        self.revision = revision
        self.lineage = RevisionLineage(
            revision,
            label=f"crystal-dedup-bank-{operation_}",
            parent_revision_id=parent_revision_id,
            parent_lineage_id=parent_lineage_id,
        )
        self.bank_name = bank_name
        self.bank_id = bank_id
        self.sequence = sequence
        self.operation = operation_
        self.decisions = decisions_
        self.structures = structures_
        self.parent_commit_id = parent_commit_id
        self.commit_id = canonical_fingerprint(_commit_content(self))


def _commit_content(commit: DedupBankCommit, /) -> dict[str, Any]:
    return {
        "kind": "crystal-dedup-bank-commit",
        "bank_name": commit.bank_name,
        "bank_id": commit.bank_id,
        "sequence": commit.sequence,
        "operation": commit.operation,
        "policy": _policy_payload(commit.policy),
        "decisions": [_decision_payload(decision) for decision in commit.decisions],
        "structures": [_structure_payload(structure) for structure in commit.structures],
        "parent": (
            None
            if commit.parent_commit_id is None
            else {
                "commit_id": commit.parent_commit_id,
                "revision_id": commit.lineage.parent_revision_id,
                "lineage_id": commit.lineage.parent_lineage_id,
            }
        ),
        "revision_id": commit.revision.revision_id,
        "lineage_id": commit.lineage.lineage_id,
    }


def _refuse_constant(value: str, /) -> NoReturn:
    raise ValueError(f"Non-finite JSON constant {value!r} is not admitted.")


def _keys(payload: object, keys: Sequence[str], label: str, /) -> dict[str, Any]:
    if not isinstance(payload, dict) or set(payload) != set(keys):
        raise DedupBankError("malformed", f"{label} has unexpected fields.")
    return payload


def _policy_from_payload(payload: object, /) -> DedupPolicy:
    record = _keys(
        payload, ("mode", "priority_order", "equivalence", "policy_id"), "policy"
    )
    fields = _keys(
        record["equivalence"],
        (
            "position_tolerance",
            "lattice_tolerance",
            "supercell",
            "enantiomorph",
            "maximum_atoms",
            "maximum_translation_order",
            "maximum_lattice_vectors",
            "maximum_basis_checks",
            "maximum_bases",
            "maximum_assignments",
            "maximum_mappings",
            "maximum_nodes",
        ),
        "equivalence policy",
    )
    equivalence = EquivalencePolicy(
        position_tolerance=float.fromhex(fields["position_tolerance"]),
        lattice_tolerance=float.fromhex(fields["lattice_tolerance"]),
        supercell=fields["supercell"],
        enantiomorph=fields["enantiomorph"],
        maximum_atoms=fields["maximum_atoms"],
        maximum_translation_order=fields["maximum_translation_order"],
        maximum_lattice_vectors=fields["maximum_lattice_vectors"],
        maximum_basis_checks=fields["maximum_basis_checks"],
        maximum_bases=fields["maximum_bases"],
        maximum_assignments=fields["maximum_assignments"],
        maximum_mappings=fields["maximum_mappings"],
        maximum_nodes=fields["maximum_nodes"],
    )
    policy = DedupPolicy(
        mode=record["mode"],
        priority_order=record["priority_order"],
        equivalence=equivalence,
    )
    if policy.policy_id != record["policy_id"]:
        raise DedupBankError(
            "identity", "The persisted dedup policy identity does not match."
        )
    return policy


def _structure_from_payload(
    payload: object, components: dict[str, MolecularComponent], /
) -> CrystalStructure:
    record = _keys(
        payload,
        ("structure_id", "content_id", "vectors", "fractional", "species", "molecules"),
        "structure",
    )
    molecules = []
    for component_id, atoms in record["molecules"]:
        component = components.get(component_id)
        if component is None:
            raise DedupBankError(
                "identity", f"Stored structure needs molecular component {component_id}."
            )
        molecules.append((component, atoms))
    vectors = np.asarray([float.fromhex(v) for v in record["vectors"]], dtype=np.float64)
    fractional = np.asarray(
        [float.fromhex(v) for v in record["fractional"]], dtype=np.float64
    )
    structure = CrystalStructure(
        vectors.reshape((3, 3)),
        fractional.reshape((-1, 3)),
        record["species"],
        structure_id=record["structure_id"],
        molecules=molecules,
    )
    if structure.content_id != record["content_id"]:
        raise DedupBankError(
            "identity", "A stored structure does not match its content id."
        )
    return structure


def _decision_from_payload(payload: object, /) -> DedupDecision:
    record = _keys(
        payload,
        (
            "structure_id",
            "content_id",
            "priority",
            "ordinal",
            "sequence",
            "role",
            "representative_id",
            "component_id",
            "merged_components",
            "comparisons",
        ),
        "decision",
    )
    comparisons = []
    for item in record["comparisons"]:
        entry = _keys(
            item,
            ("other_id", "status", "result_id", "certificate_id", "enantiomorphic"),
            "comparison",
        )
        comparisons.append(
            DedupComparison(
                entry["other_id"],
                EquivalenceStatus(entry["status"]),
                entry["result_id"],
                entry["certificate_id"],
                entry["enantiomorphic"],
            )
        )
    return DedupDecision(
        record["structure_id"],
        record["content_id"],
        priority=float.fromhex(record["priority"]),
        ordinal=record["ordinal"],
        sequence=record["sequence"],
        role=record["role"],
        representative_id=record["representative_id"],
        component_id=record["component_id"],
        merged_components=record["merged_components"],
        comparisons=comparisons,
    )


def _commit_from_payload(
    payload: object, components: dict[str, MolecularComponent], /
) -> DedupBankCommit:
    record = _keys(
        payload,
        (
            "kind",
            "bank_name",
            "bank_id",
            "sequence",
            "operation",
            "policy",
            "decisions",
            "structures",
            "parent",
            "revision_id",
            "lineage_id",
            "commit_id",
        ),
        "commit",
    )
    parent = record["parent"]
    parents: tuple[str | None, str | None, str | None] = (None, None, None)
    if parent is not None:
        entry = _keys(parent, ("commit_id", "revision_id", "lineage_id"), "parent")
        parents = (entry["commit_id"], entry["revision_id"], entry["lineage_id"])
    commit = DedupBankCommit(
        _policy_from_payload(record["policy"]),
        bank_name=record["bank_name"],
        sequence=record["sequence"],
        operation=record["operation"],
        decisions=tuple(_decision_from_payload(item) for item in record["decisions"]),
        structures=tuple(
            _structure_from_payload(item, components) for item in record["structures"]
        ),
        parent_commit_id=parents[0],
        parent_revision_id=parents[1],
        parent_lineage_id=parents[2],
    )
    if {**_commit_content(commit), "commit_id": commit.commit_id} != record:
        raise DedupBankError("identity", "A persisted commit does not match its content.")
    return commit


@final
class DedupBank(StrictModule, NonTrainableState):
    """Verified in-memory view of a persistent dedup bank at its head revision."""

    root: str = eqx.field(static=True)
    limits: DedupBankLimits
    commits: tuple[DedupBankCommit, ...]

    @property
    def head(self) -> DedupBankCommit:
        return self.commits[-1]

    @property
    def policy(self) -> DedupPolicy:
        return self.head.policy

    @property
    def decisions(self) -> tuple[DedupDecision, ...]:
        return tuple(decision for commit in self.commits for decision in commit.decisions)

    @property
    def structures(self) -> tuple[CrystalStructure, ...]:
        return tuple(
            structure for commit in self.commits for structure in commit.structures
        )

    def decision(self, structure_id: str, /) -> DedupDecision | None:
        for decision in self.decisions:
            if decision.structure_id == structure_id:
                return decision
        return None

    def components(self) -> tuple[tuple[str, str, tuple[str, ...]], ...]:
        """Return ``(component_id, representative_id, member_ids)`` per live component.

        Membership is the closure of recorded certified edges; it does not
        certify every member pair.  The representative is the member of best
        declared priority.
        """
        decisions = self.decisions
        parent = _component_parents(decisions)
        groups: dict[str, list[DedupDecision]] = {}
        for decision in decisions:
            groups.setdefault(_find(parent, decision.component_id), []).append(decision)
        result = []
        for component_id, members in groups.items():
            best = min(
                members,
                key=lambda d: _priority_key(
                    self.policy, d.priority, d.ordinal, d.structure_id
                ),
            )
            result.append(
                (component_id, best.structure_id, tuple(d.structure_id for d in members))
            )
        return tuple(result)


def _priority_key(
    policy: DedupPolicy, priority: float, ordinal: int, structure_id: str, /
) -> tuple[float, int, str]:
    value = priority if policy.priority_order == "ascending" else -priority
    return (value, ordinal, structure_id)


def _component_parents(decisions: Sequence[DedupDecision], /) -> dict[str, str]:
    parent: dict[str, str] = {}
    for decision in decisions:
        parent.setdefault(decision.component_id, decision.component_id)
        for merged in decision.merged_components:
            parent[_find(parent, merged)] = _find(parent, decision.component_id)
    return parent


def _find(parent: dict[str, str], key: str, /) -> str:
    while parent.get(key, key) != key:
        key = parent[key]
    return key


def _root(root: str | os.PathLike[str], /) -> str:
    text = os.fspath(root)
    if not isinstance(text, str) or not text or "\x00" in text or remote_location(text):
        raise ValueError("Dedup bank roots must be local filesystem paths.")
    return os.path.abspath(os.path.expanduser(text))


def _commit_path(root: str, sequence: int, /) -> Path:
    return Path(root) / "commits" / f"{sequence:020d}.json"


def _publish(root: str, commit: DedupBankCommit, limits: DedupBankLimits, /) -> None:
    data = canonical_json(
        {**_commit_content(commit), "commit_id": commit.commit_id}
    ).encode("ascii")
    if len(data) > limits.max_commit_bytes:
        raise DedupBankError("limit", "The dedup commit exceeds max_commit_bytes.")
    try:
        publish_bytes(
            _commit_path(root, commit.sequence),
            data,
            maximum_bytes=limits.max_commit_bytes,
            mode="exclusive",
        )
    except FileExistsError as error:
        raise DedupBankError(
            "conflict", f"Dedup revision {commit.sequence} was already published."
        ) from error


def create_dedup_bank(
    root: str | os.PathLike[str],
    /,
    *,
    name: str,
    policy: DedupPolicy,
    limits: DedupBankLimits,
) -> DedupBank:
    """Create an empty bank by exclusively publishing its sequence-zero commit."""
    if not isinstance(limits, DedupBankLimits):
        raise TypeError("limits must be DedupBankLimits.")
    root_ = _root(root)
    commit = DedupBankCommit(
        policy,
        bank_name=name,
        sequence=0,
        operation="create",
        decisions=(),
        structures=(),
    )
    _publish(root_, commit, limits)
    return DedupBank(root_, limits, (commit,))


def open_dedup_bank(
    root: str | os.PathLike[str],
    /,
    *,
    limits: DedupBankLimits,
    components: Sequence[MolecularComponent] = (),
) -> DedupBank:
    """Open and verify every commit of a bank; molecular components are resupplied."""
    if not isinstance(limits, DedupBankLimits):
        raise TypeError("limits must be DedupBankLimits.")
    root_ = _root(root)
    known = {component.component_id: component for component in components}
    try:
        with open_directory_beneath(
            "commits", trusted_root=root_, maximum_depth=1
        ) as opened:
            names = os.listdir(opened.descriptor)
    except (OSError, ValueError, RuntimeError) as error:
        raise DedupBankError(
            "malformed", "The path is not a readable dedup bank."
        ) from error
    sequences = []
    for entry in names:
        match = _COMMIT_NAME.fullmatch(entry)
        if match is not None:
            sequences.append(int(match[1]))
        elif _COMMIT_TEMPORARY.fullmatch(entry) is None:
            raise DedupBankError(
                "malformed", f"Unexpected entry {entry!r} in the commit directory."
            )
    sequences.sort()
    if not sequences or sequences != list(range(len(sequences))):
        raise DedupBankError("malformed", "The dedup commit sequence is incomplete.")
    if len(sequences) > limits.max_commits:
        raise DedupBankError("limit", "The bank exceeds max_commits.")
    commits: list[DedupBankCommit] = []
    for sequence in sequences:
        try:
            resource = read_bounded_resource(
                f"commits/{sequence:020d}.json",
                trusted_root=root_,
                limits=ResourceLimits(
                    max_bytes=limits.max_commit_bytes,
                    max_depth=2,
                    max_nodes=1,
                    max_attributes=0,
                    max_losses=0,
                ),
            )
        except ResourceReadError as error:
            reason: DedupFailure = "limit" if error.reason == "limit" else "malformed"
            raise DedupBankError(
                reason, f"Dedup commit {sequence} cannot be read."
            ) from error
        try:
            payload = json.loads(
                resource.data.decode("ascii"), parse_constant=_refuse_constant
            )
        except (UnicodeDecodeError, ValueError) as error:
            raise DedupBankError(
                "malformed", f"Dedup commit {sequence} is not JSON."
            ) from error
        if canonical_json(payload).encode("ascii") != resource.data:
            raise DedupBankError(
                "malformed", f"Dedup commit {sequence} is not canonical JSON."
            )
        try:
            commit = _commit_from_payload(payload, known)
        except DedupBankError:
            raise
        except (TypeError, ValueError, KeyError) as error:
            raise DedupBankError(
                "malformed", f"Dedup commit {sequence} is invalid: {error}"
            ) from error
        if commit.sequence != sequence:
            raise DedupBankError(
                "malformed", f"Dedup commit {sequence} names another sequence."
            )
        if commits:
            previous = commits[-1]
            if (
                commit.parent_commit_id != previous.commit_id
                or commit.lineage.parent_revision_id != previous.revision.revision_id
                or commit.lineage.parent_lineage_id != previous.lineage.lineage_id
                or commit.bank_id != previous.bank_id
            ):
                raise DedupBankError(
                    "identity", f"Dedup commit {sequence} breaks the lineage."
                )
        commits.append(commit)
    return DedupBank(root_, limits, tuple(commits))


@final
class DedupSubmission(StrictModule, NonTrainableState):
    """Result of one submission: the new bank head and one decision per candidate.

    ``decisions`` follow the submitted candidate order; an already-decided
    identity returns its durable earlier decision and adds nothing.
    """

    bank: DedupBank
    decisions: tuple[DedupDecision, ...]
    published: bool = eqx.field(static=True)


def _comparison(other: CrystalStructure, result: EquivalenceResult, /) -> DedupComparison:
    return DedupComparison(
        other.structure_id,
        result.status,
        result.result_id,
        None if result.certificate is None else result.certificate.certificate_id,
        result.enantiomorphic,
    )


def submit_to_dedup_bank(
    bank: DedupBank, candidates: Sequence[DedupCandidate], /
) -> DedupSubmission:
    """Decide a batch in declared priority order and publish one new revision."""
    if not isinstance(bank, DedupBank):
        raise TypeError("bank must be a DedupBank.")
    candidates_ = tuple(candidates)
    if any(not isinstance(candidate, DedupCandidate) for candidate in candidates_):
        raise TypeError("candidates must be DedupCandidate values.")
    identities = [candidate.structure.structure_id for candidate in candidates_]
    if len(set(identities)) != len(identities):
        raise ValueError("A submission names each structure identity once.")
    policy = bank.policy
    existing = {decision.structure_id: decision for decision in bank.decisions}
    decided: dict[str, DedupDecision] = {}
    fresh: list[DedupCandidate] = []
    for candidate in candidates_:
        previous = existing.get(candidate.structure.structure_id)
        if previous is None:
            fresh.append(candidate)
        elif previous.content_id != candidate.structure.content_id:
            raise DedupBankError(
                "identity",
                f"Structure {previous.structure_id} was decided with different content.",
            )
        else:
            decided[previous.structure_id] = previous
    if len(existing) + len(fresh) > bank.limits.max_entries:
        raise DedupBankError("limit", "The submission exceeds max_entries.")
    fresh.sort(
        key=lambda c: _priority_key(
            policy, c.priority, c.ordinal, c.structure.structure_id
        )
    )
    sequence = bank.head.sequence + 1
    stored = list(bank.structures)
    stored_atoms = sum(structure.atom_count for structure in stored)
    by_structure = {decision.structure_id: decision for decision in bank.decisions}
    new_decisions: list[DedupDecision] = []
    new_structures: list[CrystalStructure] = []
    parent = _component_parents(bank.decisions)
    for candidate in fresh:
        structure = candidate.structure
        comparisons: list[DedupComparison] = []
        role: DedupRole
        if policy.mode == "greedy":
            match = None
            for other in stored:
                result = certify_equivalence(structure, other, policy.equivalence)
                comparisons.append(_comparison(other, result))
                if result.status == EquivalenceStatus.EQUIVALENT:
                    match = other
                    break
            if match is None:
                role, representative, component, merged = (
                    "representative",
                    structure.structure_id,
                    structure.structure_id,
                    (),
                )
            else:
                owner = by_structure[match.structure_id]
                role, representative, component, merged = (
                    "duplicate",
                    match.structure_id,
                    owner.component_id,
                    (),
                )
        else:
            joined: list[str] = []
            for other in stored:
                result = certify_equivalence(structure, other, policy.equivalence)
                comparisons.append(_comparison(other, result))
                if result.status == EquivalenceStatus.EQUIVALENT:
                    root = _find(parent, by_structure[other.structure_id].component_id)
                    if root not in joined:
                        joined.append(root)
            if not joined:
                role, representative, component, merged = (
                    "representative",
                    structure.structure_id,
                    structure.structure_id,
                    (),
                )
            else:
                founders = {
                    decision.component_id: index
                    for index, decision in enumerate((*bank.decisions, *new_decisions))
                    if decision.component_id == decision.structure_id
                }
                joined.sort(key=lambda key: founders[key])
                role, representative, component, merged = (
                    "member",
                    joined[0],
                    joined[0],
                    tuple(joined[1:]),
                )
        decision = DedupDecision(
            structure.structure_id,
            structure.content_id,
            priority=candidate.priority,
            ordinal=candidate.ordinal,
            sequence=sequence,
            role=role,
            representative_id=representative,
            component_id=component,
            merged_components=merged,
            comparisons=comparisons,
        )
        parent.setdefault(component, component)
        for key in merged:
            parent[_find(parent, key)] = _find(parent, component)
        if policy.mode == "components" or role == "representative":
            stored_atoms += structure.atom_count
            if stored_atoms > bank.limits.max_stored_atoms:
                raise DedupBankError("limit", "The bank exceeds max_stored_atoms.")
            stored.append(structure)
            new_structures.append(structure)
        by_structure[structure.structure_id] = decision
        new_decisions.append(decision)
        decided[structure.structure_id] = decision
    if not new_decisions:
        return DedupSubmission(bank, tuple(decided[i] for i in identities), False)
    if sequence + 1 > bank.limits.max_commits:
        raise DedupBankError("limit", "The bank would exceed max_commits.")
    head = bank.head
    commit = DedupBankCommit(
        policy,
        bank_name=head.bank_name,
        sequence=sequence,
        operation="submit",
        decisions=new_decisions,
        structures=new_structures,
        parent_commit_id=head.commit_id,
        parent_revision_id=head.revision.revision_id,
        parent_lineage_id=head.lineage.lineage_id,
    )
    _publish(bank.root, commit, bank.limits)
    updated = DedupBank(bank.root, bank.limits, (*bank.commits, commit))
    return DedupSubmission(updated, tuple(decided[i] for i in identities), True)


__all__ = [
    "create_dedup_bank",
    "DedupBank",
    "DedupBankCommit",
    "DedupBankError",
    "DedupBankLimits",
    "DedupCandidate",
    "DedupComparison",
    "DedupDecision",
    "DedupMode",
    "DedupPolicy",
    "DedupRole",
    "DedupSubmission",
    "open_dedup_bank",
    "PriorityOrder",
    "submit_to_dedup_bank",
]
