#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Persistent certified deduplication: priority, policies, restart and conflicts.

Controls: an explicit basis/origin/permutation copy is a duplicate of the
better-priority representative; a chain X ~ Y ~ Z whose ends are not
certified separates greedy representatives from component closure; reopening
the bank from disk reproduces every decision identity and continues the same
lineage; a resubmitted identity publishes nothing; conflicting writers,
conflicting content and corrupted commits are refused.
"""

from pathlib import Path

import numpy as np
import pytest

from phydrax.atomistic.crystal import (
    create_dedup_bank,
    CrystalStructure,
    DedupBankError,
    DedupBankLimits,
    DedupCandidate,
    DedupMode,
    DedupPolicy,
    EquivalencePolicy,
    EquivalenceStatus,
    open_dedup_bank,
    submit_to_dedup_bank,
)


_CELL = np.asarray([[5.3, 0.0, 0.0], [-1.4, 5.9, 0.0], [-0.9, 0.7, 7.3]])
_FRACTIONAL = np.asarray(
    [(0.11, 0.23, 0.37), (0.61, 0.17, 0.82), (0.29, 0.71, 0.44), (0.83, 0.52, 0.09)]
)
_SPECIES = (8, 14, 26, 29)
_LIMITS = DedupBankLimits(
    max_commit_bytes=4_000_000, max_commits=64, max_entries=256, max_stored_atoms=10_000
)


def _policy(mode: DedupMode) -> DedupPolicy:
    return DedupPolicy(
        mode=mode,
        priority_order="ascending",
        equivalence=EquivalencePolicy(
            position_tolerance=0.05,
            lattice_tolerance=1.0e-3,
            supercell="primitive-reduction",
            enantiomorph="proper-only",
            maximum_atoms=64,
            maximum_translation_order=4,
            maximum_lattice_vectors=100_000,
            maximum_basis_checks=1_000_000,
            maximum_bases=256,
            maximum_assignments=10_000,
            maximum_mappings=16,
            maximum_nodes=10_000,
        ),
    )


def _structure(
    structure_id: str, *, displaced: float = 0.0, atom: int = 2
) -> CrystalStructure:
    fractional = _FRACTIONAL.copy()
    fractional[atom] += np.asarray((displaced, 0.0, 0.0)) @ np.linalg.inv(_CELL)
    return CrystalStructure(_CELL, fractional, _SPECIES, structure_id=structure_id)


def _moved(structure_id: str) -> CrystalStructure:
    transform = np.asarray([[1, 0, 1], [0, 1, 0], [0, 0, 1]])
    fractional = _FRACTIONAL @ np.linalg.inv(transform).T + (0.4, 0.1, -0.3)
    order = [3, 1, 0, 2]
    return CrystalStructure(
        transform.T @ _CELL,
        fractional[order],
        np.asarray(_SPECIES)[order],
        structure_id=structure_id,
    )


def _distinct(structure_id: str) -> CrystalStructure:
    return _structure(structure_id, displaced=0.6, atom=1)


def test_greedy_priority_restart_and_idempotent_resubmission(tmp_path: Path) -> None:
    root = tmp_path / "greedy"
    bank = create_dedup_bank(
        root, name="greedy-bank", policy=_policy("greedy"), limits=_LIMITS
    )
    candidates = [
        DedupCandidate(_structure("a"), priority=1.0, ordinal=0),
        DedupCandidate(_moved("a-moved"), priority=0.5, ordinal=1),
        DedupCandidate(_distinct("c"), priority=2.0, ordinal=2),
    ]
    submission = submit_to_dedup_bank(bank, candidates)
    assert submission.published
    roles = {d.structure_id: (d.role, d.representative_id) for d in submission.decisions}
    assert roles == {
        "a-moved": ("representative", "a-moved"),
        "a": ("duplicate", "a-moved"),
        "c": ("representative", "c"),
    }
    duplicate = submission.decisions[0]
    assert duplicate.comparisons[-1].status is EquivalenceStatus.EQUIVALENT
    assert duplicate.comparisons[-1].certificate_id is not None
    reopened = open_dedup_bank(root, limits=_LIMITS)
    assert reopened.head.commit_id == submission.bank.head.commit_id
    assert [d.decision_id for d in reopened.decisions] == [
        d.decision_id for d in submission.bank.decisions
    ]
    assert [s.structure_id for s in reopened.structures] == ["a-moved", "c"]
    again = submit_to_dedup_bank(
        reopened, [DedupCandidate(_structure("a"), priority=1.0, ordinal=0)]
    )
    assert not again.published
    assert again.decisions[0].decision_id == duplicate.decision_id
    assert len(list((root / "commits").iterdir())) == 2
    later = submit_to_dedup_bank(
        reopened, [DedupCandidate(_distinct("c-copy"), priority=0.1, ordinal=3)]
    )
    assert later.decisions[0].role == "duplicate"
    assert later.decisions[0].representative_id == "c"
    assert later.bank.head.lineage.parent_lineage_id == reopened.head.lineage.lineage_id


def test_greedy_and_component_policies_differ_on_a_non_transitive_chain(
    tmp_path: Path,
) -> None:
    chain = [
        DedupCandidate(_structure("x"), priority=0.0, ordinal=0),
        DedupCandidate(_structure("y", displaced=0.04), priority=1.0, ordinal=1),
        DedupCandidate(_structure("z", displaced=0.08), priority=2.0, ordinal=2),
    ]
    greedy = submit_to_dedup_bank(
        create_dedup_bank(
            tmp_path / "g", name="g", policy=_policy("greedy"), limits=_LIMITS
        ),
        chain,
    )
    by_id = {d.structure_id: d for d in greedy.decisions}
    assert by_id["y"].role == "duplicate" and by_id["y"].representative_id == "x"
    assert by_id["z"].role == "representative"
    assert by_id["z"].comparisons[0].status is not EquivalenceStatus.EQUIVALENT
    components = submit_to_dedup_bank(
        create_dedup_bank(
            tmp_path / "c", name="c", policy=_policy("components"), limits=_LIMITS
        ),
        chain,
    )
    by_id = {d.structure_id: d for d in components.decisions}
    assert by_id["z"].role == "member" and by_id["z"].component_id == "x"
    statuses = {c.other_id: c.status for c in by_id["z"].comparisons}
    assert statuses["y"] is EquivalenceStatus.EQUIVALENT
    assert statuses["x"] is not EquivalenceStatus.EQUIVALENT
    ((component_id, representative, members),) = components.bank.components()
    assert component_id == "x" and representative == "x"
    assert set(members) == {"x", "y", "z"}


def test_conflicting_writers_content_and_corruption_are_refused(tmp_path: Path) -> None:
    root = tmp_path / "bank"
    bank = create_dedup_bank(root, name="bank", policy=_policy("greedy"), limits=_LIMITS)
    with pytest.raises(DedupBankError) as created:
        create_dedup_bank(root, name="bank", policy=_policy("greedy"), limits=_LIMITS)
    assert created.value.reason == "conflict"
    first = submit_to_dedup_bank(
        bank, [DedupCandidate(_structure("a"), priority=0.0, ordinal=0)]
    )
    with pytest.raises(DedupBankError) as raced:
        submit_to_dedup_bank(
            bank, [DedupCandidate(_distinct("b"), priority=0.0, ordinal=1)]
        )
    assert raced.value.reason == "conflict"
    with pytest.raises(DedupBankError) as conflicting:
        submit_to_dedup_bank(
            first.bank, [DedupCandidate(_distinct("a"), priority=0.0, ordinal=0)]
        )
    assert conflicting.value.reason == "identity"
    commit = root / "commits" / f"{1:020d}.json"
    commit.write_bytes(
        commit.read_bytes().replace(b'"role":"representative"', b'"role":"duplicate"')
    )
    with pytest.raises(DedupBankError):
        open_dedup_bank(root, limits=_LIMITS)
