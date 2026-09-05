#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from dataclasses import dataclass

from ...._fingerprint import canonical_fingerprint
from .._construct import NucleicAcidConstruct, NucleotideKey


@dataclass(frozen=True, slots=True)
class StrandComplexPartition:
    """Canonical partition of physically labelled strand copies.

    Permuting input blocks or members does not change identity. Identical
    sequences on distinct strand IDs remain distinct physical copies.
    """

    strand_ids: tuple[str, ...]
    complexes: tuple[tuple[str, ...], ...]

    def __post_init__(self):
        strands = tuple(self.strand_ids)
        if not strands or len(set(strands)) != len(strands):
            raise ValueError("A partition requires unique declared strand identities.")
        if any(not isinstance(value, str) or not value for value in strands):
            raise ValueError("Strand identities must be nonempty strings.")
        blocks = tuple(tuple(block) for block in self.complexes)
        flat = tuple(value for block in blocks for value in block)
        if (
            any(not block for block in blocks)
            or len(flat) != len(strands)
            or set(flat) != set(strands)
        ):
            raise ValueError(
                "Every declared strand must occur in exactly one nonempty complex."
            )
        order = {value: i for i, value in enumerate(strands)}
        canonical = tuple(
            sorted(
                (tuple(sorted(block, key=order.__getitem__)) for block in blocks),
                key=lambda block: order[block[0]],
            )
        )
        object.__setattr__(self, "strand_ids", strands)
        object.__setattr__(self, "complexes", canonical)

    @property
    def complex_count(self) -> int:
        return len(self.complexes)

    def fingerprint(self) -> str:
        return canonical_fingerprint(
            ("labelled-strand-partition", self.strand_ids, self.complexes)
        )


def _partition(
    construct: NucleicAcidConstruct, pairs: tuple[tuple[int, int], ...]
) -> StrandComplexPartition:
    strands = construct.strand_ids
    strand_index = {name: i for i, name in enumerate(strands)}
    keys = construct.nucleotide_keys
    parent = list(range(len(strands)))

    def root(i):
        while parent[i] != i:
            i = parent[i]
        return i

    for i, j in pairs:
        first = root(strand_index[keys[i].strand_id])
        second = root(strand_index[keys[j].strand_id])
        parent[max(first, second)] = min(first, second)
    blocks: dict[int, list[str]] = {}
    for i, name in enumerate(strands):
        blocks.setdefault(root(i), []).append(name)
    return StrandComplexPartition(
        strands, tuple(tuple(block) for block in blocks.values())
    )


def _noncrossing(pairs: tuple[tuple[int, int], ...]) -> bool:
    return not any(i < k < j < l or k < i < l < j for i, j in pairs for k, l in pairs)


@dataclass(frozen=True, slots=True, init=False)
class SecondaryStructureState:
    """One-partner, ordered-planar pairing and its induced strand partition.

    Planarity is relative to declared strand order and 5′→3′ positions. This is
    a consuming CTMC restriction, not a replacement for full base interactions.
    A complex contains exactly a connected component of intermolecular pairs;
    encounter complexes without pairs are not hidden additional states.
    """

    construct: NucleicAcidConstruct
    pairs: tuple[tuple[NucleotideKey, NucleotideKey], ...]
    partition: StrandComplexPartition
    numeric_pairs: tuple[tuple[int, int], ...]

    def __init__(
        self,
        construct: NucleicAcidConstruct,
        pairs: tuple[tuple[NucleotideKey, NucleotideKey], ...] = (),
        *,
        partition: StrandComplexPartition | None = None,
    ):
        keys = construct.nucleotide_keys
        index = {key: i for i, key in enumerate(keys)}
        if any(
            len(pair) != 2 or pair[0] not in index or pair[1] not in index
            for pair in pairs
        ):
            raise ValueError("Every pair must bind two keys from this construct.")
        numeric = tuple(
            sorted(
                tuple(sorted((index[first], index[second]))) for first, second in pairs
            )
        )
        used = tuple(i for pair in numeric for i in pair)
        if len(used) != len(set(used)):
            raise ValueError(
                "A nucleotide cannot self-pair or have multiple pairing partners."
            )
        if not _noncrossing(numeric):
            raise ValueError("Ordered-planar secondary kinetics refuses crossing pairs.")
        induced = _partition(construct, numeric)
        if partition is not None and partition != induced:
            raise ValueError(
                "Declared complex partition disagrees with pair connectivity."
            )
        object.__setattr__(self, "construct", construct)
        object.__setattr__(self, "numeric_pairs", numeric)
        object.__setattr__(self, "pairs", tuple((keys[i], keys[j]) for i, j in numeric))
        object.__setattr__(self, "partition", induced)

    @property
    def pair_count(self) -> int:
        return len(self.pairs)

    def fingerprint(self) -> str:
        return canonical_fingerprint(
            (
                "secondary-structure-state",
                self.construct.fingerprint(),
                self.numeric_pairs,
                self.partition.fingerprint(),
            )
        )


@dataclass(frozen=True, slots=True)
class SecondaryMove:
    """A reversible pair toggle; join/split is derived, never guessed."""

    kind: str
    pair: tuple[NucleotideKey, NucleotideKey]
    before: SecondaryStructureState
    after: SecondaryStructureState


__all__ = ["SecondaryMove", "SecondaryStructureState", "StrandComplexPartition"]
