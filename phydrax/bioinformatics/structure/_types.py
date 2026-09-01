#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from enum import IntEnum, StrEnum


class EntityKind(StrEnum):
    """mmCIF entity classification retained without lossy normalization."""

    POLYMER = "polymer"
    NON_POLYMER = "non-polymer"
    BRANCHED = "branched"
    WATER = "water"
    UNKNOWN = "unknown"


class PolymerKind(StrEnum):
    """Biopolymer chemistry relevant to topology and sequence interpretation."""

    PROTEIN_L = "polypeptide(L)"
    PROTEIN_D = "polypeptide(D)"
    DNA = "polydeoxyribonucleotide"
    RNA = "polyribonucleotide"
    DNA_RNA_HYBRID = "polydeoxyribonucleotide/polyribonucleotide hybrid"
    POLYSACCHARIDE_D = "polysaccharide(D)"
    POLYSACCHARIDE_L = "polysaccharide(L)"
    OTHER = "other"
    NONE = "none"


class BondOrder(IntEnum):
    """Portable covalent bond order; aromaticity is represented independently."""

    UNKNOWN = 0
    SINGLE = 1
    DOUBLE = 2
    TRIPLE = 3
    QUADRUPLE = 4


class ConnectionKind(StrEnum):
    """Semantic origin of a resolved structure connection."""

    COVALENT = "covalent"
    DISULFIDE = "disulfide"
    METAL_COORDINATION = "metal-coordination"
    HYDROGEN_BOND = "hydrogen-bond"
    OTHER = "other"


class StructureStatus(IntEnum):
    """Observable status for structure compilation and bounded analyses."""

    SUCCESS = 0
    INVALID_RECORD = 1
    CAPACITY_EXCEEDED = 2
    UNRESOLVED_CHEMISTRY = 3
    UNRESOLVED_REFERENCE = 4
    NO_VALID_MODEL = 5
    NONFINITE = 6
    DEGENERATE_GEOMETRY = 7
    UNSUPPORTED = 8


class SecondaryStructureKind(IntEnum):
    """Coarse geometric secondary-structure assignment."""

    UNKNOWN = 0
    COIL = 1
    HELIX = 2
    STRAND = 3
    TURN = 4
    RNA_PAIRED = 5
    RNA_UNPAIRED = 6


class AlignmentStatus(IntEnum):
    """Status of a weighted rigid alignment."""

    SUCCESS = 0
    INSUFFICIENT_POINTS = 1
    DEGENERATE = 2
    NONFINITE = 3


__all__ = [
    "AlignmentStatus",
    "BondOrder",
    "ConnectionKind",
    "EntityKind",
    "PolymerKind",
    "SecondaryStructureKind",
    "StructureStatus",
]
