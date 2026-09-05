# Copyright © 2026 PHYDRA, Inc. All rights reserved.
from __future__ import annotations

from dataclasses import dataclass

from ..._fingerprint import canonical_fingerprint


_CANONICAL = frozenset("ACDEFGHIKLMNPQRSTVWY")


def _identifier(value: str, name: str) -> None:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError(f"{name} must be a nonempty canonical string.")


@dataclass(frozen=True, slots=True, order=True)
class ResidueKey:
    """Construct-local identity; zero-based sequence position, not author numbering."""

    chain_id: str
    position: int

    def __post_init__(self):
        _identifier(self.chain_id, "chain_id")
        if (
            isinstance(self.position, bool)
            or not isinstance(self.position, int)
            or self.position < 0
        ):
            raise ValueError("position must be a nonnegative integer.")


@dataclass(frozen=True, slots=True)
class ProteinConstruct:
    """Ordered canonical amino-acid chains, independent of coordinate coverage."""

    chain_ids: tuple[str, ...]
    sequences: tuple[str, ...]

    def __post_init__(self):
        if not isinstance(self.chain_ids, tuple) or not isinstance(self.sequences, tuple):
            raise TypeError("chain_ids and sequences must be immutable tuples.")
        if not self.chain_ids or len(self.chain_ids) != len(self.sequences):
            raise ValueError("Each declared chain needs one nonempty sequence.")
        if len(set(self.chain_ids)) != len(self.chain_ids):
            raise ValueError("Chain/copy identities must be unique.")
        for chain, sequence in zip(self.chain_ids, self.sequences, strict=True):
            _identifier(chain, "chain_id")
            if (
                not isinstance(sequence, str)
                or not sequence
                or set(sequence) - _CANONICAL
            ):
                raise ValueError(
                    "Only explicit uppercase canonical amino-acid sequences are supported."
                )

    @property
    def residue_keys(self) -> tuple[ResidueKey, ...]:
        return tuple(
            ResidueKey(chain, position)
            for chain, sequence in zip(self.chain_ids, self.sequences, strict=True)
            for position in range(len(sequence))
        )

    @property
    def residue_count(self) -> int:
        return sum(map(len, self.sequences))

    def fingerprint(self) -> str:
        return canonical_fingerprint(
            {
                "kind": "protein-construct",
                "chains": list(zip(self.chain_ids, self.sequences, strict=True)),
            }
        )


@dataclass(frozen=True, slots=True, order=True)
class ProteinAtomKey:
    residue: ResidueKey
    atom_name: str

    def __post_init__(self):
        if not isinstance(self.residue, ResidueKey):
            raise TypeError("residue must be a ResidueKey.")
        _identifier(self.atom_name, "atom_name")

    def record(self) -> tuple[str, int, str]:
        return self.residue.chain_id, self.residue.position, self.atom_name


__all__ = ["ProteinConstruct", "ResidueKey", "ProteinAtomKey"]
