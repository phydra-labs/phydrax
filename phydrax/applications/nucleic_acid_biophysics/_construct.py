# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Host-only canonical DNA/RNA identities and directed covalent connectivity."""

from __future__ import annotations

from dataclasses import dataclass

from ..._fingerprint import canonical_fingerprint


@dataclass(frozen=True, slots=True, order=True)
class NucleotideKey:
    strand_id: str
    position: int

    def __post_init__(self):
        if not self.strand_id or self.strand_id != self.strand_id.strip():
            raise ValueError("strand_id must be a nonempty canonical identifier.")
        if (
            isinstance(self.position, bool)
            or not isinstance(self.position, int)
            or self.position < 0
        ):
            raise ValueError("Nucleotide positions are zero-based nonnegative integers.")


@dataclass(frozen=True, slots=True)
class NucleicAcidConstruct:
    """Declared 5′→3′ strand order; only unmodified canonical DNA/RNA chemistry.

    Circularity closes the last-to-first edge. Linear ends are uncapped declared
    termini, not a protonation/charge assignment. Modified chemistry requires an
    explicit supported realization, never U/T substitution.
    """

    strand_ids: tuple[str, ...]
    sequences: tuple[str, ...]
    polymer_types: tuple[str, ...]
    circular: tuple[bool, ...]

    def __post_init__(self):
        fields = (self.strand_ids, self.sequences, self.polymer_types, self.circular)
        if any(not isinstance(value, tuple) for value in fields):
            raise TypeError("Construct fields must be immutable tuples.")
        if not self.strand_ids or any(
            len(value) != len(self.strand_ids) for value in fields
        ):
            raise ValueError(
                "Each strand requires sequence, polymer type and circularity."
            )
        if len(set(self.strand_ids)) != len(self.strand_ids):
            raise ValueError("Strand identifiers must be unique.")
        for strand, sequence, polymer, circular in zip(*fields, strict=True):
            NucleotideKey(strand, 0)
            if polymer not in ("DNA", "RNA"):
                raise ValueError("Polymer type must explicitly be DNA or RNA.")
            if (
                not isinstance(sequence, str)
                or not sequence
                or set(sequence) - set("ACGT" if polymer == "DNA" else "ACGU")
            ):
                raise ValueError(
                    "Sequence must use canonical uppercase bases of the declared chemistry."
                )
            if not isinstance(circular, bool) or (circular and len(sequence) < 2):
                raise ValueError(
                    "Circularity must be boolean; circular strands need at least two nucleotides."
                )

    @property
    def nucleotide_keys(self) -> tuple[NucleotideKey, ...]:
        return tuple(
            NucleotideKey(strand, i)
            for strand, sequence in zip(self.strand_ids, self.sequences, strict=True)
            for i in range(len(sequence))
        )

    @property
    def nucleotide_count(self) -> int:
        return sum(map(len, self.sequences))

    @property
    def bases(self) -> tuple[str, ...]:
        return tuple(base for sequence in self.sequences for base in sequence)

    @property
    def directed_edges(self) -> tuple[tuple[NucleotideKey, NucleotideKey], ...]:
        return tuple(
            (NucleotideKey(strand, i), NucleotideKey(strand, (i + 1) % len(sequence)))
            for strand, sequence, circular in zip(
                self.strand_ids, self.sequences, self.circular, strict=True
            )
            for i in range(len(sequence) if circular else len(sequence) - 1)
        )

    @property
    def termini(self) -> tuple[tuple[NucleotideKey, str], ...]:
        return tuple(
            end
            for strand, sequence, circular in zip(
                self.strand_ids, self.sequences, self.circular, strict=True
            )
            if not circular
            for end in (
                (NucleotideKey(strand, 0), "5-prime"),
                (NucleotideKey(strand, len(sequence) - 1), "3-prime"),
            )
        )

    def fingerprint(self) -> str:
        return canonical_fingerprint(
            {
                "kind": "nucleic-acid-construct",
                "strands": self.strand_ids,
                "sequences": self.sequences,
                "polymers": self.polymer_types,
                "circular": self.circular,
            }
        )


__all__ = ["NucleotideKey", "NucleicAcidConstruct"]
