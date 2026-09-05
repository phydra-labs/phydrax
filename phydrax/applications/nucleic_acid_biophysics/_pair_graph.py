# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Full annotated interactions; restricted secondary exports never erase edges."""

from __future__ import annotations

from dataclasses import dataclass

from ..._fingerprint import canonical_fingerprint
from ._construct import NucleicAcidConstruct, NucleotideKey


@dataclass(frozen=True, slots=True)
class BaseInteraction:
    left: NucleotideKey
    right: NucleotideKey
    kind: str
    annotation: str
    source_id: str

    def __post_init__(self):
        if self.left == self.right or any(
            not value or value != value.strip()
            for value in (self.kind, self.annotation, self.source_id)
        ):
            raise ValueError(
                "An interaction needs distinct sites and explicit kind, annotation and source."
            )


@dataclass(frozen=True, slots=True)
class BaseInteractionGraph:
    construct: NucleicAcidConstruct
    interactions: tuple[BaseInteraction, ...]

    def __post_init__(self):
        keys = set(self.construct.nucleotide_keys)
        if not isinstance(self.interactions, tuple) or len(set(self.interactions)) != len(
            self.interactions
        ):
            raise ValueError(
                "Interactions must be a tuple of distinct annotated records."
            )
        if any(
            edge.left not in keys or edge.right not in keys for edge in self.interactions
        ):
            raise ValueError("Interaction refers outside the construct.")

    def fingerprint(self) -> str:
        records = sorted(
            (
                e.left.strand_id,
                e.left.position,
                e.right.strand_id,
                e.right.position,
                e.kind,
                e.annotation,
                e.source_id,
            )
            for e in self.interactions
        )
        return canonical_fingerprint(
            {"construct": self.construct.fingerprint(), "interactions": records}
        )

    def to_dot_bracket(self) -> str:
        """Lossless only for linear, noncrossing, one-partner canonical pairs.

        Noncanonical, stacking and multiply annotated interactions remain in the
        graph and cause refusal rather than silently disappearing in this view.
        """
        if any(self.construct.circular):
            raise ValueError("Dot-bracket export cannot retain circular connectivity.")
        order = {key: i for i, key in enumerate(self.construct.nucleotide_keys)}
        pairs = []
        used = set()
        for edge in self.interactions:
            if edge.kind != "pair" or edge.annotation != "canonical":
                raise ValueError(
                    "Dot-bracket cannot retain full interaction annotations."
                )
            i, j = sorted((order[edge.left], order[edge.right]))
            if i in used or j in used:
                raise ValueError("Dot-bracket requires one partner per nucleotide.")
            used.update((i, j))
            pairs.append((i, j))
        if any(a < c < b < d or c < a < d < b for a, b in pairs for c, d in pairs):
            raise ValueError("This dot-bracket profile cannot represent pseudoknots.")
        symbols = ["."] * self.construct.nucleotide_count
        for i, j in pairs:
            symbols[i], symbols[j] = "(", ")"
        result, offset = [], 0
        for sequence in self.construct.sequences:
            result.append("".join(symbols[offset : offset + len(sequence)]))
            offset += len(sequence)
        return "&".join(result)


__all__ = ["BaseInteraction", "BaseInteractionGraph"]
