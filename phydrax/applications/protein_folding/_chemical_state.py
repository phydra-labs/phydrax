# Copyright © 2026 PHYDRA, Inc. All rights reserved.
from __future__ import annotations

from dataclasses import dataclass

from ..._fingerprint import canonical_fingerprint
from ._construct import _identifier, ProteinAtomKey, ProteinConstruct


# Chemical nomenclature, not a parameter table. Hydrogens are supplied by the
# declared chemical realization; their names are force-field dependent.
_SIDECHAINS = {
    "A": "CB",
    "R": "CB CG CD NE CZ NH1 NH2",
    "N": "CB CG OD1 ND2",
    "D": "CB CG OD1 OD2",
    "C": "CB SG",
    "E": "CB CG CD OE1 OE2",
    "Q": "CB CG CD OE1 NE2",
    "G": "",
    "H": "CB CG ND1 CD2 CE1 NE2",
    "I": "CB CG1 CG2 CD1",
    "L": "CB CG CD1 CD2",
    "K": "CB CG CD CE NZ",
    "M": "CB CG SD CE",
    "F": "CB CG CD1 CD2 CE1 CE2 CZ",
    "P": "CB CG CD",
    "S": "CB OG",
    "T": "CB OG1 CG2",
    "W": "CB CG CD1 CD2 NE1 CE2 CE3 CZ2 CZ3 CH2",
    "Y": "CB CG CD1 CD2 CE1 CE2 CZ OH",
    "V": "CB CG1 CG2",
}
_ELEMENT = {"C": 6, "N": 7, "O": 8, "S": 16}


@dataclass(frozen=True, slots=True)
class ResolvedProteinChemistry:
    """Caller-audited complete all-atom inventory, never a protonation predictor.

    The admitted profile is uncapped canonical-L single-chain protein without
    disulfides, covalent ligands, solvent, ions or PTMs. Each residue's explicit
    protonation/tautomer label and expected hydrogen count come from the cited
    chemical realization, not from pH or coordinate guessing.
    """

    construct: ProteinConstruct
    atom_keys: tuple[ProteinAtomKey, ...]
    atomic_numbers: tuple[int, ...]
    residue_states: tuple[str, ...]
    hydrogen_counts: tuple[int, ...]
    n_terminal_state: str
    c_terminal_state: str
    source_id: str
    profile: str = "canonical-L-single-chain-explicit"

    def __post_init__(self):
        if (
            self.profile != "canonical-L-single-chain-explicit"
            or len(self.construct.chain_ids) != 1
        ):
            raise ValueError(
                "Unsupported chemistry profile; only uncapped canonical-L single chains are admitted."
            )
        for name, value in (
            ("atom_keys", self.atom_keys),
            ("atomic_numbers", self.atomic_numbers),
            ("residue_states", self.residue_states),
            ("hydrogen_counts", self.hydrogen_counts),
        ):
            if not isinstance(value, tuple):
                raise TypeError(f"{name} must be an immutable tuple.")
        _identifier(self.source_id, "chemical realization source_id")
        if self.n_terminal_state not in ("NH2", "NH3+") or self.c_terminal_state not in (
            "COOH",
            "COO-",
        ):
            raise ValueError(
                "Only explicitly uncapped amine/carboxyl termini are supported."
            )
        residues = self.construct.residue_keys
        if len(self.residue_states) != len(residues) or len(self.hydrogen_counts) != len(
            residues
        ):
            raise ValueError(
                "Protonation labels and hydrogen inventories must cover every residue."
            )
        if len(self.atom_keys) != len(self.atomic_numbers) or len(
            set(self.atom_keys)
        ) != len(self.atom_keys):
            raise ValueError(
                "Chemical atom identities must be unique and match atomic numbers."
            )
        if any(key.residue not in residues for key in self.atom_keys):
            raise ValueError("Chemical atoms reference an undeclared residue.")
        for residue, letter, state, h_count in zip(
            residues,
            self.construct.sequences[0],
            self.residue_states,
            self.hydrogen_counts,
            strict=True,
        ):
            _identifier(state, "residue chemical state")
            if state not in (
                "standard",
                "protonated",
                "deprotonated",
                "delta-tautomer",
                "epsilon-tautomer",
                "thiol",
            ):
                raise ValueError(
                    "Unsupported residue chemistry: modifications and disulfides require a separate profile."
                )
            if letter == "C" and state != "thiol":
                raise ValueError(
                    "Cysteine must explicitly retain its thiol chemistry in this profile."
                )
            if letter == "H" and state not in (
                "protonated",
                "delta-tautomer",
                "epsilon-tautomer",
            ):
                raise ValueError(
                    "Histidine requires an explicit tautomer/protonation state."
                )
            if isinstance(h_count, bool) or not isinstance(h_count, int) or h_count <= 0:
                raise ValueError(
                    "Expected hydrogen counts must be explicitly positive integers."
                )
            atoms = [
                (key.atom_name, number)
                for key, number in zip(self.atom_keys, self.atomic_numbers, strict=True)
                if key.residue == residue
            ]
            required = set(("N CA C O " + _SIDECHAINS[letter]).split())
            if residue == residues[-1]:
                required.add("OXT")
            heavy = {name for name, number in atoms if number != 1}
            if heavy != required:
                raise ValueError(
                    f"Incomplete or unsupported heavy-atom chemistry at {residue}: "
                    f"missing={required - heavy}, extra={heavy - required}."
                )
            if sum(number == 1 for _, number in atoms) != h_count:
                raise ValueError(
                    "Hydrogen inventory differs from the declared chemical realization."
                )
            if any(number != 1 and number != _ELEMENT[name[0]] for name, number in atoms):
                raise ValueError("Atom names and element chemistry disagree.")

    def fingerprint(self) -> str:
        return canonical_fingerprint(
            {
                "kind": "resolved-protein-chemistry",
                "construct": self.construct.fingerprint(),
                "atoms": [
                    (key.record(), z)
                    for key, z in zip(self.atom_keys, self.atomic_numbers, strict=True)
                ],
                "states": self.residue_states,
                "hydrogens": self.hydrogen_counts,
                "termini": (self.n_terminal_state, self.c_terminal_state),
                "source": self.source_id,
                "profile": self.profile,
            }
        )


__all__ = ["ResolvedProteinChemistry"]
