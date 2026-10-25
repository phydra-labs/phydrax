#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Optional RDKit host graph admission for molecular crystal components.

RDKit is used only as a host molecular-graph container and SMILES parser.  It
never supplies conformer embedding, force-field optimization, masses,
radii, symmetry or packing: those are native (`phydrax.atomistic.crystal`) or
caller-admitted data.  Every source semantic is either converted exactly,
recorded as an `AdapterLoss`, or refused with an `AdapterError`:

* explicit hydrogens are required, or added on request by RDKit's valence
  model (recorded as a synthesized loss);
* bond types map to orders 1/2/3/aromatic; dative, zero-order and query
  bonds are refused;
* tetrahedral chiral tags on four-coordinate atoms convert by the RDKit
  convention (looking from the first bonded neighbor, ``CCW`` lists the other
  neighbors anticlockwise), giving parity ``+1`` for the first three bonded
  neighbors; three-coordinate and non-tetrahedral tags are refused;
* ``STEREOCIS``/``STEREOTRANS`` convert exactly through their stereo atoms;
  CIP-labeled ``STEREOE``/``STEREOZ`` are refused unless
  ``ez_conversion="legacy-cip-rank"`` explicitly admits RDKit's legacy CIP
  ranks (recorded as an assumption);
* alternatively ``stereo_source="conformer"`` measures every stereo element
  from a 3D source conformer;
* isotopes, enhanced stereo groups other than absolute, dummy/query atoms and
  multiple fragments are refused.
"""

from __future__ import annotations

import importlib
import importlib.util
from collections.abc import Sequence
from types import ModuleType
from typing import Any, assert_never, Literal, TypeAlias

import numpy as np

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ...interchange import AdapterError, AdapterLoss, AdapterReport, AdapterStatus
from ...typing import parse
from ..crystal._input import (
    AROMATIC_BOND,
    ElementTable,
    MolecularComponent,
    MolecularConformer,
    UndeclaredStereoPolicy,
    ValencePolicy,
)


RDKitStereoSource: TypeAlias = Literal["tags", "conformer"]
RDKitEZConversion: TypeAlias = Literal["refuse", "legacy-cip-rank"]

_SOURCE_FORMAT = "rdkit-mol"
_TARGET_FORMAT = "phydrax.atomistic.crystal.MolecularComponent"


def is_rdkit_available() -> bool:
    """Return whether RDKit can be imported, without importing it."""
    return importlib.util.find_spec("rdkit") is not None


def require_rdkit() -> ModuleType:
    """Import ``rdkit.Chem`` or refuse with an optional-dependency status."""
    if not is_rdkit_available():
        raise AdapterError(
            AdapterStatus.OPTIONAL_DEPENDENCY_UNAVAILABLE,
            "RDKit is not installed; it is an optional host graph parser.",
        )
    return importlib.import_module("rdkit.Chem")


def _refuse(message: str, /) -> AdapterError:
    return AdapterError(AdapterStatus.UNSUPPORTED_REQUIRED_SEMANTIC, message)


def _bond_order(chem: ModuleType, bond: Any, /) -> int:
    kind = bond.GetBondType()
    types = chem.rdchem.BondType
    if kind == types.SINGLE:
        return 1
    if kind == types.DOUBLE:
        return 2
    if kind == types.TRIPLE:
        return 3
    if kind == types.AROMATIC:
        return AROMATIC_BOND
    raise _refuse(f"RDKit bond type {kind} has no admitted order.")


def _tetrahedral_from_tags(
    chem: ModuleType, molecule: Any, /
) -> list[tuple[int, int, int, int, int]]:
    tags = chem.rdchem.ChiralType
    entries: list[tuple[int, int, int, int, int]] = []
    for atom in molecule.GetAtoms():
        tag = atom.GetChiralTag()
        if tag == tags.CHI_UNSPECIFIED:
            continue
        if tag not in (tags.CHI_TETRAHEDRAL_CW, tags.CHI_TETRAHEDRAL_CCW):
            raise _refuse(f"Atom {atom.GetIdx()} has non-tetrahedral chiral tag {tag}.")
        index = atom.GetIdx()
        neighbors = [bond.GetOtherAtomIdx(index) for bond in atom.GetBonds()]
        if len(neighbors) != 4:
            raise _refuse(
                f"Atom {index} carries a chiral tag with {len(neighbors)} explicit "
                "neighbors; only four-coordinate tags convert unambiguously."
            )
        sign = 1 if tag == tags.CHI_TETRAHEDRAL_CCW else -1
        entries.append((index, neighbors[0], neighbors[1], neighbors[2], sign))
    return entries


def _double_bonds_from_tags(
    chem: ModuleType,
    molecule: Any,
    ez_conversion: RDKitEZConversion,
    assumptions: list[str],
    /,
) -> list[tuple[int, int, int, int, int]]:
    stereo = chem.rdchem.BondStereo
    entries: list[tuple[int, int, int, int, int]] = []
    ranks: list[int] | None = None
    for bond in molecule.GetBonds():
        kind = bond.GetStereo()
        if kind in (stereo.STEREONONE, stereo.STEREOANY):
            continue
        begin, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if kind in (stereo.STEREOCIS, stereo.STEREOTRANS):
            first, last = tuple(bond.GetStereoAtoms())
            entries.append(
                (first, begin, end, last, 1 if kind == stereo.STEREOCIS else -1)
            )
            continue
        if kind not in (stereo.STEREOE, stereo.STEREOZ):
            raise _refuse(f"Bond {begin}-{end} has unsupported stereo {kind}.")
        if ez_conversion == "refuse":
            raise _refuse(
                f"Bond {begin}-{end} carries a CIP-labeled {kind}; supply STEREOCIS/"
                "STEREOTRANS, stereo_source='conformer', or "
                "ez_conversion='legacy-cip-rank'."
            )
        if ranks is None:
            chem.AssignStereochemistry(molecule, cleanIt=True, force=True)
            ranks = [int(atom.GetProp("_CIPRank")) for atom in molecule.GetAtoms()]
            assumptions.append(
                "E/Z labels were converted with RDKit legacy CIP ranks (_CIPRank)."
            )
        substituents = []
        for atom, other in ((begin, end), (end, begin)):
            options = [
                neighbor.GetIdx()
                for neighbor in molecule.GetAtomWithIdx(atom).GetNeighbors()
                if neighbor.GetIdx() != other
            ]
            ordered = sorted(options, key=lambda value: ranks[value], reverse=True)
            if len(ordered) == 2 and ranks[ordered[0]] == ranks[ordered[1]]:
                raise _refuse(f"Bond {begin}-{end} substituents have equal CIP rank.")
            substituents.append(ordered[0])
        entries.append(
            (
                substituents[0],
                begin,
                end,
                substituents[1],
                1 if kind == stereo.STEREOZ else -1,
            )
        )
    return entries


def _stereo_from_geometry(
    component: MolecularComponent, positions: np.ndarray, /
) -> tuple[list[tuple[int, int, int, int, int]], list[tuple[int, int, int, int, int]]]:
    tetrahedral = []
    for center in component.unspecified_tetrahedral:
        first, second, third = sorted(
            atom
            for bond in component.bonds
            if center in bond
            for atom in bond
            if atom != center
        )[:3]
        volume = np.linalg.det(positions[[first, second, third]] - positions[center])
        if abs(volume) < 1.0e-3:
            raise _refuse(f"Conformer stereo at atom {center} is degenerate (planar).")
        tetrahedral.append((center, first, second, third, 1 if volume > 0.0 else -1))
    double = []
    for begin, end in component.unspecified_double_bonds:
        first = min(
            atom
            for bond in component.bonds
            if begin in bond
            for atom in bond
            if atom not in (begin, end)
        )
        last = min(
            atom
            for bond in component.bonds
            if end in bond
            for atom in bond
            if atom not in (begin, end)
        )
        axis = positions[end] - positions[begin]
        axis = axis / np.linalg.norm(axis)
        left = positions[first] - positions[begin]
        right = positions[last] - positions[end]
        left = left - np.dot(left, axis) * axis
        right = right - np.dot(right, axis) * axis
        cosine = float(
            np.dot(left, right) / (np.linalg.norm(left) * np.linalg.norm(right))
        )
        if abs(cosine) < 0.5:
            raise _refuse(
                f"Conformer double bond {begin}={end} is twisted beyond 60 degrees."
            )
        double.append((first, begin, end, last, 1 if cosine > 0.0 else -1))
    return tetrahedral, double


def molecular_component_from_rdkit(
    molecule: Any,
    /,
    *,
    masses: ElementTable | Sequence[float],
    spin_multiplicity: int,
    add_hydrogens: bool = False,
    stereo_source: RDKitStereoSource = "tags",
    conformer_id: int = -1,
    ez_conversion: RDKitEZConversion = "refuse",
    undeclared_stereo: UndeclaredStereoPolicy = "refuse",
    valence: ValencePolicy = "standard",
    name: str = "molecule",
    source_id: str | None = None,
) -> tuple[MolecularComponent, Any, AdapterReport]:
    """Admit one connected RDKit molecule as a `MolecularComponent`.

    Returns the component, the (possibly hydrogen-completed) RDKit molecule
    whose atom order the component uses, and the adapter report.
    """
    chem = require_rdkit()
    if not isinstance(molecule, chem.Mol):
        raise TypeError("molecule must be an rdkit.Chem.Mol.")
    source = parse(stereo_source, RDKitStereoSource, "stereo_source")
    conversion = parse(ez_conversion, RDKitEZConversion, "ez_conversion")
    if not isinstance(add_hydrogens, bool):
        raise TypeError("add_hydrogens must be a bool.")
    losses: list[AdapterLoss] = []
    assumptions: list[str] = []
    if molecule.GetNumAtoms() == 0 or len(chem.GetMolFrags(molecule)) != 1:
        raise _refuse("The RDKit molecule must be nonempty and connected.")
    if any(atom.GetNumImplicitHs() for atom in molecule.GetAtoms()):
        if not add_hydrogens:
            raise _refuse(
                "The RDKit molecule has implicit hydrogens; complete them explicitly or "
                "pass add_hydrogens=True."
            )
        molecule = chem.AddHs(molecule, addCoords=False)
        losses.append(
            AdapterLoss(
                "hydrogens",
                "import",
                "synthesized",
                "Implicit hydrogens were made explicit by RDKit's valence model.",
                changes_interpretation=False,
            )
        )
    for atom in molecule.GetAtoms():
        if atom.GetAtomicNum() < 1:
            raise _refuse(f"Atom {atom.GetIdx()} is a dummy or query atom.")
        if atom.GetIsotope():
            raise _refuse(f"Atom {atom.GetIdx()} carries isotope {atom.GetIsotope()}.")
        if atom.HasQuery():
            raise _refuse(f"Atom {atom.GetIdx()} is a query atom.")
    groups = molecule.GetStereoGroups()
    if any(
        group.GetGroupType() != chem.rdchem.StereoGroupType.STEREO_ABSOLUTE
        for group in groups
    ):
        raise _refuse("Enhanced relative (AND/OR) stereo groups are not admitted.")
    numbers = [atom.GetAtomicNum() for atom in molecule.GetAtoms()]
    charges = [atom.GetFormalCharge() for atom in molecule.GetAtoms()]
    bonds = [
        (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()) for bond in molecule.GetBonds()
    ]
    orders = [_bond_order(chem, bond) for bond in molecule.GetBonds()]
    positions: np.ndarray | None = None
    match source:
        case "tags":
            tetrahedral = _tetrahedral_from_tags(chem, molecule)
            double = _double_bonds_from_tags(chem, molecule, conversion, assumptions)
            component = MolecularComponent(
                numbers,
                bonds,
                bond_orders=orders,
                masses=masses,
                spin_multiplicity=spin_multiplicity,
                formal_charges=charges,
                tetrahedral_stereo=tetrahedral,
                double_bond_stereo=double,
                undeclared_stereo=undeclared_stereo,
                valence=valence,
                name=name,
            )
        case "conformer":
            if molecule.GetNumConformers() == 0:
                raise _refuse("stereo_source='conformer' requires a source conformer.")
            conformer = molecule.GetConformer(conformer_id)
            if not conformer.Is3D():
                raise _refuse("stereo_source='conformer' requires a 3D conformer.")
            positions = np.asarray(conformer.GetPositions(), dtype=np.float64)
            probe = MolecularComponent(
                numbers,
                bonds,
                bond_orders=orders,
                masses=masses,
                spin_multiplicity=spin_multiplicity,
                formal_charges=charges,
                undeclared_stereo="unspecified",
                valence=valence,
                name=name,
            )
            tetrahedral, double = _stereo_from_geometry(probe, positions)
            assumptions.append(
                "Stereo of every detected stereo candidate was measured from the "
                "source conformer (angstrom); RDKit chiral tags were not used."
            )
            component = MolecularComponent(
                numbers,
                bonds,
                bond_orders=orders,
                masses=masses,
                spin_multiplicity=spin_multiplicity,
                formal_charges=charges,
                tetrahedral_stereo=tetrahedral,
                double_bond_stereo=double,
                undeclared_stereo=undeclared_stereo,
                valence=valence,
                name=name,
            )
        case _ as unreachable:
            assert_never(unreachable)
    if molecule.GetPropsAsDict():
        losses.append(
            AdapterLoss(
                "properties",
                "import",
                "dropped",
                "RDKit molecule properties were not retained: "
                + ", ".join(sorted(molecule.GetPropsAsDict())),
                changes_interpretation=False,
            )
        )
    resolved_source = (
        canonical_fingerprint(
            {
                "kind": "rdkit-mol-source",
                "molblock": chem.MolToMolBlock(molecule),
                "positions": None
                if positions is None
                else array_tree_fingerprint(positions),
            }
        )
        if source_id is None
        else str(source_id)
    )
    report = AdapterReport(
        AdapterStatus.DECLARED_LOSS if losses else AdapterStatus.LOSSLESS,
        _SOURCE_FORMAT,
        _TARGET_FORMAT,
        source_id=resolved_source,
        target_id=component.component_id,
        preserved_fields=(
            "atomic-numbers",
            "formal-charges",
            "bonds",
            "bond-orders",
            "atom-order",
            "tetrahedral-stereo",
            "double-bond-stereo",
        ),
        assumptions=(
            "Masses are caller-admitted; RDKit periodic-table data is not used.",
            *assumptions,
        ),
        losses=losses,
    )
    return component, molecule, report


def molecular_component_from_smiles(
    smiles: str,
    /,
    *,
    masses: ElementTable | Sequence[float],
    spin_multiplicity: int,
    ez_conversion: RDKitEZConversion = "refuse",
    undeclared_stereo: UndeclaredStereoPolicy = "refuse",
    valence: ValencePolicy = "standard",
    name: str = "molecule",
) -> tuple[MolecularComponent, Any, AdapterReport]:
    """Parse SMILES with RDKit and admit it with explicit hydrogens.

    Hydrogens are completed by RDKit (recorded as a loss); stereo comes from
    the SMILES tags.  No coordinates are produced: use the native conformer
    workflow for geometry.
    """
    chem = require_rdkit()
    if not isinstance(smiles, str) or not smiles.strip():
        raise ValueError("smiles must be a non-empty string.")
    molecule = chem.MolFromSmiles(smiles)
    if molecule is None:
        raise AdapterError(AdapterStatus.MALFORMED_SOURCE, f"Invalid SMILES {smiles!r}.")
    return molecular_component_from_rdkit(
        molecule,
        masses=masses,
        spin_multiplicity=spin_multiplicity,
        add_hydrogens=True,
        ez_conversion=ez_conversion,
        undeclared_stereo=undeclared_stereo,
        valence=valence,
        name=name,
        source_id=canonical_fingerprint({"kind": "smiles-source", "smiles": smiles}),
    )


def conformers_from_rdkit(
    component: MolecularComponent,
    molecule: Any,
    /,
    *,
    provenance: str,
    minimum_distance: float = 0.5,
) -> tuple[MolecularConformer, ...]:
    """Admit every 3D RDKit conformer as an input `MolecularConformer`.

    ``molecule`` must be the atom-ordered molecule returned with
    ``component``; coordinates are read as angstrom.  Geometries that violate
    the component's declared stereo are refused by `MolecularConformer`.
    """
    chem = require_rdkit()
    if not isinstance(molecule, chem.Mol):
        raise TypeError("molecule must be an rdkit.Chem.Mol.")
    if not isinstance(component, MolecularComponent):
        raise TypeError("component must be a MolecularComponent.")
    numbers = tuple(atom.GetAtomicNum() for atom in molecule.GetAtoms())
    if numbers != component.atomic_numbers:
        raise ValueError("molecule atom order does not match the component.")
    conformers = []
    for conformer in molecule.GetConformers():
        if not conformer.Is3D():
            raise _refuse(f"RDKit conformer {conformer.GetId()} is not 3D.")
        conformers.append(
            MolecularConformer(
                component,
                np.asarray(conformer.GetPositions(), dtype=np.float64),
                provenance=f"{provenance} rdkit-conformer={conformer.GetId()}",
                source="input",
                minimum_distance=minimum_distance,
            )
        )
    if not conformers:
        raise _refuse("The RDKit molecule has no conformers.")
    return tuple(conformers)


__all__ = [
    "RDKitEZConversion",
    "RDKitStereoSource",
    "conformers_from_rdkit",
    "is_rdkit_available",
    "molecular_component_from_rdkit",
    "molecular_component_from_smiles",
    "require_rdkit",
]
