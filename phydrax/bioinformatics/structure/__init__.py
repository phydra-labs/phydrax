#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Macromolecular records, topology, lowering, and structural analysis."""

from ._alignment import (
    align_coordinates,
    align_structure_models,
    RigidAlignmentResult,
)
from ._ensemble import (
    analyze_structure_ensemble,
    StructureEnsembleResult,
)
from ._interfaces import (
    analyze_chain_interfaces,
    ChainInterfaceResult,
)
from ._lowering import (
    lower_macromolecular_record,
    StructureLoweringPlan,
    StructureLoweringResult,
)
from ._record import (
    AssemblyGenerator,
    AssemblyOperation,
    AtomRecord,
    BondRecord,
    ChainRecord,
    ChemicalComponent,
    ChemicalComponentAtom,
    ChemicalComponentBond,
    EntityRecord,
    MacromolecularRecord,
    MissingAtomRecord,
    MissingResidueRecord,
    ResidueRecord,
)
from ._secondary import (
    assign_geometric_secondary_structure,
    ContactAnalysisPlan,
    residue_contacts,
    ResidueContactResult,
    SecondaryStructureResult,
)
from ._topology import (
    MacromolecularStructure,
)
from ._types import (
    AlignmentStatus,
    BondOrder,
    ConnectionKind,
    EntityKind,
    PolymerKind,
    SecondaryStructureKind,
    StructureStatus,
)


__all__ = [
    "align_coordinates",
    "align_structure_models",
    "AlignmentStatus",
    "analyze_chain_interfaces",
    "analyze_structure_ensemble",
    "AssemblyGenerator",
    "AssemblyOperation",
    "assign_geometric_secondary_structure",
    "AtomRecord",
    "BondOrder",
    "BondRecord",
    "ChainInterfaceResult",
    "ChainRecord",
    "ChemicalComponent",
    "ChemicalComponentAtom",
    "ChemicalComponentBond",
    "ConnectionKind",
    "ContactAnalysisPlan",
    "EntityKind",
    "EntityRecord",
    "lower_macromolecular_record",
    "MacromolecularRecord",
    "MacromolecularStructure",
    "MissingAtomRecord",
    "MissingResidueRecord",
    "PolymerKind",
    "residue_contacts",
    "ResidueContactResult",
    "ResidueRecord",
    "RigidAlignmentResult",
    "SecondaryStructureKind",
    "SecondaryStructureResult",
    "StructureEnsembleResult",
    "StructureLoweringPlan",
    "StructureLoweringResult",
    "StructureStatus",
]
