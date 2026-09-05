"""Optional host-side atomistic adapters.

Structure, force-field, trajectory, and assembly boundaries remain lazy imports.
"""

from ._adapters import (
    force_field_from_mapping,
    force_field_to_mapping,
    from_openff_interchange,
    from_openmm_system,
    from_parmed_structure,
    to_openff_interchange,
    to_openmm_system,
)
from ._ase import (
    ASE_PARTICLE_ID_ARRAY,
    ASE_SOURCE_ID_INFO,
    from_ase_atoms,
    is_ase_available,
    require_ase,
    to_ase_atoms,
)
from ._core import (
    AtomisticInterchangeBundle,
    AtomisticInterchangeReport,
    UnsupportedAtomisticContentError,
)
from ._ipi import (
    IPIListener,
    IPIRequest,
    IPIResponse,
    IPISession,
    IPITransportPlan,
    IPITransportStatus,
    serve_ipi_once,
    TransportedExternalAtomisticProvider,
)
from ._mdanalysis import (
    atomistic_frame_from_mdanalysis,
    atomistic_metadata_from_mdanalysis,
    mdanalysis_selection,
    mdanalysis_universe_from_frames,
)
from ._packmol import (
    PackmolAssemblyPlan,
    PackmolAssemblyResult,
    PackmolComponentPlan,
    PackmolRegionConstraint,
)
from ._structure_records import (
    PDBAtomRecord,
    read_pdb_atom_records,
    select_pdb_model,
)
from ._trajectory_io import ExtendedXYZTrajectoryPlan, H5MDTrajectoryPlan


__all__ = [
    "ASE_PARTICLE_ID_ARRAY",
    "ASE_SOURCE_ID_INFO",
    "AtomisticInterchangeBundle",
    "atomistic_frame_from_mdanalysis",
    "atomistic_metadata_from_mdanalysis",
    "AtomisticInterchangeReport",
    "ExtendedXYZTrajectoryPlan",
    "H5MDTrajectoryPlan",
    "IPIListener",
    "IPIRequest",
    "IPIResponse",
    "IPISession",
    "IPITransportPlan",
    "IPITransportStatus",
    "PackmolAssemblyPlan",
    "PackmolAssemblyResult",
    "PackmolComponentPlan",
    "PackmolRegionConstraint",
    "PDBAtomRecord",
    "TransportedExternalAtomisticProvider",
    "mdanalysis_selection",
    "mdanalysis_universe_from_frames",
    "UnsupportedAtomisticContentError",
    "force_field_from_mapping",
    "from_ase_atoms",
    "force_field_to_mapping",
    "from_openff_interchange",
    "from_openmm_system",
    "from_parmed_structure",
    "is_ase_available",
    "require_ase",
    "read_pdb_atom_records",
    "select_pdb_model",
    "serve_ipi_once",
    "to_openff_interchange",
    "to_openmm_system",
    "to_ase_atoms",
]
