"""Optional host-side atomistic adapters.

Structure, force-field, trajectory, and assembly boundaries remain lazy imports.
The ASE calculator resolves lazily because its module imports ASE.
"""

from importlib import import_module
from typing import Any

from ..._jax_config import process_numerics
from ._adapters import (
    force_field_from_mapping,
    force_field_to_mapping,
    from_openff_interchange,
    from_openmm_system,
    from_parmed_structure,
    to_openff_interchange,
    to_openmm_system,
)
from ._aimnet2_checkpoint import (
    AIMNET2_PROVIDER_RELEASES,
    AIMNET2_SOURCE_PIN,
    AIMNet2CheckpointConversion,
    AIMNet2ExtractionLimits,
    AIMNet2LegacyOverrides,
    AIMNet2OracleConfiguration,
    AIMNet2OracleEvaluation,
    AIMNet2ProviderRuntime,
    AIMNet2Source,
    AIMNet2SourceFormat,
    AIMNet2SourceProvenance,
    AIMNet2SourceRecord,
    AIMNet2SourceRefusedError,
    convert_aimnet2_checkpoint,
    evaluate_aimnet2_source,
    extract_aimnet2_source,
    read_safetensors_tensors,
    read_torch_v2_source,
)
from ._ase import (
    ASE_PARTICLE_ID_ARRAY,
    ASE_SOURCE_ID_INFO,
    from_ase_atoms,
    is_ase_available,
    require_ase,
    to_ase_atoms,
)
from ._cif import (
    ChargedTypeSymbolPolicy,
    cif_crystal_structure,
    CIFAtomDim,
    CIFBond,
    CIFCrystalStructure,
    CIFSiteDim,
    parse_symmetry_operation,
)
from ._core import (
    AtomisticInterchangeBundle,
    AtomisticInterchangeReport,
    UnsupportedAtomisticContentError,
)
from ._ipi import (
    IPIInverseCellPolicy,
    IPIListener,
    IPIRequest,
    IPISession,
    IPITransportMode,
    IPITransportPlan,
    IPITransportStatus,
    IPIVirialPolicy,
    serve_ipi_once,
    TransportedExternalAtomisticProvider,
)
from ._mace_checkpoint import (
    convert_mace_checkpoint,
    create_mace_provider_fixture,
    evaluate_mace_source,
    MACE_LES_RELEASES,
    MACE_PROVIDER_RELEASES,
    mace_source_gradients,
    MACECheckpointConversion,
    MACEConversionLimits,
    MACEEvaluationDtype,
    MACEProviderCase,
    MACEProviderConfiguration,
    MACEProviderEvaluation,
    MACEProviderFixture,
    MACEProviderGradients,
    MACEProviderRuntime,
    MACESource,
    MACESourceKind,
    MACESourceProvenance,
    MACESourceRefusedError,
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
from ._rdkit import (
    conformers_from_rdkit,
    is_rdkit_available,
    molecular_component_from_rdkit,
    molecular_component_from_smiles,
    RDKitEZConversion,
    RDKitStereoSource,
    require_rdkit,
)
from ._source_admission import TrustedTorchPickleSource
from ._structure_records import (
    PDBAtomRecord,
    read_pdb_atom_records,
    select_pdb_model,
)
from ._trajectory_io import ExtendedXYZTrajectoryPlan, H5MDTrajectoryPlan
from ._uma_checkpoint import (
    convert_uma_checkpoint,
    extract_uma_checkpoint,
    UMACheckpointConversion,
    UMAConversionLimits,
    UMAExtractionRuntime,
    UMASource,
    UMASourceExtraction,
    UMASourceProvenance,
    UMASourceRefusedError,
)


_ASE_CALCULATOR_EXPORTS = ("NativeASECalculator", "NativeASECalculatorPlan")


def __getattr__(name: str) -> Any:
    with process_numerics():
        if name in _ASE_CALCULATOR_EXPORTS:
            # Resolving an export whose optional provider is absent fails as an import;
            # AdapterError remains the conversion-boundary status for ASE adapters.
            if not is_ase_available():
                raise ImportError(f"{name} requires optional dependency 'ase'.")
            value = getattr(import_module("._ase_calculator", __package__), name)
            globals()[name] = value
            return value
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))


__all__ = [
    "AIMNET2_PROVIDER_RELEASES",
    "AIMNET2_SOURCE_PIN",
    "AIMNet2CheckpointConversion",
    "AIMNet2ExtractionLimits",
    "AIMNet2LegacyOverrides",
    "AIMNet2OracleConfiguration",
    "AIMNet2OracleEvaluation",
    "AIMNet2ProviderRuntime",
    "AIMNet2Source",
    "AIMNet2SourceFormat",
    "AIMNet2SourceProvenance",
    "AIMNet2SourceRecord",
    "AIMNet2SourceRefusedError",
    "ChargedTypeSymbolPolicy",
    "CIFAtomDim",
    "CIFBond",
    "CIFCrystalStructure",
    "CIFSiteDim",
    "MACE_LES_RELEASES",
    "RDKitEZConversion",
    "RDKitStereoSource",
    "UMACheckpointConversion",
    "UMAConversionLimits",
    "UMAExtractionRuntime",
    "UMASource",
    "UMASourceExtraction",
    "UMASourceProvenance",
    "UMASourceRefusedError",
    "cif_crystal_structure",
    "conformers_from_rdkit",
    "convert_aimnet2_checkpoint",
    "convert_uma_checkpoint",
    "evaluate_aimnet2_source",
    "extract_aimnet2_source",
    "extract_uma_checkpoint",
    "is_rdkit_available",
    "molecular_component_from_rdkit",
    "molecular_component_from_smiles",
    "parse_symmetry_operation",
    "read_safetensors_tensors",
    "read_torch_v2_source",
    "require_rdkit",
    "convert_mace_checkpoint",
    "create_mace_provider_fixture",
    "evaluate_mace_source",
    "MACE_PROVIDER_RELEASES",
    "mace_source_gradients",
    "MACECheckpointConversion",
    "MACEConversionLimits",
    "MACEEvaluationDtype",
    "MACEProviderCase",
    "MACEProviderConfiguration",
    "MACEProviderEvaluation",
    "MACEProviderFixture",
    "MACEProviderGradients",
    "MACEProviderRuntime",
    "MACESource",
    "MACESourceKind",
    "MACESourceProvenance",
    "MACESourceRefusedError",
    "TrustedTorchPickleSource",
    "ASE_PARTICLE_ID_ARRAY",
    "ASE_SOURCE_ID_INFO",
    "AtomisticInterchangeBundle",
    "atomistic_frame_from_mdanalysis",
    "atomistic_metadata_from_mdanalysis",
    "AtomisticInterchangeReport",
    "ExtendedXYZTrajectoryPlan",
    "H5MDTrajectoryPlan",
    "IPIInverseCellPolicy",
    "IPIListener",
    "IPIRequest",
    "IPISession",
    "IPITransportMode",
    "IPITransportPlan",
    "IPITransportStatus",
    "IPIVirialPolicy",
    "NativeASECalculator",
    "NativeASECalculatorPlan",
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
