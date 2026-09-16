"""Host-only, schema-specific HEP interchange adapters."""

from ._accelerator import accelerator_bunch_from_openpmd_columns, write_madx_sequence
from ._analysis import (
    binned_likelihood_to_pyhf_workspace,
    pyhf_workspace_json,
    weighted_histogram_to_hepdata,
)
from ._calorimeter import (
    CaloChallengeImport,
    CaloChallengeProfile,
    import_calochallenge_hdf5,
)
from ._columnar import (
    columnar_host_event_report,
    host_events_from_awkward,
    host_events_from_records,
    host_events_to_awkward,
    host_events_to_records,
)
from ._common import (
    HEPColumnProfile,
    HEPExportResult,
    HEPImportResult,
    HEPOptionalDependencyError,
    import_event_columns,
)
from ._framework import DataTierReference, FrameworkEventContext
from ._hepmc import write_hepmc3_ascii
from ._lhef import read_lhef, write_lhef
from ._operations import (
    HEPDatasetSnapshot,
    HEPFileReplica,
    HEPPreservationBundle,
    HEPSoftwareEnvironment,
    HEPWorkloadExport,
    WorkloadBackend,
)
from ._root import read_root_event_tree
from ._slha import (
    parse_slha,
    serialize_slha,
    SLHABlock,
    SLHADecay,
    SLHADecayChannel,
    SLHADiagnostics,
    SLHADocument,
    SLHAEntry,
    spectrum_observables_from_slha,
)
from ._slha2 import (
    interpret_slha2,
    SLHA2Matrix,
    SLHA2QuantumNumbers,
    SLHA2SemanticModel,
)


__all__ = [
    "CaloChallengeImport",
    "CaloChallengeProfile",
    "DataTierReference",
    "FrameworkEventContext",
    "HEPDatasetSnapshot",
    "HEPFileReplica",
    "HEPPreservationBundle",
    "HEPSoftwareEnvironment",
    "HEPWorkloadExport",
    "HEPColumnProfile",
    "HEPExportResult",
    "HEPImportResult",
    "HEPOptionalDependencyError",
    "binned_likelihood_to_pyhf_workspace",
    "accelerator_bunch_from_openpmd_columns",
    "columnar_host_event_report",
    "host_events_from_awkward",
    "host_events_from_records",
    "host_events_to_awkward",
    "host_events_to_records",
    "import_calochallenge_hdf5",
    "import_event_columns",
    "interpret_slha2",
    "parse_slha",
    "read_lhef",
    "read_root_event_tree",
    "pyhf_workspace_json",
    "serialize_slha",
    "SLHABlock",
    "SLHADecay",
    "SLHADecayChannel",
    "SLHADiagnostics",
    "SLHADocument",
    "SLHAEntry",
    "spectrum_observables_from_slha",
    "SLHA2Matrix",
    "SLHA2QuantumNumbers",
    "SLHA2SemanticModel",
    "write_hepmc3_ascii",
    "write_lhef",
    "write_madx_sequence",
    "WorkloadBackend",
    "weighted_histogram_to_hepdata",
]
