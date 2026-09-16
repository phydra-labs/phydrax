"""Typed bounded semantics shared by HEP production applications."""

from . import statistics
from ._associations import AssociationTable
from ._capabilities import HEPProviderBinding
from ._events import (
    ParticleEventBatch,
    ParticleEventPlan,
    ParticleEventStatus,
    PreparedParticleEvents,
)
from ._host_events import (
    EventPackingReport,
    EventPackingStatus,
    HostAttribute,
    HostEventPackingResult,
    HostEventRecord,
    HostEventWeight,
    HostParticleRecord,
    HostVertexRecord,
    pack_host_events,
    unpack_particle_events,
)
from ._identity import (
    ParticleCatalogueReference,
    ParticleRole,
    ReproducibilityGrade,
)
from ._operations import (
    BeamConditionSnapshot,
    HEPRunContext,
    merge_process_normalizations,
    ProcessNormalization,
)
from ._qualification import (
    build_hep_qualification_bundle,
    HEP_RELEASE_GATES,
    HEPQualificationBundle,
)
from ._scale_bvp import (
    evaluate_scale_bvp,
    ScaleBVPEvaluation,
    ScaleBVPPlan,
    ScaleBVPResult,
    ScaleBVPStatus,
    solve_scale_bvp,
)
from ._species import lookup_particle_species, ParticleSpeciesLookup, ParticleSpeciesTable
from ._spectrum import (
    SpectrumApproximationProfile,
    SpectrumCalculationResult,
    SpectrumDiagnostics,
    SpectrumObservableTable,
    SpectrumStatus,
)
from ._spectrum_provider import (
    execute_spectrum_provider,
    ExternalSpectrumProvider,
    spectrum_result_from_provider,
    SpectrumProviderExecution,
    SpectrumProviderPlan,
    SpectrumProviderStatus,
)
from ._spectrum_qualification import (
    particle_spectrum_candidate_profiles,
    particle_spectrum_candidate_support_tuples,
)
from ._systematics import SystematicConfiguration, SystematicKind, SystematicSource
from ._weights import (
    CrossSectionLedger,
    EventAccountingStatus,
    EventWeightSet,
    summarize_event_weights,
    WeightVariationKind,
)


__all__ = [
    "AssociationTable",
    "BeamConditionSnapshot",
    "CrossSectionLedger",
    "EventAccountingStatus",
    "EventWeightSet",
    "EventPackingReport",
    "EventPackingStatus",
    "ExternalSpectrumProvider",
    "HEPProviderBinding",
    "HEPRunContext",
    "HEPQualificationBundle",
    "HEP_RELEASE_GATES",
    "HostAttribute",
    "HostEventPackingResult",
    "HostEventRecord",
    "HostEventWeight",
    "HostParticleRecord",
    "HostVertexRecord",
    "ParticleCatalogueReference",
    "ParticleEventBatch",
    "ParticleEventPlan",
    "ParticleEventStatus",
    "ParticleRole",
    "ParticleSpeciesLookup",
    "ParticleSpeciesTable",
    "ProcessNormalization",
    "PreparedParticleEvents",
    "particle_spectrum_candidate_profiles",
    "particle_spectrum_candidate_support_tuples",
    "ReproducibilityGrade",
    "ScaleBVPEvaluation",
    "ScaleBVPPlan",
    "ScaleBVPResult",
    "ScaleBVPStatus",
    "SpectrumApproximationProfile",
    "SpectrumCalculationResult",
    "SpectrumDiagnostics",
    "SpectrumObservableTable",
    "SpectrumProviderExecution",
    "SpectrumProviderPlan",
    "SpectrumProviderStatus",
    "SpectrumStatus",
    "WeightVariationKind",
    "SystematicConfiguration",
    "statistics",
    "SystematicKind",
    "SystematicSource",
    "lookup_particle_species",
    "evaluate_scale_bvp",
    "execute_spectrum_provider",
    "build_hep_qualification_bundle",
    "pack_host_events",
    "summarize_event_weights",
    "merge_process_normalizations",
    "solve_scale_bvp",
    "spectrum_result_from_provider",
    "unpack_particle_events",
]
