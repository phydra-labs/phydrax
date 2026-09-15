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
from ._species import lookup_particle_species, ParticleSpeciesLookup, ParticleSpeciesTable
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
    "ReproducibilityGrade",
    "WeightVariationKind",
    "SystematicConfiguration",
    "statistics",
    "SystematicKind",
    "SystematicSource",
    "lookup_particle_species",
    "build_hep_qualification_bundle",
    "pack_host_events",
    "summarize_event_weights",
    "merge_process_normalizations",
    "unpack_particle_events",
]
